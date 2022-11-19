#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/12
# @Author: qing
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import reduce_losses

from method.prompttune.datadefine.dataset import NliDataset
from method.prompttune.model.gpt2_model_pt import GPT2ModelPT
from method.prompttune.tasks.prompt_tune import prompt_tune_train
from method.prompttune.paras.custom_arguments import add_custom_args


def model_provider():
    """Build the model."""

    print_rank_0('building PT model ...')
    model = GPT2ModelPT(num_tokentypes=0, parallel_output=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['input_ids', 'label_ids', 'context_lengths', 'loss_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['input_ids'].long()
    target = data_b['label_ids'].long()
    context_lengths = data_b['context_lengths'].long()
    loss_mask = data_b['loss_mask'].float()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    loss_mask = loss_mask[:, :-1].contiguous()
    context_lengths[context_lengths>=args.seq_length] = args.seq_length-1

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    # loss_mask = torch.ones(tokens.shape, dtype=torch.float, device=tokens.device)
    # loss_mask[tokens == tokenizer.pad_id] = 0.0
    # loss_mask = loss_mask.unsqueeze(dim=-1)
    return tokens, labels, target, loss_mask, attention_mask, position_ids, context_lengths

def get_preds(logits, loss_mask):
    tokenizer = get_tokenizer()
    cand_ids = NliDataset.get_cand_ids(tokenizer)

    output = torch.sum(logits * loss_mask.unsqueeze(-1), 1) / torch.sum(loss_mask, -1).unsqueeze(-1)
    scores = output[:, cand_ids]
    preds = torch.argmax(scores, dim=-1)

    # tensor_list = [torch.zeros_like(preds) for _ in range(mpu.get_data_parallel_world_size())]
    # torch.distributed.all_gather(tensor_list, preds, mpu.get_data_parallel_group())
    # preds = torch.stack(tensor_list, 0).view(-1).cpu().detach().numpy()

    return preds


def forward_step(data_iterator, model, parallel_output=None):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    timers('batch generator').start()
    tokens, labels, target, loss_mask, attention_mask, position_ids, context_lengths = get_batch(data_iterator)
    timers('batch generator').stop()

    loss_, logits = model(tokens, position_ids, attention_mask, labels=labels, forward_method_parallel_output=parallel_output)
    loss = torch.sum(loss_*loss_mask)/loss_mask.sum()
    reduced_loss = reduce_losses([loss])

    # print_rank_0(f"loss_mask shape: {loss_mask.shape}.")
    # print_rank_0(f"tokens shape: {tokens.shape}.")
    # print_rank_0(f"loss_ shape: {loss_.shape}.")
    # print_rank_0(f"loss shape: {loss.shape}.")

    preds = get_preds(logits, loss_mask)

    return loss, {'lm loss': reduced_loss[0]}, target, preds


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0('> building train, validation, and test datasets for pangu ...')

    train_ds = NliDataset(args, tokenizer, "train.json")
    valid_ds = NliDataset(args, tokenizer, "dev.json")
    test_ds = NliDataset(args, tokenizer, "dev.json")
    print_rank_0("> finished creating pangu finetune datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    prompt_tune_train(train_valid_test_datasets_provider,
                      model_provider,
                      forward_step,
                      extra_args_provider=add_custom_args,
                      args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})