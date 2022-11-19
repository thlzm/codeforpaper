#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/5
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
from method.finetune.datadefine.dataset import NliDataset
from method.finetune.model.gpt2_model_ft_head import GPT2ModelFTHead
from method.finetune.tasks.finetune import finetune


def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    # model = GPT2Model(num_tokentypes=0, parallel_output=True)
    model = GPT2ModelFTHead(num_tokentypes=0, parallel_output=True)

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['input_ids', 'label_ids', 'context_lengths']
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
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    context_lengths[context_lengths>=args.seq_length] = args.seq_length-1

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    # # Loss mask for finetune
    # loss_mask_ft = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
    # loss_mask_ft[tokens == tokenizer.pad_id] = 0.0
    # loss_mask_ft = loss_mask_ft.unsqueeze(dim=-1)

    return tokens, labels, target, loss_mask, attention_mask, position_ids, context_lengths


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, target, loss_mask, attention_mask, position_ids, context_lengths = get_batch(data_iterator)
    # print(f"tokens shape: {tokens.shape}")
    timers('batch generator').stop()
    # Forward model.
    # losses = model(tokens, position_ids, attention_mask, labels=labels)
    losses, predicts = model(tokens, position_ids, attention_mask, context_lengths, target=target)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}, target, predicts


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
    finetune(train_valid_test_datasets_provider, model_provider, forward_step, args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})