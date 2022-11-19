#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/13
# @Author: qing
import torch
from tqdm import tqdm
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args, mpu
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPT2Model
from megatron.fp16 import FP16_Module
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import get_ltor_masks_and_position_ids


def get_model():
    """Build the model."""
    args = get_args()

    # Build model on cpu.
    model = GPT2Model(num_tokentypes=0, parallel_output=False)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        rank_id = mpu.get_model_parallel_rank()
        parameter_num = sum([p.nelement() for p in model.parameters()])
        print(f" > number of parameters on model parallel rank {rank_id}: {parameter_num}.", flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training."""
    if args.DDP_impl == 'torch':
        i = torch.cuda.current_device()
        model = torchDDP(model, device_ids=[i], output_device=i, process_group=mpu.get_data_parallel_group())
        return model

    if args.DDP_impl == 'local':
        model = LocalDDP(model)
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. Exiting.'.format(args.DDP_impl))

def do_infer_ppl(load_data_fn):
    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    args = get_args()
    tokenizer = get_tokenizer()

    model = get_model()
    load_checkpoint(model, None, None)
    model.eval()

    dataloader = load_data_fn()

    torch.distributed.barrier()

    loss_all, idx_all, label_all = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, disable=(torch.distributed.get_rank() != 0)):
            input_ids = batch['input_ids'].long().cuda()
            loss_mask = batch['loss_mask'].float().cuda()

            label_ids = batch['label_ids'].cuda()
            idxs = batch['idxs'].cuda()

            labels = input_ids[:, 1:].contiguous()
            tokens = input_ids[:, :-1].contiguous()
            loss_mask = loss_mask[:, :-1].contiguous()

            attention_mask, _, position_ids = get_ltor_masks_and_position_ids(tokens,
                                                                              tokenizer.eod,
                                                                              args.reset_position_ids,
                                                                              args.reset_attention_mask,
                                                                              args.eod_mask_loss)

            loss_ = model(tokens, position_ids, attention_mask, labels=labels)

            loss = torch.sum(loss_ * loss_mask, dim=-1) / loss_mask.sum(dim=-1)

            # print(f"loss: {loss}")
            # print(f"loss.shape: {loss.shape}")

            loss_list = [torch.zeros_like(loss) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(loss_list, loss, mpu.get_data_parallel_group())
            loss_list = torch.stack(loss_list, 0).view(-1).tolist()

            idxs_list = [torch.zeros_like(idxs) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(idxs_list, idxs, mpu.get_data_parallel_group())
            idxs_list = torch.stack(idxs_list, 0).view(-1).tolist()

            label_list = [torch.zeros_like(label_ids) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(label_list, label_ids, mpu.get_data_parallel_group())
            label_list = torch.stack(label_list, 0).view(-1).tolist()

            # print(f"loss_list: {loss_list}")
            # print(f"idxs_list: {idxs_list}")
            # print(f"label_list: {label_list}")

            loss_all.extend(loss_list)
            idx_all.extend(idxs_list)
            label_all.extend(label_list)

    return loss_all, idx_all, label_all