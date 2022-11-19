#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/16
# @Author: qing
import os
import sys
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import get_args, mpu
from megatron.data.samplers import DistributedBatchSampler
from method.incontext.datadefine.dataset import NliDataset
from method.incontext.tasks.infer_ppl import do_infer_ppl


def make_data_loader(dataset):
    """Buld dataloader given an input dataset."""
    if dataset is None:
        return None
    args = get_args()

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=False,
                                            rank=rank,
                                            world_size=world_size)

    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       collate_fn=dataset.collate)


def load_data():
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0('> building test dataset for nli ...')
    test_ds = NliDataset(args, tokenizer, "dev.json")
    print_rank_0("> finished creating nli incontext dataset ...")

    print_rank_0('> building test dataloader for nli ...')
    test_dataloader = make_data_loader(test_ds)
    print_rank_0("> finished creating nli incontext dataloader ...")

    return test_dataloader


def evaluate_results(loss_all, idx_all, label_all):
    idx_set = list(set(idx_all))

    loss_all = np.array(loss_all)
    idx_all = np.array(idx_all)
    label_all = np.array(label_all)

    print_rank_0(f"loss_all[:75]: {loss_all[:75]}.")
    print_rank_0(f"idx_all[:75]: {idx_all[:75]}.")
    print_rank_0(f"label_all[:75]: {label_all[:75]}.")

    cnt = 0
    for idx in idx_set:
        losses = loss_all[idx_all==idx]
        pred_idx = np.argmin(losses)
        if label_all[idx_all==idx][pred_idx]:
            cnt += 1

    acc = cnt/len(idx_set)

    return acc


def do_infer_ppl_nli():
    loss_all, idx_all, label_all = do_infer_ppl(load_data)
    acc = evaluate_results(loss_all, idx_all, label_all)

    print_rank_0(f"Evaluate acc:\n {acc}.")


if __name__=="__main__":
    do_infer_ppl_nli()
