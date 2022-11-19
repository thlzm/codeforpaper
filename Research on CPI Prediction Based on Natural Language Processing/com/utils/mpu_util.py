#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/24
# @Author: qing
import torch

from megatron.mpu.initialize import get_model_parallel_group, get_model_parallel_world_size, get_model_parallel_rank


def mp_gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_.clone().detach()
    torch.distributed.all_gather(tensor_list, input_, group=get_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output