#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/6
# @Author: qing
import torch
import torch.nn.functional as F

from torch import nn

from megatron import get_args
from megatron.module import MegatronModule
from megatron.model.utils import init_method_normal
from method.finetune.model.gpt2_model_ft import GPT2ModelFT


class GPT2ModelFTHead(MegatronModule):

    def __init__(self, num_tokentypes=0, parallel_output=True, num_labels=3):
        super(GPT2ModelFTHead, self).__init__()
        args = get_args()

        self.gpt = GPT2ModelFT(num_tokentypes=num_tokentypes, parallel_output=parallel_output)
        self.liner = nn.Linear(args.hidden_size, num_labels, bias=False)

        self.init_method= init_method_normal(args.init_method_std)
        self.init_method(self.liner.weight)

        # for param in self.gpt.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, position_ids, attention_mask, context_lengths, labels=None, target=None, tokentype_ids=None,
                layer_past=None, get_key_value=False, forward_method_parallel_output=None):

        lm_output = self.gpt(input_ids, position_ids, attention_mask)
        # lm_output_ft = torch.stack([torch.mean(lm_output[i, :context_lengths[i][0], :], dim=0) for i in range(lm_output.shape[0])])
        lm_output_ft = torch.stack([lm_output[i, context_lengths[i][0]-1, :] for i in range(lm_output.shape[0])])
        # lm_output_ft = torch.mean(lm_output_ft, dim=0)

        output = self.liner(lm_output_ft)
        predict = torch.max(output.data, 1)[1]

        if target is not None:
            loss = F.cross_entropy(output, target=target.view(-1))
            return loss, predict
        else:
            return output, predict

    # def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
    #
    #     state_dict_ = {}
    #     state_dict_[self._language_model_key] \
    #         = self.language_model.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
    #     return state_dict_
    #
    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.gpt.load_state_dict(state_dict, strict=strict)