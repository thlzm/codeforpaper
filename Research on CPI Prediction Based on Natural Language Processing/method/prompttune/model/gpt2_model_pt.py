#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/12
# @Author: qing
import torch

from com.utils.mpu_util import mp_gather
from megatron import mpu
from megatron.model import GPT2Model
from megatron.model.language_model import parallel_lm_logits


def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


class GPT2ModelPT(GPT2Model):

    def __init__(self, num_tokentypes=0, parallel_output=True):
        super().__init__(num_tokentypes=num_tokentypes, parallel_output=parallel_output)

    def forward(self, input_ids, position_ids, attention_mask, labels=None, tokentype_ids=None, layer_past=None,
                get_key_value=False,forward_method_parallel_output=None):

        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        layer_past=layer_past,
                                        get_key_value=get_key_value)

        if get_key_value:
            lm_output, presents = lm_output

        lm_output = torch.add(lm_output,0)

        parallel_output = self.parallel_output
        if forward_method_parallel_output is not None:
            parallel_output = forward_method_parallel_output

        output = parallel_lm_logits(lm_output, self.language_model.embedding.word_embeddings.weight, parallel_output)

        if get_key_value:
            output = [output, presents]

        if labels is None:
            return output
        else:
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = mpu.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)

            if parallel_output:
                output_gather = mp_gather(output)
            else:
                output_gather = output

            return loss, output_gather

