#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/10/8
# @Author: qing
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))

from megatron import get_args, mpu
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.fp16 import FP16_Module
from megatron.model import GPT2Model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import get_ltor_masks_and_position_ids

class InferGenerate(object):

    def __init__(self):
        initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

        self.args = get_args()
        self.tokenizer = get_tokenizer()

        self.model = self.get_model()
        load_checkpoint(self.model, None, None)
        self.model.eval()

    def top_k_logits(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        # This function has been mostly taken from huggingface conversational ai code at
        # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        batch_size = logits.size()[0]
        if top_p > 0.0:
            logits = logits.view(batch_size, -1).contiguous()
            for logit in logits:
                sorted_logits, sorted_indices = torch.sort(logit, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logit[indices_to_remove] = filter_value

            logits = logits.view(batch_size, -1).contiguous()

        return logits

    def get_model(self):
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

    def get_single_generate_data(self, input_str, max_len):
        origin_input_ids = self.tokenizer.tokenize(input_str)
        origin_input_ids = np.array(origin_input_ids).reshape(1, -1)
        if origin_input_ids.shape[-1] >= max_len:
            origin_input_ids = origin_input_ids[:, -max_len:]

        bs, valid_length = origin_input_ids.shape
        context_len = valid_length
        pad_length = self.args.seq_length - valid_length
        input_ids = np.pad(origin_input_ids, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, self.tokenizer.pad_id))

        input_ids = torch.tensor(input_ids, dtype=torch.long).contiguous().cuda()
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(input_ids,
                                                                          self.tokenizer.eod,
                                                                          self.args.reset_position_ids,
                                                                          self.args.reset_attention_mask,
                                                                          self.args.eod_mask_loss)
        input_data = (input_ids, position_ids, attention_mask, valid_length)
        return input_data, context_len

    def do_single_generate(self, input_data, temperature=1, top_k=None, top_p=None, max_num=50):
        input_ids, position_ids, attention_mask, valid_length = input_data
        inputs = input_ids

        cnt = 0
        with torch.no_grad():
            while valid_length < self.args.seq_length:
                logits = self.model(inputs, position_ids, attention_mask)

                next_token_logits = logits[:, valid_length - 1, :] / temperature

                if top_k is None and top_p is None:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                else:
                    next_token_logscores = self.top_k_logits(next_token_logits, top_k=top_k, top_p=top_p)
                    probs = F.softmax(next_token_logscores, dim=-1)
                    # probs = next_token_logscores / torch.sum(next_token_logscores, dim=-1)
                    next_token = torch.multinomial(probs.float(), num_samples=1).squeeze(1)

                if next_token[0] == self.tokenizer.eod or valid_length == self.args.seq_length - 1 or cnt >= max_num:
                    outputs = input_ids
                    break

                input_ids[:, valid_length] = next_token
                valid_length += 1
                cnt += 1

        return outputs

    def generate(self, input_str, temperature=1, top_k=None, top_p=None, max_len=1000, max_num=50):
        input_data, context_len = self.get_single_generate_data(input_str, max_len)

        outputs = self.do_single_generate(input_data, temperature=temperature, top_k=top_k, top_p=top_p, max_num=max_num)
        outputs = outputs.cpu().numpy()
        length = np.sum(outputs != self.tokenizer.pad_id)
        outputs = outputs[0, context_len:length]

        generate_text = "".join(self.tokenizer.convert_ids_to_tokens(outputs.tolist()))
        return generate_text

if __name__=="__main__":
    examples = {"自由生成": ["讲一个关于小镇失业青年的故事，关键词：失业、互联网、相亲、中年危机。故事的主角叫大黄，有一天"]}
    generater = InferGenerate()
    input_text = examples["自由生成"][0]
    output_text = generater.generate(input_text)
    print(f"input_text: \n{input_text} \noutput_text: \n{output_text}")
