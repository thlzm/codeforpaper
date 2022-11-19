#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/10/8
# @Author: qing
import os
import sys
import time
import datetime
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
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.utils import get_ltor_masks_and_position_ids
from model.pangu_evolution.model.model_define import PanguModelEnhance

import pandas as pd


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
        model = PanguModelEnhance(num_tokentypes=0, parallel_output=False)

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
        input_ids = np.pad(origin_input_ids, ((0, 0), (0, pad_length)), 'constant',
                           constant_values=(0, self.tokenizer.pad_id))

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
        inputs = input_ids, None

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
        print(datetime.datetime.now(), "生成中~")
        input_data, context_len = self.get_single_generate_data(input_str, max_len)

        outputs = self.do_single_generate(input_data, temperature=temperature, top_k=top_k, top_p=top_p,
                                          max_num=max_num)
        outputs = outputs.cpu().numpy()
        length = np.sum(outputs != self.tokenizer.pad_id)
        outputs = outputs[0, context_len:length]

        generate_text = "".join(self.tokenizer.convert_ids_to_tokens(outputs.tolist()))
        return generate_text


def process_my_data(my_generater, data_dir):
    # def generate_keywords(series):
    #     return [my_generater.generate(sentence) for sentence in series["content"]]

    df_macro = pd.read_pickle(os.path.join(data_dir, "macro_origianl_to_baike.pkl"))
    df_micro = pd.read_pickle(os.path.join(data_dir, "micro_origianl_to_baike.pkl"))
    # df_macro["keywords"] = df_macro.apply(generate_keywords, axis=1)
    # df_micro["keywords"] = df_micro.apply(generate_keywords, axis=1)
    df_macro["keywords"] = df_macro.apply(
        lambda series: [my_generater.generate(f"关键词识别：\n摘要：{sentence}\n关键词：") for sentence in series["content"]], axis=1)
    df_macro.to_pickle(os.path.join(data_dir, "df_macro_keywords.pkl"))
    df_micro["keywords"] = df_micro.apply(
        lambda series: [my_generater.generate(f"关键词识别：\n摘要：{sentence}\n关键词：") for sentence in series["content"]], axis=1)
    df_micro.to_pickle(os.path.join(data_dir, "df_micro_keywords.pkl"))


if __name__ == "__main__":
    examples = {"文本分类": ["文本分类：\n基本上可以说是诈骗\n选项：积极，消极\n答案："],
                "关键词生成": [
                    "关键词识别：\n摘要：为解决传统均匀FFT波束形成算法引起的3维声呐成像分辨率降低的问题,该文提出分区域FFT波束形成算法.远场条件下,以保证成像分辨率为约束条件,以划分数量最少为目标,采用遗传算法作为优化手段将成像区域划分为多个区域.在每个区域内选取一个波束方向,获得每一个接收阵元收到该方向回波时的解调输出,以此为原始数据在该区域内进行传统均匀FFT波束形成.对FFT计算过程进行优化,降低新算法的计算量,使其满足3维成像声呐实时性的要求.仿真与实验结果表明,采用分区域FFT波束形成算法的成像分辨率较传统均匀FFT波束形成算法有显著提高,且满足实时性要求.\n关键词：",
                    "关键词识别：\n摘要：消费者物价指数（CPI），与就业形势报告（非农）结合在一起，就成了金融市场上被仔细研究的另一个热门的经济指标，因为通货膨胀影响着每一个人，它决定着消费者花费多少来购买商品和服务，左右着商业经营的成本，极大地破坏着个人或企业的投资，影响着退休人员的生活质量。而且，对通货膨胀的展望有助于设立劳动合同和制定政府的财政政策。\n关键词：",
                    # "关键词识别：\n摘要：各省（区、市）都有固定的价格调查人员和临时调查员按统一规定进行价格收集工作。调查点确定以后，各市、县价格调查人员就要按照规定时间对选定的商店、市场和服务网点的商品或服务价格，采用“三定”原则进行收集调查登记，“三定”原则即定点、定时、定人直接采价。定点，就是到已选定的调查点，即固定的调查商店和农贸市场，以保障价格资料来源的稳定性和可比性。定时，即在固定的日子和时间来采价，这是保证基期价格和报告期价格在时间上具有可比性，因为采集价格的时间不同，商品的价格也存在差异。这一点鲜活商品体现的最为明显，比如鲜菜，通常是上午刚上市时价格高一些，晚上收市时价格则低一些。因此，在进行价格调查时，不但每个月的调查次数和日期应保持一致，每次调查的时间也应相对固定。定人，就是在一定时期内由固定调查人员去调查，这是为了避免因调查人员的频繁变动而引起的人为价格调查误差，保持价格资料的稳定性、连续性和可比性。同时各地也常常利用价格采集点的计算机管理系统作为辅助性调查工具，同一规格品的价格必须同质可比，即产品性质基本相同可以进行比较；如果商品的挂牌价格与实际成交价格不一致，应调查采集实际成交价格；对于与居民生活密切相关、价格变动比较频繁的商品（如鲜菜、鲜果等鲜活食品），至少每5天调查一次价格；一般性商品每月调查采集2－3次价格\n关键词：",
                    # "关键词识别：\n摘要：消费者物价指数（CPI）涵盖全国城乡居民生活消费的食品、烟酒及用品、衣着、家庭设备用品及维修服务、医疗保健和个人用品、交通和通信、娱乐教育文化用品及服务、居住等八大类、262个基本分类的商品与服务价格。数据来源于全国31个省（区、市）500个市县、6.3万家价格调查点，包括食杂店、百货店、超市、便利店、专业市场、专卖店、购物中心以及农贸市场与服务消费单位等\n关键词：",
                ],
                }
    generater = InferGenerate()
    # for input_text in examples["关键词生成"]:
    #     output_text = generater.generate(input_text)
    #     print(f"input_text: \n{input_text} \noutput_text: \n{output_text}")
    print("@@@@@@@@@@@@@@@@@@@测试结束，现在开始处理数据@@@@@@@@@@@@@@@@@")
    process_my_data(generater, "/content/drive/MyDrive/pangu-alpha-applications/model/pangu_evolution/inference/data")

    # # 手动输入
    # while True:
    #     print("@" * 20)
    #     input_text = input("请输入文本：")
    #     output_text = generater.generate(input_text)
    #     print(f"input_text: \n{input_text} \noutput_text: \n{output_text}")
