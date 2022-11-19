#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/11
# @Author: qing

import os
import json
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, args, tokenizer, split, ratio=1, num=-1):
        self.args = args
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.split = split
        self.pad_id = tokenizer.pad_id
        self.sep_id = self.tokenizer.encoder['<sep>']
        self.eod_id = self.tokenizer.encoder['<eod>']

        self.data, self.max_len = self.process_data()
        if num > 0: self.data = self.data[:num]

    def collate(self, samples):
        bs = len(samples)

        model_data = {
            "input_ids": torch.ones(bs, self.args.seq_length, dtype=torch.long) * self.pad_id,
            "label_ids": torch.ones(bs, 1, dtype=torch.long),
            "context_lengths": torch.ones(bs, 1, dtype=torch.long),
            "loss_mask": torch.zeros(bs, self.args.seq_length, dtype=torch.long)
        }

        for i, samp in enumerate(samples):
            seq_len = len(samp["input_ids"])
            model_data["input_ids"][i][:seq_len] = torch.tensor(samp["input_ids"], dtype=torch.long)
            model_data["label_ids"][i][0] = samp["label_ids"]
            model_data["context_lengths"][i][0] = seq_len
            model_data["loss_mask"][i, seq_len-2] = 1

        return model_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NliDataset(BaseDataset):

    LABELS = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    template = "句子一：{sentence1}。句子二：{sentence2}。句子一与句子二的关系是：1、矛盾，2、蕴含，3、中立。答案：{label}"

    def __init__(self, args, tokenizer, split, ratio=1, num=-1):
        super().__init__(args, tokenizer, split, ratio, num)

    @staticmethod
    def get_cand_ids(tokenizer):
        cand_ids = [tokenizer.encode(str(i)) for i in range(1, 4)]
        cand_ids = [cand_id[0] for cand_id in cand_ids]
        # print(f"cand_ids: {cand_ids}.")
        return cand_ids

    def process_data(self):
        data = []
        sizes = []

        data_path = os.path.join(self.args.data_path, self.split)

        with open(data_path, "r", encoding="utf-8") as f:
            data_lines = f.readlines()
        print(f"All {self.split} case num: {len(data_lines)}.")

        for i, instance in enumerate(data_lines):
            instance = json.loads(instance)

            sentence1 = instance["sentence1"]
            sentence2 = instance["sentence2"]
            label = instance.get("label", "-100")

            if label not in self.LABELS:
                continue

            label_id = self.LABELS.get(label, -100)
            input_id_str = self.template.replace("{sentence1}", sentence1).replace("{sentence2}", sentence2).replace("{label}", str(label_id+1))
            input_id = self.tokenizer.encode(input_id_str)

            sizes.append(len(input_id))

            data.append({
                "idx": i,
                "input_ids": input_id,
                "label_ids": label_id,
            })

        max_len = max(sizes)

        return data, max_len