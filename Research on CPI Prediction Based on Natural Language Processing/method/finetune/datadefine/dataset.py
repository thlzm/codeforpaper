#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/5
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
            "context_lengths": torch.ones(bs, 1, dtype=torch.long)
        }

        for i, samp in enumerate(samples):
            seq_len = len(samp["input_ids"])
            model_data["input_ids"][i][:seq_len] = torch.tensor(samp["input_ids"], dtype=torch.long)
            model_data["label_ids"][i][0] = samp["label_ids"]
            model_data["context_lengths"][i][0] = seq_len

        return model_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NliDataset(BaseDataset):

    LABELS = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

    def __init__(self, args, tokenizer, split, ratio=1, num=-1):
        super().__init__(args, tokenizer, split, ratio, num)

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
                # print(f"Worng label: {label}.")
                continue

            input_id_1 = self.tokenizer.encode(sentence1)
            input_id_2 = self.tokenizer.encode(sentence2)

            # 用两个在预训练中没用到过的token做句子分隔和结束，结束token用于输到分类器
            input_id = input_id_1 + [self.sep_id] + input_id_2 + [self.eod_id]

            label_id = self.LABELS.get(label, -100)
            sizes.append(len(input_id))

            data.append({
                "idx": i,
                "input_ids": input_id,
                "label_ids": label_id,
            })

        max_len = max(sizes)

        return data, max_len