#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/13
# @Author: qing
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
            "label_ids": torch.ones(bs, dtype=torch.long),
            "idxs": torch.ones(bs, dtype=torch.long),
            "context_lengths": torch.ones(bs, 1, dtype=torch.long),
            "loss_mask": torch.zeros(bs, self.args.seq_length, dtype=torch.long)
        }

        for i, samp in enumerate(samples):
            seq_len = len(samp["input_ids"])
            loss_mask_index_left = samp["loss_mask_index_left"]
            model_data["input_ids"][i][:seq_len] = torch.tensor(samp["input_ids"], dtype=torch.long)
            model_data["label_ids"][i] = samp["label_ids"]
            model_data["idxs"][i] = samp["idx"]
            model_data["context_lengths"][i][0] = seq_len
            model_data["loss_mask"][i][loss_mask_index_left:seq_len] = 1

        return model_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NliDataset(BaseDataset):

    LABELS = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
    template = ["{sentence1}？", "{label}，", "{sentence2}"]

    def __init__(self, args, tokenizer, split, ratio=1, num=-1):
        self.candicate_lb_ids = None
        self.max_num = 100000
        super().__init__(args, tokenizer, split, ratio, num)

    def get_candicate_lb_ids(self):
        if not self.candicate_lb_ids:
            self.candicate_lb_ids = [self.tokenizer.encode(l) for l in ["错，", "对，", "或，"]]
        return self.candicate_lb_ids

    def process_data(self):
        data = []
        sizes = []

        data_path = os.path.join(self.args.data_path, self.split)

        with open(data_path, "r", encoding="utf-8") as f:
            data_lines = f.readlines()
        print(f"All {self.split} case num: {len(data_lines)}.")

        for i, instance in enumerate(data_lines):
            if i>=self.max_num: break

            instance = json.loads(instance)

            sentence1 = instance["sentence1"]
            sentence2 = instance["sentence2"]
            label = instance.get("label", "-100")

            if label not in self.LABELS:
                continue

            s1_str = self.template[0].replace("{sentence1}", sentence1)
            s2_str = self.template[2].replace("{sentence2}", sentence2)
            label_idx = self.LABELS.get(label, -100)

            s1_id = self.tokenizer.encode(s1_str)
            s2_id = self.tokenizer.encode(s2_str)
            candicate_lb_ids = self.get_candicate_lb_ids()

            for j, candicate_lb_id in enumerate(candicate_lb_ids):
                input_id = s1_id + candicate_lb_id + s2_id
                loss_mask_index_left = len(s1_id + candicate_lb_id)

                label_id = int(j==label_idx)

                sizes.append(len(input_id))

                data.append({
                    "idx": i,
                    "input_ids": input_id,
                    "label_ids": label_id,
                    "loss_mask_index_left": loss_mask_index_left
                })

        max_len = max(sizes)

        return data, max_len