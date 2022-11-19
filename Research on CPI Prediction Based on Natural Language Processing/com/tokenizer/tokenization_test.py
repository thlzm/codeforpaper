#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/11
# @Author: qing
import os

from com.config.path_config import PathConfig
from megatron.tokenizer.tokenization_jieba import JIEBATokenizer


def test_tokenization_jieba(text):
    path_config = PathConfig()
    vocab_path = os.path.join(path_config.root_path, "megatron/bpe_4w_pcl/vocab")

    tokenizer = JIEBATokenizer(vocab_path)
    input_ids = tokenizer.encode(text)

    print(f"input str: {text}.")
    print(f"encode id: {input_ids}.")


if __name__ == '__main__':
    text = "句子一与句子二的关系是：1、矛盾，2、蕴含，3、中立。答案：3"
    test_tokenization_jieba(text)