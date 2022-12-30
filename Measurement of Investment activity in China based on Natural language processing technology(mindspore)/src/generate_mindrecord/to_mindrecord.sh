#!/bin/bash

python generate_tnews_mindrecord.py \
  --data_dir=data/tnews/ \
  --task_name=tnews \
  --vocab_file=data/tnews/vocab.txt \
  --output_dir=data/tnews \
  --do_train=False \
  --do_eval=False \
  --do_predict=True 