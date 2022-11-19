#! /bin/bash
export CUDA_VISIBLE_DEVICES="0"
export MASTER_PORT=6600
WORKING_DIR=/content/drive/MyDrive/pangu-alpha-applications
MODEL_PATH=/content/drive/MyDrive/pangu-alpha-evolution_2.6b_fp16
python -u ${WORKING_DIR}/model/pangu_evolution/inference/infer_generate.py \
       --model-parallel-size 1 \
       --num-layers 31 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --seq-length 1024 \
       --load $MODEL_PATH \
       --vocab-file ${WORKING_DIR}/megatron/bpe_4w_pcl/vocab \
       --merge-file gpt2-merges.txt \
       --no-load-rng \
       --make-vocab-size-divisible-by 1 \
       --fp16 \

