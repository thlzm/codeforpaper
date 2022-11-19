#! /bin/bash
export CUDA_VISIBLE_DEVICES="0,1"
export MASTER_PORT=6600

WORKING_DIR=/userhome/gpt3/PanGu-Alpha-Applications
MODEL_PATH=/root/pangu/model/pretrain/pangu_fp16_8mp_350m_pretrain056_051_1209/merged

# 2.6B
#Layers=31
#Hsize=2560
#Heads=32

# 350M
Layers=23
Hsize=1024
Heads=16

python -u ${WORKING_DIR}/model/pangu/inference/infer_generate.py \
       --model-parallel-size 1 \
       --num-layers $Layers \
       --hidden-size $Hsize \
       --num-attention-heads $Heads \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --seq-length 1024 \
       --load $MODEL_PATH \
       --vocab-file ${WORKING_DIR}/megatron/bpe_4w_pcl/vocab \
       --merge-file gpt2-merges.txt \
       --no-load-rng \
       --make-vocab-size-divisible-by 1 \
       --fp16 \
