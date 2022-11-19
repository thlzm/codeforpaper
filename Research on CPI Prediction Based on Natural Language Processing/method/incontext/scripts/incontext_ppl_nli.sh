#! /bin/bash
#export CUDA_VISIBLE_DEVICES="0,1"
#export NCCL_DEBUG="INFO"

WORKING_DIR=/userhome/gpt3/PanGu-Alpha-Application
MODEL_PATH=/root/pangu/model/Pangu-alpha_2.6B_mgt
DATA_PATH=/root/pangu/data/cmnli_public

GPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ${WORKING_DIR}/method/incontext/tasks/infer_ppl_nli.py \
       --model-parallel-size 1 \
       --num-layers 31 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --batch-size 3 \
       --seq-length 1024 \
       --load $MODEL_PATH \
       --data-path $DATA_PATH \
       --vocab-file ${WORKING_DIR}/megatron/bpe_4w_pcl/vocab \
       --merge-file gpt2-merges.txt \
       --no-load-rng \
