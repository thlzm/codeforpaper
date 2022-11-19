#! /bin/bash
#export CUDA_VISIBLE_DEVICES="0,1"
#export NCCL_DEBUG="INFO"

WORKING_DIR=/userhome/gpt3/PanGu-Alpha-Application
MODEL_PATH=/root/pangu/model/pangu_fp16_8mp_2b6
MODEL_SAVE_PATH=/root/pangu/model/pangu_fp16_8mp_2b6_pt_cmnli
DATA_PATH=/root/pangu/data/cmnli_public
TENSORBOARD_PATH=/root/pangu/log/tensorboard_nli_pt

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ${WORKING_DIR}/method/prompttune/tasks/prompt_tune_nli.py \
       --model-parallel-size 8 \
       --num-layers 31 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --batch-size 64 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 50000 \
       --lr-decay-iters 1000 \
       --save $MODEL_SAVE_PATH \
       --load $MODEL_PATH \
       --data-path $DATA_PATH \
       --tensorboard-dir $TENSORBOARD_PATH \
       --vocab-file ${WORKING_DIR}/megatron/bpe_4w_pcl/vocab \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00005 \
       --lr-decay-style cosine \
       --min-lr 1.0e-6 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 500 \
       --eval-iters 200 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --seed 1234 \
       --fp16 \
       --finetune \
       --DDP-impl "torch"\
       --checkpoint-activations \
       --distribute-checkpointed-activations \
       --fp16-lm-cross-entropy \
       --use-cpu-initialization \
       --make-vocab-size-divisible-by 1 \
