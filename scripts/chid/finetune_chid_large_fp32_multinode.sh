#!/bin/bash

DATA_DIR="/data/chid/preprocessed/"
CHECKPOINT_PATH="/data/checkpoints/CPM-large"
RESULTS_DIR="results/"
MODEL_NAME="finetune-chid-large-fp32-mnode"
TOKENIZER_PATH="bpe_3w_new/"
MPSIZE=2
NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8

NLAYERS=32
NHIDDEN=2560
NATT=32
MAXSEQLEN=1024

CUR_PATH=$(realpath $0)
CUR_DIR=$(dirname ${CUR_PATH})
DS_CONFIG="${CUR_DIR}/../ds_config/ds_finetune_large_fp32.json"
HOST_FILE="${CUR_DIR}/../host_files/hostfile"

deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --master_port ${1-1122} --hostfile ${HOST_FILE} finetune_chid.py \
       --do_train \
       --do_eval \
       --data_dir ${DATA_DIR} \
       --model-parallel-size ${MPSIZE} \
       --num-layers ${NLAYERS} \
       --hidden-size ${NHIDDEN} \
       --load ${CHECKPOINT_PATH} \
       --num-attention-heads ${NATT} \
       --seq-length ${MAXSEQLEN} \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --out-seq-length 512 \
       --tokenizer-path ${TOKENIZER_PATH} \
       --vocab-size 30000 \
       --lr 0.00001 \
       --warmup 0.1 \
       --batch-size 4 \
       --deepspeed \
       --deepspeed_config ${DS_CONFIG} \
       --log-interval 5 \
       --eval-interval 1000 \
       --seed 23333 \
       --results_dir ${RESULTS_DIR} \
       --model_name ${MODEL_NAME} \
       --epoch 10 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing
