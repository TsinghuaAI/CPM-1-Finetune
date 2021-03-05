#!/bin/bash

DATA_DIR="/data/STC/preprocessed/"
CHECKPOINT_PATH="/data/checkpoints/CPM-large"
RESULTS_DIR="results/"
MODEL_NAME="finetune-dial-large-fp32"
TOKENIZER_PATH="bpe_3w_new/"
MPSIZE=2
NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8

NLAYERS=32
NHIDDEN=2560
NATT=32
MAXSEQLEN=200

CUR_PATH=$(realpath $0)
CUR_DIR=$(dirname ${CUR_PATH})
DS_CONFIG="${CUR_DIR}/../ds_config/ds_finetune_lm_large_fp32.json"
HOST_FILE="${CUR_DIR}/../host_files/hostfile"

deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --master_port ${1-1122} --hostfile ${HOST_FILE} finetune_lm.py \
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
       --tokenizer-path ${TOKENIZER_PATH} \
       --vocab-size 30000 \
       --lr 0.00001 \
       --warmup 0.1 \
       --batch-size 16 \
       --deepspeed \
       --deepspeed_config ${DS_CONFIG} \
       --log-interval 5 \
       --eval-interval 100 \
       --seed 23333 \
       --results_dir ${RESULTS_DIR} \
       --model_name ${MODEL_NAME} \
       --epoch 1 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing
