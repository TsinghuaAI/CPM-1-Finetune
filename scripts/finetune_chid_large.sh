#!/bin/bash

DATA_DIR="/data/gyx/chid/preprocessed_qa_cands_end"
CHECKPOINT_PATH="/mnt/nfs/home/zzy/checkpoints/CPM-large"
RESULTS_DIR="results"
MODEL_NAME="finetune-test"
MPSIZE=2
NLAYERS=32
NHIDDEN=2560
NATT=32
MAXSEQLEN=1024

CUR_PATH=$(realpath $0)
CUR_DIR=$(dirname ${CUR_PATH})
DS_CONFIG="${CUR_DIR}/ds_finetune_large.json"

python3 -m torch.distributed.launch --master_port ${1-1122} --nproc_per_node 4 finetune_chid.py \
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
       --fp16 \
       --out-seq-length 512 \
       --tokenizer-path ~/bpe/bpe_3w/ \
       --vocab-size 30000 \
       --lr 0.00001 \
       --warmup 0.1 \
       --batch-size 2 \
       --deepspeed \
       --deepspeed_config ${DS_CONFIG} \
       --log-interval 1 \
       --eval-interval 3 \
       --seed 23333 \
       --results_dir ${RESULTS_DIR} \
       --model_name ${MODEL_NAME} \
       --epoch 1 \
       --eval_ckpt_path "/mnt/nfs/home/gyx/gpt-finetune/results/finetune-test-2020-12-05-10:23:22/dev_step-3"
