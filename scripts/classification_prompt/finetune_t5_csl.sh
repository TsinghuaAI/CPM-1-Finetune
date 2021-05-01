#! /bin/bash

WORKING_DIR=/mnt/sfs_turbo/CPM-Finetune

if [[ $DLS_TASK_NUMBER == 1 ]]; then
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NNODES=1
    NODE_RANK=0
else
    MASTER_HOST="$BATCH_CUSTOM0_HOSTS"
    MASTER_ADDR="${MASTER_HOST%%:*}"
    MASTER_PORT="${MASTER_HOST##*:}"
    NNODES="$DLS_TASK_NUMBER"
    NODE_RANK="$DLS_TASK_INDEX"
fi

GPUS_PER_NODE=8
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# OPTIONS_NCCL="NCCL_DEBUG=info"

# Change for multinode config
MP_SIZE=4

ORIGIN_DATA_PATH="${WORKING_DIR}/large_data/"
DATA_EXT=".json"
CACHE_PATH="/cache/"
DATA_PATH="/mnt/sfs_turbo/data/CLUE/csl"

CONFIG_PATH="${WORKING_DIR}/configs/model/enc_dec_xlarge_8_config.json"
# CKPT_PATH="/mnt/sfs_turbo/enc-dec-pretrain/checkpoints/checkpoint-4-19"
CKPT_PATH="${WORKING_DIR}/results/t5_finetune_csl_lr0.000005const/"

SAVE_PATH="${WORKING_DIR}/results/t5_finetune_csl_lr0.000005const/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_finetune_t5.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_new"

BATCH_SIZE=4
GRAD_ACC=8
LR=0.000005
TRAIN_ITER=20000
EPOCHS=5

ENC_LEN=512
DEC_LEN=256


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --enc-seq-length ${ENC_LEN}"
OPTS+=" --dec-seq-length ${DEC_LEN}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-ext ${DATA_EXT}"
OPTS+=" --data-name csl"
OPTS+=" --data-impl mmap"
OPTS+=" --lazy-loader"
OPTS+=" --tokenizer-type GPT2BPETokenizer"
OPTS+=" --split 949,50,1"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 600"
OPTS+=" --eval-interval 100"
OPTS+=" --eval-iters 10"
OPTS+=" --log-interval 10"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
# OPTS+=" --do_train"
# OPTS+=" --do_valid"
# OPTS+=" --do_eval"
OPTS+=" --do_infer"
OPTS+=" --epochs ${EPOCHS}"

CMD="python -m torch.distributed.launch ${DISTRIBUTED_ARGS} ${WORKING_DIR}/finetune_t5.py ${OPTS}"

# echo "Copying Data"
# cp -r ${ORIGIN_DATA_PATH} ${CACHE_PATH}
# ls ${DATA_PATH}
# echo "Copy End"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log

set +x
