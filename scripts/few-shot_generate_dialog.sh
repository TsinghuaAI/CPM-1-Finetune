#!/bin/bash

DATA_DIR="data/STC/"
CHECKPOINT_PATH="checkpoints/CPM-large"
RESULTS_DIR="results/few-shot_dialog/"
TOKENIZER_PATH="bpe_3w_new/"
MPSIZE=2
NLAYERS=32
NHIDDEN=2560
NATT=32
MAXSEQLEN=1024

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0.9

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/ds_dialog_config.json"

python -m torch.distributed.launch --nproc_per_node 2 --master_port 1255 generate_samples_dialog_fewshot.py \
       --data_dir ${DATA_DIR} \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --seq-length $MAXSEQLEN \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --cache-dir cache \
       --out-seq-length 50 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --tokenizer-path ${TOKENIZER_PATH} \
       --vocab-size 30000 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --results_dir ${RESULTS_DIR}
