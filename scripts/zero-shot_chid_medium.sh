#!/bin/bash

CHECKPOINT_PATH="/mnt/nfs/home/zzy/checkpoints/CPM-medium"
MPSIZE=1
NLAYERS=24
NHIDDEN=1024
NATT=16
MAXSEQLEN=1024

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/ds_finetune.json"

python3 -m torch.distributed.launch --master_port 1235 --nproc_per_node 4 finetune_chid.py \
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
       --out-seq-length 512 \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --tokenizer-path ~/bpe/bpe_3w/ \
       --vocab-size 30000 \
       --lr 0.0001 \
       --warmup 0.05 \
       --batch-size 8 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --log-interval 10 \
       --seed 23333 \
       --alpha 0 \
       --save results/ \
