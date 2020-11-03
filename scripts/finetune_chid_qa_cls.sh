#!/bin/bash

CHECKPOINT_PATH="/mnt/nfs/home/zzy/checkpoints/3B-new-bpe-fat"
MPSIZE=2
NLAYERS=32
NHIDDEN=2560
NATT=32
MAXSEQLEN=1024

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/ds_finetune.json"

python3 -m torch.distributed.launch --master_port ${1-1122} --nproc_per_node 8 finetune_chid_qa_cls.py \
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
       --lr 0.000005 \
       --warmup 0.1 \
       --batch-size 3 \
       --deepspeed \
       --deepspeed_config ${config_json} \
       --log-interval 10 \
       --seed 23333 \
       --save results/
