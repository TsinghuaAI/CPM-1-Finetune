# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate responses for contexts in dialogs (unsupervised model)"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
import json
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint
#from data_utils import make_tokenizer
from data_utils.tokenization_gpt2 import GPT2Tokenizer
from configure_data import configure_data
import mpu
import deepspeed

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0

from pretrain_gpt2 import *

def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask)

    return tokens, attention_mask, position_ids

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()
	
    return logits


def generate_samples(model, tokenizer, args, device, test_data):

    prompt = [["如果我答应你这些你会选什么？", "我什么都不选你开心就好"], ["如果不累就继续跑…又一次的出发……", "这几年老了不少啊，保重！"], \
              ["这真的是送给我的？！我太喜欢啦！", "亲爱的，你终于回来了，嘛嘛嘛"], ["亲爱的，你不在的日子我会好好照顾自己。", "遗憾啊了解不到内幕了你们一路顺风"]]
    prompt_text_ids = []
    sep_token_id = tokenizer.encoder["<sep>"]
    eod_token_id = tokenizer.encoder["<eod>"]
    bos_token_id = tokenizer.encoder["<s>"]
    eos_token_id = tokenizer.encoder["</s>"]
    post_prefix_ids = tokenizer.encode("对话上文:")
    resp_prefix_ids = tokenizer.encode("回复:")
    cnt = 0
    prompt_text_len = 0
    for pr_pair in prompt:
        post_ids, resp_ids = tokenizer.encode(pr_pair[0]), tokenizer.encode(pr_pair[1])
        prompt_text_ids.extend(post_prefix_ids)
        prompt_text_ids.extend([bos_token_id])
        prompt_text_ids.extend(post_ids)
        prompt_text_ids.extend([eos_token_id])
        prompt_text_ids.extend(resp_prefix_ids)
        prompt_text_ids.extend([bos_token_id])
        prompt_text_ids.extend(resp_ids)
        prompt_text_ids.extend([eos_token_id])
        prompt_text_ids.extend([sep_token_id])
        prompt_text_len += len(pr_pair[0]) + len(pr_pair[1]) + len("对话上文:") + len("回复:")
        cnt += 1

    context_count=0
    model.eval()
    print('\nNumber of STC test set: ', len(test_data))
    f_gen = open(args.results_dir + 'gen.txt', 'w')
    f_ref = open(args.results_dir + 'ref.txt', 'w')
    with torch.no_grad():
        for data_pair in test_data:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            raw_text = ''.join(data_pair[0].split(' '))
            context_tokens = tokenizer.encode(raw_text)
            context_tokens = prompt_text_ids + post_prefix_ids + [bos_token_id] + context_tokens + [eos_token_id] + resp_prefix_ids + [bos_token_id]
            len_raw_text = len(raw_text) + len("对话上文:") + len("回复:")
            context_length = len(context_tokens)

            pad_id = tokenizer.encoder['<pad>']
            args.eod_token = tokenizer.encoder['<eod>']
            if context_length < args.seq_length:
                context_tokens.extend([pad_id] * (args.seq_length - context_length))

            context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
            context_length_tensor = torch.cuda.LongTensor([context_length])

            torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())

            context_length = context_length_tensor[0].item()
            tokens, attention_mask, position_ids=get_batch(context_tokens_tensor, device, args)

            start_time = time.time()

            counter = 0
            org_context_length = context_length

            while counter < (org_context_length + args.out_seq_length):
                logits = model(tokens, position_ids, attention_mask)
                logits = logits[:, context_length - 1, :] / args.temperature
                logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)            
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)
                tokens[0, context_length] = prev[0] 
                torch.distributed.broadcast(tokens, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
                context_length += 1
                counter += 1

                output_tokens_list = tokens.view(-1).contiguous()
                decode_tokens = tokenizer.decode(output_tokens_list.tolist())
                decode_tokens_only_resp = decode_tokens[prompt_text_len + len_raw_text:]
                token_end = decode_tokens_only_resp.rfind("\n")

                if token_end != -1:
                   break
                
            if mpu.get_model_parallel_rank() == 0:
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nCount: ", context_count)
                print("\nContext:", raw_text, flush=True)
                output_tokens_list = tokens.view(-1).contiguous()
                decode_tokens = tokenizer.decode(output_tokens_list.tolist())
                if decode_tokens.rfind("\n") > decode_tokens.rfind("回复:"):
                    trim_decode_tokens = decode_tokens[decode_tokens.rfind("回复:")+3: decode_tokens.rfind("\n")]
                else:
                    trim_decode_tokens = decode_tokens[decode_tokens.rfind("回复:")+3: ]
                print("\nGPT2:", trim_decode_tokens, flush=True)
                print("\nTruth:", "".join(data_pair[1].split(" ")), flush=True)
                f_gen.write(trim_decode_tokens + "\n")
                f_ref.write("".join(data_pair[1].split(" "))+'\n')
                f_gen.flush()
                f_ref.flush()

            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

    f_gen.close()
    f_ref.close()

def prepare_tokenizer(args):

    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    args.tokenizer_num_tokens = tokenizer.num_tokens
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    after = tokenizer.num_tokens
    while after % mpu.get_model_parallel_world_size() != 0:
        after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer


def get_model_wo_parallel(args):
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=False)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    #To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model



def setup_model_and_optimizer_wo_parallel(args):
    """Setup model and optimizer."""

    model = get_model_wo_parallel(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False
        )

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def main():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'),
                              os.path.join(args.tokenizer_path, 'chinese_vocab.model'))

    # Model, optimizer, and learning rate.
    model = setup_model_and_optimizer_wo_parallel(args)[0]

    #setting default batch size to 1
    args.batch_size = 1

    # read STC test set
    with open(args.data_dir + 'STC_test.json', 'r') as f_test:
        raw_data = json.loads(f_test.read())
        raw_data = raw_data['test']

    #generate samples
    generate_samples(model, tokenizer, args, torch.cuda.current_device(), raw_data)
    

if __name__ == "__main__":
    main()
