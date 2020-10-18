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

"""Sample Generate GPT2"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
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
from data.samplers import DistributedBatchSampler

from torch.utils.data import TensorDataset

from pretrain_gpt2 import *

def get_batch(context_tokens, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()

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

def load_data(data_path, data_type, tokenizer):
    import json

    args = get_args()
    batch_size = args.batch_size
    args.batch_size = 1

    filename = os.path.join(data_path, data_type+'.json')
    objs = []
    with open(filename) as fin:
        for line in fin:
            # {"sentence1": "双十一花呗提额在哪", "sentence2": "里可以提花呗额度", "label": "0"}
            objs.append(json.loads(line.strip()))

    pad_id = tokenizer.encoder['<pad>']
    eod_token = tokenizer.encoder['<eod>']
    args.eod_token = tokenizer.encoder['<eod>']
    sep_token = tokenizer.encoder['<sep>']

    all_tokens = []
    all_labels = []
    all_attention_mask = []
    all_position_ids = []
    for obj in objs:
        sentence1 = obj['sentence1']
        sentence2 = obj['sentence2']
        sentence1 = tokenizer.encode(sentence1)
        sentence2 = tokenizer.encode(sentence2)
        # TODO random truncate
        tokens = sentence1 + [sep_token] + sentence2
        tokens = tokens[:256]
        tokens = tokens + [eod_token]

        token_length = len(tokens)
        if token_length < 256:
            tokens.extend([pad_id] * (256 - token_length))
        tokens_tensor = torch.LongTensor(tokens)
        tokens, attention_mask, position_ids = get_batch(tokens_tensor, args)

        all_tokens.append(tokens)
        all_attention_mask.append(attention_mask)
        all_position_ids.append(position_ids)
        
        if obj['label'] == '0':
            all_labels.append([1])
        else:
            all_labels.append([2])

    all_tokens = torch.stack(all_tokens).cpu()
    all_attention_mask = torch.stack(all_attention_mask).cpu()
    all_position_ids = torch.stack(all_position_ids).cpu()
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    dataset = TensorDataset(all_tokens, all_attention_mask, all_position_ids, all_labels)

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    args.batch_size = batch_size
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Use a random sampler with distributed batch sampler.
    if data_type == 'train':
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)

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
    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'), os.path.join(args.tokenizer_path, 'merges.txt'), os.path.join(args.tokenizer_path, 'chinese_vocab.model'))

    # load data
    train_dataloader = load_data('/data/zzy/afqmc', 'train', tokenizer)
    dev_dataloader = load_data('/data/zzy/afqmc', 'dev', tokenizer)

    args.train_iters = len(train_dataloader)

    # Model, optimizer, and learning rate.
    # TODO: maybe need to reinitialize optimizer
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

    epoch = 3
    device = torch.cuda.current_device()
    for _ in range(epoch):
        for batch in train_dataloader:
            tokens, attention_mask, position_ids, labels = [x.to(device) for x in batch]
            attention_mask = attention_mask[0, :, :, :, :]
            tokens = tokens.squeeze(1)
            position_ids = position_ids.squeeze(1)
            output = model(tokens, position_ids, attention_mask[0])
            output = output[tokens == 7, :].unsqueeze(1)
            losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
            loss = losses.mean()
            model.backward(loss)
            model.step()

            # if torch.distributed.get_rank() == 0:
            #     res = output.squeeze(1).cpu().detach().numpy()[:, 1:3]
            #     labels = labels.view(-1).cpu().detach().numpy()
            #     res = [1==y if x[0] > x[1] else 2==y for x, y in zip(res,labels)]
            #     print("acc", sum(res)/len(res), "loss", loss)
                

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            tokens, attention_mask, position_ids, labels = [x.to(device) for x in batch]
            attention_mask = attention_mask[0, :, :, :, :]
            tokens = tokens.squeeze(1)
            position_ids = position_ids.squeeze(1)
            output = model(tokens, position_ids, attention_mask[0])
            output = output[tokens == 7, :]

            tensor_list = [torch.zeros_like(output), torch.zeros_like(output)]
            torch.distributed.all_gather(tensor_list, output, mpu.get_data_parallel_group())

            if torch.distributed.get_rank() == 0:
                output = torch.stack(tensor_list, 0).view([-1, 15000])
                res = output.cpu().detach().numpy()[:, 1:3]
                labels = labels.view(-1).cpu().detach().numpy()
                res = [1==y if x[0] > x[1] else 2==y for x, y in zip(res, labels)]
                correct += sum(res)
                total += len(res)
    
    if torch.distributed.get_rank() == 0:
        print(correct, total)

if __name__ == "__main__":
    main()



