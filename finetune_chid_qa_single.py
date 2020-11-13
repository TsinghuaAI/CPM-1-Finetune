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
import json
import pickle
from tqdm import tqdm
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint
# from data_utils import make_tokenizer
from data_utils.tokenization_gpt2 import GPT2Tokenizer
from configure_data import configure_data
import mpu
import deepspeed
import json
import time

from tqdm import tqdm
from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
from data.samplers import DistributedBatchSampler, RandomSampler
# from sklearn.metrics import accuracy_score

from torch.utils.data import TensorDataset

from pretrain_gpt2 import *

def yprint(str):
    print("\033[43;30m{}\033[0m".format(str))

class CHIDDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, split, tokenizer, ratio=1):
        self.split = split
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.args = args
        self.world_size = args.world_size

        self.pad_id = tokenizer.encoder['<pad>']
        self.eod_token = tokenizer.encoder['<eod>']
        args.eod_token = tokenizer.encoder['<eod>']

        with open(data_path, "r") as f:
            self.num_ids, data = json.load(f)
        self.seq, self.ints, self.sizes = self.process(data)

        self.max_size = max(self.sizes)

    def process(self, data):
        seq = []
        ints = []
        sizes = []
        for d in tqdm(data[:int(self.ratio * len(data))]):
            loss_mask = [0] * (len(d["sent"]) - 2) + [1]

            seq.append({
                "input_ids": d["sent"][:-1],
            })
            ints.append({
                "loss_mask": loss_mask,
                "labels": d["sent"][1:],
                "truth": d["truth"],
                "full_attn_range": d["cands_len"]
            })
            sizes.append(len(d["sent"]) - 1)

        return seq, ints, sizes

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        return self.seq[idx], self.ints[idx], self.sizes[idx]

    def collate(self, samples):
        bs = len(samples)
        seq = [s[0] for s in samples]
        ints = [s[1] for s in samples]
        sizes = [s[2] for s in samples]
        # max_size = max(sizes)
        max_size = self.max_size

        attn_mask = torch.tril(torch.ones((max_size, max_size))).unsqueeze(0)
        # for i, x in enumerate(ints):
        #     s, e = x["full_attn_range"]
        #     attn_mask[i, s:e, :e] = torch.ones(e-s, e)
        position_ids = torch.arange(max_size, dtype=torch.long).unsqueeze(0).repeat(bs, 1)

        if self.args.fp16:
            attn_mask = attn_mask.half()

        batch_seq = {
            "input_ids": torch.ones(bs, max_size).long() * self.pad_id,
            "attention_mask": attn_mask.unsqueeze(1),
            "position_ids": position_ids,
        }

        no_model_seq = {
            "labels": torch.ones(bs, max_size).long() * self.pad_id,
            "truth": torch.zeros(bs).long(),
            "loss_mask": torch.zeros(bs, max_size).float()
        }

        for i, samp in enumerate(seq):
            batch_seq["input_ids"][i, :len(samp["input_ids"])] = torch.tensor(samp["input_ids"])
        for i, samp in enumerate(ints):
            no_model_seq["labels"][i, :len(samp["labels"])] = torch.tensor(samp["labels"])
            no_model_seq["truth"][i] = torch.tensor(samp["truth"])
            no_model_seq["loss_mask"][i, :len(samp["loss_mask"])] = torch.tensor(samp["loss_mask"])

        return batch_seq, no_model_seq

def load_data(data_path, data_type, tokenizer, ratio=1):
    args = get_args()
    batch_size = args.batch_size
    args.batch_size = 1


    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    args.batch_size = batch_size
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Dataset
    filename = os.path.join(data_path, data_type+'.json')
    dataset = CHIDDataset(args, filename, data_type, tokenizer, ratio=ratio)
    
    # Use a random sampler with distributed batch sampler.
    if data_type == 'train':
        sampler = RandomSampler(dataset)
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
                                       pin_memory=True,
                                       collate_fn=dataset.collate), dataset

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

    # get the tokenizer
    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'), os.path.join(args.tokenizer_path, 'merges.txt'), os.path.join(args.tokenizer_path, 'chinese_vocab.model'))

    # load data
    train_dataloader, _ = load_data('/data/gyx/chid/preprocessed_qa_cands_end', 'train', tokenizer, 1)
    dev_dataloader, dev_dataset = load_data('/data/gyx/chid/preprocessed_qa_cands_end', 'test', tokenizer, 1)

    with open(args.deepspeed_config, "r") as f:
        deepspeed_conf = json.load(f)

    epoch = 10
    grad_acc = deepspeed_conf["gradient_accumulation_steps"]
    args.train_iters = len(train_dataloader) * epoch / grad_acc

    # Model, optimizer, and learning rate.
    # TODO: maybe need to reinitialize optimizer
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

    device = torch.cuda.current_device()

    results_dir = "results/"

    cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    model_dir = os.path.join(results_dir, "{}-{}-{}-{}".format(args.model_name, cur_time, args.lr, mpu.get_data_parallel_world_size() * grad_acc * args.batch_size))

    if torch.distributed.get_rank() == 0:
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, "train_log.txt"), "w") as f:
            f.write("Train losses:\n")

        with open(os.path.join(model_dir, "eval_log.txt"), "w") as f:
            f.write("Eval losses:\n")

    torch.distributed.barrier()

    num_ids = torch.tensor(dev_dataset.num_ids).to(device)
    total_loss = 0
    logging_loss = 0
    global_step = 0
    total_step = 0
    # trn_all_truth = []
    # trn_all_preds = []
    for e in range(epoch):
        model.train()
        for batch, no_model_batch in tqdm(train_dataloader, disable=torch.distributed.get_rank() != 0):
            for k in batch:
                batch[k] = batch[k].to(device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(device)
            
            output = model(**batch)
            output = torch.sum(output * no_model_batch["loss_mask"].unsqueeze(-1), 1) / torch.sum(no_model_batch["loss_mask"], -1).unsqueeze(-1)
            labels = (torch.sum(no_model_batch["labels"] * no_model_batch["loss_mask"], 1) / torch.sum(no_model_batch["loss_mask"], -1)).long()
            losses = mpu.vocab_parallel_cross_entropy(output.unsqueeze(1).contiguous().float(), labels.unsqueeze(1))
            loss = torch.mean(losses)

            model.backward(loss)
            model.step()

            torch.distributed.all_reduce(loss.data, group=mpu.get_data_parallel_group())
            loss.data = loss.data / mpu.get_data_parallel_world_size()
            total_loss += loss.item() / grad_acc

            if total_step % grad_acc == 0:
                global_step += 1
                if global_step != 0 and global_step % args.log_interval == 0:
                    if torch.distributed.get_rank() == 0:
                        train_log = "epoch {}, global step {}, total step {}, train lm loss: {}".format(e, global_step, epoch * len(train_dataloader), (total_loss - logging_loss) / args.log_interval)
                        yprint(train_log)
                        with open(os.path.join(model_dir, "train_log.txt"), "a") as f:
                            f.write(train_log + "\n")
                        
                        # yprint("Acc: {}".format(sum([int(p == l) for p, l in zip(trn_all_preds, trn_all_truth)]) / len(trn_all_truth)))
                        
                        # trn_all_preds = []
                        # trn_all_truth = []

                    logging_loss = total_loss
   
                if global_step != 0 and global_step % args.eval_interval == 0:
                    model.eval()
                    all_truth = []
                    all_preds = []
                    with torch.no_grad():
                        for batch, no_model_batch in tqdm(dev_dataloader, desc="Evaluating"):
                            for k in batch:
                                batch[k] = batch[k].to(device)
                            for k in no_model_batch:
                                no_model_batch[k] = no_model_batch[k].to(device)

                            output = model(**batch)
                            output = torch.sum(output * no_model_batch["loss_mask"].unsqueeze(-1), 1) / torch.sum(no_model_batch["loss_mask"], -1).unsqueeze(-1)

                            tensor_list = [torch.zeros_like(output) for _ in range(mpu.get_data_parallel_world_size())]
                            torch.distributed.all_gather(tensor_list, output, mpu.get_data_parallel_group())

                            tensor_list_truth = [torch.zeros_like(no_model_batch["truth"], dtype=torch.long) for _ in range(mpu.get_data_parallel_world_size())]
                            torch.distributed.all_gather(tensor_list_truth, no_model_batch["truth"], mpu.get_data_parallel_group())

                            if torch.distributed.get_rank() == 0:
                                scores = torch.stack(tensor_list, 0).view(-1, 30000) # for convience, the truth labels only appears in the first part of the model
                                truth = torch.stack(tensor_list_truth, 0).view(-1)
                                scores = scores[:, num_ids]

                                preds = torch.argmax(scores, dim=-1)

                                all_truth.extend(truth.detach().cpu().tolist())
                                all_preds.extend(preds.detach().cpu().tolist())

                        if torch.distributed.get_rank() == 0:
                            acc = sum([int(p == l) for p, l in zip(all_preds, all_truth)]) / len(all_truth)
                            eval_log = "Epoch: {}, Global step: {}, Acc: {}".format(e, global_step, acc)
                            yprint(eval_log)
                            eval_results_dir = os.path.join(model_dir, "eval_step{}".format(global_step))
                            os.makedirs(eval_results_dir, exist_ok=True)
                            with open(os.path.join(eval_results_dir, "eval_result.txt"), "w") as f:
                                f.write(eval_log)
                            with open(os.path.join(model_dir, "eval_log.txt"), "a") as f:
                                f.write(eval_log)
                            with open(os.path.join(eval_results_dir, "pred.txt"), "w") as f:
                                f.write(str(all_preds))
                            with open(os.path.join(eval_results_dir, "truth.txt"), "w") as f:
                                f.write(str(all_truth))

                        torch.distributed.barrier()

                    if args.save:
                        args.save = model_dir
                        save_checkpoint(global_step, model, optimizer, lr_scheduler, args)

            total_step += 1

if __name__ == "__main__":
    # args = get_args()

    # # Pytorch distributed.
    # initialize_distributed(args)

    # # Random seeds for reproducability.
    # set_random_seed(args.seed)

    # # get the tokenizer
    # tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'), os.path.join(args.tokenizer_path, 'merges.txt'), os.path.join(args.tokenizer_path, 'chinese_vocab.model'))

    # train_dataloader = load_data('/data/gyx/chid/preprocessed', 'train', tokenizer, ratio=0.01)

    # for batch in train_dataloader:
    #     print(batch)
    #     exit(0)

    main()
