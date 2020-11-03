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

        if os.path.exists(data_path + "{}_cache.pkl".format(ratio)):
            if torch.distributed.get_rank() == 0:
                yprint("Load from cache: {}".format(data_path + "{}_cache.pkl".format(ratio)))
            with open(data_path + "{}_cache.pkl".format(ratio), "rb") as f:
                self.seq, self.sizes, self.truth_labels = pickle.load(f)
        else:
            with open(data_path, "r") as f:
                data = json.load(f)
                # train: {"contents": ["谈到巴萨目前的成就", ...], "sids": [0, 1, 2, 3, ...], "labels": []}
                # dev: {"contents": ["中国青年报：篮协改革切莫因噎废食", ...], "sids": [0, 0, ..., 1, 1, ...], "labels": [5, 1, 4, 3, ...]}
            if torch.distributed.get_rank() == 0:
                yprint("Preprocessing dataset")
            self.seq, self.sizes, self.truth_labels = self.process(data)
            if torch.distributed.get_rank() == 0:
                yprint("Save to cache: {}".format(data_path + "{}_cache.pkl".format(ratio)))
                with open(data_path + "{}_cache.pkl".format(ratio), "wb") as f:
                    pickle.dump((self.seq, self.sizes, self.truth_labels), f)

        self.max_size = {}
        for k in self.sizes[0]:
            self.max_size[k] = max([s[k] for s in self.sizes])

    def process(self, data):
        contents = data["contents"]
        sids = data["sids"]
        truth_labels = data["labels"]
        cids = data["cids"]
        sizes = []
        seq = []
        for content, sid, cid in zip(tqdm(contents[:int(self.ratio * len(contents))], desc="Processing"), sids, cids):
            if self.split == "train":
                sample, neg_sample = content
            else:
                sample = content
            
            input_ids = self.tokenizer.encode(sample)
            input_ids = input_ids + [self.eod_token]
            length = len(input_ids) - 1
            sizes.append({
                "input_ids": length
            })
            seq.append({
                "sid": sid,
                "cid": cid,
                "input_ids": input_ids[:-1],
                "loss_mask": [1.0] * length,
                "labels": input_ids[1:],
            })


            if self.split == "train":
                neg_input_ids = self.tokenizer.encode(neg_sample)
                neg_input_ids = neg_input_ids + [self.eod_token]
                neg_length = len(neg_input_ids) - 1
                sizes[-1].update({
                    "neg_input_ids": neg_length
                })
                seq[-1].update({
                    "neg_input_ids": neg_input_ids[:-1],
                    "neg_labels": neg_input_ids[1:],
                    "neg_loss_mask": [1.0] * neg_length,
                })


        if torch.distributed.get_rank() == 0:
            for k in sizes[0]:
                yprint(max([s[k] for s in sizes]))

        return seq, sizes, truth_labels

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        return self.seq[idx], self.sizes[idx]

    def collate(self, samples):
        bs = len(samples)
        seq = [s[0] for s in samples]
        sizes = [s[1] for s in samples]
        # max_size = max(sizes)
        max_size = self.max_size["input_ids"]
        attn_mask, pos_ids = build_attn_mask_pos_ids(self.args, bs, max_size)

        batch_seq = {
            "input_ids": torch.ones(bs, max_size).long() * self.pad_id,
            "attention_mask": attn_mask,
            "position_ids":pos_ids
        }

        no_model_seq = {
            "sids": torch.zeros(bs).long(),
            "cids": torch.zeros(bs).long(),
            "loss_mask": torch.zeros(bs, max_size).float(),
            "labels": torch.ones(bs, max_size).long() * self.pad_id,
        }

        if self.split == "train":
            neg_max_size = self.max_size["neg_input_ids"]
            neg_attn_mask, neg_pos_ids = build_attn_mask_pos_ids(self.args, bs, neg_max_size)
            
            batch_seq_neg = {
                "input_ids": torch.ones(bs, neg_max_size).long() * self.pad_id,
                "attention_mask": neg_attn_mask,
                "position_ids": neg_pos_ids
            }
            no_model_seq_neg = {
                "loss_mask": torch.zeros(bs, neg_max_size).float(),
                "labels": torch.ones(bs, neg_max_size).long() * self.pad_id,
            }

        for i, samp in enumerate(seq):
            batch_seq["input_ids"][i, :len(samp["input_ids"])] = torch.tensor(samp["input_ids"])
            no_model_seq["loss_mask"][i, :len(samp["loss_mask"])] = torch.tensor(samp["loss_mask"])
            no_model_seq["labels"][i, :len(samp["labels"])] = torch.tensor(samp["labels"])
            no_model_seq["sids"][i] = torch.tensor(samp["sid"])
            no_model_seq["cids"][i] = torch.tensor(samp["cid"])
            if self.split == "train":
                batch_seq_neg["input_ids"][i, :len(samp["neg_input_ids"])] = torch.tensor(samp["neg_input_ids"])
                no_model_seq_neg["loss_mask"][i, :len(samp["neg_loss_mask"])] = torch.tensor(samp["neg_loss_mask"])
                no_model_seq_neg["labels"][i, :len(samp["neg_labels"])] = torch.tensor(samp["neg_labels"])

        if self.split == "train":
            return batch_seq, batch_seq_neg, no_model_seq, no_model_seq_neg
        else:
            return batch_seq, no_model_seq

def build_attn_mask_pos_ids(args, batch_size, max_size):
    attn_mask = torch.tril(torch.ones((max_size, max_size))).unsqueeze(0)

    position_ids = torch.arange(max_size, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    if args.fp16:
        attn_mask = attn_mask.half()

    return attn_mask, position_ids

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
    train_dataloader, _ = load_data('/data/gyx/chid/preprocessed_with_neg', 'train', tokenizer, 0.1)
    dev_dataloader, dev_dataset = load_data('/data/gyx/chid/preprocessed_with_neg', 'dev', tokenizer, 1)

    with open("scripts/ds_finetune.json", "r") as f:
        deepspeed_conf = json.load(f)

    epoch = 3
    grad_acc = deepspeed_conf["gradient_accumulation_steps"]
    args.train_iters = len(train_dataloader) * epoch / grad_acc

    # Model, optimizer, and learning rate.
    # TODO: maybe need to reinitialize optimizer
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

    device = torch.cuda.current_device()

    results_dir = "results/"

    cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    model_dir = os.path.join(results_dir, "lm-hinge{}".format(cur_time))

    if torch.distributed.get_rank() == 0:
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, "train_log.txt"), "w") as f:
            f.write("Train losses:\n")

    total_loss = 0
    logging_loss = 0
    global_step = 0
    total_step = 0
    for e in range(epoch):
        model.train()
        for batch, batch_neg, no_model_batch, no_model_batch_neg in tqdm(train_dataloader, disable=torch.distributed.get_rank() != 0):
            for k in batch:
                batch[k] = batch[k].to(device)
            for k in batch_neg:
                batch_neg[k] = batch_neg[k].to(device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(device)
            for k in no_model_batch_neg:
                no_model_batch_neg[k] = no_model_batch_neg[k].to(device)

            output = model(**batch)
            lm_losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), no_model_batch["labels"])
            loss_mask = no_model_batch["loss_mask"]
            lm_loss = torch.sum(lm_losses * loss_mask, dim=-1) / loss_mask.sum(dim=-1)

            neg_output = model(**batch_neg)
            neg_lm_losses = mpu.vocab_parallel_cross_entropy(neg_output.contiguous().float(), no_model_batch_neg["labels"])
            neg_loss_mask = no_model_batch_neg["loss_mask"]
            neg_lm_loss = torch.sum(neg_lm_losses * neg_loss_mask, dim=-1) / neg_loss_mask.sum(dim=-1)

            loss = 1 + lm_loss - neg_lm_loss
            loss = torch.where(loss > 0, loss, torch.zeros_like(loss).to(device))
            loss = torch.mean(loss)

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
                        
                    logging_loss = total_loss
                    
            total_step += 1

        model.eval()
        all_sids = []
        all_cids = []
        all_losses = []
        with torch.no_grad():
            for batch, no_model_batch in tqdm(dev_dataloader, desc="Evaluating", disable=torch.distributed.get_rank() != 0):
                for k in batch:
                    batch[k] = batch[k].to(device)
                for k in no_model_batch:
                    no_model_batch[k] = no_model_batch[k].to(device)
                
                output = model(**batch)
                losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), no_model_batch["labels"])
                loss_mask = no_model_batch["loss_mask"]
                loss = torch.sum(losses * loss_mask, dim=-1) / loss_mask.sum(dim=-1)

                loss_tensor_list = [torch.zeros_like(loss).to(device) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(loss_tensor_list, loss.data, group=mpu.get_data_parallel_group())
                all_losses.extend(loss_tensor_list)

                sids = no_model_batch["sids"]
                sid_tensor_list = [torch.zeros_like(sids) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(sid_tensor_list, sids.data, group=mpu.get_data_parallel_group())
                all_sids.extend(sid_tensor_list)

                cids = no_model_batch["cids"]
                cid_tensor_list = [torch.zeros_like(cids) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(cid_tensor_list, cids.data, group=mpu.get_data_parallel_group())
                all_cids.extend(cid_tensor_list)

        if torch.distributed.get_rank() == 0:
            all_losses = torch.stack(all_losses).view(-1).cpu().detach().numpy()
            all_sids = torch.stack(all_sids).view(-1).cpu().detach().numpy()
            all_cids = torch.stack(all_cids).view(-1).cpu().detach().numpy()

            truth_labels = dev_dataset.truth_labels
            preds = [[] for _ in truth_labels]

            for sid, cid, loss in zip(all_sids, all_cids, all_losses):
                preds[sid].append((cid, loss))

            preds = [min(p, key=lambda x: x[1])[0] for p in preds if len(p) > 0]

            yprint("Acc: {}".format(sum([int(p == l) for p, l in zip(preds, truth_labels)]) / len(truth_labels)))
            eval_results_dir = os.path.join(model_dir, "eval_e{}".format(e))
            os.makedirs(eval_results_dir, exist_ok=True)
            with open(os.path.join(eval_results_dir, "eval_result.txt"), "w") as f:
                f.write("Acc: {}\n".format(sum([int(p == l) for p, l in zip(preds, truth_labels)]) / len(truth_labels)))

            with open(os.path.join(eval_results_dir, "pred.txt"), "w") as f:
                f.write(str(preds))
            with open(os.path.join(eval_results_dir, "truth.txt"), "w") as f:
                f.write(str(truth_labels))

        torch.distributed.barrier()
        if args.save:
            save_checkpoint(global_step, model, optimizer, lr_scheduler, args)

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
