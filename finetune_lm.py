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

"""Finetune CPM For Language Modeling"""
# Flag to use Pytorch ddp which uses overlapping communication and computation.

import os
import numpy as np
import torch
import time
import json
from tqdm import tqdm
from arguments import get_args
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from data_utils.tokenization_gpt2 import GPT2Tokenizer

import mpu
import json
import time

from tqdm import tqdm
from data.samplers import DistributedBatchSampler, RandomSampler

from utils import initialize_distributed, set_random_seed, setup_model_and_optimizer, yprint


class GenDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, split, tokenizer: GPT2Tokenizer, ratio=1):
        self.split = split
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.args = args
        self.world_size = args.world_size
        self.seq_length = args.seq_length

        self.pad_id = tokenizer.encoder['<pad>']
        self.eod_token = tokenizer.encoder['<eod>']
        args.eod_token = tokenizer.encoder['<eod>']

        with open(data_path, "r") as f:
            data = f.readlines()
        self.samples = self.process(data)

    def process(self, data):
        samples = []
        for doc in tqdm(data[:int(self.ratio * len(data))], disable=(torch.distributed.get_rank() != 0)):
            token_ids = self.tokenizer.encode(doc)
            token_ids.append(self.eod_token)
            start = 0
            while start + self.seq_length + 1 < len(token_ids):
                samples.append(token_ids[start: start + self.seq_length + 1])
                start = start + self.seq_length + 1
            samples.append(token_ids[start:] + [self.pad_id] * (self.seq_length + 1 - (len(token_ids) - start)))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate(self, samps):
        bs = len(samps)

        # triangle attention mask
        attn_mask = torch.tril(torch.ones((self.seq_length, self.seq_length))).unsqueeze(0)
        position_ids = torch.arange(self.seq_length, dtype=torch.long).unsqueeze(0).repeat(bs, 1)

        if self.args.fp16:
            attn_mask = attn_mask.half()

        # the data that need to go through the model
        batch_sample = {
            "input_ids": torch.ones(bs, self.seq_length).long() * self.pad_id,
            "attention_mask": attn_mask.unsqueeze(1),
            "position_ids": position_ids,
        }

        # the data that do not need to go through the model
        no_model_sample = {
            "labels": torch.ones(bs, self.seq_length).long() * self.pad_id,
            "loss_mask": torch.zeros(bs, self.seq_length).float()
        }

        for i, samp in enumerate(samps):
            assert len(samp) == self.seq_length + 1, (len(samp), self.seq_length)
            batch_sample["input_ids"][i] = torch.tensor(samp[:-1], dtype=torch.long)
            no_model_sample["labels"][i] = torch.tensor(samp[1:], dtype=torch.long)
            no_model_sample["loss_mask"][i] = (no_model_sample["labels"][i] != self.pad_id).float()

        return batch_sample, no_model_sample


def load_data(args, data_type, tokenizer, ratio=1):
    data_path = args.data_dir

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Dataset
    filename = os.path.join(data_path, data_type + '.txt')
    dataset = GenDataset(args, filename, data_type, tokenizer, ratio=ratio)
    
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


def evaluate(args, model, dataloader, device, mode="dev"):
    model.eval()
    all_losses = []
    with torch.no_grad():
        for batch, no_model_batch in tqdm(dataloader, desc="Evaluating {}".format(mode), disable=(torch.distributed.get_rank() != 0)):
            for k in batch:
                batch[k] = batch[k].to(device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(device)

            output = model(**batch)
            labels = no_model_batch["labels"]

            # cross_entropy loss
            losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
            loss_mask = no_model_batch["loss_mask"]
            losses = losses * loss_mask
                
            loss = torch.sum(losses, dim=-1) / loss_mask.sum(dim=-1)

            all_losses.extend(loss.tolist())


    return np.mean(all_losses)


def main():
    """Main training program."""

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
    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'), os.path.join(args.tokenizer_path, 'chinese_vocab.model'))

    # load train data
    if args.do_train:
        train_dataloader, _ = load_data(args, 'train', tokenizer, 1)
        dev_dataloader, _ = load_data(args, 'valid', tokenizer, 1)

        with open(args.deepspeed_config, "r") as f:
            deepspeed_conf = json.load(f)

        epoch = args.epoch
        grad_acc = deepspeed_conf["gradient_accumulation_steps"]
        args.train_iters = len(train_dataloader) * epoch / grad_acc

        # Model, optimizer, and learning rate.
        # TODO: maybe need to reinitialize optimizer
    elif args.do_eval:
        # Set an arbitrary positive integer since the optimizer and the scheduler will not be used when do eval.
        args.train_iters = 1

    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    device = torch.cuda.current_device()

    # give a time stemp to the model
    cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    results_dir = os.path.join(args.results_dir, "{}-{}".format(args.model_name, cur_time))
    os.makedirs(results_dir, exist_ok=True)

    if args.do_train and torch.distributed.get_rank() == 0:

        with open(os.path.join(results_dir, "train_log.txt"), "w") as f:
            f.write("Train losses:\n")

        with open(os.path.join(results_dir, "dev_log.txt"), "w") as f:
            f.write("Dev accs:\n")

    if args.do_train:
        total_loss, logging_loss, best_dev_loss = 0.0, 0.0, 1000000
        global_step, total_step, best_step = 0, 0, 0
        
        for e in range(epoch):
            model.train()
            for batch, no_model_batch in tqdm(train_dataloader, disable=(torch.distributed.get_rank() != 0)):
                for k in batch:
                    batch[k] = batch[k].to(device)
                for k in no_model_batch:
                    no_model_batch[k] = no_model_batch[k].to(device)

                output = model(**batch)
                labels = no_model_batch["labels"]
                losses = mpu.vocab_parallel_cross_entropy(output.contiguous().float(), labels)

                loss_mask = no_model_batch["loss_mask"].view(-1)
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

                model.backward(loss)
                model.step()

                torch.distributed.all_reduce(loss.data, group=mpu.get_data_parallel_group())
                loss.data = loss.data / mpu.get_data_parallel_world_size()
                total_loss += loss.item() / grad_acc

                if total_step % grad_acc == 0:
                    global_step += 1
                    if global_step != 0 and global_step % args.log_interval == 0:
                        # logging
                        if torch.distributed.get_rank() == 0:
                            train_log = "Epoch {}, global step {}, total step {}, train lm loss: {}".format(e, global_step, epoch * len(train_dataloader), (total_loss - logging_loss) / args.log_interval)
                            yprint(train_log)
                            with open(os.path.join(results_dir, "train_log.txt"), "a") as f:
                                f.write(train_log + "\n")

                        logging_loss = total_loss
    
                    if global_step != 0 and global_step % args.eval_interval == 0:
                        # evaluate on the dev
                        dev_loss = evaluate(args, model, dev_dataloader, device, mode="dev")
                        dev_results_dir = os.path.join(results_dir, "dev_step-{}".format(global_step))

                        if dev_loss < best_dev_loss:
                            best_dev_loss = dev_loss
                            best_step = global_step

                        if torch.distributed.get_rank() == 0:
                            # we will only write the log file once
                            dev_log = "Epoch: {}, Global step: {}, Loss: {}, PPL: {}".format(e, global_step, dev_loss, np.exp(dev_loss))
                            yprint(dev_log)
                            os.makedirs(dev_results_dir, exist_ok=True)
                            with open(os.path.join(dev_results_dir, "dev_result.txt"), "w") as f:
                                f.write(dev_log + "\n")
                            with open(os.path.join(results_dir, "dev_log.txt"), "a") as f:
                                f.write(dev_log + "\n")

                        torch.distributed.barrier()
                        
                        args.save = dev_results_dir
                        save_checkpoint(global_step, model, optimizer, lr_scheduler, args)

                total_step += 1

        with open(os.path.join(results_dir, "final_dev_result.txt"), "a") as f:
            f.write("Best Loss: {} Best PPL: {}, Best step: {}\n".format(best_dev_loss, np.exp(best_dev_loss), best_step))

    if args.do_eval:
        # evaluate on the test
        test_dataloader, test_dataset = load_data(args, 'test', tokenizer, 1)

        if args.do_train:
            # if do training, then evaluate the one with the max acc on dev set.
            eval_ckpt_path = os.path.join(results_dir, "dev_step-{}".format(best_step))
            args.load = eval_ckpt_path
        else:
            # if only do eval, then evaluate the one specified by the user.
            args.load = args.eval_ckpt_path            
        
        load_checkpoint(model=model, optimizer=None, lr_scheduler=None, args=args)
        eval_loss = evaluate(args, model, test_dataloader, device, mode="test")

        if torch.distributed.get_rank() == 0:
            eval_log = "Checkpoint from {}: Loss: {} PPL: {}".format(args.load, eval_loss, np.exp(eval_loss))
            yprint(eval_log)
            with open(os.path.join(results_dir, "eval_log.txt"), "w") as f:
                f.write(eval_log + "\n")

        torch.distributed.barrier()


if __name__ == "__main__":

    main()
