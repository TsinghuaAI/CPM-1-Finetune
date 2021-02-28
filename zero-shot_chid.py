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
import torch
import time
import json
from tqdm import tqdm
from arguments import get_args
from utils import Timers
from data_utils.tokenization_gpt2 import GPT2Tokenizer
import mpu
import json

from tqdm import tqdm
from data.samplers import DistributedBatchSampler, RandomSampler

from utils import initialize_distributed, set_random_seed, setup_model_and_optimizer, yprint


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
            data = json.load(f)

        self.samples, self.sizes, self.truth_labels = self.process(data)

        self.max_size = max(self.sizes)

    def process(self, data):
        contents = data["contents"]
        sids = data["sids"]
        truth_labels = data["labels"]
        cids = data["cids"]
        sizes = []
        samples = []
        for content, sid, cid in zip(tqdm(contents[:int(self.ratio * len(contents))], desc="Processing", disable=(torch.distributed.get_rank() != 0)), sids, cids):
            input_ids = content
            input_ids = input_ids + [self.eod_token]
            length = len(input_ids) - 1
            sizes.append(length)
            samples.append({
                "sid": sid,
                "cid": cid,
                "input_ids": input_ids[:-1],
                "loss_mask": [1.0] * length,
                "labels": input_ids[1:]
            })

        return samples, sizes, truth_labels

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        return self.samples[idx], self.sizes[idx]

    def collate(self, x):
        bs = len(x)
        samps = [s[0] for s in x]
        sizes = [s[1] for s in x]
        max_size = self.max_size

        attn_mask = torch.tril(torch.ones((max_size, max_size))).unsqueeze(0)
        position_ids = torch.arange(max_size, dtype=torch.long).unsqueeze(0).repeat(bs, 1)
        if self.args.fp16:
            attn_mask = attn_mask.half()

        batch_sample = {
            "input_ids": torch.ones(bs, max_size).long() * self.pad_id,
            "attention_mask": attn_mask,
            "position_ids": position_ids
        }

        no_model_sample = {
            "sids": torch.zeros(bs).long(),
            "cids": torch.zeros(bs).long(),
            "loss_mask": torch.zeros(bs, max_size).float(),
            "labels": torch.ones(bs, max_size).long() * self.pad_id,
        }

        for i, samp in enumerate(samps):
            batch_sample["input_ids"][i, :len(samp["input_ids"])] = torch.tensor(samp["input_ids"])
            no_model_sample["loss_mask"][i, :len(samp["loss_mask"])] = torch.tensor(samp["loss_mask"])
            no_model_sample["labels"][i, :len(samp["labels"])] = torch.tensor(samp["labels"])
            no_model_sample["sids"][i] = torch.tensor(samp["sid"])
            no_model_sample["cids"][i] = torch.tensor(samp["cid"])

        return batch_sample, no_model_sample


def load_data(args, data_type, tokenizer, ratio=1):
    data_path = args.data_dir
    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
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

    # load data
    test_dataloader, test_dataset = load_data(args, 'test', tokenizer, 1)
    # Set an arbitrary positive integer since the optimizer and the scheduler will not be used when do eval.
    args.train_iters = 1

    # Model
    model, _, _ = setup_model_and_optimizer(args)

    device = torch.cuda.current_device()

    # give a time stemp to the model
    cur_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    results_dir = os.path.join(args.results_dir, "{}-{}".format(args.model_name, cur_time))

    if torch.distributed.get_rank() == 0:
        os.makedirs(results_dir, exist_ok=True)

    model.eval()
    all_sids = []
    all_cids = []
    all_losses = []
    with torch.no_grad():
        for batch, no_model_batch in tqdm(test_dataloader, desc="Evaluating", disable=(torch.distributed.get_rank() != 0)):
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

        truth_labels = test_dataset.truth_labels
        preds = [[] for _ in truth_labels]

        for sid, cid, loss in zip(all_sids, all_cids, all_losses):
            preds[sid].append((cid, loss))

        preds = [min(p, key=lambda x: x[1])[0] for p in preds if len(p) > 0]

        yprint("Acc: {}".format(sum([int(p == l) for p, l in zip(preds, truth_labels)]) / len(truth_labels)))
        with open(os.path.join(results_dir, "zero-shot_result.txt"), "w") as f:
            f.write("Acc: {}\n".format(sum([int(p == l) for p, l in zip(preds, truth_labels)]) / len(truth_labels)))

    torch.distributed.barrier()

if __name__ == "__main__":

    main()
