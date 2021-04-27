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

"""Pretrain Enc-Dec"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.
USE_TORCH_DDP = False

from datetime import datetime
import os
import random
import math
import numpy as np
import torch
import json
from tqdm import tqdm

import deepspeed

from arguments import get_args
from data_utils.tokenization_enc_dec import EncDecTokenizer
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import EncDecModel, EncDecConfig
from model import enc_dec_get_params_for_weight_decay_optimization

if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from model import DistributedDataParallel as DDP
import mpu
from apex.optimizers import FusedAdam as Adam
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from utils import report_memory
from utils import print_args
from utils import print_rank_0, save_rank_0
import torch.distributed as dist

from data.enc_dec_dataset import build_train_valid_test_datasets
from data.samplers import DistributedBatchSampler, RandomSampler

from T5Dataset import AFQMCDataset, C3Dataset, C3Dataset2, CHIDDataset, CHIDDataset2, CMNLIDataset, CMRCDataset, CSLDataset, CombinedDataset, IFLYTEKDataset, OCNLIDataset, T5Dataset, TNewsDataset, WSCDataset, WSCDataset2

import torch.nn.functional as F

import time


def get_model(args, vocab_size):
    """Build the model."""

    print_rank_0('building Enc-Dec model ...')
    config = EncDecConfig.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    model = EncDecModel(config,
                        parallel_output=True,
                        checkpoint_activations=args.checkpoint_activations,
                        checkpoint_num_layers=args.checkpoint_num_layers)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
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


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    param_groups = enc_dec_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                        lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        optimizer = Adam(param_groups,
                         lr=args.lr, weight_decay=args.weight_decay)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               gradient_accumulation_steps=args.gradient_accumulation_steps)

    return lr_scheduler


def setup_model_and_optimizer(args, vocab_size, ds_config):
    """Setup model and optimizer."""

    model = get_model(args, vocab_size)
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
            dist_init_required=False,
            config_params=ds_config
        )

    print(args.load)
    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=False):
    for k in model_batch:
        model_batch[k] = model_batch[k].to(device)
    for k in no_model_batch:
        no_model_batch[k] = no_model_batch[k].to(device)

    if keep_enc_hidden:
        enc_outputs = model(**model_batch, only_encoder=True)
        enc_hidden_states = enc_outputs["encoder_last_hidden_state"]
        output = model(**model_batch, enc_hidden_states=enc_hidden_states)
    else:
        output = model(**model_batch)
    
    logits = output["lm_logits"]
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), no_model_batch["labels"])
    loss_mask = no_model_batch["loss_mask"]
    losses = (losses * loss_mask).sum(-1) / loss_mask.sum(-1)
    loss = losses.mean()

    if keep_enc_hidden:
        return loss, logits, enc_hidden_states
    else:
        return loss, logits


def backward_step(args, loss, model, optimizer):
    # backward
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)


def train(args, data_config, tokenizer, model, optimizer, lr_scheduler,
          train_dataset, train_dataloader, dev_dataset, dev_dataloader, device):
    """Train the model."""

    eval_func = data_config[args.data_name]["eval_func"]

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss = 0.0

    step, global_step = 1, 1

    for e in range(args.epochs):
        model.train()
        for model_batch, no_model_batch in train_dataloader:

            loss, _ = forward_step(args, model_batch, no_model_batch, model, device)
            if torch.distributed.get_rank() == 0:
                print(loss)

            backward_step(args, loss, model, optimizer)

            # Update losses.
            total_loss += loss.item()

            if args.deepspeed:
                model.step()
            else:
                optimizer.step()
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()

            # Logging.
            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                avg_lm_loss = total_loss / (args.log_interval * args.gradient_accumulation_steps)
                log_string = 'epoch {:3d}/{:3d} |'.format(e, args.epochs)
                log_string += ' global iteration {:8d}/{:8d} |'.format(global_step, args.train_iters)
                log_string += ' learning rate {:.3} |'.format(learning_rate)
                log_string += ' lm loss {:.6} |'.format(avg_lm_loss)
                if args.fp16:
                    log_string += ' loss scale {:.1f} |'.format(optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
                print_rank_0(log_string)
                save_rank_0(args, log_string)
                total_loss = 0.0

            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_checkpoint(global_step, model, optimizer, lr_scheduler, args)

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0 and args.do_valid:
                prefix = 'iteration {} | '.format(global_step)
                eval_loss, acc = eval_func(args, tokenizer, dev_dataset, dev_dataloader, model, device, mode="dev")
                log_string = prefix + " eval_loss: " + str(eval_loss) + " | eval acc: " + str(acc)
                print_rank_0(log_string)
                save_rank_0(args, log_string)

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

    return global_step


def evaluate(args, tokenizer: EncDecTokenizer, eval_dataset, eval_data_loader, model, device, mode='dev'):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            loss, logits = forward_step(args, model_batch, no_model_batch, model, device)

            total_loss += loss.item()

            logits_list = [torch.zeros_like(logits) for _ in range(mpu.get_model_parallel_world_size())]
            torch.distributed.all_gather(logits_list, logits, mpu.get_model_parallel_group())

            gathered_logits = torch.cat(logits_list, dim=-1)

            pred_token_logits = gathered_logits[:, 1, :]
            preds = torch.argmax(pred_token_logits, dim=-1)
            labels = no_model_batch["labels"][:, 1]
            
            gathered_preds = [torch.zeros_like(preds) for _ in range(mpu.get_data_parallel_world_size())]
            gathered_labels = [torch.zeros_like(labels) for _ in range(mpu.get_data_parallel_world_size())]

            torch.distributed.all_gather(gathered_preds, preds.contiguous(), mpu.get_data_parallel_group())
            torch.distributed.all_gather(gathered_labels, labels.contiguous(), mpu.get_data_parallel_group())

            all_preds.extend(gathered_preds)
            all_labels.extend(gathered_labels)

            step += 1

    total_loss /= step

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)

    return total_loss, acc


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


def evaluate_gen(args, tokenizer: EncDecTokenizer, eval_dataset: T5Dataset, eval_data_loader, model, device, mode="dev"):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_preds = []
    all_labels = []
    all_truth = []

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            
            loss, logits, enc_hidden_states = forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=True)

            total_loss += loss.item()

            # for generating responses
            # we only use the <go> token, so truncate other tokens
            dec_input_ids = model_batch['dec_input_ids'][..., :1]
            dec_attention_mask = model_batch['dec_attention_mask'][..., :1, :1]
            # # we use past_key_values, so only the current token mask is needed
            cross_attention_mask = model_batch['cross_attention_mask'][..., :1, :]

            unfinished_sents = model_batch['enc_input_ids'].new(model_batch['enc_input_ids'].size(0)).fill_(1)
            output_ids = model_batch['enc_input_ids'].new_zeros([model_batch['enc_input_ids'].size(0), 0])
            past_key_values = None

            gen_len = 0
            while gen_len < args.dec_seq_length:
                if unfinished_sents.max() == 0:
                    tokens_to_add = tokenizer.pad_id * (1 - unfinished_sents)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

                else:
                    dec_outputs = model(
                        dec_input_ids=dec_input_ids,
                        dec_attention_mask=dec_attention_mask,
                        cross_attention_mask=cross_attention_mask,
                        enc_hidden_states=enc_hidden_states,
                        past_key_values=past_key_values,
                    )
                    lm_logits = dec_outputs["lm_logits"]
                    past_key_values = dec_outputs['past_key_values']

                    gathered_lm_logits = [torch.zeros_like(lm_logits).to(device) for _ in range(mpu.get_model_parallel_world_size())]
                    torch.distributed.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_model_parallel_group())

                    lm_logits = torch.cat(gathered_lm_logits, dim=-1)

                    next_token_logits = lm_logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    # next_token_logscores = top_k_logits(next_token_logits, top_k=args.top_k, top_p=args.top_p)
                    # probs = F.softmax(next_token_logscores, dim=-1)
                    # next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                    tokens_to_add = next_token * unfinished_sents + tokenizer.pad_id * (1 - unfinished_sents)

                    dec_input_ids = tokens_to_add.unsqueeze(-1)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                    # let the current token attend to all previous tokens
                    dec_attention_mask = torch.cat([dec_attention_mask, dec_attention_mask[:, :, :, -1:]], dim=-1)

                gen_len += 1
                unfinished_sents.mul_(tokens_to_add.ne(tokenizer.get_sentinel_id(1)).long())
            
            gathered_preds = [torch.zeros_like(output_ids) for _ in range(mpu.get_data_parallel_world_size())]
            gathered_labels = [torch.zeros_like(no_model_batch["labels"]) for _ in range(mpu.get_data_parallel_world_size())]

            torch.distributed.all_gather(gathered_preds, output_ids, mpu.get_data_parallel_group())
            torch.distributed.all_gather(gathered_labels, no_model_batch["labels"], mpu.get_data_parallel_group())

            all_preds.extend(gathered_preds)
            all_labels.extend(gathered_labels)

            if args.data_name in ["wsc2"]:
                gathered_truth = [torch.zeros_like(no_model_batch["truth"]) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(gathered_truth, no_model_batch["truth"], mpu.get_data_parallel_group())
                all_truth.extend(gathered_truth)
                
            step += 1

    total_loss /= step

    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    all_labels = torch.cat(all_labels, dim=0).cpu().tolist()

    all_preds = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_preds]
    all_labels = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_labels]

    with open(os.path.join(args.save, "preds.txt"), "w") as f:
        for p in all_preds:
            f.write(tokenizer.decode(p) + "\n")
    
    with open(os.path.join(args.save, "labels.txt"), "w") as f:
        for l in all_labels:
            f.write(tokenizer.decode(l) + "\n")

    if args.data_name in ["wsc2"]:
        all_truth = torch.cat(all_truth, dim=0).cpu().tolist()
        acc = wsc2_metric(tokenizer, all_preds, all_labels, all_truth)
    elif args.data_name in ["cmrc"]:
        acc = cmrc_metric(tokenizer, all_preds, eval_dataset.data)
    else:
        acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)

    return total_loss, acc


def wsc2_metric(tokenizer: EncDecTokenizer, all_preds, all_labels, all_truth):
    all_preds = [p[1:-1] for p in all_preds]
    all_labels = [l[1:-1] for l in all_labels]
    res = []
    for p, l, t in zip(all_preds, all_labels, all_truth):
        p = tokenizer.decode(p)
        l = tokenizer.decode(l)
        pp = int((p in l) or (l in p))
        res.append(int(pp == t))

    acc = sum(res) / len(res)
    return acc


def cmrc_metric(tokenizer: EncDecTokenizer, all_preds, data):
    print("Doing cmrc metric")        
    all_preds = [tokenizer.decode(p[1:-1]) for p in all_preds]
    res = [int(p in d["truth"]) for p, d in zip(all_preds, data)]

    acc = sum(res) / len(res)
    return acc


def evaluate_and_print_results(tokenizer, prefix, data_iterator, model,
                               args, timers, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    lm_loss = evaluate(tokenizer, data_iterator, model, args, timers, verbose)
    lm_ppl = math.exp(min(20, lm_loss))
    string = '-' * 100 + "\n"
    string += ' validation loss at {} | '.format(prefix)
    string += 'LM loss: {:.6} | '.format(lm_loss)
    string += 'LM PPL: {:.6}'.format(lm_ppl)
    length = len(string) + 1
    string = '-' * length + "\n" + string + "\n" + '-' * length
    print_rank_0(string)
    save_rank_0(args, string)

    return lm_loss


def set_deepspeed_activation_checkpointing(args):

    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    deepspeed.init_distributed()

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def load_data(args, data_config, data_type, tokenizer, ratio=1):
    data_path = os.path.join(args.data_path, data_type + args.data_ext)

    dataset = data_config[args.data_name]["dataset"](args, tokenizer, data_path, data_type, ratio=ratio, prefix=args.data_prefix, cache_path=data_config[args.data_name]["cache_path"])

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    if data_type == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)

    data_loader = torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       collate_fn=dataset.collate)

    # Torch dataloader.
    return data_loader, dataset


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    os.makedirs(args.save, exist_ok=True)

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain Enc-Dec model')
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # setup tokenizer
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, tokenizer.vocab_size, ds_config)
    device = torch.cuda.current_device()

    data_config = {
        "tnews": {
            "dataset": TNewsDataset,
            "eval_func": evaluate,
            "cache_path": None
        },
        "afqmc": {
            "dataset": AFQMCDataset,
            "eval_func": evaluate,
            "cache_path": None
        },
        "ocnli": {
            "dataset": OCNLIDataset,
            "eval_func": evaluate,
            "cache_path": None
        },
        "iflytek": {
            "dataset": IFLYTEKDataset,
            "eval_func": evaluate_gen,
            "cache_path": None
        },
        "cmnli": {
            "dataset": CMNLIDataset,
            "eval_func": evaluate,
            "cache_path": None
        },
        "csl": {
            "dataset": CSLDataset,
            "eval_func": evaluate,
            "cache_path": None
        },
        "chid": {
            "dataset": CHIDDataset,
            "eval_func": evaluate_gen,
            "cache_path": args.data_path
        },
        "cmrc": {
            "dataset": CMRCDataset,
            "eval_func": evaluate_gen,
            "cache_path": None
        },
        "c3": {
            "dataset": C3Dataset,
            "eval_func": evaluate_gen,
            "cache_path": None
        },
        "c32": {
            "dataset": C3Dataset2,
            "eval_func": evaluate,
            "cache_path": None
        },
        "wsc": {
            "dataset": WSCDataset,
            "eval_func": evaluate,
            "cache_path": None
        },
        "wsc2": {
            "dataset": WSCDataset2,
            "eval_func": evaluate_gen,
            "cache_path": None
        },
        "chid2": {
            "dataset": CHIDDataset2,
            "eval_func": evaluate,
            "cache_path": None
        },
        "combined": {
            "dataset": CombinedDataset,
            "eval_func": evaluate,
            "cache_path": args.data_path
        }
    }

    if args.do_train:
        train_dataloader, train_dataset = load_data(args, data_config, 'train', tokenizer, ratio=1)
        dev_dataloader, dev_dataset  = load_data(args, data_config, 'dev', tokenizer, ratio=1)
        train(args, data_config, tokenizer, model, optimizer, lr_scheduler, train_dataset, train_dataloader, dev_dataset, dev_dataloader, device)

    if args.do_eval:
        eval_dataloader, eval_dataset = load_data(args, data_config, 'dev', tokenizer, ratio=1)
        eval_func = data_config[args.data_name]["eval_func"]

        loss, acc = eval_func(args, tokenizer, eval_dataset, eval_dataloader, model, device, mode="test")

        log_string = "Eval result: loss: {:.6} | acc: {:.4}".format(loss, acc)
        print_rank_0(log_string)
        save_rank_0(args, log_string)

if __name__ == "__main__":
    main()
