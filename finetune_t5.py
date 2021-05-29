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
from infer_funcs import infer_afqmc, infer_c32, infer_chid2, infer_cmnli, infer_cmrc, infer_csl, infer_iflytek, infer_ocnli, infer_tnews, infer_wsc2
import os
import random
import math
import numpy as np
import torch
import json
from tqdm import tqdm
import shutil

import deepspeed

from arguments import get_args
from data_utils.tokenization_enc_dec import EncDecTokenizer, MT5EncDecTokenizer
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import EncDecModel, EncDecConfig
from model import enc_dec_get_params_for_weight_decay_optimization, enc_dec_get_params_for_prompt_optimization

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

from T5Dataset import AFQMCDataset, C3Dataset, C3Dataset2, CHIDDataset, CHIDDataset2, CHIDDataset3, CMNLIDataset, CMRCDataset, CSLDataset, CSLDataset2, CombinedDataset, IFLYTEKDataset, OCNLIDataset, T5Dataset, TNewsDataset, WSCDataset, WSCDataset2, WSCDataset3
from T5Dataset import QuoteRDataset, SogouLogDataset
import torch.nn.functional as F

import time


def get_model(args, vocab_size, prompt_config=None):
    """Build the model."""

    print_rank_0('building Enc-Dec model ...')
    config = EncDecConfig.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    model = EncDecModel(config,
                        parallel_output=True,
                        checkpoint_activations=args.checkpoint_activations,
                        checkpoint_num_layers=args.checkpoint_num_layers,
                        data_hack="chid" if args.data_name == "chid3" else None,
                        prompt_config=prompt_config)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())
    if args.prompt_tune:
        model.init_prompt_embeds()

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


def get_optimizer(model, args, prompt_config=None):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    if args.prompt_tune:
        param_groups = enc_dec_get_params_for_prompt_optimization(model)
    else:
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

    if torch.distributed.get_rank() == 0:
        print(optimizer.param_groups)

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


def setup_model_and_optimizer(args, vocab_size, ds_config, prompt_config=None):
    """Setup model and optimizer."""

    model = get_model(args, vocab_size, prompt_config)
    optimizer = get_optimizer(model, args, prompt_config)
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


def forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=False, do_infer=False):
    for k in model_batch:
        model_batch[k] = model_batch[k].to(device)
    for k in no_model_batch:
        # if type(no_model_batch[k]) == list:
        #     continue
        no_model_batch[k] = no_model_batch[k].to(device)

    if keep_enc_hidden:
        enc_outputs = model(**model_batch, only_encoder=True)
        enc_hidden_states = enc_outputs["encoder_last_hidden_state"]
        output = model(**model_batch, enc_hidden_states=enc_hidden_states)
    else:
        output = model(**model_batch)
    
    logits = output["lm_logits"]
    forw_out = {
        "logits": logits
    }
    if keep_enc_hidden:
        forw_out["enc_hidden_states"] = enc_hidden_states
    
    if not do_infer:
        losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), no_model_batch["labels"])
        # if torch.distributed.get_rank() == 0:
        #     print(losses)

        loss_mask = no_model_batch["loss_mask"]
        losses = (losses * loss_mask).sum(-1) / loss_mask.sum(-1)
        loss = losses.mean()

        forw_out["loss"] = loss
    
    return forw_out


totalnum = 0
rightnum = 0
class _RankCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score, loss_mask):
        losses = F.cross_entropy(score, torch.zeros(score.size()[0], dtype=torch.long).to(score.device)) * loss_mask
        global totalnum, rightnum
        totalnum += int(score.size()[0])
        rightnum += int((torch.max(score, dim = 1)[1] == 0).sum())
        print_rank_0("accuracy: %s\t%s\t%s" % (rightnum, totalnum, rightnum/totalnum))
        torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.SUM, group=mpu.get_model_parallel_group())
        softmax = F.softmax(score, dim = 1)
        ctx.save_for_backward(softmax, loss_mask)

        return losses

    @staticmethod
    def backward(ctx, grad_output):
        softmax, loss_mask = ctx.saved_tensors
        grad_input = softmax
        grad_input[:, 0] -= 1.0
        grad_input.mul_(loss_mask)
        grad_input.mul_(grad_output)

        return grad_input, None

def forward_rank_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=False, do_infer=False, quoter_valid = False):
    for k in model_batch:
        model_batch[k] = model_batch[k].to(device)
    for k in no_model_batch:
        # if type(no_model_batch[k]) == list:
        #     continue
        no_model_batch[k] = no_model_batch[k].to(device)

    if keep_enc_hidden:
        enc_outputs = model(**model_batch, only_encoder=True)
        enc_hidden_states = enc_outputs["encoder_last_hidden_state"]
        output = model(**model_batch, enc_hidden_states=enc_hidden_states)
    else:
        output = model(**model_batch)
    
    logits = output["lm_logits"]
    forw_out = {
        "logits": logits
    }
    if keep_enc_hidden:
        forw_out["enc_hidden_states"] = enc_hidden_states

    assert logits.size()[1] == 2
    if not do_infer:
        rank = mpu.get_model_parallel_rank()
        relevant_logit = logits[:,-1,400]
        # if valid:
        #     forw_out["score"] = relevant_logit
        #     return forw_out
        if rank != 0:
            loss_mask = torch.tensor(0, dtype=torch.long).to(logits.device)
        else:
            loss_mask = torch.tensor(1, dtype=torch.long).to(logits.device)
        if quoter_valid:
            forw_out["score"] = relevant_logit
            forw_out["loss"] = 0
            return forw_out
        score = relevant_logit.view(-1, 2) / args.temp

        losses = _RankCrossEntropy.apply(score, loss_mask)

        loss = losses.mean()

        forw_out["loss"] = loss
        forw_out["score"] = score
    
    return forw_out



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

    best_accs = []

    for e in range(args.epochs):
        model.train()
        for model_batch, no_model_batch in train_dataloader:
            if args.data_name in ["sogou-log", "quoter"]:
                forw_out = forward_rank_step(args, model_batch, no_model_batch, model, device)
            else:
                forw_out = forward_step(args, model_batch, no_model_batch, model, device)
            loss = forw_out["loss"]
            
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
                eval_loss, acc = eval_func(args, tokenizer, data_config, dev_dataset, dev_dataloader, model, device, mode="dev")
                model.train()
                log_string = prefix + " eval_loss: " + str(eval_loss) + " | eval acc: " + str(acc)
                print_rank_0(log_string)
                save_rank_0(args, log_string)

                if args.max_save > 0:
                    i = 0
                    while i < len(best_accs):
                        if best_accs[i][1] < acc:
                            break
                        i += 1
                    if len(best_accs) < args.max_save or i < len(best_accs):
                        best_accs.insert(i, (global_step, acc))
                        if len(best_accs) > args.max_save:
                            step_to_be_rm, acc_to_be_rm = best_accs[-1]
                            if torch.distributed.get_rank() == 0:
                                shutil.rmtree(os.path.join(args.save, "acc_{}_{:.3}".format(step_to_be_rm, acc_to_be_rm)))
                        save_checkpoint(global_step, model, optimizer, lr_scheduler, args, save_dir=os.path.join(args.save, "acc_{}_{:.3}".format(global_step, acc)))
                        best_accs = best_accs[:args.max_save]

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

    return global_step


def evaluate(args, tokenizer: EncDecTokenizer, data_config, eval_dataset, eval_data_loader, model, device, mode='dev'):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_idx = []
    all_preds = []
    all_labels = []
    all_L = []

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            forw_out = forward_step(args, model_batch, no_model_batch, model, device, do_infer=(mode=="infer"))
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            logits_list = [torch.zeros_like(forw_out["logits"]) for _ in range(mpu.get_model_parallel_world_size())]
            torch.distributed.all_gather(logits_list, forw_out["logits"], mpu.get_model_parallel_group())

            gathered_logits = torch.cat(logits_list, dim=-1)

            pred_token_logits = gathered_logits[:, 1, :]
            # pred_token_logits = -F.log_softmax(pred_token_logits, dim=-1)
            # labels = no_model_batch["labels"][:, 1]
            # if torch.distributed.get_rank() == 0:
            #     L = [p[x] for p, x in zip(pred_token_logits.cpu().tolist(), labels.cpu().tolist())]
            #     print(L)
            #     all_L.extend(L)

            preds = torch.argmax(pred_token_logits, dim=-1)
            gathered_preds = [torch.zeros_like(preds) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_preds, preds.contiguous(), mpu.get_data_parallel_group())
            all_preds.extend(gathered_preds)
            
            gathered_idx = [torch.zeros_like(no_model_batch["idx"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_idx, no_model_batch["idx"].contiguous(), mpu.get_data_parallel_group())
            all_idx.extend(gathered_idx)

            if mode != "infer":
                labels = no_model_batch["labels"][:, 1]
                gathered_labels = [torch.zeros_like(labels) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(gathered_labels, labels.contiguous(), mpu.get_data_parallel_group())
                all_labels.extend(gathered_labels)

            step += 1

    total_loss /= step

    all_idx = torch.cat(all_idx, dim=0).cpu().tolist()
    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()

    # if torch.distributed.get_rank() == 0:
    #     print(sum(all_L) / len(all_L))
    # #     print(all_preds)

    if mode == "infer":
        return data_config[args.data_name]["infer_func"](args, tokenizer, all_idx, all_preds)
    else:
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()

        acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)

        return total_loss, acc


def evaluate_rank(args, tokenizer: EncDecTokenizer, data_config, eval_dataset, eval_data_loader, model, device, mode='dev'):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_idx = []
    all_preds = []
    all_labels = []
    all_L = []

    total = torch.tensor(0, dtype=torch.long).to(device)
    right = torch.tensor(0, dtype=torch.long).to(device)
    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            forw_out = forward_rank_step(args, model_batch, no_model_batch, model, device, do_infer=(mode=="infer"))
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            score = forw_out["score"] # batch, 2
            total += score.size()[0]
            right += (torch.max(score, dim = 1)[1] == 0).int().sum()

            step += 1
    rank = mpu.get_model_parallel_rank()
    if rank != 0:
        total *= 0
        right *= 0
    torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(right, op=torch.distributed.ReduceOp.SUM, group=mpu.get_data_parallel_group())

    total_loss /= step
    return total_loss, float(right.float() / total.float())


def evaluate_quoter(args, tokenizer: EncDecTokenizer, data_config, eval_dataset, eval_data_loader, model, device, mode='dev'):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_qidx = []
    all_cidx = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            forw_out = forward_rank_step(args, model_batch, no_model_batch, model, device, do_infer=(mode=="infer"), quoter_valid=True)
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            score = forw_out["score"] # batch
            gathered_logits = [torch.zeros_like(score) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_logits, score.contiguous(), mpu.get_data_parallel_group())
            all_logits.extend(gathered_logits)
            
            gathered_qidx = [torch.zeros_like(no_model_batch["qidx"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_qidx, no_model_batch["qidx"].contiguous(), mpu.get_data_parallel_group())
            all_qidx.extend(gathered_qidx)

            gathered_cidx = [torch.zeros_like(no_model_batch["cidx"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_cidx, no_model_batch["cidx"].contiguous(), mpu.get_data_parallel_group())
            all_cidx.extend(gathered_cidx)

            gathered_labels = [torch.zeros_like(no_model_batch["label"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_labels, no_model_batch["label"].contiguous(), mpu.get_data_parallel_group())
            all_labels.extend(gathered_labels)

            step += 1
    
    rank = mpu.get_model_parallel_rank()

    all_qidx = torch.cat(all_qidx, dim=0).cpu().tolist()
    all_cidx = torch.cat(all_cidx, dim=0).cpu().tolist()
    all_logits = torch.cat(all_logits, dim=0).cpu().tolist()
    all_labels = torch.cat(all_labels, dim=0).cpu().tolist()

    pred = {}
    for qid, cid, logit, label in zip(all_qidx, all_cidx, all_logits, all_labels):
        if qid not in pred:
            pred[qid] = {"label": label, "cand": [], "qid": qid}
        assert pred[qid]["label"] == label
        pred[qid]["cand"].append((logit, cid))
    mrr = 0.0
    for qid in pred:
        pred[qid]["cand"].sort(key = lambda x:x[0], reverse=True)
        print_rank_0(json.dumps(pred[qid]))
        for ind, cand in enumerate(pred[qid]["cand"]):
            if cand[1] == pred[qid]["label"] - 1:
                mrr += 1.0 / (ind + 1)

    total_loss /= step
    return total_loss, mrr / len(all_labels)



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


def evaluate_gen(args, tokenizer: EncDecTokenizer, data_config, eval_dataset: T5Dataset, eval_data_loader, model, device, mode="dev"):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_preds = []
    all_labels = []
    all_idx = []

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            
            forw_out = forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=True, do_infer=(mode=="infer"))
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            enc_hidden_states = forw_out["enc_hidden_states"]

            # for generating responses
            # we only use the <go> token, so truncate other tokens
            dec_input_ids = model_batch['dec_input_ids'][..., :1]
            dec_attention_mask = model_batch['dec_attention_mask'][..., :1, :1]
            # we use past_key_values, so only the current token mask is needed
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
                    # print_rank_0(next_token_logits)
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
            torch.distributed.all_gather(gathered_preds, output_ids, mpu.get_data_parallel_group())
            all_preds.extend(gathered_preds)
            
            gathered_idx = [torch.zeros_like(no_model_batch["idx"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_idx, no_model_batch["idx"].contiguous(), mpu.get_data_parallel_group())
            all_idx.extend(gathered_idx)

            if mode != "infer":
                gathered_labels = [torch.zeros_like(no_model_batch["labels"]) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(gathered_labels, no_model_batch["labels"], mpu.get_data_parallel_group())
                all_labels.extend(gathered_labels)

            step += 1

    total_loss /= step

    all_idx = torch.cat(all_idx, dim=0).cpu().tolist()
    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    all_preds = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_preds]
    if args.data_name in ["wsc2"]:
        all_preds = [tokenizer.decode(p[1:-1]) for p in all_preds]
        all_preds = [int(p in d["cand_ids"] or d["cand_ids"] in p) for p, d in zip(all_preds, eval_dataset.data)]

    if mode == "infer":
        return data_config[args.data_name]["infer_func"](args, tokenizer, all_idx, all_preds)
    else:
        if args.data_name == "wsc2":
            all_labels = [d["truth"] for d in eval_dataset.data]
        else:
            all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
            all_labels = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_labels]
        
        if args.data_name in ["cmrc"]:
            acc = cmrc_metric(tokenizer, all_preds, eval_dataset.data)
        else:
            acc = sum([int(p == l) for p, l in zip(all_preds, all_labels)]) / len(all_preds)

        return total_loss, acc


def evaluate_span_extraction(args, tokenizer: EncDecTokenizer, data_config, eval_dataset: T5Dataset, eval_data_loader, model, device, mode="dev"):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.0
    step = 0

    all_preds = []
    all_labels = []
    all_idx = []


    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            # span_from_batch = model_batch["enc_input_ids"].tolist()
            # print("before the model calculation ", no_model_batch["span_from"][0].sum(), no_model_batch["span_from"][1].sum(), no_model_batch["span_from"][2].sum(), no_model_batch["span_from"][3].sum())
            forw_out = forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=True, do_infer=(mode=="infer"))
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            enc_hidden_states = forw_out["enc_hidden_states"]

            # for generating responses
            # we only use the <go> token, so truncate other tokens
            dec_input_ids = model_batch['dec_input_ids'][..., :1]
            dec_attention_mask = model_batch['dec_attention_mask'][..., :1, :1]
            # we use past_key_values, so only the current token mask is needed
            cross_attention_mask = model_batch['cross_attention_mask'][..., :1, :]

            unfinished_sents = model_batch['enc_input_ids'].new(model_batch['enc_input_ids'].size(0)).fill_(1)
            output_ids = model_batch['enc_input_ids'].new_zeros([model_batch['enc_input_ids'].size(0), 0])
            past_key_values = None

            gen_len = 0
            # now_pos = None
            # print("after the model calculation ", no_model_batch["span_from"][0].sum(), no_model_batch["span_from"][1].sum(), no_model_batch["span_from"][2].sum(), no_model_batch["span_from"][3].sum())
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
                    # print_rank_0(next_token_logits)
                    # next_token = torch.argmax(next_token_logits, dim=-1)
                    # if (next_token == 5).any():
                    #     print(next_token)
                    # next_token, now_pos = limit_vocab(next_token_logits, no_model_batch["span_from"], now_pos)
                    # next_token = torch.tensor(next_token, dtype=torch.long).to(device)
                    # next_token = limit_vocab1(next_token_logits, no_model_batch["span_from"])

                    # normal = torch.argmax(next_token_logits, dim=-1)
                    next_token = torch.argmax(next_token_logits + no_model_batch["span_from"] * 1000, dim=-1)
                    # print_rank_0("next_token %s %s" % (gen_len, torch.argmax(next_token_logits, dim=-1).tolist()))
                    # print_rank_0("next_token_logits %s %s" % (gen_len, next_token_logits.tolist()))
                    # print_rank_0("next_token + logits %s %s" % (gen_len, torch.argmax(next_token_logits + no_model_batch["span_from"] * 1000, dim=-1).tolist()))
                    # print_rank_0("next_token_logits + logits %s %s" % (gen_len, (next_token_logits + no_model_batch["span_from"] * 1000).tolist()))
                    # print_rank_0("==" * 20)

                    # if (normal == 5).any():
                    #     print(next_token, torch.argmax(next_token_logits, dim=-1))
                    #     print(next_token_logits[:,:10])
                    #     print(no_model_batch["span_from"][:,:10])

                    tokens_to_add = next_token * unfinished_sents + tokenizer.pad_id * (1 - unfinished_sents)

                    dec_input_ids = tokens_to_add.unsqueeze(-1)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                    # let the current token attend to all previous tokens
                    dec_attention_mask = torch.cat([dec_attention_mask, dec_attention_mask[:, :, :, -1:]], dim=-1)

                gen_len += 1
                unfinished_sents.mul_(tokens_to_add.ne(tokenizer.get_sentinel_id(1)).long())
            
            gathered_preds = [torch.zeros_like(output_ids) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_preds, output_ids, mpu.get_data_parallel_group())
            all_preds.extend(gathered_preds)

            # print_rank_0(gathered_preds)
            
            gathered_idx = [torch.zeros_like(no_model_batch["idx"]) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(gathered_idx, no_model_batch["idx"].contiguous(), mpu.get_data_parallel_group())
            all_idx.extend(gathered_idx)

            if mode != "infer":
                gathered_labels = [torch.zeros_like(no_model_batch["labels"]) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(gathered_labels, no_model_batch["labels"], mpu.get_data_parallel_group())
                all_labels.extend(gathered_labels)

            step += 1

    total_loss /= step

    all_idx = torch.cat(all_idx, dim=0).cpu().tolist()
    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    all_preds = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_preds]
    if args.data_name in ["wsc2"]:
        all_preds = [tokenizer.decode(p[1:-1]) for p in all_preds]
        all_preds = [int(p in d["cand_ids"] or d["cand_ids"] in p) for p, d in zip(all_preds, eval_dataset.data)]

    if mode == "infer":
        return data_config[args.data_name]["infer_func"](args, tokenizer, all_idx, all_preds)
    else:
        if args.data_name == "wsc2":
            all_labels = [d["truth"] for d in eval_dataset.data]
        else:
            all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
            all_labels = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in all_labels]
        
        if args.data_name in ["cmrc", "cmrc2"]:
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
    print(all_preds)
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


def load_data(args, data_config, data_type, tokenizer, prompt_config=None, ratio=1, drop_last=True, do_infer=False):
    data_path = os.path.join(args.data_path, data_type + args.data_ext)

    dataset = data_config[args.data_name]["dataset"](
        args,
        tokenizer,
        data_path,
        data_type,
        ratio=ratio,
        prefix=args.data_prefix,
        cache_path=data_config[args.data_name]["cache_path"],
        do_infer=do_infer,
        prompt_config=prompt_config)

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
                                            drop_last=drop_last,
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
    device = torch.cuda.current_device()

    # setup tokenizer
    if args.mt5:
        tokenizer = MT5EncDecTokenizer(args.tokenizer_path)
    else:
        tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size

    prompt_config = None
    if args.prompt_tune:
        with open(args.prompt_config, "r") as f:
            prompt_config = json.load(f)
            for t in ["enc", "dec"]:
                prompt_config[t]["init_ids"] = tokenizer.encode(prompt_config[t]["init_tokens"])
                pad_num = prompt_config[t]["prompt_len"] - len(prompt_config[t]["init_ids"])
                prompt_config[t]["init_ids"].extend(tokenizer.convert_tokens_to_ids([prompt_config[t]["default_init_token"] for _ in range(pad_num)]))
                prompt_config[t]["init_ids"] = torch.tensor(prompt_config[t]["init_ids"], dtype=torch.long).to(device)

    data_config = {
        "tnews": {
            "dataset": TNewsDataset,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_tnews
        },
        "afqmc": {
            "dataset": AFQMCDataset,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_afqmc
        },
        "ocnli": {
            "dataset": OCNLIDataset,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_ocnli
        },
        "iflytek": {
            "dataset": IFLYTEKDataset,
            "eval_func": evaluate_gen,
            "cache_path": None,
            "infer_func": infer_iflytek
        },
        "cmnli": {
            "dataset": CMNLIDataset,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_cmnli
        },
        "csl": {
            "dataset": CSLDataset,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_csl
        },
        "chid": {
            "dataset": CHIDDataset,
            "eval_func": evaluate_gen,
            "cache_path": args.data_path,
            "infer_func": None
        },
        "cmrc": {
            "dataset": CMRCDataset,
            "eval_func": evaluate_gen,
            "cache_path": None,
            "infer_func": infer_cmrc
        },
        "cmrc2": {
            "dataset": CMRCDataset,
            "eval_func": evaluate_span_extraction,
            "cache_path": None,
            "infer_func": infer_cmrc
        },
        "c3": {
            "dataset": C3Dataset,
            "eval_func": evaluate_gen,
            "cache_path": None,
            "infer_func": None
        },
        "c32": {
            "dataset": C3Dataset2,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_c32
        },
        "wsc": {
            "dataset": WSCDataset,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": None
        },
        "wsc2": {
            "dataset": WSCDataset2,
            "eval_func": evaluate_gen,
            "cache_path": None,
            "infer_func": infer_wsc2
        },
        "chid2": {
            "dataset": CHIDDataset2,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_chid2
        },
        "combined": {
            "dataset": CombinedDataset,
            "eval_func": evaluate,
            "cache_path": args.data_path
        },
        "csl2": {
            "dataset": CSLDataset2,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_csl
        },
        "wsc3": {
            "dataset": WSCDataset3,
            "eval_func": evaluate_gen,
            "cache_path": None,
            "infer_func": infer_wsc2
        },
        "chid3": {
            "dataset": CHIDDataset3,
            "eval_func": evaluate,
            "cache_path": None,
            "infer_func": infer_chid2
        },
        "sogou-log": {
            "dataset": SogouLogDataset,
            "eval_func": evaluate_rank,
            "cache_path": None,
            "infer_func": infer_csl
        },
        "quoter": {
            "dataset": QuoteRDataset,
            "eval_func": evaluate_quoter,
            "cache_path": None,
            "infer_func": infer_csl
        }
    }

    if args.do_train:
        train_dataloader, train_dataset = load_data(args, data_config, 'train', tokenizer, prompt_config, ratio=1)
        dev_dataloader, dev_dataset  = load_data(args, data_config, 'dev', tokenizer, prompt_config, ratio=1) #, drop_last=False)
        if args.train_iters == -1:
            args.train_iters = len(train_dataset) * args.epochs // (mpu.get_data_parallel_world_size() * args.batch_size * args.gradient_accumulation_steps)
    else:
        args.train_iters = 10 # a magic number

    log_string = "Total train epochs {} | Total train iters {} | ".format(args.epochs, args.train_iters)
    print_rank_0(log_string)
    save_rank_0(args, log_string)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, tokenizer.vocab_size, ds_config, prompt_config)
        
    if args.do_train:
        train(args, data_config, tokenizer, model, optimizer, lr_scheduler, train_dataset, train_dataloader, dev_dataset, dev_dataloader, device)

    if args.do_eval:
        eval_dataloader, eval_dataset = load_data(args, data_config, 'dev', tokenizer, prompt_config, ratio=1)
        eval_func = data_config[args.data_name]["eval_func"]

        loss, acc = eval_func(args, tokenizer, data_config, eval_dataset, eval_dataloader, model, device, mode="test")

        log_string = "Eval result: loss: {:.6} | acc: {:.4}".format(loss, acc)
        print_rank_0(log_string)
        save_rank_0(args, log_string)

    if args.do_infer:
        infer_dataloader, infer_dataset = load_data(args, data_config, "test", tokenizer, prompt_config, ratio=1, drop_last=True, do_infer=True)
        eval_func = data_config[args.data_name]["eval_func"]

        sample_num = eval_func(args, tokenizer, data_config, infer_dataset, infer_dataloader, model, device, mode="infer")

        log_string = "Inferenced {} samples".format(sample_num)
        print_rank_0(log_string)
        save_rank_0(args, log_string)


if __name__ == "__main__":
    main()
