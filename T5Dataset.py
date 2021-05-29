from os import POSIX_FADV_SEQUENTIAL, replace
from preprocess_chid_finetune import process_one_sent
import torch
import json
import re
import os
import random
from tqdm import tqdm
from torch._C import dtype
from torch.utils.data import Dataset
from data_utils.tokenization_enc_dec import EncDecTokenizer,MT5EncDecTokenizer
import pickle
import mpu
import math
from utils import print_rank_0, save_rank_0


class T5Dataset(Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        self.args = args
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.path = path
        self.pad_id = tokenizer.pad_id
        self.prefix = prefix
        self.prefix_ids = self.tokenizer.encode(prefix) if prefix is not None else []
        self.enc_seq_length = args.enc_seq_length - len(self.prefix_ids)
        self.add_target_post=add_target_post
        self.split = split
        self.do_infer = do_infer
        self.idx = 0
        self.prompt_config = prompt_config
        if cache_path is not None:
            cache_path = os.path.join(cache_path, "cache_{}_{}.pkl".format(path.replace("/", "_"), ratio))
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.data, self.max_enc_len, self.max_dec_len = pickle.load(f)
            else:
                self.data, self.max_enc_len, self.max_dec_len = self.process_data()
                with open(cache_path, "wb") as f:
                    pickle.dump((self.data, self.max_enc_len, self.max_dec_len), f)
        else:
            self.data, self.max_enc_len, self.max_dec_len = self.process_data()

        if prompt_config is not None:
            self.data, self.max_enc_len, self.max_dec_len = self.add_prompt_ids(self.data, self.max_enc_len, self.max_dec_len)

        if do_infer:
            total_eval_batch_size = mpu.get_data_parallel_world_size() * args.batch_size
            total_data_num = math.ceil(len(self.data) / total_eval_batch_size) * total_eval_batch_size
            while len(self.data) < total_data_num:
                tmp = self.data[0].copy()
                tmp["idx"] = -1
                self.data.append(tmp)

        print_str = "Path: {} | Ratio:{} | Max enc len: {} | Max dec len: {} | Data num: {}".format(path, ratio, self.max_enc_len, self.max_dec_len, len(self.data))
        print_rank_0(print_str)
        save_rank_0(args, print_str)

    def process_data(self):
        raise NotImplementedError

    def add_prompt_ids(self, data, max_enc_len, max_dec_len):
        enc_prompt_ids = [- (i + 1) for i in range(self.prompt_config["enc"]["prompt_len"])]
        dec_prompt_ids = [- (i + 1) for i in range(self.prompt_config["dec"]["prompt_len"])]
        pad_ids = [self.tokenizer.pad_id for _ in range(self.prompt_config["dec"]["prompt_len"])]

        for d in data:
            d["enc_input_ids"] = enc_prompt_ids + d["enc_input_ids"]
            d["dec_input_ids"] = dec_prompt_ids + d["dec_input_ids"]
            d["label_ids"] = pad_ids + d["label_ids"]
        # print(enc_prompt_ids)
        # print(dec_prompt_ids)
        # print("=" * 20)
        max_enc_len += self.prompt_config["enc"]["prompt_len"]
        max_dec_len += self.prompt_config["dec"]["prompt_len"]

        return data, max_enc_len, max_dec_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate(self, samples):
        bs = len(samples)
        model_data = {
            "enc_input_ids": torch.ones(bs, self.max_enc_len, dtype=torch.long) * self.pad_id,
            "enc_attention_mask": torch.zeros(bs, 1, self.max_enc_len, self.max_enc_len),
            "dec_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_dec_len),
            "cross_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_enc_len),
            "dec_input_ids": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
        }
        if not self.do_infer:
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
                "labels": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
                "loss_mask": torch.zeros(bs, self.max_dec_len),
                # "span_from": [samp["span_from"] for samp in samples] # list of list
                "span_from": torch.zeros(bs, len(self.tokenizer), dtype=torch.long),
            }
        else:
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
                # "span_from": [samp["span_from"] for samp in samples] # list of list
                "span_from": torch.zeros(bs, len(self.tokenizer), dtype=torch.long),
            }

        for i, samp in enumerate(samples):
            enc_len, dec_len = len(samp["enc_input_ids"]), len(samp["dec_input_ids"])
            model_data["enc_input_ids"][i][:enc_len] = torch.tensor(samp["enc_input_ids"], dtype=torch.long)
            model_data["dec_input_ids"][i][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)
            model_data["enc_attention_mask"][i][0, :enc_len, :enc_len] = 1.0
            model_data["dec_attention_mask"][i][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
            model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = 1.0
            no_model_data["idx"][i] = samp["idx"]
            if "span_from" in samp:
                # print(samp["idx"], samp["span_from"])
                for tokenid in samp["span_from"]:
                    no_model_data["span_from"][i][tokenid] = 1
            # print("in the collefn", no_model_data["span_from"][i].sum())
            if not self.do_infer:
                no_model_data["labels"][i][:len(samp["label_ids"])] = torch.tensor(samp["label_ids"], dtype=torch.long)
                if self.prompt_config is not None:
                    no_model_data["loss_mask"][i][self.prompt_config["dec"]["prompt_len"]:len(samp["label_ids"])] = 1.0
                else:
                    no_model_data["loss_mask"][i][:len(samp["label_ids"])] = 1.0

        if self.args.fp16:
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].half()
            model_data["dec_attention_mask"] = model_data["dec_attention_mask"].half()
            model_data["cross_attention_mask"] = model_data["cross_attention_mask"].half()

        return model_data, no_model_data


class TNewsDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(TNewsDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = {
            "100": "故事",
            "101": "文化",
            "102": "娱乐",
            "103": "体育",
            "104": "金融",
            "106": "房地产",
            "107": "汽车",
            "108": "教育",
            "109": "科技",
            "110": "军事",
            "112": "旅游",
            "113": "世界",
            "114": "股票",
            "115": "农业",
            "116": "游戏"            
        }

        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            context = self.prefix_ids + (self.tokenizer.encode(d["keywords"].replace(",", "，")) + [12] + self.tokenizer.encode(d["sentence"]))[:self.enc_seq_length]
            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": d["id"] if self.do_infer else self.idx,
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class QuoteRDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False, prompt_config=None):
        super(QuoteRDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        data, self.neg_data = [], []
        enc_sizes = []
        dec_sizes = []
        with open(self.path, "r") as f:
            data_lines = f.readlines()

        with open(os.path.join(self.args.data_path, "baihua_acl_0111.csv"), "r") as f:
            cand_lines = f.readlines()
            all_candidates = [line.strip().split(",")[0] for line in cand_lines]
        all_candidates[0] = all_candidates[0][1:] # remove \ufeff
        all_candidates_token = {cand: self.tokenizer.encode(cand) for cand in all_candidates}
        print("all_cand_num: ", len(all_candidates_token))
        self.all_truth = {}

        did = 0
        for line in data_lines[:int(self.ratio * len(data_lines))]:
            line = json.loads(line)
            prefix = self.tokenizer.encode(line["prefix"])
            postfix = self.tokenizer.encode(line["postfix"])
            if self.split == "train":# or self.split == "dev":
                # positve sample
                pos_inputx = prefix + [495] + all_candidates_token[line["answer"]] + [495] + postfix
                target = [1, self.tokenizer.get_sentinel_id(0)] #+ self.tokenizer.encode("正确")
                # if self.add_target_post:
                #     target += [self.tokenizer.get_sentinel_id(1)]
                enc_sizes.append(len(pos_inputx))
                dec_sizes.append(len(target))
                negs = []
                for cand in line["candidates"]:
                    if cand != line["answer"]:
                        neg_inputx = prefix + [495] + all_candidates_token[cand] + [495] + postfix
                        negs.append(neg_inputx)
                        
                        enc_sizes.append(len(neg_inputx))
                data.append({
                    "idx": self.idx,
                    "pos_input_idx": pos_inputx,
                    "neg_input_idx": negs,
                    "dec_input_ids": target,
                })
                self.idx += 1
            else:
                label = line["label"]
                if label > len(line["candidates"]):
                    continue
                assert line["candidates"][label - 1] == line["answer"]
                for cid, cand in enumerate(line["candidates"]):
                    cand_inputx = prefix + [495] + all_candidates_token[cand] + [495] + postfix
                    target = [1, self.tokenizer.get_sentinel_id(0)]
                    data.append({
                        "qidx": did,
                        "cidx": cid,
                        "enc_input_ids": cand_inputx,
                        "dec_input_ids": target,
                        "label": label,
                    })
                    enc_sizes.append(len(cand_inputx))
                    dec_sizes.append(len(target))
                did += 1
        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)
        random.shuffle(data)
        print_rank_0("an data example")
        print_rank_0(data[0])
        print_rank_0("==" * 20)
        return data, max_enc_len, max_dec_len

    def collate(self, samples):
        bs = len(samples)
        neg_num = 1
        if self.split == "train":# or self.split == "dev":
            model_data = {
                "enc_input_ids": torch.ones(bs, neg_num + 1, self.max_enc_len, dtype=torch.long) * self.pad_id,
                "enc_attention_mask": torch.zeros(bs, neg_num + 1, 1, self.max_enc_len, self.max_enc_len),
                "dec_attention_mask": torch.zeros(bs, neg_num + 1, 1, self.max_dec_len, self.max_dec_len),
                "cross_attention_mask": torch.zeros(bs, neg_num + 1, 1, self.max_dec_len, self.max_enc_len),
                "dec_input_ids": torch.ones(bs, neg_num + 1, self.max_dec_len, dtype=torch.long) * self.pad_id,
            }
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
                # "labels": torch.zeros(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
                "loss_mask": torch.zeros(bs, self.max_dec_len),
            }
        else:
            model_data = {
                "enc_input_ids": torch.ones(bs, self.max_enc_len, dtype=torch.long) * self.pad_id,
                "enc_attention_mask": torch.zeros(bs, 1, self.max_enc_len, self.max_enc_len),
                "dec_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_dec_len),
                "cross_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_enc_len),
                "dec_input_ids": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
            }
            no_model_data = {
                "qidx": torch.zeros(bs, dtype=torch.long),
                "cidx": torch.zeros(bs, dtype=torch.long),
                "label": torch.zeros(bs, dtype=torch.long),
            }
        if self.split == "train":# or self.split == "dev":
            for i, samp in enumerate(samples):
                pos_enc_len, dec_len = len(samp["pos_input_idx"]), len(samp["dec_input_ids"])
                model_data["enc_input_ids"][i][0][:pos_enc_len] = torch.tensor(samp["pos_input_idx"], dtype=torch.long)
                model_data["dec_input_ids"][i][0][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)
                model_data["enc_attention_mask"][i][0][0, :pos_enc_len, :pos_enc_len] = 1.0
                model_data["dec_attention_mask"][i][0][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
                model_data["cross_attention_mask"][i][0][0, :dec_len, :pos_enc_len] = 1.0

                negs = random.sample(samp["neg_input_idx"], neg_num)
                
                for nid, neg in enumerate(negs):
                    neg_enc_len = len(neg)
                    model_data["enc_input_ids"][i][nid + 1][:neg_enc_len] = torch.tensor(neg, dtype=torch.long)
                    model_data["dec_input_ids"][i][nid + 1][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)
                    model_data["enc_attention_mask"][i][nid + 1][0, :neg_enc_len, :neg_enc_len] = 1.0
                    model_data["dec_attention_mask"][i][nid + 1][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
                    model_data["cross_attention_mask"][i][nid + 1][0, :dec_len, :neg_enc_len] = 1.0
                no_model_data["idx"][i] = samp["idx"]
        else:
            for i, samp in enumerate(samples):
                enc_len, dec_len = len(samp["enc_input_ids"]), len(samp["dec_input_ids"])
                model_data["enc_input_ids"][i][:enc_len] = torch.tensor(samp["enc_input_ids"], dtype=torch.long)
                model_data["dec_input_ids"][i][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)

                model_data["enc_attention_mask"][i][0, :enc_len, :enc_len] = 1.0
                model_data["dec_attention_mask"][i][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
                model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = 1.0
                no_model_data["qidx"][i] = samp["qidx"]
                no_model_data["cidx"][i] = samp["cidx"]
                no_model_data["label"][i] = samp["label"]

        if self.args.fp16:
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].half()
            model_data["dec_attention_mask"] = model_data["dec_attention_mask"].half()
            model_data["cross_attention_mask"] = model_data["cross_attention_mask"].half()
        if self.split == "train":# or self.split == "dev":
            model_data["enc_input_ids"] = model_data["enc_input_ids"].view(bs * (neg_num + 1), self.max_enc_len)
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].view(bs * (neg_num + 1), 1, self.max_enc_len, self.max_enc_len)
            model_data["dec_attention_mask"] = model_data["dec_attention_mask"].view(bs * (neg_num + 1), 1, self.max_dec_len, self.max_dec_len)
            model_data["cross_attention_mask"] = model_data["cross_attention_mask"].view(bs * (neg_num + 1), 1, self.max_dec_len, self.max_enc_len)
            model_data["dec_input_ids"] = model_data["dec_input_ids"].view(bs * (neg_num + 1), self.max_dec_len)

        return model_data, no_model_data


class SogouLogDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(SogouLogDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)


    def keep(self, data): # 把一些确定是无效的数据删掉
        if "锟斤拷" in data["query"] or "�" in data["query"]:
            return False
        # if len(data["query"]) <= 1:
        #     return False
        # if data["query"] in data["pos_doc"] and data["query"] in data["neg_doc"]:
        #     return False
        # if data["query"] in data["neg_doc"] and data["query"] not in data["pos_doc"]:
        #     return False
        if data["pos_doc"] == data["neg_doc"]:
            return False
        return True
    

    # def __getitem__(self, idx):
    #     if not self.do_infer:
    #         return random.choice(self.data)
    #     else:
    #         return self.data[idx]
    
    def pre_process(self, data):
        ret = {}
        for key in data:
            if isinstance(data[key], str):
                ret[key] = data[key].replace("\u3000", " ")
        return ret

    def process_data(self):
        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()
        if self.do_infer:
            self.datanum = len(lines)
        else:
            self.datanum = int(0.2 * len(lines)) # 数据量有点太大了，如果不是做inference的话，那就只用20%的数据
        for line in lines[:self.datanum]:
            d = json.loads(line)
            d = self.pre_process(d)
            if self.do_infer:
                qid = d["query_id"]
                for cand in d["doc"]:
                    cid = int(cand["sogou_id"].split('-')[-1])
                    inputx = self.prefix_ids + self.tokenizer.encode("查询：") + (self.tokenizer.encode(d["query"])) + self.tokenizer.encode("标题：") + self.tokenizer.encode(cand["title"])
                    target = [1, self.tokenizer.get_sentinel_id(0)]
                    data.append({
                        "qidx": qid,
                        "cidx": cid,
                        "enc_input_ids": inputx,
                        "dec_input_ids": target
                    })
                    enc_sizes.append(len(inputx))
                    dec_sizes.append(len(target))
            else:
                if not self.keep(d):
                    continue
                inputx_pos = self.prefix_ids + self.tokenizer.encode("查询：") + self.tokenizer.encode(d["query"]) + self.tokenizer.encode("标题：") + self.tokenizer.encode(d["pos_doc"])
                inputx_neg = self.prefix_ids + self.tokenizer.encode("查询：") + self.tokenizer.encode(d["query"]) + self.tokenizer.encode("标题：") + self.tokenizer.encode(d["neg_doc"])

                # context = self.prefix_ids + (self.tokenizer.encode(d["keywords"].replace(",", "，")) + [12] + self.tokenizer.encode(d["sentence"]))[:self.enc_seq_length]
                target = [1, self.tokenizer.get_sentinel_id(0)] #+ (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])

                if len(inputx_pos) > 150 or len(inputx_neg) > 150:
                    continue
                data.append({
                    "idx": self.idx,
                    "pos_input_idx": inputx_pos,
                    "neg_input_idx": inputx_neg,
                    "dec_input_ids": target,
                })
                enc_sizes.append(len(inputx_pos))
                enc_sizes.append(len(inputx_neg))
                dec_sizes.append(len(target))
                # dec_sizes.append(1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

    def collate(self, samples):
        bs = len(samples)
        if not self.do_infer:
            model_data = {
                "enc_input_ids": torch.ones(bs, 2, self.max_enc_len, dtype=torch.long) * self.pad_id,
                "enc_attention_mask": torch.zeros(bs, 2, 1, self.max_enc_len, self.max_enc_len),
                "dec_attention_mask": torch.zeros(bs, 2, 1, self.max_dec_len, self.max_dec_len),
                "cross_attention_mask": torch.zeros(bs, 2, 1, self.max_dec_len, self.max_enc_len),
                "dec_input_ids": torch.ones(bs, 2, self.max_dec_len, dtype=torch.long) * self.pad_id,
            }
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
                "labels": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
                "loss_mask": torch.zeros(bs, self.max_dec_len),
            }
        else:
            model_data = {
                "enc_input_ids": torch.ones(bs, self.max_enc_len, dtype=torch.long) * self.pad_id,
                "enc_attention_mask": torch.zeros(bs, 1, self.max_enc_len, self.max_enc_len),
                "dec_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_dec_len),
                "cross_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_enc_len),
                "dec_input_ids": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
            }
            no_model_data = {
                "qidx": torch.zeros(bs, dtype=torch.long),
                "cidx": torch.zeros(bs, dtype=torch.long),
            }
        if not self.do_infer:
            for i, samp in enumerate(samples):
                pos_enc_len, neg_enc_len = len(samp["pos_input_idx"]), len(samp["neg_input_idx"])
                dec_len = len(samp["dec_input_ids"])
                model_data["enc_input_ids"][i][0][:pos_enc_len] = torch.tensor(samp["pos_input_idx"], dtype=torch.long)
                model_data["enc_input_ids"][i][1][:neg_enc_len] = torch.tensor(samp["neg_input_idx"], dtype=torch.long)

                model_data["dec_input_ids"][i][0][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)
                model_data["dec_input_ids"][i][1][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)

                model_data["enc_attention_mask"][i][0][0, :pos_enc_len, :pos_enc_len] = 1.0
                model_data["enc_attention_mask"][i][1][0, :neg_enc_len, :neg_enc_len] = 1.0

                model_data["dec_attention_mask"][i][0][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
                model_data["dec_attention_mask"][i][1][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))

                model_data["cross_attention_mask"][i][0][0, :dec_len, :pos_enc_len] = 1.0
                model_data["cross_attention_mask"][i][1][0, :dec_len, :neg_enc_len] = 1.0
                no_model_data["idx"][i] = samp["idx"]
        else:
            for i, samp in enumerate(samples):
                enc_len, dec_len = len(samp["enc_input_ids"]), len(samp["dec_input_ids"])
                model_data["enc_input_ids"][i][:enc_len] = torch.tensor(samp["enc_input_ids"], dtype=torch.long)
                model_data["dec_input_ids"][i][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)

                model_data["enc_attention_mask"][i][0, :enc_len, :enc_len] = 1.0
                model_data["dec_attention_mask"][i][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
                model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = 1.0
                no_model_data["qidx"][i] = samp["qidx"]
                no_model_data["cidx"][i] = samp["cidx"]
                

        if self.args.fp16:
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].half()
            model_data["dec_attention_mask"] = model_data["dec_attention_mask"].half()
            model_data["cross_attention_mask"] = model_data["cross_attention_mask"].half()
        if not self.do_infer:
            model_data["enc_input_ids"] = model_data["enc_input_ids"].view(bs * 2, self.max_enc_len)
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].view(bs * 2, 1, self.max_enc_len, self.max_enc_len)
            model_data["dec_attention_mask"] = model_data["dec_attention_mask"].view(bs * 2, 1, self.max_dec_len, self.max_dec_len)
            model_data["cross_attention_mask"] = model_data["cross_attention_mask"].view(bs * 2, 1, self.max_dec_len, self.max_enc_len)
            model_data["dec_input_ids"] = model_data["dec_input_ids"].view(bs * 2, self.max_dec_len)

        return model_data, no_model_data


class OCNLIDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(OCNLIDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = {
            "entailment": "相似",
            "contradiction": "矛盾",
            "neutral": "中立"
        }
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.do_infer or d["label"] in ["entailment", "contradiction", "neutral"]:
                context = self.prefix_ids + [39] + self.tokenizer.encode(d["sentence1"])[:self.enc_seq_length // 2 - 4] + [41, 62, 39] + self.tokenizer.encode(d["sentence2"])[:self.enc_seq_length // 2 - 4] + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
                target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": d["id"] if self.do_infer else self.idx,  
                    "enc_input_ids": context,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:]
                })
                enc_sizes.append(len(context))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class AFQMCDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(AFQMCDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = {
            "0": "不同",
            "1": "相同"
        }
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            context = self.prefix_ids + [39] + self.tokenizer.encode(d["sentence1"])[:self.enc_seq_length // 2 - 4] + [41, 62, 39] + self.tokenizer.encode(d["sentence2"])[:self.enc_seq_length // 2 - 4] + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": d["id"] if self.do_infer else self.idx,
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class IFLYTEKDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False, prompt_config=None):
        super(IFLYTEKDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = {
            "2": "免费wifi",
            "23": "竞技游戏"
        }
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            context = self.prefix_ids + self.tokenizer.encode(d["sentence"])[:self.enc_seq_length]
            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]] if d["label"] in self.label_word_map else d["label_des"]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": d["id"] if self.do_infer else self.idx,
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class CMNLIDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CMNLIDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = {
            "entailment": "相似",
            "contradiction": "矛盾",
            "neutral": "中立"
        }
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            if self.do_infer or d["label"] in ["entailment", "contradiction", "neutral"]:
                context = self.prefix_ids + [39] + self.tokenizer.encode(d["sentence1"])[:self.enc_seq_length // 2 - 4] + [41, 62, 39] + self.tokenizer.encode(d["sentence2"])[:self.enc_seq_length // 2 - 4] + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
                target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": d["id"] if self.do_infer else self.idx,
                    "enc_input_ids": context,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:]
                })
                enc_sizes.append(len(context))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class CSLDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CSLDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = {
            "0": "错误",
            "1": "正确",
        }
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            context = self.tokenizer.encode(d["abst"])
            key_words = self.tokenizer.encode("关键词：")
            for x in d["keyword"]:
                key_words += self.tokenizer.encode(x) + [16]
            key_words = key_words[:-1]
            context = self.prefix_ids + context[:self.enc_seq_length - len(key_words)] + key_words
            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": d["id"] if self.do_infer else self.idx,
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


def cut_to_max_len(prefix, postfix, max_len):
    if len(prefix) + len(postfix) <= max_len:
        return prefix, postfix

    overflow_num = len(prefix)  + len(postfix) - max_len

    overflow_num_prefix = int((len(prefix) / (len(prefix) + len(postfix))) * overflow_num)
    overflow_num_postfix = int((len(postfix) / (len(prefix) + len(postfix))) * overflow_num)
        
    if overflow_num_prefix + overflow_num_postfix < overflow_num:
        if len(prefix) > len(postfix):
            overflow_num_prefix += 1
        else:
            overflow_num_postfix += 1

    assert overflow_num_prefix + overflow_num_postfix >= overflow_num, (overflow_num_prefix, overflow_num_postfix, overflow_num)

    return prefix[overflow_num_prefix:], postfix[:len(postfix) - overflow_num_postfix]


class CHIDDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False, prompt_config=None):
        super(CHIDDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()
        
        ans_d = None
        if not self.do_infer:
            with open(self.path.replace(".json", "_answer.json"), "r") as f:
                ans_d = json.load(f)

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            for sent in d["content"]:
                samples, tmp_enc_sizes, tmp_dec_sizes = self.process_one_sent(sent, ans_d, d["candidates"])
                data.extend(samples)
                enc_sizes.extend(tmp_enc_sizes)
                dec_sizes.extend(tmp_dec_sizes)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

    def process_one_sent(self, sent, answers, cands):
        pattern = re.compile(r"#idiom(\d+)#")
        start = 0
        samples = []
        enc_sizes, dec_sizes = [], []
        cands_ids = self.tokenizer.encode("选项：")
        for i, cand in enumerate(cands):
            cands_ids.extend(self.tokenizer.encode(cand.strip()) + [16])

        while True:
            m = pattern.search(sent, start)
            if m is None:
                break
            
            context_ids = self.tokenizer.encode("上下文：")
    
            prefix = self.tokenizer.encode(re.sub(pattern, "", sent[:m.start()]))
            postfix = self.tokenizer.encode(re.sub(pattern, "", sent[m.end():]))
    
            max_len = self.enc_seq_length - len(cands_ids) - len(context_ids) - 1
            prefix, postfix = cut_to_max_len(prefix, postfix, max_len)
            context_ids.extend(prefix + [self.tokenizer.get_sentinel_id(0)] + postfix)
    
            ids = cands_ids + context_ids
    
            assert len(ids) <= self.enc_seq_length, (len(ids), max_len, len(prefix), len(postfix))

            ids = self.prefix_ids + ids

            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(cands[answers[m.group()]]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            
            samples.append({
                "idx": int(m.group(1)) if self.do_infer else self.idx,
                "enc_input_ids": ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1
    
            start = m.end()

        return samples, enc_sizes, dec_sizes


class CHIDDataset2(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CHIDDataset2, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()
        
        ans_d = None
        if not self.do_infer:
            with open(self.path.replace(".json", "_answer.json"), "r") as f:
                ans_d = json.load(f)

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            for sent in d["content"]:
                samples, tmp_enc_sizes, tmp_dec_sizes = self.process_one_sent(sent, ans_d, d["candidates"])
                data.extend(samples)
                enc_sizes.extend(tmp_enc_sizes)
                dec_sizes.extend(tmp_dec_sizes)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

    def process_one_sent(self, sent, answers, cands):

        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031, # 八
            1189, # 九
            1320
        ]

        pattern = re.compile(r"#idiom(\d+)#")
        start = 0
        samples = []
        enc_sizes, dec_sizes = [], []
        cands_ids = self.tokenizer.encode("选项：")
        for i, cand in enumerate(cands):
            cand = list(cand.strip())
            cands_ids.extend([number_map[i], 20] + self.tokenizer.convert_tokens_to_ids(cand) + [16])

        while True:
            m = pattern.search(sent, start)
            if m is None:
                break
            
            context_ids = self.tokenizer.encode("上下文：")
    
            prefix = self.tokenizer.encode(re.sub(pattern, "", sent[:m.start()]))
            postfix = self.tokenizer.encode(re.sub(pattern, "", sent[m.end():]))
    
            max_len = self.enc_seq_length - len(cands_ids) - len(context_ids) - 1
            prefix, postfix = cut_to_max_len(prefix, postfix, max_len)
            context_ids.extend(prefix + [self.tokenizer.get_sentinel_id(0)] + postfix)
    
            ids = cands_ids + context_ids
    
            assert len(ids) <= self.enc_seq_length, (len(ids), max_len, len(prefix), len(postfix))

            ids = self.prefix_ids + ids

            target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[answers[m.group()]]] if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            
            samples.append({
                "idx": int(m.group(1)) if self.do_infer else self.idx,
                "enc_input_ids": ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1
    
            start = m.end()

        return samples, enc_sizes, dec_sizes


class CHIDDataset3(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CHIDDataset3, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()
        
        ans_d = None
        if not self.do_infer:
            with open(self.path.replace(".json", "_answer.json"), "r") as f:
                ans_d = json.load(f)

        with open(os.path.join(self.args.data_path, "idiom_2_id.json"), "r") as f:
            idiom_2_id = json.load(f)

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            for sent in d["content"]:
                samples, tmp_enc_sizes, tmp_dec_sizes = self.process_one_sent(sent, ans_d, d["candidates"], idiom_2_id)
                data.extend(samples)
                enc_sizes.extend(tmp_enc_sizes)
                dec_sizes.extend(tmp_dec_sizes)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

    def process_one_sent(self, sent, answers, cands, idiom_2_id):

        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031, # 八
            1189, # 九
            1320
        ]

        pattern = re.compile(r"#idiom(\d+)#")
        start = 0
        samples = []
        enc_sizes, dec_sizes = [], []
        cands_ids = self.tokenizer.encode("选项：")
        for i, cand in enumerate(cands):
            # cand = list(cand.strip())
            cands_ids.extend([number_map[i], 20] + [idiom_2_id[cand] + self.tokenizer.vocab_size] + [16])

        while True:
            m = pattern.search(sent, start)
            if m is None:
                break
            
            context_ids = self.tokenizer.encode("上下文：")
    
            prefix = self.tokenizer.encode(re.sub(pattern, "", sent[:m.start()]))
            postfix = self.tokenizer.encode(re.sub(pattern, "", sent[m.end():]))
    
            max_len = self.enc_seq_length - len(cands_ids) - len(context_ids) - 1
            prefix, postfix = cut_to_max_len(prefix, postfix, max_len)
            context_ids.extend(prefix + [self.tokenizer.get_sentinel_id(0)] + postfix)
    
            ids = cands_ids + context_ids
    
            assert len(ids) <= self.enc_seq_length, (len(ids), max_len, len(prefix), len(postfix))

            ids = self.prefix_ids + ids

            target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[answers[m.group()]]] if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            
            samples.append({
                "idx": int(m.group(1)) if self.do_infer else self.idx,
                "enc_input_ids": ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(ids))
            dec_sizes.append(len(target) - 1)
            self.idx += 1
    
            start = m.end()

        return samples, enc_sizes, dec_sizes


class CMRCDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False, prompt_config=None):
        super(CMRCDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        with open(self.path, "r") as f:
            jobj = json.load(f)["data"]

        data = []
        enc_sizes, dec_sizes = [], []

        for dd in jobj[:int(self.ratio * len(jobj))]:
            dd = dd["paragraphs"]
            for d in dd:
                context = d["context"]
                for qa in d["qas"]:
                    question_ids = self.tokenizer.encode(qa["question"])
                    sidx = qa["id"].split("_")
                    sidx = int(sidx[1]) * 100000 + int(sidx[3])
                    all_answers = []
                    if not self.do_infer:
                        answer = ""
                        answer_start = 0
                        for x in qa["answers"]:
                            if x["answer_start"] != -1:
                                answer = x["text"]
                                answer_start = x["answer_start"]
                                break
                        all_answers = [x["text"] for x in qa["answers"]]

                        prefix = self.tokenizer.encode(context[:answer_start])
                        postfix = self.tokenizer.encode(context[answer_start+len(answer):])

                        fake_answer_ids = self.tokenizer.encode(context[answer_start:answer_start+len(answer)])
                        answer_ids = self.tokenizer.encode(answer)
                    else:
                        prefix = self.tokenizer.encode(context)
                        postfix = []
                        fake_answer_ids = []

                    enc_input_ids = self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("文章：")

                    max_len = self.enc_seq_length - len(fake_answer_ids) - len(enc_input_ids)
                    context_ids = prefix + fake_answer_ids + postfix
                    prefix, postfix = cut_to_max_len(prefix, postfix, max_len)

                    enc_input_ids.extend(prefix + fake_answer_ids + postfix)
                    enc_input_ids = self.prefix_ids + enc_input_ids
                    target = [1, self.tokenizer.get_sentinel_id(0)] + (answer_ids if not self.do_infer else [self.tokenizer.pad_id])
                    if self.add_target_post:
                        target = target + [self.tokenizer.get_sentinel_id(1)]
                    data.append({
                        "idx": sidx if self.do_infer else self.idx,
                        "enc_input_ids": enc_input_ids,
                        "dec_input_ids": target[:-1],
                        "label_ids": target[1:],
                        "truth": all_answers,
                        "span_from": set(context_ids + [1, self.tokenizer.get_sentinel_id(0), self.tokenizer.get_sentinel_id(1), self.tokenizer.pad_id]),
                        # "span_from": target + [self.tokenizer.pad_id], # context_ids + answer_ids + [1, self.tokenizer.get_sentinel_id(0), self.tokenizer.pad_id],
                    })
                    # print("in the dataset", len(data[-1]["span_from"]))
                    enc_sizes.append(len(enc_input_ids))
                    dec_sizes.append(len(target) - 1)
                    self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

    # def collate(self, samples):
    #     model_data, no_model_data = super().collate(samples)
    #     max_truth_len = max([len(s["truth"]) for s in samples])
    #     no_model_data["truth"] = torch.ones(len(samples), max_truth_len, dtype=torch.long) * self.tokenizer.pad_id
    #     for i, s in enumerate(samples):
    #         no_model_data["truth"][i, :len(s["truth"])] = torch.tensor(s["truth"], dtype=torch.long)

    #     return model_data, no_model_data


class C3Dataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False, prompt_config=None):
        super(C3Dataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            context_ids = self.tokenizer.encode(d[0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []
                for choice in qa["choice"]:
                    choice_ids.extend(self.tokenizer.encode(choice) + [18])
                enc_input_ids = self.prefix_ids + self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids
                # NOTE: This can be dangerous
                enc_input_ids = enc_input_ids[:self.enc_seq_length]
                target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(qa["answer"]) if not self.do_infer else [self.tokenizer.pad_id])
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": qa["id"] if self.do_infer else self.idx,
                    "enc_input_ids": enc_input_ids,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:]
                })

                enc_sizes.append(len(enc_input_ids))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class C3Dataset2(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(C3Dataset2, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        print(type(self.tokenizer))
        
        if isinstance(self.tokenizer, MT5EncDecTokenizer):
            number_map = [ # MT5  # 对应了 一、二、三、四、五、六、七、八
                1374, 3178, 2092, 6216, 5737, 10534, 15390, 7704
            ]
            # number_map = [ # MT5 extra_id_99, extra_id_98, ..., extra_id_93
            #     250000, 250001, 250002, 250003, 250004, 250005, 250006, 250007
            # ]
        else:
            number_map = [ # CPM  # 对应了 一、二、三、四、五、六、七、八
                50, 230, 156, 349, 443, 803, 950, 1031
            ]
            # 对应了 一些unused的token
            # number_map = [ # CPM s_180, ..., s_187
            #     26230, 26231, 26232, 26233, 26234, 26235, 26236, 26237
            # ]
            

        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            if type(d[0]) == list:
                context_ids = self.tokenizer.encode(''.join(d[0]))
            else:
                context_ids = self.tokenizer.encode(d[0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []
                for i, choice in enumerate(qa["choice"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                enc_input_ids = self.prefix_ids + self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids
                # enc_input_ids = self.prefix_ids + self.tokenizer.encode("阅读文章：") + context_ids + self.tokenizer.encode("回答问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids
                # NOTE: This can be dangerous
                if self.split == "train":
                    enc_input_ids = enc_input_ids[:self.enc_seq_length]
                target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[qa["choice"].index(qa["answer"])]] if not self.do_infer else [self.tokenizer.pad_id])
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": qa["id"] if self.do_infer and "id" in qa else self.idx,
                    "enc_input_ids": enc_input_ids,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                })

                enc_sizes.append(len(enc_input_ids))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

class C3Dataset3(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(C3Dataset3, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        print(type(self.tokenizer))
        
        if isinstance(self.tokenizer, MT5EncDecTokenizer):
            self.number_map = [ # MT5  # 对应了 一、二、三、四、五、六、七、八
                1374, 3178, 2092, 6216, 5737, 10534, 15390, 7704
            ]
        else:
            self.number_map = [ # CPM  # 对应了 一、二、三、四、五、六、七、八
                50, 230, 156, 349, 443, 803, 950, 1031
            ]

        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            if type(d[0]) == list:
                context_ids = self.tokenizer.encode(''.join(d[0]))
            else:
                context_ids = self.tokenizer.encode(d[0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []
                choice_len = 0
                for i, choice in enumerate(qa["choice"]):
                    choice_ids.append([20] + self.tokenizer.encode(choice) + [18])
                    choice_len += len(choice_ids[-1]) + 1
                label = qa["choice"].index(qa["answer"])
                prefix = self.prefix_ids + self.tokenizer.encode("问题：")
                choice_prefix = self.tokenizer.encode("选项：")
                context_prefix = self.tokenizer.encode("文章：")
                data.append({
                    "idx": qa["id"] if self.do_infer and "id" in qa else self.idx,
                    "prefix": prefix,
                    "question_ids": question_ids,
                    "choice_prefix": choice_prefix,
                    "choice_ids": choice_ids,
                    "context_prefix": context_prefix,
                    "context_ids": context_ids,
                    "label": label,
                })
                
                enc_sizes.append(len(prefix) + len(question_ids) + len(choice_prefix) + len(choice_ids) + len(context_prefix) + len(context_ids))
                dec_sizes.append(2)
                self.idx += 1
        max_enc_len = min(max(enc_sizes), self.enc_seq_length)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len

    def collate(self, samples):
        insamples = []
        for samp in samples:
            anslist = list(range(len(samp["choice_ids"])))
            random.shuffle(anslist)
            inpchoice_idx = []
            for i in range(len(anslist)):
                inpchoice_idx.extend([self.number_map[i]] + samp["choice_ids"][anslist[i]])
            enc_input_ids = samp["prefix"] + samp["question_ids"] + samp["choice_prefix"] + inpchoice_idx + samp["context_prefix"] + samp["context_ids"]
            enc_input_ids = enc_input_ids[:self.enc_seq_length]

            target = [1, self.tokenizer.get_sentinel_id(0)] + ([self.number_map[anslist.index(samp["label"])]] if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            insamples.append({
                "idx": samp["idx"],
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:],
            })
        return super().collate(insamples)


class WSCDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(WSCDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []

        self.label_map = {
            "true": "正确",
            "false": "错误"
        }

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            context = d["text"]
            context = context[:d["target"]["span2_index"]] + d["target"]["span1_text"] + context[d["target"]["span2_index"] + len(d["target"]["span2_text"]):]
            context = self.prefix_ids + self.tokenizer.encode(context)
            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": d["id"] if self.do_infer else self.idx,
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class CombinedDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CombinedDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        
        self.data_config = {
            "tnews": {"dataset": TNewsDataset, "prefix": "新闻主题分类："},
            "afqmc": {"dataset": AFQMCDataset, "prefix": "金融语义相似度："},
            "ocnli": {"dataset": OCNLIDataset, "prefix": "中文原版语言推理："},
            "iflytek": {"dataset": IFLYTEKDataset, "prefix": "多主题分类："},
            "cmnli": {"dataset": CMNLIDataset, "prefix": "多类型语言推理："},
            "csl": {"dataset": CSLDataset, "prefix": "关键词识别："},
            "chid": {"dataset": CHIDDataset, "prefix": "成语填空："},
            "cmrc": {"dataset": CMRCDataset, "prefix": "抽取式阅读理解："},
            "c3": {"dataset": C3Dataset, "prefix": "选择式阅读理解："},
            "wsc": {"dataset": WSCDataset, "prefix": "词义消歧："},
        }

        data = []
        enc_sizes, dec_sizes = [], []

        for data_name, data_info in self.data_config.items():
            path = self.path.split("/")
            path = path[:-1] + [data_name] + [path[-1]]
            data_path = "/".join(path)
            dataset = data_info["dataset"](self.args, self.tokenizer, data_path, prefix=data_info["prefix"], add_target_post=True)
            data.extend(dataset.data)
            enc_sizes.append(dataset.max_enc_len)
            dec_sizes.append(dataset.max_dec_len)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        random.shuffle(data)

        return data, max_enc_len, max_dec_len


class WSCDataset2(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False, prompt_config=None):
        super(WSCDataset2, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            
            if self.do_infer or self.split in ["dev", "test"] or d["label"] == "true":
                context = d["text"]
                if self.prompt_config is not None:
                    prompt_len = self.prompt_config["enc"]["prompt_len"]
                    context = [-(i + 1) for i in range(prompt_len)] + self.tokenizer.encode(context[:d["target"]["span2_index"]]) + [495] + self.tokenizer.encode(d["target"]["span2_text"]) + [495] + self.tokenizer.encode(context[d["target"]["span2_index"] + len(d["target"]["span2_text"]):])
                else:
                    context = self.prefix_ids + context
                target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(d["target"]["span1_text"]) if not self.do_infer else [self.tokenizer.pad_id])
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": d["id"] if self.do_infer else self.idx,
                    "enc_input_ids": context,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                    "cand_ids": d["target"]["span1_text"],
                    "truth": (1 if d["label"] == "true" else 0) if not self.do_infer else None
                })

                enc_sizes.append(len(context))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class CSLDataset2(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(CSLDataset2, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):

        self.label_word_map = {
            "0": "错误",
            "1": "正确",
        }
    
        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            context = self.tokenizer.encode(d["abst"])
            # key_words = self.tokenizer.encode("关键词：")
            key_words = []
            for x in d["keyword"]:
                key_words += self.tokenizer.encode(x) + [16]
            # key_words = key_words[:-1]
            if self.prompt_config:
                prompt_len = self.prompt_config["enc"]["prompt_len"]
                context = self.prefix_ids + [-(i + 1) for i in range(prompt_len // 2)] + key_words + [-(i + 1) for i in range(prompt_len // 2, prompt_len)] + context[:self.enc_seq_length - len(key_words)]
            else:
                context = self.prefix_ids + key_words + context[:self.enc_seq_length - len(key_words)]
                
            target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(self.label_word_map[d["label"]]) if not self.do_infer else [self.tokenizer.pad_id])
            if self.add_target_post:
                target += [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "idx": d["id"] if self.do_infer else self.idx,
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)
            self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class WSCDataset3(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False, prompt_config=None):
        super(WSCDataset3, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)

    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            
            if self.do_infer or self.split in ["dev", "test"] or d["label"] == "true":
                context = d["text"]
                if self.prompt_config:
                    prompt_len = self.prompt_config["enc"]["prompt_len"]
                    context = [-(i + 1) for i in range(prompt_len - 2)] + self.tokenizer.encode(context[:d["target"]["span2_index"]]) + [-(prompt_len - 1)] + self.tokenizer.encode(d["target"]["span2_text"]) + [-prompt_len] + self.tokenizer.encode(context[d["target"]["span2_index"] + len(d["target"]["span2_text"]):])
                else:
                    context = self.tokenizer.encode(context[:d["target"]["span2_index"]]) + [495] + self.tokenizer.encode(d["target"]["span2_text"]) + [495] + self.tokenizer.encode(context[d["target"]["span2_index"] + len(d["target"]["span2_text"]):])
                context = self.prefix_ids + context
                target = [1, self.tokenizer.get_sentinel_id(0)] + (self.tokenizer.encode(d["target"]["span1_text"]) if not self.do_infer else [self.tokenizer.pad_id])
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": d["id"] if self.do_infer else self.idx,
                    "enc_input_ids": context,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:],
                    "cand_ids": d["target"]["span1_text"],
                    "truth": (1 if d["label"] == "true" else 0) if not self.do_infer else None
                })

                enc_sizes.append(len(context))
                dec_sizes.append(len(target) - 1)
                self.idx += 1

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len