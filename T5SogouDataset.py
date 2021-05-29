from .T5Dataset import T5Dataset
import torch
import json
import re
import os
import random
from data_utils.tokenization_enc_dec import EncDecTokenizer,MT5EncDecTokenizer
import pickle
import mpu
import math
from utils import print_rank_0, save_rank_0



class SogouLogDataset2(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False, prompt_config=None):
        super(SogouLogDataset2, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer, prompt_config)
        if not do_infer:
            self.data_num = int(0.2 * len(self.data))
        else:
            self.data_num = len(self.data)

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
    
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if not self.do_infer:
            return random.choice(self.data)
        else:
            return self.data[idx]
    
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
            datanum = len(lines)
        else:
            datanum = int(0.2 * len(lines)) # 数据量有点太大了，如果不是做inference的话，那就只用20%的数据
        for line in lines[:datanum]:
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
                    "idx": d["id"] if self.do_infer else self.idx,
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
