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
from data_utils.tokenization_enc_dec import EncDecTokenizer
import pickle
import mpu
import math
from utils import print_rank_0, save_rank_0

class T5Dataset(Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False):
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
            "dec_input_ids": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id
        }
        if not self.do_infer:
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
                "labels": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
                "loss_mask": torch.zeros(bs, self.max_dec_len)
            }
        else:
            no_model_data = {
                "idx": torch.zeros(bs, dtype=torch.long),
            }

        for i, samp in enumerate(samples):
            enc_len, dec_len = len(samp["enc_input_ids"]), len(samp["dec_input_ids"])
            model_data["enc_input_ids"][i][:enc_len] = torch.tensor(samp["enc_input_ids"], dtype=torch.long)
            model_data["dec_input_ids"][i][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)
            model_data["enc_attention_mask"][i][0, :enc_len, :enc_len] = 1.0
            model_data["dec_attention_mask"][i][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
            model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = 1.0
            no_model_data["idx"][i] = samp["idx"]
            if not self.do_infer:
                no_model_data["labels"][i][:len(samp["label_ids"])] = torch.tensor(samp["label_ids"], dtype=torch.long)
                no_model_data["loss_mask"][i][:len(samp["label_ids"])] = 1.0

        if self.args.fp16:
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].half()
            model_data["dec_attention_mask"] = model_data["dec_attention_mask"].half()
            model_data["cross_attention_mask"] = model_data["cross_attention_mask"].half()

        return model_data, no_model_data


class TNewsDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False):
        super(TNewsDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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


class OCNLIDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False):
        super(OCNLIDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False):
        super(AFQMCDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False):
        super(IFLYTEKDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False):
        super(CMNLIDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False):
        super(CSLDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False):
        super(CHIDDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False):
        super(CHIDDataset2, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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


class CMRCDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False):
        super(CMRCDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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
                        "truth": all_answers
                    })

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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False):
        super(C3Dataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)
    
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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False):
        super(C3Dataset2, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)
    
    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []
        
        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031 # 八
        ]

        with open(self.path, "r") as f:
            jobj = json.load(f)

        for d in jobj[:int(self.ratio * len(jobj))]:
            context_ids = self.tokenizer.encode(d[0])
            for qa in d[1]:
                question_ids = self.tokenizer.encode(qa["question"])
                choice_ids = []
                for i, choice in enumerate(qa["choice"]):
                    choice_ids.extend([number_map[i], 20] + self.tokenizer.encode(choice) + [18])
                enc_input_ids = self.prefix_ids + self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids
                # NOTE: This can be dangerous
                enc_input_ids = enc_input_ids[:self.enc_seq_length]
                target = [1, self.tokenizer.get_sentinel_id(0)] + ([number_map[qa["choice"].index(qa["answer"])]] if not self.do_infer else [self.tokenizer.pad_id])
                if self.add_target_post:
                    target += [self.tokenizer.get_sentinel_id(1)]
                data.append({
                    "idx": qa["id"] if self.do_infer else self.idx,
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


class WSCDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False):
        super(WSCDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=False, cache_path=None, do_infer=False):
        super(CombinedDataset, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, prefix=None, add_target_post=True, cache_path=None, do_infer=False):
        super(WSCDataset2, self).__init__(args, tokenizer, path, split, ratio, prefix, add_target_post, cache_path, do_infer)

    def process_data(self):
        data = []
        enc_sizes, dec_sizes = [], []

        with open(self.path, "r") as f:
            lines = f.readlines()

        for line in lines[:int(self.ratio * len(lines))]:
            d = json.loads(line)
            
            if self.do_infer or self.split in ["dev", "test"] or d["label"] == "true":
                context = d["text"]
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
