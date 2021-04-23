from os import POSIX_FADV_SEQUENTIAL, replace
from preprocess_chid_finetune import process_one_sent
import torch
import json
import re
from tqdm import tqdm
from torch._C import dtype
from torch.utils.data import Dataset
from data_utils.tokenization_enc_dec import EncDecTokenizer

class T5Dataset(Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        self.args = args
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.path = path
        self.pad_id = tokenizer.pad_id
        self.data, self.max_enc_len, self.max_dec_len = self.process_data()

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
        no_model_data = {
            "labels": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
            "loss_mask": torch.zeros(bs, self.max_dec_len)
        }

        for i, samp in enumerate(samples):
            enc_len, dec_len = len(samp["enc_input_ids"]), len(samp["dec_input_ids"])
            model_data["enc_input_ids"][i][:enc_len] = torch.tensor(samp["enc_input_ids"], dtype=torch.long)
            model_data["dec_input_ids"][i][:dec_len] = torch.tensor(samp["dec_input_ids"], dtype=torch.long)
            model_data["enc_attention_mask"][i][0, :enc_len, :enc_len] = 1.0
            model_data["dec_attention_mask"][i][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
            model_data["cross_attention_mask"][i][0, :dec_len, :enc_len] = 1.0
            no_model_data["labels"][i][:len(samp["label_ids"])] = torch.tensor(samp["label_ids"], dtype=torch.long)
            no_model_data["loss_mask"][i][:len(samp["label_ids"])] = 1.0

        if self.args.fp16:
            model_data["enc_attention_mask"] = model_data["enc_attention_mask"].half()
            model_data["dec_attention_mask"] = model_data["dec_attention_mask"].half()
            model_data["cross_attention_mask"] = model_data["cross_attention_mask"].half()

        return model_data, no_model_data


class TNewsDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        super(TNewsDataset, self).__init__(args, tokenizer, path, ratio)

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
            "115": "建筑",
            "116": "游戏"            
        }

        data = []
        enc_sizes = []
        dec_sizes = []
        
        with open(self.path, "r") as f:
            lines = f.readlines()
        
        for line in tqdm(lines[:int(self.ratio * len(lines))], disable=(torch.distributed.get_rank() != 0), desc="loading Dataset"):
            d = json.loads(line)
            context = self.tokenizer.encode(d["sentence"])[:self.args.enc_seq_length]
            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]])
            data.append({
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class OCNLIDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        super(OCNLIDataset, self).__init__(args, tokenizer, path, ratio)

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
        
        for line in tqdm(lines[:int(self.ratio * len(lines))], disable=(torch.distributed.get_rank() != 0), desc="loading Dataset"):
            d = json.loads(line)
            if d["label"] in ["entailment", "contradiction", "neutral"]:
                context = [39] + self.tokenizer.encode(d["sentence1"])[:self.args.enc_seq_length // 2 - 4] + [41, 62, 39] + self.tokenizer.encode(d["sentence2"])[:self.args.enc_seq_length // 2 - 4] + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
                target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]])
                data.append({
                    "enc_input_ids": context,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:]
                })
                enc_sizes.append(len(context))
                dec_sizes.append(len(target) - 1)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class AFQMCDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        super(AFQMCDataset, self).__init__(args, tokenizer, path, ratio)

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
        
        for line in tqdm(lines[:int(self.ratio * len(lines))], disable=(torch.distributed.get_rank() != 0), desc="loading Dataset"):
            d = json.loads(line)
            context = [39] + self.tokenizer.encode(d["sentence1"])[:self.args.enc_seq_length // 2 - 4] + [41, 62, 39] + self.tokenizer.encode(d["sentence2"])[:self.args.enc_seq_length // 2 - 4] + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]])
            data.append({
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class IFLYTEKDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        super(IFLYTEKDataset, self).__init__(args, tokenizer, path, ratio)

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
        
        for line in tqdm(lines[:int(self.ratio * len(lines))], disable=(torch.distributed.get_rank() != 0), desc="loading Dataset"):
            d = json.loads(line)
            context = self.tokenizer.encode(d["sentence"])[:self.args.enc_seq_length]
            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]] if d["label"] in self.label_word_map else d["label_des"]) + [self.tokenizer.get_sentinel_id(1)]
            data.append({
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class CMNLIDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        super(CMNLIDataset, self).__init__(args, tokenizer, path, ratio)

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
        
        for line in tqdm(lines[:int(self.ratio * len(lines))], disable=(torch.distributed.get_rank() != 0), desc="loading Dataset"):
            d = json.loads(line)
            if d["label"] in ["entailment", "contradiction", "neutral"]:
                context = [39] + self.tokenizer.encode(d["sentence1"])[:self.args.enc_seq_length // 2 - 4] + [41, 62, 39] + self.tokenizer.encode(d["sentence2"])[:self.args.enc_seq_length // 2 - 4] + [41, 11, 1348, self.tokenizer.get_sentinel_id(0)]
                target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]])
                data.append({
                    "enc_input_ids": context,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:]
                })
                enc_sizes.append(len(context))
                dec_sizes.append(len(target) - 1)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class CSLDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        super(CSLDataset, self).__init__(args, tokenizer, path, ratio)

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
        
        for line in tqdm(lines[:int(self.ratio * len(lines))], disable=(torch.distributed.get_rank() != 0), desc="loading Dataset"):
            d = json.loads(line)
            context = self.tokenizer.encode(d["abst"])
            key_words = self.tokenizer.encode("关键词：")
            for x in d["keyword"]:
                key_words += self.tokenizer.encode(x) + [16]
            key_words = key_words[:-1]
            context = context[:self.args.enc_seq_length - len(key_words)] + key_words
            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]])
            data.append({
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)

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
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        super(CHIDDataset, self).__init__(args, tokenizer, path, ratio)

    def process_data(self):
        data = []
        enc_sizes = []
        dec_sizes = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        with open(self.path.replace(".json", "_answer.json"), "r") as f:
            ans_d = json.load(f)

        for line in tqdm(lines[:int(self.ratio * len(lines))], disable=(torch.distributed.get_rank() != 0), desc="loading Dataset"):
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
        pattern = re.compile(r"#idiom\d+#")
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
    
            max_len = self.args.enc_seq_length - len(cands_ids) - len(context_ids) - 1
            prefix, postfix = cut_to_max_len(prefix, postfix, max_len)
            context_ids.extend(prefix + [self.tokenizer.get_sentinel_id(0)] + postfix)
    
            ids = cands_ids + context_ids
    
            assert len(ids) <= self.args.enc_seq_length, (len(ids), max_len, len(prefix), len(postfix))

            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(cands[answers[m.group()]]) + [self.tokenizer.get_sentinel_id(1)]

            samples.append({
                "enc_input_ids": ids,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(ids))
            dec_sizes.append(len(target) - 1)
    
            start = m.end()

        return samples, enc_sizes, dec_sizes


class CMRCDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        super(CMRCDataset, self).__init__(args, tokenizer, path, ratio)

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
                    answer = ""
                    answer_start = 0
                    for x in qa["answers"]:
                        if x["answer_start"] != -1:
                            answer = x["text"]
                            answer_start = x["answer_start"]
                            break
                        
                    prefix = self.tokenizer.encode(context[:answer_start])
                    postfix = self.tokenizer.encode(context[answer_start+len(answer):])

                    fake_answer_ids = self.tokenizer.encode(context[answer_start:answer_start+len(answer)])
                    answer_ids = self.tokenizer.encode(answer)
                    enc_input_ids = self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("文章：")

                    max_len = self.args.enc_seq_length - len(fake_answer_ids) - len(enc_input_ids)
                    prefix, postfix = cut_to_max_len(prefix, postfix, max_len)

                    enc_input_ids.extend(prefix + fake_answer_ids + postfix)
                    target = [1, self.tokenizer.get_sentinel_id(0)] + answer_ids + [self.tokenizer.get_sentinel_id(1)]

                    data.append({
                        "enc_input_ids": enc_input_ids,
                        "dec_input_ids": target[:-1],
                        "label_ids": target[1:]
                    })

                    enc_sizes.append(len(enc_input_ids))
                    dec_sizes.append(len(target) - 1)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class C3Dataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        super(C3Dataset, self).__init__(args, tokenizer, path, ratio)
    
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
                answer_ids = self.tokenizer.encode(qa["answer"])
                enc_input_ids = self.tokenizer.encode("问题：") + question_ids + self.tokenizer.encode("选项：") + choice_ids + self.tokenizer.encode("文章：") + context_ids
                # NOTE: This can be dangerous
                enc_input_ids = enc_input_ids[:self.args.enc_seq_length]
                target = [1 + self.tokenizer.get_sentinel_id(0)] + answer_ids + [self.tokenizer.get_sentinel_id(1)]

                data.append({
                    "enc_input_ids": enc_input_ids,
                    "dec_input_ids": target[:-1],
                    "label_ids": target[1:]
                })

                enc_sizes.append(len(enc_input_ids))
                dec_sizes.append(len(target) - 1)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len


class WSCDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        super(WSCDataset, self).__init__(args, tokenizer, path, ratio)

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
            context = self.tokenizer.encode(context)
            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_map[d["label"]])

            data.append({
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })

            enc_sizes.append(len(context))
            dec_sizes.append(len(target) - 1)

        max_enc_len = max(enc_sizes)
        max_dec_len = max(dec_sizes)

        return data, max_enc_len, max_dec_len