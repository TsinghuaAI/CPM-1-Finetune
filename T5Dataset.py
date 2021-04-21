import torch
import json
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
            model_data["enc_attention_mask"][0, :enc_len, :enc_len] = 1.0
            model_data["dec_attention_mask"][0, :dec_len, :dec_len] = torch.tril(torch.ones(dec_len, dec_len))
            model_data["cross_attention_mask"][0, :dec_len, :enc_len] = 1.0
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
