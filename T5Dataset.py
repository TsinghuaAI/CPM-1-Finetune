import torch
import json
from tqdm import tqdm
from torch._C import dtype
from torch.utils.data import Dataset
from data_utils.tokenization_enc_dec import EncDecTokenizer

class T5Dataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __item__(self, idx):
        pass


class TNewsDataset(T5Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, ratio=1):
        self.tokenizer = tokenizer
        self.args = args

        with open(path, "r") as f:
            lines = f.readlines()

        self.data = []
        self.enc_sizes = []
        self.dec_sizes = []
        self.pad_id = self.tokenizer.pad_id

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

        for line in tqdm(lines[:int(ratio * len(lines))], disable=(torch.distributed.get_rank() != 0), desc="loading Dataset"):
            d = json.loads(line)
            context = self.tokenizer.encode(d["sentence"])[:args.enc_seq_length]
            target = [1, self.tokenizer.get_sentinel_id(0)] + self.tokenizer.encode(self.label_word_map[d["label"]])
            self.data.append({
                "enc_input_ids": context,
                "dec_input_ids": target[:-1],
                "label_ids": target[1:]
            })
            self.enc_sizes.append(len(context))
            self.dec_sizes.append(len(target) - 1)

        self.max_enc_len = max(self.enc_sizes)
        self.max_dec_len = max(self.dec_sizes)

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