import json
import re
import os
import random
from tqdm import tqdm
from data_utils.tokenization_gpt2 import GPT2Tokenizer

def process_one_sent(sent, answers, cands, tokenizer, num_ids, split):
    pattern = re.compile(r"#idiom\d+#")
    res = pattern.findall(sent)
    start = 0
    L = []
    while True:
        m = pattern.search(sent, start)
        if m is None:
            break
        
        cands_ids = []
        cands_poses = []
        s, e = 0, 0
        for i, cand in enumerate(cands):
            # tmp = tokenizer.encode(cand.strip()) + [tokenizer.encoder["<sep>"]]
            tmp = [tokenizer.encoder["<sep>"]]
            e += len(tmp)
            cands_poses.append((s, e))
            s = e
            cands_ids.extend(tmp)

        prefix = re.sub(pattern, "", sent[:m.start()])
        postfix = re.sub(pattern, "", sent[m.end():])

        # ids = cands_ids + tokenizer.encode(prefix.strip()) + [tokenizer.encoder["<mask>"]]
        ids = cands_ids

        mask_pos = len(ids) - 1

        ids = ids + tokenizer.encode(postfix.strip())[-20:] + [tokenizer.encoder["<eod>"]]

        L.append({
            "sent": ids,
            "truth": answers[m.group()],
            "cands_len": len(cands_ids),
            "mask_pos": mask_pos,
            "cands_poses": cands_ids
        })

        start = m.end()
    
    return L

def preprocess(data, tokenizer, split):

    num_ids = [tokenizer.encode("选项{}:".format(i))[1] for i in range(10)]

    lines, ans_d = data
    
    all_data = []
    for line in tqdm(lines, desc="Preprocessing {}".format(split)):
        jobj = json.loads(line)
        for sent in jobj["content"]:
            samples = process_one_sent(sent, ans_d, jobj["candidates"], tokenizer, num_ids, split)
            all_data.extend(samples)

    with open(os.path.join(preprocesed_dir, "{}.json".format(split)), "w") as f:
        json.dump((num_ids, all_data), f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    data_dir = "/data/gyx/chid/"
    ans_data_dir = "/data/gyx/chid/"

    preprocesed_dir = "/data/gyx/chid/preprocessed_qa_cls_1000_not_too_naive/"

    tokenizer_path = "/mnt/nfs/home/gyx/bpe/bpe_3w"

    tokenizer = GPT2Tokenizer(os.path.join(tokenizer_path, 'vocab.json'), os.path.join(tokenizer_path, 'merges.txt'), os.path.join(tokenizer_path, 'chinese_vocab.model'))

    os.makedirs(preprocesed_dir, exist_ok=True)

    for split in ["train", "dev"]:
        with open(os.path.join(data_dir, "{}.json".format(split)), "r") as f:
            lines = f.readlines()[:1000]

        with open(os.path.join(ans_data_dir, "{}_answer.json".format(split)), "r") as f:
            ans_d = json.load(f)

        preprocess((lines, ans_d), tokenizer, split)
