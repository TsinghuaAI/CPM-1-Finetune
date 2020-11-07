import json
import re
import os
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
        cand_poses = []
        for i, cand in enumerate(cands):
            cands_ids.extend(tokenizer.encode(cand.strip()))
            cand_poses.append(len(cands_ids) - 1)
            cands_ids.append(tokenizer.encoder["<sep>"])

        prefix = re.sub(pattern, "", sent[:m.start()])
        postfix = re.sub(pattern, "", sent[m.end():])

        context_ids = tokenizer.encode(prefix.strip())
        mask_pos = len(context_ids) - 1 if len(context_ids) > 0 else 0
        context_ids = context_ids + [tokenizer.encoder["<mask>"]] + tokenizer.encode(postfix.strip()) + [tokenizer.encoder["<eod>"]]
        sent_pos = len(context_ids) - 2

        ids = context_ids + cands_ids

        cand_poses = [x + len(context_ids) for x in cand_poses]

        L.append({
            "sent": ids,
            "truth": answers[m.group()],
            "cand_poses": cand_poses,
            "mask_pos": mask_pos,
            "sent_pos": sent_pos
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

    preprocesed_dir = "/data/gyx/chid/preprocessed_qa_all_attn_1000/"

    tokenizer_path = "/mnt/nfs/home/gyx/bpe/bpe_3w"

    tokenizer = GPT2Tokenizer(os.path.join(tokenizer_path, 'vocab.json'), os.path.join(tokenizer_path, 'merges.txt'), os.path.join(tokenizer_path, 'chinese_vocab.model'))

    os.makedirs(preprocesed_dir, exist_ok=True)

    for split in ["train", "dev"]:
        with open(os.path.join(data_dir, "{}.json".format(split)), "r") as f:
            lines = f.readlines()

        with open(os.path.join(ans_data_dir, "{}_answer.json".format(split)), "r") as f:
            ans_d = json.load(f)

        preprocess((lines, ans_d), tokenizer, split)
