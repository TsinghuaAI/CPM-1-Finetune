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
        for i, cand in enumerate(cands):
            cands_ids.extend(tokenizer.encode("选项{}:".format(i)))
            cands_ids.extend(tokenizer.encode(cand.strip()))
            cands_ids.append(tokenizer.encoder["<sep>"])

        prefix = re.sub(pattern, "", sent[:m.start()])
        postfix = re.sub(pattern, "", sent[m.end():])

        context_ids = tokenizer.encode("上下文:") + tokenizer.encode(prefix.strip()) + [tokenizer.encoder["<mask>"]] + tokenizer.encode(postfix.strip()) + [tokenizer.encoder["<eod>"]]

        ques_ids = [tokenizer.encoder["<mask>"]] + tokenizer.encode("答案是:") + [num_ids[answers[m.group()]]]
        
        ids = context_ids + cands_ids + ques_ids

        L.append({
            "sent": ids,
            "truth": answers[m.group()],
            "cands_len": (len(context_ids), len(context_ids) + len(cands_ids)),
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

    preprocesed_dir = "/data/gyx/chid/preprocessed_qa_cands_end/"

    tokenizer_path = "/mnt/nfs/home/gyx/bpe/bpe_3w"

    tokenizer = GPT2Tokenizer(os.path.join(tokenizer_path, 'vocab.json'), os.path.join(tokenizer_path, 'merges.txt'), os.path.join(tokenizer_path, 'chinese_vocab.model'))

    os.makedirs(preprocesed_dir, exist_ok=True)

    for split in ["test"]:
        with open(os.path.join(data_dir, "{}.json".format(split)), "r") as f:
            lines = f.readlines()

        with open(os.path.join(ans_data_dir, "{}_answer.json".format(split)), "r") as f:
            ans_d = json.load(f)

        preprocess((lines, ans_d), tokenizer, split)
