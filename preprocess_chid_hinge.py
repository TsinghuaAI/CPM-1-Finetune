import json
import re
import os
import random
from tqdm import tqdm

def process_one_sent_train(sent, answers, neg_ans, candidates):
    pattern = re.compile(r"#idiom\d+#")
    s = re.sub(pattern, lambda m: candidates[answers[m.group()]], sent)
    neg_s = re.sub(pattern, lambda m: candidates[neg_ans[m.group()]], sent)

    return s, neg_s

def process_one_sent_eval(sent, answers, candidates):
    pattern = re.compile(r"#idiom\d+#")
    res = pattern.findall(sent)
    start = 0
    L = []
    while True:
        m = pattern.search(sent, start)
        if m is None:
            break
        L.append({
            "cands": [],
            "truth": answers[m.group()]
        })
        for idm in candidates:
            cand = sent[:m.start()] + idm + sent[m.end():]
            cand = re.sub(pattern, "", cand)
            L[-1]["cands"].append(cand)
        start = m.end()
    
    return L

def get_rand_except(L, x):
    y = random.choice(L)
    while y == x:
        y = random.choice(L)

    return y
    

data_dir = "/data/gyx/chid/"
ans_data_dir = "/data/gyx/chid/"

preprocesed_dir = "/data/gyx/chid/preprocessed_with_neg/"

os.makedirs(preprocesed_dir, exist_ok=True)

for split in ["train", "dev"]:
    with open(os.path.join(data_dir, "{}.json".format(split)), "r") as f:
        lines = f.readlines()

    with open(os.path.join(ans_data_dir, "{}_answer.json".format(split)), "r") as f:
        ans_d = json.load(f)
        neg_ans_d = ans_d.copy()
        for k in neg_ans_d:
            neg_ans_d[k] = get_rand_except(list(range(10)), neg_ans_d[k])

    all_data = {
        "contents": [],
        "sids": [],
        "labels": [],
        "cids": []
    }
    sid = 0
    for line in tqdm(lines, desc="Preprocessing {}".format(split)):
        jobj = json.loads(line)
        for sent in jobj["content"]:
            if split == "train":
                sample, neg_sample = process_one_sent_train(sent, ans_d, neg_ans_d, jobj["candidates"])
                all_data["contents"].append((sample, neg_sample))
                all_data["sids"].append(sid)
                all_data["cids"].append(0)
                sid += 1
            else:
                sample_L = process_one_sent_eval(sent, ans_d, jobj["candidates"])
                for samp in sample_L:
                    all_data["contents"].extend(samp["cands"])
                    all_data["sids"].extend([sid for _ in samp["cands"]])
                    all_data["cids"].extend([i for i in range(len(samp["cands"]))])
                    all_data["labels"].append(samp["truth"])
                    sid += 1

    with open(os.path.join(preprocesed_dir, "{}.json".format(split)), "w") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
