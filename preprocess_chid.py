import json
import re
import os
from tqdm import tqdm

def process_one_sent_train(sent, answers, candidates):
    pattern = re.compile(r"#idiom\d+#")
    s = re.sub(pattern, lambda m: candidates[answers[m.group()]], sent)

    return s

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

data_dir = "/data/gyx/chid/"
ans_data_dir = "/data/gyx/chid/"

preprocesed_dir = "/data/gyx/chid/preprocessed/"

os.makedirs(preprocesed_dir, exist_ok=True)

for split in ["train", "dev"]:
    with open(os.path.join(data_dir, "{}.json".format(split)), "r") as f:
        lines = f.readlines()

    with open(os.path.join(ans_data_dir, "{}_answer.json".format(split)), "r") as f:
        ans_d = json.load(f)

    all_data = {
        "contents": [],
        "sids": [],
        "labels": []
    }
    sid = 0
    for line in tqdm(lines, desc="Preprocessing {}".format(split)):
        jobj = json.loads(line)
        for sent in jobj["content"]:
            if split == "train":
                sample = process_one_sent_train(sent, ans_d, jobj["candidates"])
                all_data["contents"].append(sample)
            else:
                sample_L = process_one_sent_eval(sent, ans_d, jobj["candidates"])
                for samp in sample_L:
                    all_data["contents"].extend(samp["cands"])
                    all_data["sids"].extend([sid for _ in samp["cands"]])
                    all_data["labels"].append(samp["truth"])
                    sid += 1

    with open(os.path.join(preprocesed_dir, "{}.json".format(split)), "w") as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
