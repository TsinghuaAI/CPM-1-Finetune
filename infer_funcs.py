import json
import os

from numpy.lib.nanfunctions import _nansum_dispatcher

from data_utils.tokenization_enc_dec import EncDecTokenizer


def infer_tnews(args, tokenizer: EncDecTokenizer, all_idx, all_preds, prefix=""):
    label_word_map = {
        "故事": ("100", "news_story"),
        "文化": ("101", "news_culture"),
        "娱乐": ("102", "news_entertainment"),
        "体育": ("103", "news_sports"),
        "金融": ("104", "news_finance"),
        "房地产": ("106", "news_house"),
        "汽车": ("107", "news_car"),
        "教育": ("108", "news_edu"),
        "科技": ("109", "news_tech"),
        "军事": ("110", "news_military"),
        "旅游": ("112", "news_travel"),
        "世界": ("113", "news_world"),
        "股票": ("114", "news_stock"),
        "农业": ("115", "news_agriculture"),
        "游戏": ("116", "news_game"),
    }
    all_preds = [tokenizer.decoder[p] for p in all_preds]
    print(all_preds)

    num = 0
    with open(os.path.join(args.save, "predicts{}.json".format(prefix)), "w") as f:
        for i, p in zip(all_idx, all_preds):
            if i != -1:
                f.write(json.dumps({
                    "id": i,
                    "label": label_word_map.get(p, ("-1", "none"))[0],
                    "label_desc": label_word_map.get(p, ("-1", "none"))[1]
                }) + "\n")
                num += 1

    return num


def infer_afqmc(args, tokenizer: EncDecTokenizer, all_idx, all_preds, prefix=""):
    label_word_map = {
        "不同": "0",
        "相同": "1"
    }
    all_preds = [tokenizer.decoder[p] for p in all_preds]

    num = 0
    with open(os.path.join(args.save, "predicts{}.json".format(prefix)), "w") as f:
        for i, p in zip(all_idx, all_preds):
            if i != -1:
                f.write(json.dumps({
                    "id": i,
                    "label": label_word_map.get(p, 0),
                }) + "\n")
                num += 1

    return num


def infer_ocnli(args, tokenizer: EncDecTokenizer, all_idx, all_preds, prefix=""):
    label_word_map = {
        "相似": "entailment",
        "矛盾": "contradiction",
        "中立": "neutral"
    }
    all_preds = [tokenizer.decoder[p] for p in all_preds]
    
    num = 0
    with open(os.path.join(args.save, "predicts{}.json".format(prefix)), "w") as f:
        for i, p in zip(all_idx, all_preds):
            if i != -1:
                f.write(json.dumps({
                    "id": i,
                    "label": label_word_map[p],
                }) + "\n")
                num += 1

    return num


def infer_iflytek(args, tokenizer: EncDecTokenizer, all_idx, all_preds, prefix=""):
    label_word_map = {
        "2": "免费wifi",
        "23": "竞技游戏"
    }

    labels = {}
    with open(os.path.join(args.data_path, "labels.json"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            labels[line["label"]] = line["label_des"]

    labels.update(label_word_map)
    labels = {v:k for k, v in labels.items()}

    all_preds = [tokenizer.decode(p[1:-1]) for p in all_preds]
    
    num = 0
    with open(os.path.join(args.save, "predicts{}.json".format(prefix)), "w") as f:
        for i, p in zip(all_idx, all_preds):
            if i != -1:
                f.write(json.dumps({
                    "id": i,
                    "label": labels.get(p, "0"),
                }) + "\n")
                num += 1
                if p not in labels:
                    print(i, p)

    return num


def infer_cmnli(args, tokenizer: EncDecTokenizer, all_idx, all_preds, prefix=""):
    label_word_map = {
        "相似": "entailment",
        "矛盾": "contradiction",
        "中立": "neutral"
    }

    all_preds = [tokenizer.decoder[p] for p in all_preds]
    
    num = 0
    with open(os.path.join(args.save, "predicts{}.json".format(prefix)), "w") as f:
        for i, p in zip(all_idx, all_preds):
            if i != -1:
                f.write(json.dumps({
                    "id": i,
                    "label": label_word_map[p],
                }) + "\n")
                num += 1

    return num


def infer_csl(args, tokenizer: EncDecTokenizer, all_idx, all_preds, prefix=""):
    label_word_map = {
        "错误": "0",
        "正确": "1",
    }

    all_preds = [tokenizer.decoder[p] for p in all_preds]
    
    num = 0
    with open(os.path.join(args.save, "predicts{}.json".format(prefix)), "w") as f:
        for i, p in zip(all_idx, all_preds):
            if i != -1:
                f.write(json.dumps({
                    "id": i,
                    "label": label_word_map[p],
                }) + "\n")
                num += 1

    return num


def infer_wsc2(args, tokenizer: EncDecTokenizer, all_idx, all_preds, prefix=""):
    label_word_map = {
        0: "false",
        1: "true",
    }

    num = 0
    with open(os.path.join(args.save, "predicts{}.json".format(prefix)), "w") as f:
        for i, p in zip(all_idx, all_preds):
            if i != -1:
                f.write(json.dumps({
                    "id": i,
                    "label": label_word_map[p],
                }) + "\n")
                num += 1

    return num


def infer_cmrc(args, tokenizer: EncDecTokenizer, all_idx, all_preds, prefix=""):
    all_preds = [tokenizer.decode(p[1:-1]) for p in all_preds]
    predicts = {}
    num = 0
    for i, p in zip(all_idx, all_preds):
        if i != -1:
            pid = i // 100000
            qid = i % 100000
            predicts["TEST_{}_QUERY_{}".format(pid, qid)] = p
            num += 1

    with open(os.path.join(args.save, "predicts{}.json".format(prefix)), "w") as f:
        json.dump(predicts, f, indent=4, ensure_ascii=False)

    return num


def infer_c32(args, tokenizer: EncDecTokenizer, all_idx, all_preds, prefix=""):
    number_map = {
        50: 0, # 一
        230: 1,
        156: 2,
        349: 3,
        443: 4,
        803: 5,
        950: 6,
        1031: 7 # 八
    }

    num = 0
    with open(os.path.join(args.save, "predicts{}.json".format(prefix)), "w") as f:
        for i, p in zip(all_idx, all_preds):
            if i != -1:
                f.write(json.dumps({
                    "id": i,
                    "label": number_map[p],
                }) + "\n")
                num += 1

    return num


def infer_chid2(args, tokenizer: EncDecTokenizer, all_idx, all_preds, prefix=""):
    number_map = {
        50: 0, # 一
        230: 1,
        156: 2,
        349: 3,
        443: 4,
        803: 5,
        950: 6,
        1031: 7, # 八
        1189: 8, # 九
        1320: 9
    }

    num = 0
    with open(os.path.join(args.save, "predicts{}.json".format(prefix)), "w") as f:
        for i, p in zip(all_idx, all_preds):
            if i != -1:
                f.write(json.dumps({
                    "id": "#idiom{}#".format(i),
                    "label": number_map[p],
                }) + "\n")
                num += 1

    return num