import sentencepiece_model_pb2 as model
import sentencepiece as spm

import json
import os

sentinel_num = 100
input_model = "bpe_3w_new/chinese_vocab.model"
output_dir = "bpe_3w_enc_dec"


m = model.ModelProto()
m.ParseFromString(open(input_model, "rb").read())

special_tokens = ["<eod>", "<pad>", "<sep>", "<cls>", "<s>", "</s>", "<mask>"]
for x in m.pieces:
    if x.piece in special_tokens:
        x.type = 1 # NORMAL

sentinel_tokens = ["<s_{}>".format(i) for i in range(sentinel_num)]
for token in sentinel_tokens:
    new_token = model.ModelProto().SentencePiece()
    new_token.piece = token
    new_token.type = 4 # USER_DEFINED
    new_token.score = 0
    m.pieces.append(new_token)

with open(os.path.join(output_dir, 'chinese_vocab.model'), 'wb') as f:
    f.write(m.SerializeToString())

vocab_json = {x.piece:i for i, x in enumerate(m.pieces)}
vocab = [(x.piece, x.score) for x in m.pieces]

with open(os.path.join(output_dir, 'vocab-chn.json'), "w") as f:
    json.dump(vocab_json, f, ensure_ascii=False)

with open(os.path.join(output_dir, 'vocab.json'), "w") as f:
    json.dump(vocab_json, f)

with open(os.path.join(output_dir, 'chinese_vocab.vocab'), "w") as f:
    for t in vocab:
        f.write("{}\t{}\n".format(t[0], t[1]))
