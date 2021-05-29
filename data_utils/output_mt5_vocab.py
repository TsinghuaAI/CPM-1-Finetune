import sentencepiece as spm
import json

sp_model = spm.SentencePieceProcessor()
sp_model.Load("/mnt/sfs_turbo/CPM-Finetune-xcj/spiece.model")

vocab = {wid: sp_model.IdToPiece(wid) for wid in range(sp_model.GetPieceSize())}
print(len(vocab))
fout = open("/mnt/sfs_turbo/CPM-Finetune-xcj/mt5vocab.txt", "w")
print(json.dumps(vocab, ensure_ascii = False, indent = 4), file = fout)
fout.close()