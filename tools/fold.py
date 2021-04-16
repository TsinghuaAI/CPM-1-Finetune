import sys
from tqdm import tqdm

text_path = sys.argv[1]
out_text_path = sys.argv[2]

max_len = 100000

with open(text_path, "r") as f:
    lines = f.readlines()

with open(out_text_path, "w") as f:
    for line in tqdm(lines):
        line = line.replace("<n>", "\n")
        line = line.strip()
        start = 0
        while start + max_len < len(line):
            cut_line = line[start:start+max_len]
            cut_line = cut_line.replace("\n", "<n>")
            f.write(cut_line + "\n")
            start = start + max_len

        cut_line = line[start:]
        cut_line = cut_line.replace("\n", "<n>")
        f.write(cut_line + "\n")