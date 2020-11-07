import numpy as np
import matplotlib.pyplot as plt
import re

with open("train_log.txt") as f:
    lines = f.readlines()

L = []

for line in lines[1:]:
    pattern = re.compile(r"(.*)(\d\.\d+)")
    obj = pattern.match(line.strip())
    L.append(float(obj.group(2)))

fig = plt.figure(figsize=(10, 10))
sub = fig.add_subplot(111)

X = [(x + 1) * 10 for x in range(len(L))]

l2, = sub.plot(X, L)

plt.savefig("qa_full_trn_loss.pdf", foramt="pdf")
