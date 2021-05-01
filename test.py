import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embeds = nn.Embedding(10, 100)
        self.fc = nn.Linear(100, 100)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.embeds[x]
        x = self.fc[x]
        x = self.act[x]
        x = self.fc2[x]

        return x

net = Net()

print(net)

for t in net.named_modules():
    pass