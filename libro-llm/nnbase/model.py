import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))

    def forward(self, x):
        return self.net(x)


model = SimpleNN()
