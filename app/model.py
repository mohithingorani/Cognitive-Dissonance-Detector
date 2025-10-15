import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid, hid // 2),
            nn.ReLU(),
            nn.Linear(hid // 2, 1),
            nn.Sigmoid()  # output 0..1 dissonance
        )

    def forward(self, x):
        return self.net(x)