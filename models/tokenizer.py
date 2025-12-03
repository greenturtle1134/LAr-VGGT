import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

class Tokenizer(nn.Module):

    def __init__(
        self,
        layers_per_pool = [2, 2, 2, 2],
    ):
        super().__init__()
        
        layers = list()
        i = 1
        for n in layers_per_pool:
            for _ in range(n):
                layers.append(nn.Conv2d(i, i*2, 3, padding=1))
                layers.append(nn.ReLU())
                i *= 2
            layers.append(nn.MaxPool2d(2))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 3:
            N, H, W = x.shape
            x = x.view(N, 1, H, W)
        else:
            N, _, H, W = x.shape
        
        for layer in self.layers:
            x = layer.forward(x)
        return x.view(N, -1)


class IdentityTokenizer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if len(x.shape) == 3:
            N, H, W = x.shape
            x = x.view(N, 1, H, W)
        else:
            N, _, H, W = x.shape

        return x.view(N, -1)
        