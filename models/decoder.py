import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

class Decoder(nn.Module):

    def __init__(
        self,
        dim_in: int,
        patch_size = 16,
        base_res = 4,
        base_channels = 64
    ):
        super().__init__()

        self.dim_in = dim_in
        self.patch_size = patch_size
        self.base_res = base_res
        self.base_channels = base_channels

        self.fc = nn.Sequential(
            nn.Linear(dim_in, base_channels * base_res * base_res),
            nn.ReLU()
        )

        # Currently hardcoded to assume 16x16 output and 64 base channels.
        self.conv = nn.Sequential(
            nn.Conv2d(base_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4, 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2, 1, 3, padding=1)
        )

    def forward(self, x):
        N, _ = x.shape

        x = self.fc(x)
        x = torch.reshape(x, (N, self.base_channels, self.base_res, self.base_res))
        x = self.conv(x)
        x = torch.reshape(x, (N, self.patch_size, self.patch_size))

        return x