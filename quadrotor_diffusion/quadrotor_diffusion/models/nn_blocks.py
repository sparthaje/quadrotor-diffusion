import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


################################################# Enbeddings #################################################

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Parameters:
        - t: [batch_size]
        """
        # Half the dimension is used for sine and cosine components
        half_dim = self.dim // 2

        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -scale)

        # Compute the position encoding for each time in t (frequency as rows time as columns)
        emb = t[:, None] * freqs[None, :]

        # Positional encoding: sin(t * freqs) and cos(t * freqs)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        return emb


############################################ 1D Convolution Blocks ############################################


class Conv1dNormalized(nn.Module):
    """
        Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    """
    Cuts down horizon by half
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """
    Doulbes the horizon dimension
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)
