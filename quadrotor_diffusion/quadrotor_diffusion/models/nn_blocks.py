import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


################################################# Activation #################################################

class PELU(nn.Module):
    """Parametric Exponential Linear Unit."""

    def __init__(self) -> None:
        """Initialize learnable parameters."""
        super().__init__()
        self.log_a = nn.Parameter(torch.zeros(1))
        self.log_b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        a = self.log_a.exp()
        b = self.log_b.exp()
        return torch.where(
            x >= 0,
            x * (a / b),
            a * (torch.exp(x / b) - 1)
        )

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


############################################### Other nice stuff ############################################

def soft_argmax(x: torch.Tensor, alpha: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable approximation to the argmax.

    Args:
        x (torch.Tensor): will compute across last dimension
        alpha: Temperature parameter. Higher => closer to one-hot.

    Returns:
        - max index (torch.Tensor): 0 ≤ t < n
        - max value (torch.Tensor): max value in x
    """
    # Compute softmax over x scaled by alpha
    w = F.softmax(alpha * x, dim=-1)

    # Weighted sum of indices: 0 * w[0] + 1 * w[1] + ... + (n-1) * w[n-1]
    indices = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype).unsqueeze(0)
    soft_argmax = torch.sum(indices * w, dim=-1)
    soft_max = torch.sum(x * w, dim=-1)

    return soft_argmax, soft_max


def soft_argmin(x: torch.Tensor, alpha: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable approximation to the argmin.

    Args:
        x (torch.Tensor): will compute across the last dimension
        alpha: Temperature parameter. Higher => closer to one-hot.

    Returns:
        - min index (torch.Tensor): 0 ≤ t < n
        - min value (torch.Tensor): min value in x
    """
    # Compute softmax over x scaled by alpha
    soft_argmin, neg_soft_min = soft_argmax(-x, alpha)

    return soft_argmin, -neg_soft_min
