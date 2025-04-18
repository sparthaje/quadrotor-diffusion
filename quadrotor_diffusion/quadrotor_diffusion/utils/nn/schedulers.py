import numpy as np
import torch


def cosine_beta_schedule(timesteps, s=0.008, clip=True, dtype=torch.float32) -> torch.Tensor:
    """
    Generates a cosine-based beta schedule as proposed in 
    https://openreview.net/forum?id=-NEXDKk8gZ.

    Parameters:
    - timesteps (int): Total number of timesteps.
    - s (float): Small offset to control the start of the schedule.
    - dtype: Data type for the returned tensor (default is torch.float32).

    Returns:
    - torch.Tensor: A tensor of beta values for each timestep.
    """
    steps = timesteps + 1
    # Generate a sequence of `steps` values from 0 to 1
    x = np.linspace(0.0, 1.0, steps)

    # Compute the cumulative product of alphas using the cosine schedule
    alphas_cumprod = np.cos((x + s) / (1 + s) * np.pi * 0.5) ** 2
    # Normalize to start from 1
    alphas_cumprod /= alphas_cumprod[0]

    # Calculate betas by taking the difference in adjacent alphas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    if not clip:
        return torch.tensor(betas, dtype=dtype)

    # Clip values to keep within the range [0, 0.999]
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)

    return torch.tensor(betas_clipped, dtype=dtype)


def stable_diffusion(timesteps, dtype=torch.float32) -> torch.Tensor:
    pass


def alpha_bar_cont(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Continuous analogue of the cosine schedule used by CTM/CM.

    Args:
        t:   Continuous time in **seconds / variance units**, assumed in (0, 1].
        s:   Small offset, defaults to 8e-3 as in Nichol & Dhariwal (2021).
    Returns:
        alpha_bar(t)  -  cumulative product of alphas at that real-valued time.
    """
    _t = t.to(torch.float32)
    return torch.cos(((_t + s) / (1.0 + s)) * np.pi/2) ** 2


def sample_times(batch: int, T: float = 1.0, eps: float = 1.0e-5, *, device=None) -> torch.Tensor:
    """Uniform sampling of continuous times in (eps, T]."""
    u = torch.rand(batch, device=device)
    return eps + (T - eps) * u
