from typing import List
from dataclasses import dataclass


@dataclass
class Unet1DArgs:
    """
    traj_dim: number of items per state (i.e. xyz = 3)
    features: Base number of features to project traj_dim to
    channel_mults: How to scale up the channels as horizon gets down scaled, e.g. [1, 2, 4, 8]
    attentions: Whether or not to put a linear attention between the resnet blocks
    """
    traj_dim: int
    features: int
    channel_mults: List[int]
    attentions: List[List[bool]]


@dataclass
class DiffusionWrapperArgs:
    """
    predict_epsilon: Whether the model should predict epsilon or x_0 directly from x_t
    loss: An nn.Module wrapper of a loss function (can be dynamic with learnable parameters)
    n_timesteps: Number of diffusion timesteps
    """
    predict_epsilon: bool
    loss: str
    n_timesteps: int
