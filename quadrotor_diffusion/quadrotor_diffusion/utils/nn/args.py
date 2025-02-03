from typing import List, Tuple
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
    loss_params: Tuple of data to pass into lsoss initialization varies by loss
    n_timesteps: Number of diffusion timesteps
    """
    predict_epsilon: bool
    loss: str
    loss_params: Tuple
    n_timesteps: int


@dataclass
class VAE_EncoderArgs:
    """
    traj_dim: number of items per state (i.e. xyz = 3)
    latent_dim: latent dimensionality for vector
    features: Base number of features to project traj_dim to
    channel_mults: How to scale up the channels as horizon gets down scaled, e.g. [1, 2, 4, 8]
    """
    traj_dim: int
    latent_dim: int
    features: int
    channel_mults: Tuple[int]


@dataclass
class VAE_DecoderArgs:
    """
    traj_dim: number of items per state (i.e. xyz = 3)
    latent_dim: latent dimensionality for vector
    features: Base number of features to project into traj_dim
    channel_mults: How to scale down the channels as horizon gets up scaled, e.g. [8, 4, 2, 1]
    """
    traj_dim: int
    latent_dim: int
    features: int
    channel_mults: Tuple[int]


@dataclass
class VAE_WrapperArgs:
    """
    loss: An nn.Module wrapper of a loss function (can be dynamic with learnable parameters)
    Loss Params: whatever parameters are releavant to the internal reconstruction loss
    beta: How much to weight KL divergence
    """
    loss: str
    loss_params: Tuple
    beta: float


@dataclass
class CourseEmbeddingArgs:
    hidden_dim: int
    n_layers: int
    embed_dim: int
    gate_input_dim: int


@dataclass
class TrainerArgs:
    """
    ema_decay: EMA rate to update model
    num_batches_no_ema: Number of batches to backprop before starting EMA
    num_batches_per_ema: Number of batches to see before updating EMA

    batch_size_per_gpu: batch size per gpu
    batches_per_backward: how many batches to accumulate gradient for before optim step
                          Note: one batch size = batch_size_per_gpu * num_gpus
    learning_rate: learning rate

    log_dir: directory to put all logs in
    save_freq: how many epochs to pass before saving

    num_gpus: number of gpus to use (Default: -1 use all of them)
    device: device to train on
    max_epochs: number of epochs to stop at
    """

    ema_decay: float
    num_batches_no_ema: int
    num_batches_per_ema: int

    batch_size_per_gpu: int
    batches_per_backward: int
    learning_rate: float

    log_dir: str
    save_freq: int

    num_gpus: int = -1
    device: str = 'cuda'
    max_epochs: int = None
