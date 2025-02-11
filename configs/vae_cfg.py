# vae_cfg

import numpy as np

from quadrotor_diffusion.utils.nn.args import TrainerArgs, VAE_EncoderArgs, VAE_DecoderArgs, VAE_WrapperArgs
from quadrotor_diffusion.utils.dataset.normalizer import GuassianNormalizer, NoNormalizer, LinearNormalizer
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorTrajectoryDataset, QuadrotorRaceTrajectoryDataset

train_args = TrainerArgs(
    ema_decay=0.995,
    num_batches_no_ema=25,
    num_batches_per_ema=25,

    batch_size_per_gpu=512,
    batches_per_backward=2,

    log_dir="logs/training",
    save_freq=5,

    learning_rate=2e-4,
    num_gpus=1,
    device="cuda:0",

    max_epochs=60,
)

vae_args = VAE_WrapperArgs(
    loss="L1Loss",
    beta=0.01,
    loss_params=None
)

encoder_args = VAE_EncoderArgs(
    3,
    6,
    64,
    (1, 2, 4),
)

decoder_args = VAE_DecoderArgs(
    3,
    6,
    64,
    (4, 2, 1)
)

# dataset = QuadrotorTrajectoryDataset('data/quadrotor_random', NoNormalizer(), order=1)
dataset = QuadrotorRaceTrajectoryDataset('data', ["linear", "u"], 360, NoNormalizer())
