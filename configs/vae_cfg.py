# vae_cfg

import numpy as np

from quadrotor_diffusion.utils.nn.args import TrainerArgs, VAE_EncoderArgs, VAE_DecoderArgs, VAE_WrapperArgs
from quadrotor_diffusion.utils.dataset.normalizer import GuassianNormalizer, NoNormalizer, LinearNormalizer
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorTrajectoryDataset

train_args = TrainerArgs(
    ema_decay=0.995,
    num_batches_no_ema=25,
    num_batches_per_ema=25,

    batch_size_per_gpu=1024,
    batches_per_backward=1,

    log_dir="logs/training",
    save_freq=5,

    learning_rate=2e-4,
    num_gpus=1,
    device="cuda:2",

    max_epochs=50,
)

vae_args = VAE_WrapperArgs(
    loss="L1Loss",
    beta=0.01,
    loss_params=None  # (10.0, 0.8, 2.5),
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

normalizer = LinearNormalizer(
    scalers=np.array([0.5, 0.5, 15.,]),
    biases=np.array([0., 0., -6.,])
)

dataset = QuadrotorTrajectoryDataset('data/quadrotor_random', NoNormalizer(), order=1)
