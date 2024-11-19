# quadrotor_pos_vae

import numpy as np

from quadrotor_diffusion.utils.nn.args import TrainerArgs, VAE_EncoderArgs, VAE_DecoderArgs, VAE_WrapperArgs
from quadrotor_diffusion.utils.dataset.normalizer import GuassianNormalizer, NoNormalizer
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorTrajectoryDataset, QuadrotorAcc

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
    device="cuda:1",

    max_epochs=160,
)

vae_args = VAE_WrapperArgs(
    loss="L1Loss",
    beta=0.2,
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

normalizer = NoNormalizer()

dataset = QuadrotorAcc('data/quadrotor_random', normalizer)
