# diffusion_cfg

import numpy as np

from quadrotor_diffusion.utils.nn.args import DiffusionWrapperArgs, Unet1DArgs, TrainerArgs
from quadrotor_diffusion.utils.dataset.normalizer import NoNormalizer
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorRaceTrajectoryDataset

unet_args = Unet1DArgs(
    traj_dim=3,
    features=64,
    channel_mults=[1, 2, 4, 8],
    attentions=[
        [False, False, False, False],
        [False],
        [False, False, False]
    ]
)

diff_args = DiffusionWrapperArgs(
    predict_epsilon=True,
    loss="L1Loss",
    n_timesteps=1000,
    loss_params=None
)

train_args = TrainerArgs(
    ema_decay=0.995,
    num_batches_no_ema=20,
    num_batches_per_ema=10,

    batch_size_per_gpu=128,
    batches_per_backward=4,

    log_dir="logs/training/",
    save_freq=5,

    learning_rate=2e-4,
    num_gpus=1,
    device="cuda:3",
    max_epochs=200,
    evaluate_every=5
)

normalizer = NoNormalizer()

dataset = QuadrotorRaceTrajectoryDataset('data', ["linear", "u"], 360, normalizer)
