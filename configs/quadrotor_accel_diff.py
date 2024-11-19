# quadrotor_accel_diff

import numpy as np

from quadrotor_diffusion.utils.nn.args import DiffusionWrapperArgs, Unet1DArgs, TrainerArgs
from quadrotor_diffusion.utils.dataset.normalizer import GuassianNormalizer, NoNormalizer, MinMaxNormalizer
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorAcc

unet_args = Unet1DArgs(
    traj_dim=3,
    features=32,
    channel_mults=[1, 2, 4, 8],
    attentions=[
        [True, True, True, False],
        [False],
        [False, True, True]
    ]
)

diff_args = DiffusionWrapperArgs(
    predict_epsilon=True,
    loss="L1Loss",
    n_timesteps=1000
)

train_args = TrainerArgs(
    ema_decay=0.995,
    num_batches_no_ema=20,
    num_batches_per_ema=10,

    batch_size_per_gpu=1024,
    batches_per_backward=1,

    log_dir="logs/training/",
    save_freq=5,

    learning_rate=2e-4,
    num_gpus=1,
    device="cuda:2",

    max_epochs=150
)

normalizer = NoNormalizer()  # MinMaxNormalizer(mins=np.array([-2.0, -2.0, -2.0]), maxes=np.array([2.0, 2.0, 2.0]))

dataset = QuadrotorAcc('data/quadrotor_random', normalizer)
