# full_state_quadrotor

import numpy as np

from quadrotor_diffusion.utils.nn.args import DiffusionWrapperArgs, Unet1DArgs, TrainerArgs
from quadrotor_diffusion.utils.dataset.normalizer import GuassianNormalizer, NoNormalizer
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorFullStateDataset

unet_args = Unet1DArgs(
    traj_dim=9,
    features=32,
    channel_mults=[1, 2, 4, 8],
    attentions=[
        [True, True, True, True],
        [True],
        [True, True, True]
    ]
)

diff_args = DiffusionWrapperArgs(
    predict_epsilon=True,
    loss="MSELoss",
    n_timesteps=1000
)

train_args = TrainerArgs(
    ema_decay=0.995,
    num_batches_no_ema=20,
    num_batches_per_ema=10,

    batch_size_per_gpu=64,
    batches_per_backward=4,

    log_dir="logs/training/",
    save_freq=5,

    learning_rate=2e-4,
    num_gpus=1,
    device="cuda:3",

    max_epochs=150
)

# GuassianNormalizer(mean=np.zeros((9,)), variance=np.array([1.3, 1.3, 0.02, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3]))
normalizer = NoNormalizer()

dataset = QuadrotorFullStateDataset('data/quadrotor_random', normalizer)
