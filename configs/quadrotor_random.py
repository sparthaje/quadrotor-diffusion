import numpy as np

from quadrotor_diffusion.utils.nn.args import DiffusionWrapperArgs, Unet1DArgs, TrainerArgs
from quadrotor_diffusion.utils.dataset.normalizer import GuassianNormalizer, NoNormalizer
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorTrajectoryDataset

unet_args = Unet1DArgs(
    traj_dim=3,
    features=32,
    channel_mults=[1, 2, 4, 8],
    attentions=[
        [False, False, False, False],
        [False],
        [False, False, False]
    ]
)

diff_args = DiffusionWrapperArgs(
    predict_epsilon=True,
    loss="MSELoss",
    n_timesteps=100
)

train_args = TrainerArgs(
    ema_decay=0.995,
    num_batches_no_ema=50,
    num_batches_per_ema=10,

    batch_size_per_gpu=32,
    batches_per_backward=2,

    log_dir="logs/training/",
    save_freq=5,

    learning_rate=2e-4,
    num_gpus=4,
    device="cuda",

    max_epochs=50
)

normalizer = GuassianNormalizer(
    mean=np.array([0., 0., 0.]),
    variance=np.array([1.3, 1.3, 0.02]),
)

dataset = QuadrotorTrajectoryDataset('data/quadrotor_random', normalizer)
