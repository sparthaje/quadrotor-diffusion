# latent_diffusion_cfg

import numpy as np

from quadrotor_diffusion.utils.nn.args import LatentDiffusionWrapperArgs, Unet1DArgs, TrainerArgs
from quadrotor_diffusion.utils.dataset.normalizer import NoNormalizer, NormalizerTuple
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorRaceTrajectoryDataset, DiffusionDataset

unet_args = Unet1DArgs(
    traj_dim=12,
    features=128,
    channel_mults=[1, 2, 4],
    attentions=[
        [False, False, False],
        [True],
        [False, False]
    ],
    context_mlp="time",
    conditioning=(3, 4)  # not number of tokens but dimension of each token (local, global)
)

diff_args = LatentDiffusionWrapperArgs(
    predict_epsilon=True,
    loss="L1Loss",
    n_timesteps=100,
    loss_params=None,
    dropout=0.2,
    conditioning_shape=(3, 4)
)

train_args = TrainerArgs(
    ema_decay=0.995,
    num_batches_no_ema=20,
    num_batches_per_ema=10,

    batch_size_per_gpu=128,
    batches_per_backward=4,

    log_dir="logs/training/",
    save_freq=5,

    learning_rate=1e-4,
    num_gpus=1,
    device="cuda:3",
    max_epochs=400,
    evaluate_every=5,
    description="Training wi1h 192 VAE, joint masking on local at all+weighted l1 loss"
)

vae_experiment = 192

normalizer = NoNormalizer()

dataset = DiffusionDataset("data", 128, normalizer)
