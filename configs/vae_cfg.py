# vae_cfg

import numpy as np

from quadrotor_diffusion.utils.nn.args import TrainerArgs, VAE_EncoderArgs, VAE_DecoderArgs, VAE_WrapperArgs
from quadrotor_diffusion.utils.dataset.normalizer import GuassianNormalizer, NoNormalizer, LinearNormalizer
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorRaceSegmentDataset

train_args = TrainerArgs(
    ema_decay=0.0,
    num_batches_no_ema=float('inf'),
    num_batches_per_ema=float('inf'),

    batch_size_per_gpu=128,
    batches_per_backward=4,

    log_dir="logs/training",
    save_freq=5,

    learning_rate=2e-4,
    num_gpus=1,
    device="cuda:1",

    max_epochs=400,
    evaluate_every=5,
    description="telomere strat 2"
)

vae_args = VAE_WrapperArgs(
    loss="Smooth",
    beta=0.1,
    loss_params=(
        "WeightedL1",
        # Weighting on the percent of horizon
        (0.1, 3.0),
        # Weighting for L1 loss on velocity and acceleration
        (0.3, 0.1),
    ),
    telomere_strategy=0
)

encoder_args = VAE_EncoderArgs(
    3,
    6,
    128,
    (1, 2, 4, 8),
)

decoder_args = VAE_DecoderArgs(
    3,
    6,
    128,
    (8, 4, 2, 1)
)

dataset = QuadrotorRaceSegmentDataset('data', ["square", "triangle", "pill"], 128, 0, NoNormalizer())
