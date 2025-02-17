# contrastive_cfg

import numpy as np

from quadrotor_diffusion.utils.nn.args import TrainerArgs, VAE_EncoderArgs, VAE_DecoderArgs, VAE_WrapperArgs, CourseEmbeddingArgs
from quadrotor_diffusion.utils.dataset.normalizer import NormalizerTuple, NoNormalizer
from quadrotor_diffusion.utils.dataset.dataset import ContrastiveEmbeddingDataset

train_args = TrainerArgs(
    ema_decay=0,
    num_batches_no_ema=float('inf'),
    num_batches_per_ema=float('inf'),

    batch_size_per_gpu=128,
    batches_per_backward=1,

    log_dir="logs/training",
    save_freq=5,

    learning_rate=2e-3,
    num_gpus=1,
    device="cuda:0",

    max_epochs=500,
)

course_embedding_args = CourseEmbeddingArgs(
    hidden_dim=64,
    n_layers=3,
    embed_dim=128,
    gate_input_dim=4,
    vae_padding=0,
)

vae_experiment = 102

normalizer = NormalizerTuple(
    normalizer_a=NoNormalizer(),
    normalizer_b=NoNormalizer(),
)

dataset = ContrastiveEmbeddingDataset("data", ["linear", "u"], 384, normalizer)
