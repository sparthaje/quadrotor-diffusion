# classifier_cfg

import torch

from quadrotor_diffusion.utils.nn.args import Unet1DArgs, TrainerArgs
from quadrotor_diffusion.utils.dataset.normalizer import NoNormalizer, NormalizerTuple
from quadrotor_diffusion.utils.dataset.dataset import CourseTrajectoryCrossEmbedding
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.file import get_checkpoint_file
from quadrotor_diffusion.models.vae_wrapper import VAE_Wrapper


train_args = TrainerArgs(
    ema_decay=0,
    num_batches_no_ema=float('inf'),
    num_batches_per_ema=float('inf'),

    batch_size_per_gpu=1,
    batches_per_backward=1,

    log_dir="logs/training/",
    save_freq=1,

    learning_rate=2e-4,
    num_gpus=1,
    device="cuda:0",
    max_epochs=200,
    evaluate_every=-1
)

VAE_experiment = 102
chkpt = get_checkpoint_file("logs/training", VAE_experiment)
vae_wrapper: VAE_Wrapper = None
vae_wrapper, _, _, _ = Trainer.load(chkpt, get_ema=False)
vae_wrapper = vae_wrapper.to(train_args.device)

unet_args = Unet1DArgs(
    traj_dim=vae_wrapper.args[1].latent_dim,
    features=64,
    channel_mults=[1, 2, 4],
    attentions=[
        [False, False, False],
        [False],
        [False, False]
    ],
    context_mlp="waypoints",
)

dataset = CourseTrajectoryCrossEmbedding(
    data_dir="data",
    course_types=["linear", "u"],
    traj_len=384,
    mini_batch_size=50,
    normalizer=NormalizerTuple(NoNormalizer(), NoNormalizer()),
)
