import os
import sys
import argparse
import importlib

import torch
import numpy as np
import matplotlib.pyplot as plt

from quadrotor_diffusion.models.diffusion_wrapper import LatentDiffusionWrapper
from quadrotor_diffusion.models.contrastive_wrapper import ContrastiveWrapper
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.nn.args import DiffusionWrapperArgs, Unet1DArgs, TrainerArgs
from quadrotor_diffusion.utils.logging import dataclass_to_table
from quadrotor_diffusion.utils.file import get_checkpoint_file
from quadrotor_diffusion.utils.plotting import create_course_grid

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True, help="Name of config file in configs/ without the .py")
parser.add_argument('-d', '--debug', action='store_true', help="Turn on debug mode.")
args = parser.parse_args()

os.environ['DEBUG'] = 'True' if args.debug else 'False'

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
config_module = importlib.import_module(f'configs.{args.config}')

unet_args: Unet1DArgs = config_module.unet_args
diff_args: DiffusionWrapperArgs = config_module.diff_args
train_args: TrainerArgs = config_module.train_args
dataset: torch.utils.data.Dataset = config_module.dataset

# Load pre-trained embeddings
embedding_experiment: int = config_module.embedding_experiment
chkpt = get_checkpoint_file("logs/training", embedding_experiment)
contrastive_wrapper, _, _, _ = Trainer.load(chkpt, get_ema=False)

print(dataclass_to_table(unet_args))
print(dataclass_to_table(diff_args))
print(dataclass_to_table(train_args))
print("\n" + "="*100 + "\n")

diff_model = LatentDiffusionWrapper((
    diff_args,
    unet_args,
    contrastive_wrapper.args[1],
    contrastive_wrapper.args[2],
    contrastive_wrapper.args[3],
    contrastive_wrapper.args[0],
))
diff_model.encoders = contrastive_wrapper
trainer = Trainer(train_args, diff_model, dataset)

trainer.test_forward_pass()
print("\n" + "="*100 + "\n")

N_epochs = train_args.max_epochs

trainer.args.max_epochs = 0
while trainer.epoch < N_epochs:
    trainer.args.max_epochs += train_args.evaluate_every
    trainer.train()

    sample_trajectories = diff_model.sample(batch_size=10, horizon=dataset[0].shape[0], device=train_args.device)
    fig, axes = create_course_grid(sample_trajectories)

    save_dir = os.path.join(trainer.args.log_dir, "samples", "training")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{trainer.epoch}.pdf"))
    plt.close(fig)
