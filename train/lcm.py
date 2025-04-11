import os
import copy
import time
import sys
import argparse
import importlib
from datetime import datetime
from typing import Tuple
import warnings
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm
import wandb

import quadrotor_diffusion.utils.nn.ema as ema
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.nn.args import LatentDiffusionWrapperArgs, Unet1DArgs, TrainerArgs
from quadrotor_diffusion.utils.file import get_checkpoint_file
from quadrotor_diffusion.models.vae_wrapper import VAE_Wrapper
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.quad_logging import (
    dataclass_to_table,
    iprint as print
)

# TODO(shreepa): Probably should fix this at some point
# Suppress FutureWarning for this specific issue
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only=False.*")

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True, help="Name of config file in configs/ without the .py")
parser.add_argument('-d', '--debug', action='store_true', help="Turn on debug mode.")
args = parser.parse_args()

os.environ['DEBUG'] = 'True' if args.debug else 'False'

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
cfg_name = args.config if "_cfg" in args.config else args.config + "_cfg"
config_module = importlib.import_module(f'configs.{cfg_name}')

unet_args: Unet1DArgs = config_module.unet_args
diff_args: LatentDiffusionWrapperArgs = config_module.diff_args
train_args: TrainerArgs = config_module.train_args
dataset: torch.utils.data.Dataset = config_module.dataset

# Load pre-trained embeddings
vae_experiment: int = config_module.vae_experiment
chkpt = get_checkpoint_file("logs/training", vae_experiment)
vae_wrapper: VAE_Wrapper = None
vae_wrapper, _, _, _ = Trainer.load(chkpt, get_ema=False)
vae_wrapper.to(train_args.device)

ldm_experiment: int = config_module.ldm_experiment
chkpt = get_checkpoint_file("logs/training", ldm_experiment)
vae_wrapper: VAE_Wrapper = None
_, ldm_wrapper, _, _ = Trainer.load(chkpt, get_ema=True)
ldm_wrapper.to(train_args.device)
