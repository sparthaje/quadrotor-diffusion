import os
import sys
import argparse
import importlib

import torch

from quadrotor_diffusion.models.vae import VAE_Wrapper
from quadrotor_diffusion.models.contrastive_wrapper import ContrastiveWrapper
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.nn.args import TrainerArgs, CourseEmbeddingArgs, VAE_WrapperArgs, VAE_EncoderArgs, VAE_DecoderArgs
from quadrotor_diffusion.utils.logging import (
    iprint as print,
    dataclass_to_table,
    old_print,
)
from quadrotor_diffusion.utils.file import get_checkpoint_file, load_course_trajectory, get_experiment_folder


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True, help="Name of config file in configs/ without the .py")
parser.add_argument('-d', '--debug', action='store_true', help="Turn on debug mode.")
args = parser.parse_args()

os.environ['DEBUG'] = 'True' if args.debug else 'False'

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
config_module = importlib.import_module(f'configs.{args.config}')

course_embedding_args: CourseEmbeddingArgs = config_module.course_embedding_args
train_args: TrainerArgs = config_module.train_args
dataset: torch.utils.data.Dataset = config_module.dataset

# Load pre-trained trajectory encoder and decoder
vae_experiment: int = config_module.vae_experiment
chkpt = get_checkpoint_file("logs/training", vae_experiment)
vae_model, _, vae_normalizer, _ = Trainer.load(chkpt)

# Load relevant VAE args from model checkpoint
vae_args: VAE_WrapperArgs = vae_model.args[0]
vae_encoder_args: VAE_EncoderArgs = vae_model.args[1]
vae_decoder_args: VAE_DecoderArgs = vae_model.args[2]

# Second normalizer for dataset is for trajectories, so we override the normalizer to be vae normalizer
# Only calls the normalizer on the get_item call
dataset.normalizer.normalizer_b = vae_normalizer

print("Loaded vae model from checkpoint: " + chkpt)

old_print(dataclass_to_table(course_embedding_args))
print(dataclass_to_table(train_args))
old_print("\n" + "="*100 + "\n")


contrastive_wrapper = ContrastiveWrapper([course_embedding_args, vae_args, vae_encoder_args, vae_decoder_args])
contrastive_wrapper.trajectory_encoder = vae_model
trainer = Trainer(train_args, contrastive_wrapper, dataset)

old_print("\n" + "="*100 + "\n")

trainer.train()
