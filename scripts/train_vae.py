import os
import sys
import argparse
import importlib

import torch

from quadrotor_diffusion.models.vae import VAE_Wrapper
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.nn.args import TrainerArgs, VAE_WrapperArgs, VAE_EncoderArgs, VAE_DecoderArgs
from quadrotor_diffusion.utils.logging import dataclass_to_table


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True, help="Name of config file in configs/ without the .py")
parser.add_argument('-d', '--debug', action='store_true', help="Turn on debug mode.")
args = parser.parse_args()

os.environ['DEBUG'] = 'True' if args.debug else 'False'

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
config_module = importlib.import_module(f'configs.{args.config}')

vae_args: VAE_WrapperArgs = config_module.vae_args
encoder_args: VAE_EncoderArgs = config_module.encoder_args
decoder_args: VAE_DecoderArgs = config_module.decoder_args
train_args: TrainerArgs = config_module.train_args
dataset: torch.utils.data.Dataset = config_module.dataset

print(dataclass_to_table(vae_args))
print(dataclass_to_table(encoder_args))
print(dataclass_to_table(decoder_args))
print(dataclass_to_table(train_args))
print("\n" + "="*100 + "\n")


vae_model = VAE_Wrapper((vae_args, encoder_args, decoder_args))
trainer = Trainer(train_args, vae_model, dataset)

print("\n" + "="*100 + "\n")

trainer.train()
