import os
import sys
import argparse
import importlib

import torch

from quadrotor_diffusion.models.diffusion import DiffusionWrapper
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.nn.args import DiffusionWrapperArgs, Unet1DArgs, TrainerArgs
from quadrotor_diffusion.utils.logging import dataclass_to_table


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

print(dataclass_to_table(unet_args))
print(dataclass_to_table(diff_args))
print(dataclass_to_table(train_args))
print("\n" + "="*100 + "\n")


diff_model = DiffusionWrapper((diff_args, unet_args))
trainer = Trainer(train_args, diff_model, dataset)

trainer.test_forward_pass()
print("\n" + "="*100 + "\n")

trainer.train()
