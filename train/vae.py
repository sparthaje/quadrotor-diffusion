import os
import sys
import argparse
import importlib
import random

import torch
import matplotlib.pyplot as plt
import wandb

from quadrotor_diffusion.models.vae_wrapper import VAE_Wrapper
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.nn.args import TrainerArgs, VAE_WrapperArgs, VAE_EncoderArgs, VAE_DecoderArgs
from quadrotor_diffusion.utils.quad_logging import dataclass_to_table


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True, help="Name of config file in configs/ without the .py")
parser.add_argument('-d', '--debug', action='store_true', help="Turn on debug mode.")
args = parser.parse_args()

wandb.login()

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

random.seed(42)
evaluation_set = [random.randint(0, len(dataset) - 1) for _ in range(10)]
random.seed(int.from_bytes(os.urandom(4), "little"))

N_epochs = trainer.args.max_epochs
trainer.args.max_epochs = 0
while trainer.epoch < N_epochs:
    trainer.args.max_epochs += trainer.args.evaluate_every
    trainer.train()

    # Step 1: Evaluate model on random samples from dataset.
    fig = plt.figure(figsize=(18, 16))

    for row in range(10):
        sample = dataset[evaluation_set[row]]
        inp = sample.float().unsqueeze(0).to(trainer.args.device)
        mu, logvar = vae_model.encode(inp)
        reconstructed = vae_model.decode(mu).squeeze(0).cpu().numpy()

        reference = dataset.normalizer.undo(sample.cpu().numpy())
        reconstructed = dataset.normalizer.undo(reconstructed)

        plt.subplot(10, 3, row * 3 + 1)
        plt.plot(reference[:, 0], label='Reference', linewidth=3.5)
        plt.plot(reconstructed[:, 0], label='Reconstructed', linewidth=1.5)
        plt.plot(reference[:, 0] - reconstructed[:, 0], label='Error', linewidth=1.5)
        plt.ylabel("x (meters)")
        plt.grid()

        plt.subplot(10, 3, row * 3 + 2)
        plt.plot(reference[:, 1], label='Reference', linewidth=3.5)
        plt.plot(reconstructed[:, 1], label='Reconstructed', linewidth=1.5)
        plt.plot(reference[:, 1] - reconstructed[:, 1], label='Error', linewidth=1.5)
        plt.ylabel("y (meters)")
        plt.grid()

        plt.subplot(10, 3, row * 3 + 3)
        plt.plot(reference[:, 2], label='Reference', linewidth=3.5)
        plt.plot(reconstructed[:, 2], label='Reconstructed', linewidth=1.5)
        plt.plot(reference[:, 2] - reconstructed[:, 2], label='Error', linewidth=1.5)
        plt.ylabel("z (meters)")
        plt.grid()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig.legend(['Reference', 'Reconstructed', 'Error'],
               loc='upper center',
               bbox_to_anchor=(0.5, 0.98),
               ncol=3,
               frameon=False)

    save_dir = os.path.join(trainer.args.log_dir, "samples", "training")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{trainer.epoch}.pdf"))
    plt.close()
