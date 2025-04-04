import os
import sys
import argparse
import importlib

import torch
import numpy as np
import matplotlib.pyplot as plt

from quadrotor_diffusion.models.diffusion_wrapper import LatentDiffusionWrapper
from quadrotor_diffusion.models.vae_wrapper import VAE_Wrapper
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.nn.args import LatentDiffusionWrapperArgs, Unet1DArgs, TrainerArgs
from quadrotor_diffusion.utils.quad_logging import dataclass_to_table
from quadrotor_diffusion.utils.file import get_checkpoint_file
from quadrotor_diffusion.utils.plotting import create_course_grid

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

print(dataclass_to_table(unet_args))
print(dataclass_to_table(diff_args))
print(dataclass_to_table(train_args))
print("\n" + "="*100 + "\n")

diff_model = LatentDiffusionWrapper((
    diff_args,
    unet_args,
))
diff_model.encoder = vae_wrapper.encode
diff_model.decoder = vae_wrapper.decode
trainer = Trainer(train_args, diff_model, dataset)

trainer.test_forward_pass()
print("\n" + "="*100 + "\n")

N_epochs = train_args.max_epochs

trainer.args.max_epochs = 0
while trainer.epoch < N_epochs:
    trainer.args.max_epochs += train_args.evaluate_every
    trainer.train()

    vae_downsample = 2 ** (len(vae_wrapper.args[1].channel_mults) - 1)
    sampling_model: LatentDiffusionWrapper = trainer.ema_model if trainer.ema_model is not None else diff_model

    idxs = np.random.choice(len(dataset), 5, replace=False)
    slices = []
    for i in idxs:
        c = dataset[i]["local_conditioning"].unsqueeze(0)
        slices.append(c)
        slices.append(c)

    local_conditioning = torch.concat(slices).to(train_args.device)
    global_conditioning = sampling_model.null_token_global.expand((10, 4, -1))

    # [10, n, 3]
    horizon_padding = 0 if vae_wrapper.args[0].telomere_strategy == 0 else vae_downsample * 2
    sample_trajectories = sampling_model.sample(
        batch_size=10,
        horizon=dataset[0]["x_0"].shape[0] + horizon_padding,
        vae_downsample=vae_downsample,
        device=train_args.device,
        local_conditioning=local_conditioning,
        global_conditioning=global_conditioning,
    )

    # Cut the local conditioning to only be the prior states
    local_conditioning = local_conditioning[:, :-local_conditioning.shape[1] // 2, :]
    sample_trajectories = torch.concat((local_conditioning, sample_trajectories), dim=1)

    fig, axes = create_course_grid(sample_trajectories)

    save_dir = os.path.join(trainer.args.log_dir, "samples", "training")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{trainer.epoch}.pdf"))
    plt.close(fig)
