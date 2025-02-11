import os
import re
import sys
import argparse
import time

import numpy as np

from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.models.diffusion_wrapper import DiffusionWrapper
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.utils.plotting import plot_states,  plot_ref_obs_states
from quadrotor_diffusion.utils.simulator import play_trajectory, render_simulation
from quadrotor_diffusion.utils.file import get_checkpoint_file, get_sample_folder
from quadrotor_diffusion.utils.nn.post_process import fit_to_recon

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)

parser.add_argument('-s', '--samples', type=int, help='Number of trajectories to generate', required=True)
parser.add_argument('-o', '--horizon', type=int, help='Horizon', required=True)

parser.add_argument('-p', '--epoch', type=int, help='Epoch number, default is biggest', default=None)
parser.add_argument('-d', '--device', type=str, help='Device to use', default="cuda")
parser.add_argument('-m', '--no-ema', action='store_true', help="Use normal model instead of ema model.")
parser.add_argument('-t', '--time_it', action='store_true', help="Time the diffusion process")

args = parser.parse_args()
sys.argv = [sys.argv[0]]

chkpt = get_checkpoint_file("logs/training", args.experiment, epoch=args.epoch)
sample_dir = get_sample_folder("logs/training", args.experiment)
print(sample_dir)

with open(os.path.join(sample_dir, "overview.txt"), "w") as f:
    f.write(
        f"Horizon: {args.horizon}\n" +
        f"Num Samples: {args.samples}\n" +
        ("Not using EMA\n" if args.no_ema else "Using EMA\n") +
        f"Device: {args.device}\n"
    )

model: DiffusionWrapper = None
ema: DiffusionWrapper = None
normalizer: Normalizer = None
trainer_args: TrainerArgs = None

diff, ema, normalizer, trainer_args = Trainer.load(chkpt)
print(f"Loaded {chkpt}")
model = diff if args.no_ema else ema
print(f"Using {normalizer}")

model.to(args.device)
start = time.time()
trajectories = model.sample_unguided(args.samples, args.horizon, args.device)
print(f"{time.time() - start:.2f} seconds to generate {args.samples} samples with {args.device}")

for i in range(trajectories.size(0)):
    sampled = trajectories[i].cpu().numpy()

    pos, vel, acc = fit_to_recon(sampled, 30)

    plot_states(
        pos, vel, acc,
        f"Sample {i} / {trajectories.size(0)}",
        os.path.join(sample_dir, f"traj_{i}.pdf")
    )

    worked, states = play_trajectory(pos, vel, acc)
    if not worked:
        print("Failed")

    plot_ref_obs_states(
        pos, vel, acc,
        states[0], states[1], states[1],
        f"Sample {i} / {trajectories.size(0)}",
        os.path.join(sample_dir, f"traj_{i}_sim.pdf")
    )

    render_simulation(states, [], reference=pos, filename=os.path.join(sample_dir, f"sample_{i}.mp4"))

N = 1000
if args.time_it:
    start = time.time()
    for _ in range(N):
        trajectories = model.sample_unguided(1, args.horizon, args.device)
    end = time.time() - start
    print(f"{end / N:.2f} seconds on avg to generate 1 sample with {args.device}")
