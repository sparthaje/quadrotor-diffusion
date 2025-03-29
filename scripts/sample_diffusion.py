import os
import re
import sys
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.models.diffusion_wrapper import DiffusionWrapper
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.quad_logging import iprint as print
from quadrotor_diffusion.utils.plotting import (
    plot_states,
    plot_ref_obs_states,
    course_base_plot,
    add_gates_to_course,
    add_trajectory_to_course
)
from quadrotor_diffusion.utils.simulator import play_trajectory, create_perspective_rendering
from quadrotor_diffusion.utils.file import get_checkpoint_file, get_sample_folder
from quadrotor_diffusion.utils.nn.post_process import fit_to_recon
from quadrotor_diffusion.utils.file import load_course_trajectory
from quadrotor_diffusion.guide_functions.baseline import compute_trajectory_alignment

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)

parser.add_argument('-s', '--samples', type=int, help='Number of trajectories to generate', required=True)
parser.add_argument('-c', '--course', type=str, help='Sample (course_type,course_number)', default="")
parser.add_argument('-o', '--horizon', type=int, help='Horizon', required=True)

parser.add_argument('-p', '--epoch', type=int, help='Epoch number, default is biggest', default=None)
parser.add_argument('-d', '--device', type=str, help='Device to use', default="cuda")
parser.add_argument('-m', '--no-ema', action='store_true', help="Use normal model instead of ema model.")
parser.add_argument('-t', '--time_it', action='store_true', help="Time the diffusion process")
parser.add_argument('-r', '--test_adj', action='store_true', help="Test adjusting one gate")

args = parser.parse_args()
sys.argv = [sys.argv[0]]

chkpt = get_checkpoint_file("logs/training", args.experiment, epoch=args.epoch)
print(chkpt)
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

guide_function = None
course_npy = []
if args.course != "":
    sample_info = args.course.split(",")
    course_npy, _, _ = load_course_trajectory(sample_info[0], sample_info[1], 0)
    course = torch.tensor(course_npy, dtype=torch.float32).to(args.device)
    course = course.expand(args.samples, -1, -1)
    s = 2.4
    def guide_function(trajectory): return s * compute_trajectory_alignment(trajectory, course)

diff, ema, normalizer, trainer_args = Trainer.load(chkpt)
print(f"Loaded {chkpt}")
model = diff if args.no_ema else ema
print(f"Using {normalizer}")

model.to(args.device)
start = time.time()
trajectories = model.sample(args.samples, args.horizon, args.device, guide=guide_function)
print(f"{time.time() - start:.2f} seconds to generate {args.samples} samples with {args.device}")

if args.test_adj:
    course_npy[2][1] += 0.4
    course[0][2][1] += 0.4
    def guide_function(trajectory): return s * compute_trajectory_alignment(trajectory, course)
    trajectories_adjusted = model.noise_and_resample(trajectories, 500, guide_function)
    trajectories, trajectories_old = trajectories_adjusted, trajectories


for i in range(trajectories.size(0)):
    sampled = trajectories[i].cpu().numpy()

    pos, vel, acc = fit_to_recon(sampled, 30)
    if args.test_adj:
        pos_old, _, _ = fit_to_recon(trajectories_old[i].cpu().numpy(), 30)

    plot_states(
        pos, vel, acc,
        f"Sample {i} / {trajectories.size(0)}",
        os.path.join(sample_dir, f"sample_{i}.pdf")
    )

    _, ax = course_base_plot()
    if guide_function is not None:
        add_gates_to_course(course_npy, ax)
    add_trajectory_to_course(pos, velocity_profile=vel)
    plt.savefig(os.path.join(sample_dir, f"sample_{i}_bev.pdf"))
    plt.close()

    if args.test_adj:
        _, ax = course_base_plot()
        if guide_function is not None:
            add_gates_to_course(course_npy, ax)
        add_trajectory_to_course(pos_old, velocity_profile=vel)
        plt.savefig(os.path.join(sample_dir, f"sample_no_adj_{i}_bev.pdf"))
        plt.close()

    worked, states = play_trajectory(pos, vel, acc)
    if not worked:
        print("Failed")

    plot_ref_obs_states(
        pos, vel, acc,
        states[0], states[1], states[1],
        f"Sample {i} / {trajectories.size(0)}",
        os.path.join(sample_dir, f"sample_{i}_sim.pdf")
    )

    create_perspective_rendering(states, course_npy, reference=pos,
                                 filename=os.path.join(sample_dir, f"sample_{i}.mp4"))

N = 1000
if args.time_it:
    start = time.time()
    for _ in range(N):
        trajectories = model.sample(1, args.horizon, args.device)
    end = time.time() - start
    print(f"{end / N:.2f} seconds on avg to generate 1 sample with {args.device}")
