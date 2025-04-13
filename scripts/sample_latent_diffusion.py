import os
import re
import sys
import argparse
import time

import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.models.diffusion_wrapper import LatentDiffusionWrapper
from quadrotor_diffusion.models.vae_wrapper import VAE_Wrapper
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.quad_logging import iprint as print
from quadrotor_diffusion.utils.plotting import plot_states,  plot_ref_obs_states, add_gates_to_course, add_trajectory_to_course, course_base_plot
from quadrotor_diffusion.utils.simulator import play_trajectory, create_perspective_rendering
from quadrotor_diffusion.utils.file import get_checkpoint_file, get_sample_folder, load_course_trajectory
from quadrotor_diffusion.planner import plan, cudnn_benchmark, SamplerType, ScoringMethod

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)

parser.add_argument('-s', '--samples', type=int, help='Number of trajectories to generate', required=True)
parser.add_argument('-c', '--course', type=str, help='Sample (course_type,course_number)', required=True)
parser.add_argument('-l', '--laps', type=int, help='How many laps to do', required=True)

parser.add_argument('-p', '--epoch', type=int, help='Epoch number, default is biggest', default=None)
parser.add_argument('-d', '--device', type=str, help='Device to use', default="cuda")
parser.add_argument('-m', '--no_ema', action='store_true', help="Use normal model instead of ema model.")

args = parser.parse_args()
sys.argv = [sys.argv[0]]

chkpt = get_checkpoint_file("logs/training", args.experiment, epoch=args.epoch)
print(chkpt)
sample_dir = get_sample_folder("logs/training", args.experiment)
print(sample_dir)

with open(os.path.join(sample_dir, "overview.txt"), "w") as f:
    f.write(
        f"Num Samples: {args.samples}\n" +
        ("Not using EMA\n" if args.no_ema else "Using EMA\n") +
        f"Device: {args.device}\n"
    )

course = []
if args.course.endswith(".npy"):
    course = np.load(args.course)
else:
    sample_info = args.course.split(",")
    course, _, _ = load_course_trajectory(sample_info[0], sample_info[1], 0)

model: LatentDiffusionWrapper = None
ema: LatentDiffusionWrapper = None
normalizer: Normalizer = None
trainer_args: TrainerArgs = None

diff, ema, normalizer, trainer_args = Trainer.load(chkpt)
print(f"Loaded {chkpt}")
print(f"Using {normalizer}")

vae_experiment: int = 174
chkpt = get_checkpoint_file("logs/training", vae_experiment)
vae_wrapper: VAE_Wrapper = None
vae_wrapper, _, _, _ = Trainer.load(chkpt, get_ema=False)
vae_wrapper.to(args.device)
vae_downsample = 2 ** (len(vae_wrapper.args[1].channel_mults) - 1)

model = diff if args.no_ema else ema
model.decoder = vae_wrapper.decode
model.to(args.device)

cudnn_benchmark(args.samples, model, vae_downsample, args.device)

current_traj = None

gate_idx = 0

computation_times = []
trajectory_times = []
trajectories = []

gates_per_lap = course.shape[0] - 1
iterations = gates_per_lap * args.laps + 1

for i in range(iterations):
    global_context = course
    if current_traj is not None:
        global_context = np.vstack((course[gate_idx + 1:], course[1:gate_idx]))

    s = time.time()
    next_traj, candidates = plan(
        args.samples,
        global_context,
        SamplerType.DDPM,
        ScoringMethod.FAST,
        model,
        vae_downsample,
        "cuda",
        current_traj=current_traj
    )
    trajectories.append(next_traj)

    computation_times.append(
        time.time() - s
    )
    trajectory_times.append(
        next_traj[0].shape[0] / 30
    )

    _, axs = course_base_plot()
    add_gates_to_course(course, axs, has_end=False)
    if current_traj:
        add_trajectory_to_course(axs, current_traj[0])
    for traj in candidates:
        add_trajectory_to_course(axs, traj.cpu().numpy(), reference=True)
    plt.savefig(
        os.path.join(sample_dir, f"iteration_{i}_candidates.pdf")
    )
    plt.close()

    _, axs = course_base_plot()
    add_gates_to_course(course, axs, has_end=False)
    add_trajectory_to_course(axs, next_traj[0])
    plt.savefig(
        os.path.join(sample_dir, f"iteration_{i}_taken_path.pdf")
    )
    plt.close()

    gate_idx += 1
    current_traj = next_traj

    # Looped back around to previous gate
    if gate_idx == len(course):
        gate_idx = 1

reference = [np.vstack([t[idx] for t in trajectories]) for idx in range(3)]
ref_pos = reference[0]
ref_vel = reference[1]
ref_acc = reference[2]

plot_states(
    ref_pos, ref_vel, ref_acc,
    f"",
    os.path.join(sample_dir, f"sample.pdf")
)

worked, states = play_trajectory(ref_pos, ref_vel, ref_acc)
if not worked:
    print("Failed")

plot_ref_obs_states(
    ref_pos, ref_vel, ref_acc,
    states[0], states[1], states[1],
    f"",
    os.path.join(sample_dir, f"sim.pdf")
)

create_perspective_rendering(states, np.vstack([course, course[-1]]), reference=ref_pos,
                             filename=os.path.join(sample_dir, f"sample.mp4"))

for idx, (t_t, c_t) in enumerate(zip(trajectory_times, computation_times[1:])):
    print(f"Plan {idx}:")
    print(f"\t{t_t:.2f} seconds on previous trajectory")
    print(f"\t{c_t:.2f} seconds to compute next trajectory")
    if c_t > t_t:
        print("\tFailed computation before next segment")
    else:
        print("\tComputation succeeded")
    print("=======================================")
