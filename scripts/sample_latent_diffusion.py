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
from quadrotor_diffusion.utils.nn.post_process import fit_to_recon

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)

parser.add_argument('-s', '--samples', type=int, help='Number of trajectories to generate', required=True)
parser.add_argument('-c', '--course', type=str, help='Sample (course_type,course_number)', required=True)

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

vae_experiment: int = 157
chkpt = get_checkpoint_file("logs/training", vae_experiment)
vae_wrapper: VAE_Wrapper = None
vae_wrapper, _, _, _ = Trainer.load(chkpt, get_ema=False)
vae_wrapper.to(args.device)
vae_downsample = 2 ** (len(vae_wrapper.args[1].channel_mults) - 1)

model = diff if args.no_ema else ema
model.decoder = vae_wrapper.decode
model.to(args.device)

# Generate first segment from starting point to gate 1


def score(trajectories: torch.Tensor, initial_position: np.array) -> int:
    initial_position = torch.tensor(initial_position, dtype=torch.float32, device="cpu")
    return torch.argmin(torch.norm(trajectories[:, 0, :] - initial_position, dim=-1)).item()


def score(trajectories: torch.Tensor, initial_position: np.array) -> int:
    """
    Returns the index of a trajectory whose initial position is within 0.03 of the given initial_position and
    which has the largest distance between its first and last points. Falls back to the trajectory with the minimal
    initial distance if no trajectory meets the threshold.
    """
    initial_position = torch.tensor(initial_position, dtype=torch.float32, device="cpu")
    dists0 = torch.norm(trajectories[:, 0, :] - initial_position, dim=-1)
    filtered_idxs = torch.nonzero(dists0 < 0.05, as_tuple=True)[0]
    if len(filtered_idxs) == 0:
        return torch.argmin(dists0).item()
    diff = torch.norm(trajectories[filtered_idxs, 0, :] - trajectories[filtered_idxs, -1, :], dim=-1)
    return filtered_idxs[torch.argmax(diff)].item()


local_conditioning = torch.tensor(np.tile(course[0], (6, 1)), dtype=torch.float32).to(args.device)
local_conditioning = local_conditioning.unsqueeze(0).expand((args.samples, -1, -1))

global_conditioning = np.vstack((course[1:], course[1:0]))
null_tokens = np.tile(np.array(5 * np.ones((1, 4))), (6 - len(global_conditioning), 1))
global_conditioning = np.vstack((global_conditioning, null_tokens))
global_conditioning = torch.tensor(global_conditioning, dtype=torch.float32).to(args.device)

# print("Warming")
# torch.backends.cudnn.benchmark = True
# for i in range(25):
#     trajectories = model.sample(args.samples, 128, vae_downsample, args.device,
#                                 local_conditioning=local_conditioning, global_conditioning=global_conditioning)
# print("Finished warming")

start = time.time()
trajectories = model.sample(args.samples, 128, vae_downsample, args.device,
                            local_conditioning=local_conditioning, global_conditioning=global_conditioning).cpu()

segments = []

segment_0 = trajectories[score(trajectories, course[0][:3])].numpy()
ending_idx = np.argmin(np.linalg.norm(segment_0 - course[1][:3], axis=1))
segment_0 = segment_0[:ending_idx]
segments.append(segment_0)

for i in range(len(course) - 1):
    gate_idx = i + 1
    local_conditioning = np.hstack((segments[-1][-6:], np.zeros((6, 1))))
    local_conditioning = torch.tensor(local_conditioning, dtype=torch.float32).to(args.device)
    local_conditioning = local_conditioning.unsqueeze(0).expand((args.samples, -1, -1))

    global_conditioning = np.vstack((course[1+gate_idx:], course[1:gate_idx]))
    null_tokens = np.tile(np.array(5 * np.ones((1, 4))), (6 - len(global_conditioning), 1))
    global_conditioning = np.vstack((global_conditioning, null_tokens))
    global_conditioning = torch.tensor(global_conditioning, dtype=torch.float32).to(args.device)

    trajectories = model.sample(args.samples, 128, vae_downsample, args.device,
                                local_conditioning=local_conditioning, global_conditioning=global_conditioning).cpu()

    trajectory = trajectories[score(trajectories, segments[-1][-1])].numpy()
    next_gate = gate_idx + 1 if gate_idx + 1 < len(course) else 1
    ending_idx = np.argmin(np.linalg.norm(trajectory - course[next_gate][:3], axis=1))
    trajectory = trajectory[:ending_idx]
    segments.append(trajectory)

segments.append(segments[1])
trajectory = np.vstack(segments)
pos, vel, acc = fit_to_recon(trajectory, 30)

print(f"Finished sampling in {time.time() - start:.4f} seconds")

plot_states(
    pos, vel, acc,
    f"Sample {i} / {trajectories.size(0)}",
    os.path.join(sample_dir, f"sample.pdf")
)

_, ax = course_base_plot()
add_gates_to_course(course, ax, has_end=False)
add_trajectory_to_course(pos, velocity_profile=vel)
plt.savefig(os.path.join(sample_dir, f"bev.pdf"))
plt.close()

worked, states = play_trajectory(pos, vel, acc)
if not worked:
    print("Failed")

plot_ref_obs_states(
    pos, vel, acc,
    states[0], states[1], states[1],
    f"Sample {i} / {trajectories.size(0)}",
    os.path.join(sample_dir, f"sim.pdf")
)

create_perspective_rendering(states, np.vstack([course, course[-1]]), reference=pos,
                             filename=os.path.join(sample_dir, f"sample.mp4"))
