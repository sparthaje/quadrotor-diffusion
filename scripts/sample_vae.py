import sys
import time
import subprocess
import os
from random import randint
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from quadrotor_diffusion.models.vae_wrapper import VAE_Wrapper
from quadrotor_diffusion.utils.nn.post_process import fit_to_recon
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.dataset.boundary_condition import PolynomialTrajectory
from quadrotor_diffusion.utils.file import get_checkpoint_file, load_course_trajectory, get_sample_folder
from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.utils.simulator import play_trajectory, create_perspective_rendering
from quadrotor_diffusion.utils.plotting import course_base_plot, add_gates_to_course, add_trajectory_to_course, plot_states, plot_reference_time_series
from quadrotor_diffusion.utils.trajectory import derive_trajectory
from quadrotor_diffusion.utils.voxels import create_occupancy_map, collision_along_trajectory


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)
parser.add_argument('-s', '--sample', type=str, help='Sample (course_type,course_number,sample_number)', required=True)

parser.add_argument('-p', '--epoch', type=int, help='Epoch number, default is biggest', default=None)
parser.add_argument('-d', '--device', type=str, help='Device to use', default="cuda")

args = parser.parse_args()

model: VAE_Wrapper = None
ema: VAE_Wrapper = None
normalizer: Normalizer = None
trainer_args: TrainerArgs = None

chkpt = get_checkpoint_file("logs/training", args.experiment)
eval, _, normalizer, trainer_args = Trainer.load(chkpt, get_ema=False)
eval = eval.to(args.device)
print("Loaded", chkpt)


sample_info = args.sample.split(",")
course: np.ndarray = None
trajectory: PolynomialTrajectory = None
course, trajectory, filename = load_course_trajectory(*sample_info)
print(f"Loaded trajectory from {filename}")

VOXEL_SIZE = 0.01
course = np.vstack((course, course[-1]))
occupancy_map = create_occupancy_map(course, voxel_size=VOXEL_SIZE)

trajectory.states.append(trajectory.states[2])
trajectory.segment_lengths.append(trajectory.segment_lengths[1])
trajectory.states.append(trajectory.states[3])
trajectory.segment_lengths.append(trajectory.segment_lengths[2])
ref_pos = trajectory.as_ref_pos()


sample_dir = get_sample_folder("logs/training", args.experiment)

gate_idx = 0
vae_segments = []
start_idx = 0
for idx, segment_length in enumerate(trajectory.segment_lengths):
    starting_gate = trajectory.states[idx]
    end_state = trajectory.states[idx+1]
    new_gate_idx = int(30 * segment_length)

    rp_slice = ref_pos[max(0, gate_idx - 5):gate_idx+112]
    dist_to_start_gate = np.linalg.norm(
        rp_slice - np.array([starting_gate.x.s, starting_gate.y.s, starting_gate.z.s]), axis=1)

    vae_segments.append((start_idx, start_idx + 112, end_state))

    ending_idx = np.argmin(
        np.linalg.norm(
            ref_pos[start_idx:start_idx+112] - np.array([end_state.x.s, end_state.y.s, end_state.z.s]),
            axis=1,
        )
    )
    start_idx = start_idx + ending_idx

    gate_idx += new_gate_idx

raw_inps = []
raw_recons = []
next_gate_idx = []

for start_idx, end_idx, end_state in vae_segments[:-1]:
    inp = ref_pos[start_idx:end_idx]
    inp = normalizer(inp)
    inp = torch.tensor(inp).float().unsqueeze(0).to(args.device)

    mu, _ = eval.encode(inp)
    x_recon = eval.decode(mu).squeeze(0).cpu().numpy()
    x_recon = normalizer.undo(x_recon)

    inp = ref_pos[start_idx:end_idx]
    raw_inps.append(inp)
    raw_recons.append(x_recon)

    ending_idx = np.argmin(
        np.linalg.norm(
            inp - np.array([end_state.x.s, end_state.y.s, end_state.z.s]),
            axis=1,
        )
    )

    next_gate_idx.append((ending_idx, end_state))

recons_trimmed = []
for raw_recon, (end_idx, _) in zip(raw_recons, next_gate_idx):
    recon_trimmed = raw_recon[:end_idx]
    recons_trimmed.append(
        recon_trimmed
    )

fitted = fit_to_recon(np.vstack(recons_trimmed), 30)
plot_states(fitted[0], fitted[1], fitted[2], filename=os.path.join(sample_dir, "reconstructed.pdf"))

plot_reference_time_series(
    os.path.join(sample_dir, "recon_vs_sample.pdf"),
    "",
    ref_pos,
    fitted[0],
)

recon_sim_filename = os.path.join(sample_dir, "reconstructed.mp4")
recon_works, drone_states_recon = play_trajectory(ref_pos=fitted[0], ref_vel=fitted[1], ref_acc=fitted[2])
print(f"Finished simulation on reconstructed data {'succesfully' if recon_works else 'unsuccesfully'}")
create_perspective_rendering(drone_states_recon, course, recon_sim_filename, ref_pos)
collision = collision_along_trajectory(drone_states_recon[0], occupancy_map, VOXEL_SIZE)
print(f"On reconstructed data had {'no' if not collision else 'a'} collision")

plot_reference_time_series(
    os.path.join(sample_dir, "recon_vs_sim.pdf"),
    "",
    fitted[0],
    drone_states_recon[0],
)

_, ax = course_base_plot()
add_gates_to_course(course, ax)
add_trajectory_to_course(fitted[0], velocity_profile=fitted[1])
add_trajectory_to_course(ref_pos, reference=True)
trajectory_fig_filename = os.path.join(sample_dir, "trajectories.pdf")
plt.savefig(trajectory_fig_filename)
