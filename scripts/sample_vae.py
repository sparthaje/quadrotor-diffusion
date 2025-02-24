import time
import subprocess
import os
from random import randint
import argparse
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

from quadrotor_diffusion.models.vae_wrapper import VAE_Wrapper
from quadrotor_diffusion.utils.nn.post_process import fit_to_recon
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.file import get_checkpoint_file, load_course_trajectory, get_sample_folder
from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.utils.simulator import play_trajectory, render_simulation, SimulatorViewingAngle
from quadrotor_diffusion.utils.plotting import course_base_plot, add_gates_to_course, add_trajectory_to_course, plot_states, plot_reference_time_series
from quadrotor_diffusion.utils.trajectory import derive_trajectory
from quadrotor_diffusion.utils.voxels import create_occupancy_map, collision_along_trajectory


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)
parser.add_argument('-s', '--sample', type=str, help='Sample (course_type,course_number,sample_number)', required=True)
parser.add_argument('-r', '--no-fit', action='store_true', help="Replay raw reconstructed data without post-processing")
parser.add_argument('-a', '--padding', type=int, default=0, help="Padding to use for encoding / decoding")
parser.add_argument('-t', '--time_it', action='store_true', help="Time the diffusion process")

parser.add_argument('-p', '--epoch', type=int, help='Epoch number, default is biggest', default=None)
parser.add_argument('-d', '--device', type=str, help='Device to use', default="cuda")
parser.add_argument('--ema', action='store_true', help="Use ema model.")

args = parser.parse_args()

model: VAE_Wrapper = None
ema: VAE_Wrapper = None
normalizer: Normalizer = None
trainer_args: TrainerArgs = None

chkpt = get_checkpoint_file("logs/training", args.experiment)
model, ema, normalizer, trainer_args = Trainer.load(chkpt)
print("Loaded", chkpt)

eval = ema if args.ema else model
eval = eval.to(args.device)

sample_info = args.sample.split(",")
course, trajectory, filename = load_course_trajectory(*sample_info)
print(f"Loaded trajectory from {filename}")

sample_dir = get_sample_folder("logs/training", args.experiment)

ref_pos = trajectory.as_ref_pos()
VOXEL_SIZE = 0.01
occupancy_map = create_occupancy_map(course, voxel_size=VOXEL_SIZE)

sample_works, drone_states_ref = play_trajectory(ref_pos)
print(f"Finished simulation on sample data {'succesfully' if sample_works else 'unsuccesfully'}")
reference_filename = os.path.join(sample_dir, "reference.mp4")
render_simulation(drone_states_ref, course, ref_pos, filename=reference_filename)
collision = collision_along_trajectory(drone_states_ref[0], occupancy_map, VOXEL_SIZE)
print(f"On sample data had {'no' if not collision else 'a'} collision")

ref_pos_end_padded = trajectory.as_ref_pos(pad_to=360)
normalized_reference = normalizer(ref_pos_end_padded)
inp = torch.tensor(normalized_reference).float()
inp = inp.unsqueeze(0)
inp = inp.to(args.device)
mu, logvar = eval.encode(inp, padding=args.padding)
model_out = eval.decode(mu, padding=args.padding).squeeze(0).cpu().numpy()
reconstructed = normalizer.undo(model_out)
reconstructed = reconstructed[:ref_pos.shape[0] - ref_pos_end_padded.shape[0], :]

fitted = (
    reconstructed,
    derive_trajectory(reconstructed, 30),
    derive_trajectory(reconstructed, 30, 3)
)
plot_states(fitted[0], fitted[1], fitted[2], filename=os.path.join(sample_dir, "reconstructed.pdf"))
plot_reference_time_series(
    os.path.join(sample_dir, "recon_vs_sample.pdf"),
    "",
    ref_pos,
    reconstructed,
)

if not args.no_fit:
    fitted = fit_to_recon(reconstructed, 30)

recon_works, drone_states_recon = play_trajectory(ref_pos=fitted[0], ref_vel=fitted[1], ref_acc=fitted[2])
print(f"Finished simulation on reconstructed data {'succesfully' if recon_works else 'unsuccesfully'}")


bev = os.path.join(sample_dir, "bev_reconstructed.mp4")
perspective = os.path.join(sample_dir, "perspective_reconstructed.mp4")
yz = os.path.join(sample_dir, "yz_reconstructed.mp4")
combined = os.path.join(sample_dir, "reconstructed.mp4")

render_simulation(drone_states_recon, course, reconstructed, SimulatorViewingAngle.BEV, filename=bev)
render_simulation(drone_states_recon, course, reconstructed, SimulatorViewingAngle.YZ, filename=yz)

subprocess.run(["ffmpeg", "-i", bev, "-i", yz,
               "-filter_complex", "hstack=inputs=2", combined], check=True)

os.remove(bev)
os.remove(yz)

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

N = 1000
if args.time_it:
    start = time.time()
    for _ in range(N):
        trajectory = torch.randn(
            (1, 384 // (2**(len(model.args[1].channel_mults) - 1)), model.args[1].latent_dim)).to(args.device)
        _ = model.decode(trajectory)
    end = time.time() - start
    print(f"{end / N:.4f} seconds on avg to decode 1 sample with {args.device}")
