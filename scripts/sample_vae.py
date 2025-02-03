import os
from random import randint
import argparse
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt

from quadrotor_diffusion.models.vae import VAE_Wrapper
from quadrotor_diffusion.utils.nn.post_process import fit_to_recon
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.file import get_checkpoint_file, load_course_trajectory, get_experiment_folder
from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.utils.simulator import play_trajectory, render_simulation
from quadrotor_diffusion.utils.nn.post_process import fit_to_recon
from quadrotor_diffusion.utils.plotting import course_base_plot, add_gates_to_course, add_trajectory_to_course


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)
parser.add_argument('-s', '--sample', type=str, help='Sample (course_type,course_number,sample_number)', required=True)

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

experiments_folder = get_experiment_folder("logs/training", args.experiment)

ref_pos = trajectory.as_ref_pos()

sample_works, drone_states_ref = play_trajectory(ref_pos)
print(f"Finished simulation on sample data {'succesfully' if sample_works else 'unsuccesfully'}")
reference_filename = os.path.join("logs/training", experiments_folder, "reference.mp4")
render_simulation(drone_states_ref, course, ref_pos, reference_filename)

padded_ref_pos = trajectory.as_ref_pos(pad_to=360)
normalized_reference = normalizer(padded_ref_pos)
inp = torch.tensor(normalized_reference).float()
inp = inp.unsqueeze(0)
inp = inp.to(args.device)
mu, logvar = eval.encode(inp, padding=32)
model_out = eval.decode(mu, padding=32).squeeze(0).cpu().numpy()
reconstructed = normalizer.undo(model_out)
reconstructed = reconstructed[:ref_pos.shape[0] - padded_ref_pos.shape[0], :]

fitted = fit_to_recon(reconstructed, 30)
recon_works, drone_states_recon = play_trajectory(ref_pos=fitted[0], ref_vel=fitted[1], ref_acc=fitted[2])
print(f"Finished simulation on reconstructed data {'succesfully' if recon_works else 'unsuccesfully'}")
recon_filename = os.path.join("logs/training", experiments_folder, "reconstructed.mp4")
render_simulation(drone_states_recon, course, reconstructed, filename=recon_filename)

_, ax = course_base_plot()
add_gates_to_course(course, ax)
add_trajectory_to_course(fitted[0], velocity_profile=fitted[1])
add_trajectory_to_course(ref_pos, reference=True)
trajectory_fig_filename = os.path.join("logs/training", experiments_folder, "trajectories.pdf")
plt.savefig(trajectory_fig_filename)
