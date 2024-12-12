from random import randint
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from quadrotor_diffusion.models.vae import VAE_Wrapper
from quadrotor_diffusion.utils.nn.post_process import fit_to_recon
from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorTrajectoryDataset
from quadrotor_diffusion.utils.file import get_checkpoint_file, save_trajectory
from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.utils.simulator import play_trajectory
from quadrotor_diffusion.utils.trajectory import derive_trajectory


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)
parser.add_argument('-s', '--sample', type=int, help='Sample number', required=True)

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
data = QuadrotorTrajectoryDataset("data/quadrotor_random", normalizer, order=0)
print("Loaded", chkpt)

eval = ema if args.ema else model
eval = eval.to(args.device)
sample = data[args.sample]
ref_pos = normalizer.undo(sample)

CUT_OFF = 50

sample_works, drone_states_ref = play_trajectory(ref_pos[:-CUT_OFF, :])
print(f"Finished simulation on sample data {'succesfully' if sample_works else 'unsuccesfully'}\n")

inp = torch.tensor(sample).float()
inp = inp.unsqueeze(0)
inp = inp.to(args.device)
mu, logvar = eval.encode(inp, padding=32)
model_out = eval.decode(mu, padding=32).squeeze(0).cpu().numpy()
reconstructed = normalizer.undo(model_out)

recon_works, drone_states_recon = play_trajectory(reconstructed[:-CUT_OFF, :])
print(f"Finished simulation on reconstructed data {'succesfully' if recon_works else 'unsuccesfully'}\n")

fit_pos, fit_vel, fit_acc = fit_to_recon(reconstructed, 30)
fit_pos = fit_pos[:-CUT_OFF, :]
fit_vel = fit_vel[:-CUT_OFF, :]
fit_acc = fit_acc[:-CUT_OFF, :]

recon_fitted_works, drone_states_recon_fitted = play_trajectory(fit_pos, fit_vel, fit_acc)
print(f"Finished simulation on fitted reconstructed data {'succesfully' if recon_works else 'unsuccesfully'}\n")

save_trajectory("sample.trajectory.npy", ref_pos, derive_trajectory(
    ref_pos, 30), derive_trajectory(ref_pos, 30, order=2))
save_trajectory("fitted.trajectory.npy", fit_pos, fit_vel, fit_acc)

# region: plot initial inputs
plt.figure(figsize=(18, 12))

# Position plots
plt.subplot(3, 3, 1)
plt.plot(ref_pos[:, 0][:-CUT_OFF], linewidth=3.5)
plt.plot(fit_pos[:, 0], linewidth=3.5)
plt.ylabel("x (meters)")
plt.grid()
plt.subplot(3, 3, 2)
plt.plot(ref_pos[:, 1][:-CUT_OFF], linewidth=3.5)
plt.plot(fit_pos[:, 1], linewidth=3.5)
plt.ylabel("y (meters)")
plt.grid()
plt.subplot(3, 3, 3)
plt.plot(ref_pos[:, 2][:-CUT_OFF], linewidth=3.5)
plt.plot(fit_pos[:, 2], linewidth=3.5)
plt.ylabel("z (meters)")
plt.grid()

# Velocity plots
plt.subplot(3, 3, 4)
plt.plot(fit_vel[:, 0], linewidth=3.5)
plt.ylabel("vx (m/s)")
plt.grid()
plt.subplot(3, 3, 5)
plt.plot(fit_vel[:, 1], linewidth=3.5)
plt.ylabel("vy (m/s)")
plt.grid()
plt.subplot(3, 3, 6)
plt.plot(fit_vel[:, 2], linewidth=3.5)
plt.ylabel("vz (m/s)")
plt.grid()

# Acceleration plots
plt.subplot(3, 3, 7)
plt.plot(fit_acc[:, 0], linewidth=3.5)
plt.ylabel("ax (m/s²)")
plt.grid()
plt.subplot(3, 3, 8)
plt.plot(fit_acc[:, 1], linewidth=3.5)
plt.ylabel("ay (m/s²)")
plt.grid()
plt.subplot(3, 3, 9)
plt.plot(fit_acc[:, 2], linewidth=3.5)
plt.ylabel("az (m/s²)")
plt.grid()

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.figlegend(['Reference', 'Fit'], loc='upper center', ncol=3, frameon=False)
plt.savefig("vae.pdf")
# endregion

plt.figure(figsize=(18, 12))

# Position plots
plt.subplot(3, 3, 1)
plt.plot(drone_states_ref[0][:, 0], linewidth=3.5)
plt.plot(drone_states_recon[0][:, 0], linewidth=3.5)
plt.plot(drone_states_recon_fitted[0][:, 0], linewidth=3.5)
plt.ylabel("x (meters)")
plt.grid()
plt.subplot(3, 3, 2)
plt.plot(drone_states_ref[0][:, 1], linewidth=3.5)
plt.plot(drone_states_recon[0][:, 1], linewidth=3.5)
plt.plot(drone_states_recon_fitted[0][:, 1], linewidth=3.5)
plt.ylabel("y (meters)")
plt.grid()
plt.subplot(3, 3, 3)
plt.plot(drone_states_ref[0][:, 2], linewidth=3.5)
plt.plot(drone_states_recon[0][:, 2], linewidth=3.5)
plt.plot(drone_states_recon_fitted[0][:, 2], linewidth=3.5)
plt.ylabel("z (meters)")
plt.grid()

# Velocity plots
plt.subplot(3, 3, 4)
plt.plot(drone_states_ref[1][:, 0], linewidth=3.5)
plt.plot(drone_states_recon[1][:, 0], linewidth=3.5)
plt.plot(drone_states_recon_fitted[1][:, 0], linewidth=3.5)
plt.ylabel("vx (m/s)")
plt.grid()
plt.subplot(3, 3, 5)
plt.plot(drone_states_ref[1][:, 1], linewidth=3.5)
plt.plot(drone_states_recon[1][:, 1], linewidth=3.5)
plt.plot(drone_states_recon_fitted[1][:, 1], linewidth=3.5)
plt.ylabel("vy (m/s)")
plt.grid()
plt.subplot(3, 3, 6)
plt.plot(drone_states_ref[1][:, 2], linewidth=3.5)
plt.plot(drone_states_recon[1][:, 2], linewidth=3.5)
plt.plot(drone_states_recon_fitted[1][:, 2], linewidth=3.5)
plt.ylabel("vz (m/s)")
plt.grid()
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.figlegend(['Reference', 'Reconstructed from Latent', 'Filtered + Interpolation Reconstructed'],
              loc='upper center', ncol=3, frameon=False)
plt.suptitle("Evaluating Sample Data in Simulator")
plt.savefig("va2.pdf")
