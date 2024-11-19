import argparse
import os

from quadrotor_diffusion.utils.plotting import plot_loss_and_time

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiments', type=int, help='Experiment number')
parser.add_argument('-l', '--losses', type=str, help='Losses to plot')
args = parser.parse_args()

training_folder_base = 'logs/training'
folders = os.listdir(training_folder_base)
training_folder = None
for folder in folders:
    if folder.startswith(f"{args.experiments}."):
        training_folder = folder
        break

if training_folder is None:
    raise NameError(f"No folder found for experiment {args.experiments}")

losses = args.losses.split(",")
training_folder = os.path.join(training_folder_base, training_folder)
plot_loss_and_time(os.path.join(training_folder, "logs.csv"), losses)
