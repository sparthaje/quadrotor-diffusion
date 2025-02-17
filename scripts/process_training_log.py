import argparse
import os

from quadrotor_diffusion.utils.plotting import plot_loss_and_time
from quadrotor_diffusion.utils.file import get_experiment_folder

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=int, help='Experiment number')
parser.add_argument('-l', '--losses', type=str, help='Losses to plot')
parser.add_argument('-s', '--log_loss', action='store_true', help="Take log of loss when plotting")
args = parser.parse_args()

training_folder_base = 'logs/training'

losses = args.losses.split(",")
training_folder = os.path.join(training_folder_base, get_experiment_folder(training_folder_base, args.experiment))
plot_loss_and_time(os.path.join(training_folder, "logs.csv"), losses, log_loss=args.log_loss)
