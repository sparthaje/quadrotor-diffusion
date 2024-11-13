import os
import re
import sys
import argparse
import time

from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.models.diffusion import DiffusionWrapper
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.utils.plotting import view_trajectories_in_3d

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=int, help='Experiment number', required=True)

parser.add_argument('-s', '--samples', type=int, help='Number of trajectories to generate', required=True)
parser.add_argument('-o', '--horizon', type=int, help='Horizon', required=True)

parser.add_argument('-p', '--epoch', type=int, help='Epoch number, default is biggest', default=None)
parser.add_argument('-d', '--device', type=str, help='Device to use', default="cuda")
parser.add_argument('-m', '--no-ema', action='store_true', help="Use normal model instead of ema model.")

args = parser.parse_args()
sys.argv = [sys.argv[0]]

train_dir = "logs/training"
exp_dir_base = "logs/experiments"

folders = os.listdir(train_dir)
training_folder = None
for folder in folders:
    if folder.startswith(f"{args.experiment}."):
        training_folder = folder
        break

if training_folder is None:
    raise NameError(f"No folder found for experiment {args.experiment}")
training_folder = os.path.join(train_dir, training_folder)

pattern = r'epoch_(\d+)_loss_([\d.]+)'
checkpoints = dict()

for file in os.listdir(training_folder):
    match = re.match(pattern, file)
    if match:
        epoch = int(match.group(1))
        checkpoints[epoch] = os.path.join(training_folder, file)

if not len(checkpoints.keys()):
    raise ValueError("No epoch files found.")

checkpoint_file = None
if args.epoch is None:
    checkpoint_file = checkpoints[max(checkpoints.keys())]
elif args.epoch in checkpoints:
    checkpoint_file = checkpoints[args.epoch]
else:
    raise ValueError(f"Epoch {args.epoch} not found.")

max_exp_num = 0
pattern = re.compile(rf"model_{args.experiment}_exp_(\d+)")

for entry in os.listdir(exp_dir_base):
    match = pattern.match(entry)
    if match:
        e_value = int(match.group(1))
        max_exp_num = max(max_exp_num, e_value)

exp_dir = os.path.join(exp_dir_base, f"model_{args.experiment}_exp_{max_exp_num + 1}")
os.mkdir(exp_dir)
with open(os.path.join(exp_dir, "overview.txt"), "w") as f:
    f.write(
        open(os.path.join(training_folder, "overview.txt"), "r").read()
    )

model: DiffusionWrapper = None
ema: DiffusionWrapper = None
normalizer: Normalizer = None
trainer_args: TrainerArgs = None

diff, ema, normalizer, trainer_args = Trainer.load(checkpoint_file)
print(f"Loaded {checkpoint_file}")
model = diff if args.no_ema else ema

model.to(args.device)
start = time.time()
trajectories = model.sample_unguided(args.samples, args.horizon, args.device)
print(f"{time.time() - start:.2f} seconds to generate {args.samples} samples with {args.device}")

for i in range(trajectories.size(0)):  # Loop through batch_size
    sampled = trajectories[i].cpu().numpy()
    traj = normalizer.undo(sampled)
    view_trajectories_in_3d(os.path.join(exp_dir, f"traj{i}.pdf"), f"{i}th sample", traj)
