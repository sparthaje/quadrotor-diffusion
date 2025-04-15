import time
import pickle

import numpy as np
import argparse
from flask import Flask, request

from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.models.diffusion_wrapper import LatentDiffusionWrapper, SamplerType
from quadrotor_diffusion.models.vae_wrapper import VAE_Wrapper
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.quad_logging import iprint as print
from quadrotor_diffusion.utils.file import get_checkpoint_file, get_sample_folder
from quadrotor_diffusion.planner import plan, cudnn_benchmark, ScoringMethod

app = Flask(__name__)

HOST = 'localhost'
PORT = 65432

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=str, help='LDM Experiment,VAE Experiment', required=True)

parser.add_argument('-s', '--samples', type=int, help='Number of trajectories to generate', required=True)
parser.add_argument('-t', '--sampler', type=str, help='DDPM / DDIM / LCM', required=True)
parser.add_argument('-b', '--behavior', type=str, help='FAST', required=True)

parser.add_argument('-p', '--epoch', type=int, help='Epoch number, default is biggest', default=None)
parser.add_argument('-d', '--device', type=str, help='Device to use', default="cuda")
parser.add_argument('-m', '--no_ema', action='store_true', help="Use normal model instead of ema model.")

args = parser.parse_args()

ldm_experiment = int(args.experiment.split(",")[0])
vae_experiment = int(args.experiment.split(",")[1])

chkpt = get_checkpoint_file("logs/training", ldm_experiment, epoch=args.epoch)
print(chkpt)

sample_dir = get_sample_folder("logs/training", ldm_experiment)
print(sample_dir)

model: LatentDiffusionWrapper = None
ema: LatentDiffusionWrapper = None
normalizer: Normalizer = None
trainer_args: TrainerArgs = None

diff, ema, normalizer, trainer_args = Trainer.load(chkpt)
print(f"Loaded {chkpt}")
print(f"Using {normalizer}")

chkpt = get_checkpoint_file("logs/training", vae_experiment)
vae_wrapper: VAE_Wrapper = None
vae_wrapper, _, _, _ = Trainer.load(chkpt, get_ema=False)
vae_wrapper.to(args.device)
vae_downsample = 2 ** (len(vae_wrapper.args[1].channel_mults) - 1)

model = diff if args.no_ema else ema
model.decoder = vae_wrapper.decode
model.to(args.device)

sampler = None
if args.sampler == "DDPM":
    sampler = SamplerType.DDPM
elif args.sampler == "DDIM":
    sampler = SamplerType.DDIM
else:
    raise NotImplementedError(f"Sampler {args.sampler} not implemented")

scoring_method = None
if args.behavior == "FAST":
    scoring_method = ScoringMethod.FAST
else:
    raise NotImplementedError(f"Behavior {args.behavior} not implemented")

cudnn_benchmark(args.samples, model, vae_downsample, args.device, sampler=sampler)


@app.route('/plan', methods=['POST'])
def plan_route():
    start = time.time()
    data = request.data
    current_traj, global_context = pickle.loads(data)
    next_traj, _ = plan(
        args.samples,
        global_context,
        sampler,
        scoring_method,
        model,
        vae_downsample,
        args.device,
        current_traj=current_traj,
    )
    print(f"Computation time {time.time() - start:.2f}")
    return pickle.dumps(next_traj)


if __name__ == "__main__":
    print("HTTP Server started")
    app.run(host='0.0.0.0', port=5000)
