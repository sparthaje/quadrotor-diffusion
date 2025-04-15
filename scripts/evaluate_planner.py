import os
import re
import sys
import argparse
import time
from dataclasses import dataclass

import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

from quadrotor_diffusion.utils.nn.training import Trainer
from quadrotor_diffusion.models.diffusion_wrapper import LatentDiffusionWrapper, SamplerType
from quadrotor_diffusion.models.vae_wrapper import VAE_Wrapper
from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.nn.args import TrainerArgs
from quadrotor_diffusion.utils.quad_logging import iprint as print, dataclass_to_table
from quadrotor_diffusion.utils.simulator import play_trajectory
from quadrotor_diffusion.utils.file import get_checkpoint_file, get_sample_folder
from quadrotor_diffusion.planner import plan, cudnn_benchmark, ScoringMethod
from quadrotor_diffusion.utils.trajectory import compute_tracking_error


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment', type=str, help='LDM Experiment,VAE Experiment', required=True)

parser.add_argument('-s', '--samples', type=int, help='Number of trajectories to generate', required=True)
parser.add_argument('-l', '--laps', type=int, help='How many laps to do', required=True)
parser.add_argument('-t', '--sampler', type=str, help='DDPM / DDIM / LCM', required=True)

parser.add_argument('-p', '--epoch', type=int, help='Epoch number, default is biggest', default=None)
parser.add_argument('-d', '--device', type=str, help='Device to use', default="cuda")
parser.add_argument('-m', '--no_ema', action='store_true', help="Use normal model instead of ema model.")

args = parser.parse_args()
sys.argv = [sys.argv[0]]

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

with open(os.path.join(sample_dir, "overview.txt"), "w") as f:
    f.write(
        f"Num Samples: {args.samples}\n" +
        ("Not using EMA\n" if args.no_ema else "Using EMA\n") +
        f"Device: {args.device}\n" +
        f"Laps: {args.laps}\n" +
        f"Sampling Method: {args.sampler}\n" +
        f"VAE Checkpoint: {chkpt}\n"
    )

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

cudnn_benchmark(args.samples, model, vae_downsample, args.device, sampler=sampler)


def evaluate(
    laps: int,
    samples: int,
    model: LatentDiffusionWrapper,
    course: str,
    device: str,
) -> tuple[bool, bool, float, float, int, float, float]:
    """
    Evaluates the planner on one course

    Args:
        laps (int): Laps to evaluate for
        samples (int): Number of samples to consider
        model (LatentDiffusionWrapper)
        course (str)
        device (str)

    Returns:
        tuple[bool, bool, float, float, float, int, float, float]:
            - Planner Failure
            - Worked in Simulation
            - ∆EP - Sum of Delta Expected Position (averaged across trajectories sampled) for each plan
            - ∆EV - Sum of Delta Expected Velocity (averaged across trajectories sampled) for each plan
            - Generated Plans
            - ATE - Average positional tracking error across the whole course
            - Total computation time for all plans
    """
    course: np.ndarray = np.load(course)
    current_traj = None

    gate_idx = 0
    trajectories = []

    gates_per_lap = course.shape[0] - 1
    iterations = gates_per_lap * laps + 1

    generated_plans = 0
    dep = 0.0
    dev = 0.0
    total_computation_time = 0.0

    for i in range(iterations):
        global_context = course
        if current_traj is not None:
            global_context = np.vstack((course[gate_idx + 1:], course[1:gate_idx]))

        s = time.time()
        try:
            next_traj, generated_samples = plan(
                samples,
                global_context,
                sampler,
                ScoringMethod.FAST,
                model,
                vae_downsample,
                device,
                current_traj=current_traj,
                return_all_samples=True
            )
        except ValueError:
            return True, False, dep, dev, generated_plans, 0.0, total_computation_time

        trajectories.append(next_traj)

        total_computation_time += time.time() - s

        dt = 1/30
        if current_traj:
            x_0_expected = current_traj[0][-1] + current_traj[1][-1] * dt + 0.5 * current_traj[2][-1] * dt ** 2
            x_0_expected = torch.tensor(x_0_expected, dtype=torch.float32).to(device)

            v_0_expected = current_traj[1][-1] + current_traj[2][-1] * dt
            v_0_expected = torch.tensor(v_0_expected, dtype=torch.float32).to(device)
        else:
            x_0_expected = torch.tensor(course[0][:3], dtype=torch.float32).to(device)

            v_0_expected = torch.zeros((3,), dtype=torch.float32).to(device)

        delta_x0 = torch.norm(generated_samples[:, 0, :] - x_0_expected, 2, dim=-1)
        vel_0 = (generated_samples[:, 1, :] - generated_samples[:, 0, :]) / dt
        delta_v0 = torch.norm(vel_0 - v_0_expected, 2, dim=-1)

        delta_x0_avg = torch.mean(delta_x0).item()
        delta_v0_avg = torch.mean(delta_v0).item()

        dep += delta_x0_avg
        dev += delta_v0_avg

        gate_idx += 1
        current_traj = next_traj
        generated_plans += 1

        # Looped back around to previous gate
        if gate_idx == len(course):
            gate_idx = 1

    reference = [np.vstack([t[idx] for t in trajectories]) for idx in range(3)]
    ref_pos = reference[0]
    ref_vel = reference[1]
    ref_acc = reference[2]

    worked, states = play_trajectory(ref_pos, ref_vel, ref_acc)

    return (
        False,
        worked,
        dep,
        dev,
        generated_plans,
        np.linalg.norm(compute_tracking_error(ref_pos, states[0])),
        total_computation_time,
    )


courses = []
for root, dirs, files in tqdm.tqdm(os.walk('data/courses/eval')):
    for file in files:
        if file.endswith('.npy'):
            courses.append(os.path.join(root, file))

# Course statistics
courses_success = 0
courses_failed_with_crash = 0
courses_planner_failed = 0
average_tracking_error = 0.
courses_total = len(courses)

# Planner stats
dep_total = 0.
dev_total = 0.
total_computation_time = 0.
generated_plans_total = 0

planner_failures = []
crash_failures = []

for course in tqdm.tqdm(courses):
    (
        planner_failed,
        worked_in_sim,
        dep,
        dev,
        generated_plans,
        tracking_error,
        computation_time,
    ) = evaluate(
        args.laps,
        args.samples,
        model,
        course,
        args.device,
    )

    if planner_failed:
        courses_planner_failed += 1
        planner_failures.append(course)

    if not planner_failed and not worked_in_sim:
        courses_failed_with_crash += 1
        crash_failures.append(course)

    if not planner_failed and worked_in_sim:
        courses_success += 1

        average_tracking_error += tracking_error

    if generated_plans > 0:
        dep_total += dep
        dev_total += dev
        total_computation_time += computation_time
        generated_plans_total += generated_plans


@dataclass
class Metrics:
    courses_total: int
    success_rate: float
    planner_failure_rate: float
    crash_rate: float
    avg_tracking_error: float
    delta_ep: float
    delta_ev: float
    planner_hz: float


metrics = Metrics(
    courses_total=courses_total,
    success_rate=100 * courses_success / courses_total,
    planner_failure_rate=100 * courses_planner_failed / courses_total,
    crash_rate=100 * courses_failed_with_crash / courses_total,
    avg_tracking_error=average_tracking_error / courses_success,
    delta_ep=dep_total / generated_plans_total,
    delta_ev=dev_total / generated_plans_total,
    planner_hz=generated_plans_total / total_computation_time
)

print(dataclass_to_table(metrics))

print("Planner failures:")
for course in planner_failures:
    print("\t" + course)

print("Crash failures:")
for course in crash_failures:
    print("\t" + course)


with open(os.path.join(sample_dir, "overview.txt"), "a") as f:
    f.write(dataclass_to_table(metrics) + "\n\n")

    f.write("Planner failures:\n")
    for course in planner_failures:
        f.write("\t" + course + "\n")
    f.write("Crash failures:\n")
    for course in crash_failures:
        f.write("\t" + course + "\n")
