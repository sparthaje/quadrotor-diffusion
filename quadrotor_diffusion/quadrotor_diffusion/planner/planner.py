import os
import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

from .samplers import SamplerType, ScoringMethod
from .scoring import (
    filter_valid_trajectories,
    fastest,
    slowest
)

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
from quadrotor_diffusion.utils.plotting import plot_states
from quadrotor_diffusion.utils.trajectory import derive_trajectory


def cudnn_benchmark(
    samples: int,
    model: LatentDiffusionWrapper,
    vae_downsample: int,
    device: str,
):
    """
    If model input sizes are fixed some optimizations can be done to speed up inference. Run this method before using the planner as long as the number of samples stays consistent

    Args:
        samples (int): Number of samples to consider
        model (LatentDiffusionWrapper): model
        vae_downsample (int): Compression rate of autoencoder
        device (str): Where to run sampling code
    """
    print("Warming with cudnn benchmark...")

    torch.backends.cudnn.benchmark = True

    local_conditioning = torch.ones((samples, 6, 3), dtype=torch.float32, device=device)
    global_conditioning = torch.ones((samples, 4, 4), dtype=torch.float32, device=device)
    for _ in range(5):
        _ = model.sample(samples, 128, vae_downsample, "cuda",
                         local_conditioning=local_conditioning, global_conditioning=global_conditioning)

    print("Finished cudnn benchmark")


def plan(
    samples: int,
    course: torch.Tensor,
    sampler: SamplerType,
    scoring: ScoringMethod,
    model: LatentDiffusionWrapper,
    vae_downsample: int,
    device: str,
    current_traj: list[np.array] = None,
) -> tuple[
    list[np.ndarray],
    torch.Tensor,
]:
    """
    Runs one iteration of the planner

    Args:
        samples (int): Number of samples to consider
        course (torch.Tensor): 
            If starting from initial position, just give course as ordered gates with course[0] being initial position else
            Course such that it is ordered so the zeroth element is the next gate from the current state, don't include the most recent gate
        sampler (SamplerType): Sampler algorithm
        model (LatentDiffusionWrapper): model
        vae_downsample (int): Compression rate of autoencoder
        device (str): Where to run sampling code
        current_traj (list[np.array], optional): Current trajectory tracking. If none will plan from c[0].

    Returns:
        tuple[ list[np.ndarray], torch.Tensor, ]: 
            - Next trajectory to follow
            - All considered samples
    """

    if current_traj:
        local_conditioning = current_traj[0][-6:]
        local_conditioning = torch.tensor(local_conditioning, dtype=torch.float32).to(device)

        null_tokens = np.tile(np.array(5 * np.ones((1, 4))), (4 - len(course), 1))
        global_conditioning = np.vstack((course, null_tokens))

        t = 1/30
        x_0_expected = current_traj[0][-1] + current_traj[1][-1] * t + 0.5 * current_traj[2][-1] * t ** 2
        x_0_expected = torch.tensor(x_0_expected, dtype=torch.float32).to(device)
        g_i = torch.tensor(course[0][:3], dtype=torch.float32).to(device)
    else:
        local_conditioning = torch.tensor(np.tile(course[0], (6, 1)), dtype=torch.float32).to(device)

        global_conditioning = np.vstack((course[1:], course[1:0]))
        null_tokens = np.tile(np.array(5 * np.ones((1, 4))), (4 - len(global_conditioning), 1))
        global_conditioning = np.vstack((global_conditioning, null_tokens))

        x_0_expected = torch.tensor(course[0][:3], dtype=torch.float32).to(device)
        g_i = torch.tensor(course[1][:3], dtype=torch.float32).to(device)

    local_conditioning = local_conditioning.unsqueeze(0).expand((samples, -1, -1))[:, :, :3]
    global_conditioning = torch.tensor(global_conditioning, dtype=torch.float32).to(
        device).unsqueeze(0).expand((samples, -1, -1))

    # Sample Candidates
    if sampler == SamplerType.DDPM:
        trajectories = model.sample(samples, 128, vae_downsample, device,
                                    local_conditioning=local_conditioning, global_conditioning=global_conditioning)
    else:
        raise ValueError(f"Sampler {sampler} not implemented")

    # Filter candidates to ones which have a valid starting state and cross the next gate
    trajectories = trajectories[filter_valid_trajectories(trajectories, x_0_expected, g_i)]

    if scoring == ScoringMethod.FAST:
        best_traj = fastest(trajectories)
    elif scoring == ScoringMethod.SLOW:
        best_traj = slowest(trajectories)
    else:
        raise ValueError(f"Scoring method {scoring} not implemented")

    trajectory = trajectories[best_traj]
    delta = trajectory[0] - x_0_expected
    trajectory -= delta

    ending_idx = (trajectory - g_i).pow(2).sum(-1).argmin().item()

    trajectory = trajectory.cpu().numpy()
    if current_traj is None:
        smooth_trajectory = fit_to_recon(trajectory, 30)
        smooth_trajectory = [x[:ending_idx] for x in smooth_trajectory]
        return smooth_trajectory, trajectories

    # Fit next trajectory considering current trajectory
    smooth_trajectory = np.vstack((current_traj[0], trajectory))
    smooth_trajectory = fit_to_recon(smooth_trajectory, 30)
    points_current_traj = current_traj[0].shape[0]
    smooth_trajectory = [x[points_current_traj:points_current_traj+ending_idx] for x in smooth_trajectory]

    return smooth_trajectory, trajectories
