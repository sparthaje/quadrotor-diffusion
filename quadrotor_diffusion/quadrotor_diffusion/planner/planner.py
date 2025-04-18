import os
import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

from .scoring import (
    ScoringMethod,
    filter_valid_trajectories,
    fastest,
    slowest,
    min_curvature
)

from quadrotor_diffusion.models.diffusion_wrapper import DiffusionWrapper, SamplerType
from quadrotor_diffusion.utils.quad_logging import iprint as print
from quadrotor_diffusion.utils.nn.post_process import fit_to_recon


def cudnn_benchmark(
    samples: int,
    model: DiffusionWrapper,
    vae_downsample: int,
    device: str,
    sampler: tuple[SamplerType, int],
    w: float,
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
        _ = model.sample(
            samples,
            128,
            device,
            local_conditioning=local_conditioning,
            global_conditioning=global_conditioning,
            sampler=sampler,
            decoder_downsample=vae_downsample,
            w=w,
        )

    print("Finished cudnn benchmark")


def plan(
    samples: int,
    course: torch.Tensor,
    sampler: tuple[SamplerType, int],
    w: float,
    scoring: ScoringMethod,
    model: DiffusionWrapper,
    vae_downsample: int,
    device: str,
    current_traj: list[np.array] = None,
    return_all_samples: bool = False,
    ignore_filter_step: bool = False,
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
        sampler (SamplerType): Sampler algorithm, number steps
        w: CFG weighting
        model (LatentDiffusionWrapper): model
        vae_downsample (int): Compression rate of autoencoder
        device (str): Where to run sampling code
        current_traj (list[np.array], optional): Current trajectory tracking. If none will plan from c[0].
        return_all_samples: Instead of second argument being all considered samples it will return all generated samples
        ignore_filter_step: Never set this to true unless debugging
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
    trajectories = model.sample(
        samples,
        128,
        device=device,
        local_conditioning=local_conditioning,
        global_conditioning=global_conditioning,
        sampler=sampler,
        decoder_downsample=vae_downsample,
        w=w,
    )

    if not ignore_filter_step:
        # Filter candidates to ones which have a valid starting state and cross the next gate
        candidates = trajectories[filter_valid_trajectories(trajectories, x_0_expected, g_i)]
    else:
        candidates = trajectories

    if scoring == ScoringMethod.FAST:
        best_traj = fastest(candidates)
    elif scoring == ScoringMethod.SLOW:
        best_traj = slowest(candidates)
    elif scoring == ScoringMethod.STRAIGHT:
        best_traj = min_curvature(candidates)
    else:
        raise ValueError(f"Scoring method {scoring} not implemented")

    trajectory = candidates[best_traj]
    delta = trajectory[0] - x_0_expected

    ending_idx = (trajectory - g_i).pow(2).sum(-1).argmin().item()
    trajectory[:ending_idx] -= delta * torch.linspace(1, 0, ending_idx, device=device)[:, None].expand(-1, 3)

    trajectories_to_return = trajectories if return_all_samples else candidates

    trajectory = trajectory.cpu().numpy()
    if current_traj is None:
        smooth_trajectory = fit_to_recon(trajectory, 30)
        smooth_trajectory = [x[:ending_idx] for x in smooth_trajectory]
        return smooth_trajectory, trajectories_to_return

    # Fit next trajectory considering current trajectory
    smooth_trajectory = np.vstack((current_traj[0], trajectory))
    smooth_trajectory = fit_to_recon(smooth_trajectory, 30)
    points_current_traj = current_traj[0].shape[0]
    smooth_trajectory = [x[points_current_traj:points_current_traj+ending_idx] for x in smooth_trajectory]

    return smooth_trajectory, trajectories_to_return
