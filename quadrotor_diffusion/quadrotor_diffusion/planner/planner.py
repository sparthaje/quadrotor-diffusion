import os

from scipy.spatial.transform import Rotation as R
import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

from .scoring import (
    ScoringMethod,
    filter_valid_trajectories,
    fastest,
    slowest,
    min_curvature,
    center_gate,
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

        if course.shape[0] < 4:
            null_tokens = np.tile(np.array(5 * np.ones((1, 4))), (4 - len(course), 1))
            global_conditioning = np.vstack((course, null_tokens))
        else:
            global_conditioning = course[:4]

        t = 1/30
        x_0_expected = current_traj[0][-1] + current_traj[1][-1] * t + 0.5 * current_traj[2][-1] * t ** 2
        x_0_expected = torch.tensor(x_0_expected, dtype=torch.float32).to(device)
        g_i = torch.tensor(course[0][:3], dtype=torch.float32).to(device)
    else:
        local_conditioning = torch.tensor(np.tile(course[0], (6, 1)), dtype=torch.float32).to(device)

        global_conditioning = np.vstack((course[1:], course[1:0]))

        if global_conditioning.shape[0] < 4:
            null_tokens = np.tile(np.array(5 * np.ones((1, 4))), (4 - len(global_conditioning), 1))
            global_conditioning = np.vstack((global_conditioning, null_tokens))
        else:
            global_conditioning = global_conditioning[:4]

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
        candidates = trajectories[filter_valid_trajectories(trajectories, x_0_expected, g_i, current_traj is None)]
    else:
        candidates = trajectories

    if scoring == ScoringMethod.FAST:
        best_traj = fastest(candidates)
    elif scoring == ScoringMethod.SLOW:
        best_traj = slowest(candidates)
    elif scoring == ScoringMethod.STRAIGHT:
        best_traj = min_curvature(candidates)
    elif scoring == ScoringMethod.CENTER:
        best_traj = center_gate(candidates, g_i)
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


def plan_traj_frame(
    samples: int,
    course: np.ndarray,
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
        course (np.ndarray): 
            If starting from initial position, just give course as ordered gates with course[0] being initial position else
            Course such that it is ordered so the zeroth element is the next gate from the current state, don't include the most recent gate
            NOTE: THIS CAN BE A VIEW!!!
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

    gates = course.copy()

    # 1. Define frame of reference

    # Planning from not first position
    if current_traj:
        dt = 1 / 30
        x_0_expected = current_traj[0][-1] + current_traj[1][-1] * dt + 0.5 * current_traj[2][-1] * dt ** 2
        expected_gate = gates[0][:3].copy()

        p2 = x_0_expected[:2]
        p1 = current_traj[0][-1][:2]
        delta = p2 - p1
        yaw = np.arctan2(delta[1], delta[0])

    # Planning from takeoff position
    else:
        x_0_expected = gates[0][:3]
        expected_gate = gates[1][:3].copy()
        yaw = gates[1][3] + np.pi / 2

    trans_xy = x_0_expected[:2]
    rotation = R.from_euler('z', -yaw).as_matrix()
    rot_xy = rotation[:2, :2]

    # 2. Define l_cond and g_cond in world frame

    if current_traj:
        l_cond = current_traj[0][-6:].copy()

    else:
        l_cond = np.tile(course[0][:3], (6, 1)).copy()
        # Trim off starting position
        gates = gates[1:]

    # 3. Transform l_cond and g_cond into traj frame

    l_cond[:, :2] = (l_cond[:, :2] - trans_xy) @ rot_xy.T

    gates[:, :2] = (gates[:, :2] - trans_xy) @ rot_xy.T
    gates[:, 3] -= yaw
    gates[:, 3] = np.arctan2(np.sin(gates[:, 3]), np.cos(gates[:, 3]))

    if gates.shape[0] > 4:
        g_cond = gates[:4]
    else:
        null_tokens = np.tile(np.array(5 * np.ones((1, 4))), (4 - len(gates), 1))
        g_cond = np.vstack((gates, null_tokens))

    # 4. Make l_cond and g_cond tensor

    l_cond = torch.tensor(l_cond, dtype=torch.float32, device=device).unsqueeze(0).repeat((samples, 1, 1))
    g_cond = torch.tensor(g_cond, dtype=torch.float32, device=device).unsqueeze(0).repeat((samples, 1, 1))

    # 5. Sample Candidates
    trajectories = model.sample(
        samples,
        128,
        device=device,
        local_conditioning=l_cond,
        global_conditioning=g_cond,
        sampler=sampler,
        decoder_downsample=vae_downsample,
        w=w,
    )

    # 6. Transform back into World frame

    rot_xy_tensor = torch.tensor(rot_xy, dtype=torch.float32, device=device)
    trans_xy_tensor = torch.tensor(trans_xy, dtype=torch.float32, device=device)

    trajectories[:, :, :2] = (trajectories[:, :, :2] @ rot_xy_tensor) + trans_xy_tensor

    # 7. Filter trajectories
    expected_gate = torch.tensor(expected_gate, dtype=torch.float32, device=device)
    x_0_expected = torch.tensor(x_0_expected, dtype=torch.float32, device=device)

    if not ignore_filter_step:
        # Filter candidates to ones which have a valid starting state and cross the next gate
        candidates = trajectories[filter_valid_trajectories(
            trajectories, x_0_expected, expected_gate, current_traj is None)]
    else:
        candidates = trajectories

    # 8. Choose best trajectory
    if scoring == ScoringMethod.FAST:
        best_traj = fastest(candidates)
    elif scoring == ScoringMethod.SLOW:
        best_traj = slowest(candidates)
    elif scoring == ScoringMethod.STRAIGHT:
        best_traj = min_curvature(candidates)
    elif scoring == ScoringMethod.CENTER:
        best_traj = center_gate(candidates, expected_gate)
    else:
        raise ValueError(f"Scoring method {scoring} not implemented")

    trajectory = candidates[best_traj]
    delta = trajectory[0] - x_0_expected

    ending_idx = (trajectory - expected_gate).pow(2).sum(-1).argmin().item()
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
