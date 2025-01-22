import os
import sys
import random
import copy
import enum
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from quadrotor_diffusion.utils.trajectory import (
    INITIAL_GATE_EXIT,
    has_knot,
)
from quadrotor_diffusion.data_generation.shared_utils import (
    DRONE_INIT_POSITION,
    DRONE_STOPPING_DIST,
    BoundarySamplingStrategy,
    add_next_gate,
)
from quadrotor_diffusion.utils.voxels import create_occupancy_map, collision_along_trajectory
from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.utils.dataset.boundary_condition import State, State3D, PolynomialTrajectory


def create_polynomial_trajectories(course: list[np.array]) -> list[PolynomialTrajectory]:
    """
    Creates one potential trajectory for a line course

    Args:
    - course (list[np.array]): [[x,y,z,theta]] (should include initial position up to the final stopped position inclusive)

    Returns:
    - list[PolynomialTrajectory]: Trajectories for the course validated in sim
    """
    MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP = 150

    init_state = State3D(
        x=State(s=course[0][0]),
        y=State(s=course[0][1]),
        z=State(s=course[0][2]),
    )
    init_trajectory = PolynomialTrajectory([init_state], [])

    # Add first gate
    trajectories = add_next_gate(course, 1, [0.8, 1.5], [0.8, 2.0], [init_trajectory],
                                 [BoundarySamplingStrategy.NEW_NORMAL_VELOCITY_AND_ACCEL], check_traj_direction=True)

    # Add second gate, choose MAX_TRAJECTORIES_TO_EXAMINE randomly, then filter to only keep ones that work in sim
    trajectories = add_next_gate(course, 2, [0.8, 1.5], [0.2, 2.0], trajectories, [
        BoundarySamplingStrategy.ZERO,
        BoundarySamplingStrategy.HOLD_VELOCITY,
        BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL
    ], check_traj_direction=True)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    # Add third gate, choose MAX_TRAJECTORIES_TO_EXAMINE randomly, then filter to only keep ones that work in sim
    trajectories = add_next_gate(course, 3, [0.8, 1.5], [0.2, 2.0], trajectories, [
        BoundarySamplingStrategy.ZERO,
        BoundarySamplingStrategy.HOLD_VELOCITY,
        BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL
    ], check_traj_direction=True)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    # Add fourth gate, choose MAX_TRAJECTORIES_TO_EXAMINE randomly, then filter to only keep ones that work in sim
    trajectories = add_next_gate(course, 4, [0.8, 1.0], [0.5, 2.0], trajectories, [
        BoundarySamplingStrategy.ZERO,
        BoundarySamplingStrategy.HOLD_VELOCITY,
        BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL
    ], check_traj_direction=True)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    # Add stopping point
    trajectories: list[PolynomialTrajectory] = add_next_gate(course, 5, [], [1.0, 2.5], trajectories,
                                                             [BoundarySamplingStrategy.ZERO], check_traj_direction=True)

    # Filter any trajectories with collisions
    occupancy_map = create_occupancy_map(course, voxel_size=0.05)
    trajectories = [t for t in trajectories if not collision_along_trajectory(
        t.as_ref_pos(), occupancy_map, voxel_size=0.05)]

    # Filter any trajectories with knots
    trajectories = [t for t in trajectories if not has_knot(t.as_ref_pos())]

    return trajectories


def generate_linear() -> tuple[list[np.array], list[PolynomialTrajectory]]:
    """
    Creates a linear course

    Returns
    - Course, Valid Trajectories
    """
    gate_1 = np.array([
        np.random.uniform(-0.25, 0.25),
        np.random.uniform(-1.0, -0.5),
        random.choice([0.3, 0.525]),
        np.random.uniform(-0.1 * np.pi, 0.1 * np.pi)
    ])

    gate_2 = np.array([
        np.random.uniform(-0.25, 0.25),
        np.random.uniform(-0.2, 0.3),
        random.choice([0.3, 0.525]),
        np.random.uniform(-0.1 * np.pi, 0.1 * np.pi)
    ])

    gate_3 = np.array([
        np.random.uniform(-0.25, 0.25),
        np.random.uniform(0.5, 1.0),
        random.choice([0.3, 0.525]),
        np.random.uniform(-0.1 * np.pi, 0.1 * np.pi)
    ])

    gate_4 = np.array([
        np.random.uniform(-0.25, 0.25),
        np.random.uniform(1.2, 1.7),
        random.choice([0.3, 0.525]),
        np.random.uniform(-0.1 * np.pi, 0.1 * np.pi)
    ])

    ending_position = DRONE_STOPPING_DIST * R.from_euler('z', gate_4[3]).as_matrix() @ INITIAL_GATE_EXIT + gate_4[:-1]
    ending_position = np.append(ending_position, gate_4[-1])

    course = [DRONE_INIT_POSITION, gate_1, gate_2, gate_3, gate_4, ending_position]
    trajectories = create_polynomial_trajectories(course)

    return course, trajectories
