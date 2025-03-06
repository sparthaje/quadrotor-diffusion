import os
import random
import enum
import copy
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from quadrotor_diffusion.utils.trajectory import (
    INITIAL_GATE_EXIT,
    has_knot,
)
from quadrotor_diffusion.data_generation.shared_utils import (
    BoundarySamplingStrategy,
    add_next_gate
)
from quadrotor_diffusion.utils.voxels import create_occupancy_map, collision_along_trajectory
from quadrotor_diffusion.utils.trajectory import INITIAL_GATE_EXIT
from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.utils.dataset.boundary_condition import State, State3D, PolynomialTrajectory
from quadrotor_diffusion.utils.plotting import (
    course_base_plot,
    add_gates_to_course,
    add_trajectory_to_course
)


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
    trajectories = add_next_gate(course, 1, [0.8, 1.0], [1.5, 2.8], [init_trajectory],
                                 [BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL], False, use_curve=False)

    # Add second gate, choose MAX_TRAJECTORIES_TO_EXAMINE randomly, then filter to only keep ones that work in sim
    trajectories = add_next_gate(course, 2, [0.8, 1.2], [1.0, 2.0], trajectories,
                                 [BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL], False, use_curve=False)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    trajectories = add_next_gate(course, 3, [0.8, 1.1], [1.8, 3.0], trajectories,
                                 [BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL], False, use_curve=False)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    trajectories = add_next_gate(course, 4, [0.8, 1.1], [0.8, 1.5], trajectories,
                                 [BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL], False, use_curve=False)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    # Add fourth gate, choose MAX_TRAJECTORIES_TO_EXAMINE randomly, then filter to only keep ones that work in sim
    trajectories = add_next_gate(course, 1, None, [1.8, 3.0], trajectories,
                                 [BoundarySamplingStrategy.USE_EXISTING_STATE], True, use_curve=False)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
    trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    # Filter any trajectories with collisions
    occupancy_map = create_occupancy_map(course, voxel_size=0.025)
    trajectories = [t for t in trajectories if not collision_along_trajectory(
        t.as_ref_pos(), occupancy_map, voxel_size=0.025)]

    # Filter any trajectories with knots
    trajectories = [t for t in trajectories if not has_knot(t.as_ref_pos())]

    return trajectories


def generate_pill() -> tuple[list[np.array], list[PolynomialTrajectory]]:
    """
    Creates a triangle course

    Returns
    - Course, Valid Trajectories
    """

    gate_1 = np.array([
        np.random.uniform(-0.6, -0.9),
        np.random.uniform(-0.8, -0.4),
        random.choice([0.3, 0.525]),
        np.random.uniform(-0.2, 0.2)
    ])

    gate_2 = np.array([
        np.random.uniform(-0.6, -0.9),
        np.random.uniform(0.4, 0.8),
        random.choice([0.3, 0.525]),
        np.random.uniform(-0.2, 0.2)
    ])

    gate_3 = np.array([
        np.random.uniform(0.6, 0.9),
        np.random.uniform(0.4, 0.8),
        random.choice([0.3, 0.525]),
        np.random.uniform(-0.2, 0.2) + np.pi
    ])
    gate_3[3] = np.arctan2(np.sin(gate_3[3]), np.cos(gate_3[3]))

    gate_4 = np.array([
        np.random.uniform(0.6, 0.9),
        np.random.uniform(-0.8, -0.4),
        random.choice([0.3, 0.525]),
        np.random.uniform(-0.2, 0.2) + np.pi
    ])
    gate_4[3] = np.arctan2(np.sin(gate_4[3]), np.cos(gate_4[3]))

    starting_position = np.array([
        np.random.uniform(-0.4, 0.4),
        np.random.uniform(-1.5, -1.8),
        random.choice([0.3, 0.525]),
        0
    ])

    course = [starting_position, gate_1, gate_2, gate_3, gate_4, gate_4]

    trajectories = create_polynomial_trajectories(course)
    course.pop(-1)
    return course, trajectories
