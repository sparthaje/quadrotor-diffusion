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
    trajectories = add_next_gate(course, 1, [0.7, 0.8], [2.0, 2.4], [init_trajectory],
                                 [BoundarySamplingStrategy.NEW_NORMAL_VELOCITY_AND_ACCEL], False)

    # Add second gate, choose MAX_TRAJECTORIES_TO_EXAMINE randomly, then filter to only keep ones that work in sim
    trajectories = add_next_gate(course, 2, [0.7, 0.8], [2.0, 2.4], trajectories,
                                 [BoundarySamplingStrategy.NEW_NORMAL_VELOCITY_AND_ACCEL], False)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    trajectories = add_next_gate(course, 3, [0.7, 0.8], [2.0, 2.4], trajectories,
                                 [BoundarySamplingStrategy.NEW_NORMAL_VELOCITY_AND_ACCEL], False)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    trajectories = add_next_gate(course, 4, [0.7, 0.8], [2.0, 2.4], trajectories,
                                 [BoundarySamplingStrategy.NEW_NORMAL_VELOCITY_AND_ACCEL], False)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    # Add fourth gate, choose MAX_TRAJECTORIES_TO_EXAMINE randomly, then filter to only keep ones that work in sim
    trajectories = add_next_gate(course, 1, None, [2.0, 2.4], trajectories,
                                 [BoundarySamplingStrategy.USE_EXISTING_STATE], True)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
    trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    # Filter any trajectories with collisions
    occupancy_map = create_occupancy_map(course, voxel_size=0.05)
    trajectories = [t for t in trajectories if not collision_along_trajectory(
        t.as_ref_pos(), occupancy_map, voxel_size=0.05)]

    # Filter any trajectories with knots
    trajectories = [t for t in trajectories if not has_knot(t.as_ref_pos())]

    return trajectories


def generate_square() -> tuple[list[np.array], list[PolynomialTrajectory]]:
    """
    Creates a triangle course

    Returns
    - Course, Valid Trajectories
    """

    system_rotation = np.random.uniform(-np.pi, np.pi)
    clockwise = random.choice([True, False])
    clockwise_shift = np.pi if clockwise else 0

    radii = np.random.uniform(0.75, 1.0, 4)
    orientation = [
        system_rotation,
        system_rotation + np.random.uniform(0.9, 1.1) * np.pi/2,
        system_rotation + np.random.uniform(1.9, 2.1) * np.pi/2,
        system_rotation + np.random.uniform(2.9, 3.1) * np.pi/2,
    ]
    orientation = [np.arctan2(np.sin(x), np.cos(x)) for x in orientation]
    g_theta = [orientation[idx] - np.pi/4 -
               np.random.uniform(-0.05, 0.05) + clockwise_shift for idx in range(len(radii))]
    gates = [
        np.array([
            radii[idx] * np.cos(orientation[idx]),
            radii[idx] * np.sin(orientation[idx]),
            random.choice([0.3, 0.525]),
            np.arctan2(np.sin(g_theta[idx]), np.cos(g_theta[idx])),
        ]) for idx in range(len(radii))
    ]

    starting_position = np.append(np.random.uniform(-0.2, 0.2, 2), np.array([random.choice([0.3, 0.525]), 0]))

    course = [starting_position, *gates, gates[0]]
    if clockwise:
        course[2], course[4] = course[4], course[2]
        starting_position_component_1 = np.random.uniform(0.4, 0.6) * \
            R.from_euler('z', gates[0][3] + np.pi).as_matrix() @ INITIAL_GATE_EXIT + gates[0][:-1]
        starting_position = np.random.uniform(-0.35, 0.35) * R.from_euler('z', gates[0][3] + np.sign(-clockwise) * np.pi / 2).as_matrix()  \
            @ INITIAL_GATE_EXIT + starting_position_component_1
        starting_position = np.append(starting_position, 0.0)
        course[0] = starting_position

    trajectories = create_polynomial_trajectories(course)
    course.pop(-1)

    return course, trajectories
