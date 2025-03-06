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
    trajectories = add_next_gate(course, 1, [0.5, 0.6], [1.5, 3.0], [init_trajectory],
                                 [BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL], False)

    # Add second gate, choose MAX_TRAJECTORIES_TO_EXAMINE randomly, then filter to only keep ones that work in sim
    trajectories = add_next_gate(course, 2, [0.8, 1.0], [1.5, 3.0], trajectories,
                                 [BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL], False)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    trajectories = add_next_gate(course, 3, [0.8, 1.5], [1.5, 3.0], trajectories,
                                 [BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL], False)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    # Add fourth gate, choose MAX_TRAJECTORIES_TO_EXAMINE randomly, then filter to only keep ones that work in sim
    trajectories = add_next_gate(course, 1, None, [1.5, 3.0], trajectories,
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


def generate_triangle() -> tuple[list[np.array], list[PolynomialTrajectory]]:
    """
    Creates a triangle course

    Returns
    - Course, Valid Trajectories
    """

    system_rotation = np.random.uniform(-np.pi, np.pi)
    clockwise = random.choice([-np.pi/2, np.pi/2])

    r_1 = np.random.uniform(0.75, 1.0)
    t_1 = clockwise + (system_rotation - np.pi / 2)
    t_1 = np.arctan2(np.sin(t_1), np.cos(t_1))
    gate_1 = np.array([
        r_1 * np.cos(system_rotation),
        r_1 * np.sin(system_rotation),
        random.choice([0.3, 0.525]),
        t_1
    ])

    r_2 = np.random.uniform(0.75, 1.0)
    gate_2_rotational_offset = np.random.uniform(0.8, 1.2) * np.pi / 1.5
    gate_3_rotational_offset = np.random.uniform(1.8, 2.2) * np.pi / 1.5

    if clockwise < 0:
        gate_2_rotational_offset, gate_3_rotational_offset = gate_3_rotational_offset, gate_2_rotational_offset

    t_2 = clockwise + (system_rotation + gate_2_rotational_offset - np.pi / 2)
    t_2 = np.arctan2(np.sin(t_2), np.cos(t_2))
    gate_2 = np.array([
        r_2 * np.cos(system_rotation + gate_2_rotational_offset),
        r_2 * np.sin(system_rotation + gate_2_rotational_offset),
        random.choice([0.3, 0.525]),
        t_2
    ])

    r_3 = np.random.uniform(0.75, 1.0)
    t_3 = clockwise + (system_rotation + gate_3_rotational_offset - np.pi/2)
    t_3 = np.arctan2(np.sin(t_3), np.cos(t_3))
    gate_3 = np.array([
        r_3 * np.cos(system_rotation + gate_3_rotational_offset),
        r_3 * np.sin(system_rotation + gate_3_rotational_offset),
        random.choice([0.3, 0.525]),
        t_3
    ])

    starting_position_component_1 = np.random.uniform(0.4, 0.6) * \
        R.from_euler('z', gate_1[3] + np.pi).as_matrix() @ INITIAL_GATE_EXIT + gate_1[:-1]
    starting_position = np.random.uniform(0.25, 0.35) * R.from_euler('z', gate_1[3] + np.sign(-clockwise) * np.pi / 2).as_matrix()  \
        @ INITIAL_GATE_EXIT + starting_position_component_1
    starting_position = np.append(starting_position, 0.0)

    course = [starting_position, gate_1, gate_2, gate_3, gate_1]
    trajectories = create_polynomial_trajectories(course)
    course.pop(-1)
    return course, trajectories
