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
    DRONE_INIT_POSITION,
    DRONE_STOPPING_DIST,
    BoundarySamplingStrategy,
    add_next_gate
)
from quadrotor_diffusion.utils.voxels import create_occupancy_map, collision_along_trajectory
from quadrotor_diffusion.utils.trajectory import INITIAL_GATE_EXIT
from quadrotor_diffusion.utils.logging import iprint as print
from quadrotor_diffusion.utils.dataset.boundary_condition import State, State3D, PolynomialTrajectory


from quadrotor_diffusion.utils.plotting import course_base_plot, add_gates_to_course, add_trajectory_to_course

os.environ["DEBUG"] = "True"


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
    trajectories = add_next_gate(course, 1, [1.2, 1.4], [2.5, 3.5], [init_trajectory],
                                 [BoundarySamplingStrategy.NEW_VELOCITY_AND_ACCEL])

    knot_one = [
        0.2 * course[1][0] + 0.8 * course[2][0],
        0.7 * course[1][1] + 0.3 * course[2][1],
        course[1][2],
        -np.pi/2
    ]
    course.insert(2, knot_one)

    trajectories = add_next_gate(course, 2, [0.4, 0.5], [1.0, 1.5], trajectories, [
        BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL])
    course.pop(2)

    trajectories = add_next_gate(course, 2, [0.8, 1.0], [1.6, 2.3], trajectories, [
        BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL])
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    trajectories = add_next_gate(course, 3, [0.8, 1.0], [2.3, 3.3], trajectories, [
        BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL])
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    knot_two = [
        course[4][0] + 0.3,
        course[4][1] - 0.5,
        course[4][2],
        0.0
    ]
    course.insert(4, knot_two)
    trajectories = add_next_gate(course, 4, [], [1.3, 2.3], trajectories, [
        BoundarySamplingStrategy.ZERO], num_segments=5)
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]
    course.pop(4)

    trajectories = add_next_gate(course, 4, [0.8, 1.0], [0.8, 1.3], trajectories, [
        BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL])
    if len(trajectories) > MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP:
        random.shuffle(trajectories)
        trajectories = trajectories[:MAX_TRAJECTORIES_TO_EXAMINE_PER_STEP]

    trajectories = add_next_gate(course, 5, [], [1.0, 1.5], trajectories, [
        BoundarySamplingStrategy.ZERO])
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


def generate_zig_zag() -> tuple[list[np.array], list[PolynomialTrajectory]]:
    """
    Creates a zig-zag course, problem is trajectories are too long (>12 seconds)

    Returns
    - Course, Valid Trajectories
    """
    gate_1 = np.array([
        np.random.uniform(-0.7, -0.4),
        np.random.uniform(-1.0, -0.8),
        random.choice([0.3, 0.525]),
        np.random.uniform(0.2, 0.7) * -np.pi / 2,
    ])

    gate_2 = np.array([
        np.random.uniform(0.4, 0.7),
        np.random.uniform(-0.3, 0.0),
        random.choice([0.3, 0.525]),
        np.random.uniform(0.2, 0.5) * np.pi / 2,
    ])

    gate_3 = np.array([
        np.random.uniform(-0.7, -0.4),
        np.random.uniform(0.5, 0.8),
        random.choice([0.3, 0.525]),
        np.random.uniform(0.2, 0.5) * -np.pi / 2,
    ])

    gate_4 = np.array([
        np.random.uniform(0.4, 0.7),
        np.random.uniform(1.2, 1.3),
        random.choice([0.3, 0.525]),
        np.random.uniform(0.2, 0.5) * np.pi / 2,
    ])

    ending_position = DRONE_STOPPING_DIST * R.from_euler('z', gate_4[3]).as_matrix() @ INITIAL_GATE_EXIT + gate_4[:-1]
    ending_position = np.append(ending_position, gate_4[-1])

    course = [DRONE_INIT_POSITION, gate_1, gate_2, gate_3, gate_4, ending_position]
    trajectories = create_polynomial_trajectories(course)

    return course, trajectories
