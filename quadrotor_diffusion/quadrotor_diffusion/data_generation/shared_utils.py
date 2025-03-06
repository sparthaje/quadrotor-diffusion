import random
import copy
import enum

import numpy as np
from scipy.spatial.transform import Rotation as R

from quadrotor_diffusion.utils.trajectory import (
    get_positions_from_boundary_conditions,
    derive_trajectory,
    evaluate_vel_accel_profile,
    spherical_to_cartesian,
    INITIAL_GATE_EXIT
)
from quadrotor_diffusion.utils.dataset.boundary_condition import State, State3D, PolynomialTrajectory
from quadrotor_diffusion.utils.simulator import play_trajectory


DRONE_INIT_POSITION = np.array([0.0, -1.5, 0.525, 0.0])
DRONE_STOPPING_DIST = 0.5


def random_vel_direction(idx: int,
                         course: list[np.array],
                         use_gate_normal_theta: bool = False,
                         use_theta_to_next_gate: bool = False) -> tuple[float, float]:
    """
    Helper function to generate a good direction for velocity out of a gate informed by the course

    Args:
    - idx (int): Which gate in course this is for
    - course (list[np.array]): List of all gates
    - use_gate_normal_theta (bool): If True, always use gate_normal_theta for vel_theta
    - use_theta_to_next_gate (bool): If True, always use theta_to_next_gate for vel_theta

    Returns:
    - tuple[float, float]: theta, psi
    """

    # Last position should just be zero velocity
    assert idx < len(course) - 1

    gate = course[idx]
    gate_position = gate[:3]
    next_position = course[idx + 1][:3]
    delta_position = next_position - gate_position

    gate_normal_theta = gate[3]
    # Get the angle towards the next gate
    theta_to_next_gate = -np.arctan2(delta_position[0], delta_position[1])

    if use_gate_normal_theta:
        vel_theta = gate_normal_theta
    elif use_theta_to_next_gate:
        vel_theta = theta_to_next_gate
    else:
        vel_theta = np.random.uniform(gate_normal_theta, theta_to_next_gate)

    is_high = gate[2] == 0.525
    next_is_high = next_position[2] == 0.525

    # Going from high to low
    if is_high and not next_is_high:
        vel_psi = np.random.uniform(np.radians(-15), 0.0)
    # Going from low to high
    elif not is_high and next_is_high:
        vel_psi = np.random.uniform(0.0, np.radians(15))
    else:
        vel_psi = 0.0

    return vel_theta, vel_psi


class BoundarySamplingStrategy(enum.Enum):
    """
    Different strategies for how to sample dynamics on boundaries
    """
    # Zero velocity and acceleration at gate
    ZERO = 0

    # Very small velocity
    NEAR_ZERO = 1

    # Hold the previous velocity through gate with no acceleration
    HOLD_VELOCITY = 2

    # Sample a new velocity at gate with no acceleration
    NEW_VELOCITY_ZERO_ACCEL = 3

    # Sample a new velocity / acceleration at the gate only if previous velocity was zero
    NEW_VELOCITY_AND_ACCEL = 4

    # Sample a new velocity / acceleration normal to the gate only if previous velocity was zero
    NEW_NORMAL_VELOCITY_AND_ACCEL = 4

    # Use a given gate_idx
    USE_EXISTING_STATE = 5


def add_next_gate(course: list[np.array],
                  gate_idx: int,
                  vel_bounds: tuple[float, float],
                  time_bounds: tuple[float, float],
                  trajectories_prev: list[PolynomialTrajectory],
                  strategies: list[BoundarySamplingStrategy],
                  last_gate: bool,
                  check_traj_direction: bool = False,
                  num_segments: int = None,
                  use_curve: bool = True) -> list[PolynomialTrajectory]:
    """
    Adds the gate_idx to list of potential trajectories

    If its the last position, the trajectories will be sim validated

    Args:
    - course (list[np.array]): Gates
    - gate_idx (int): Gate to add
    - vel_bounds (tuple[float, float]): Upper and Lower bound for vel at gate if new vel
    - time_bounds (tuple[float, float]): Upper and Lower bound for time segments for previous gate to current
    - trajectories_prev (list[PolynomialTrajectory]): List of all trajectories that go from starting point to gate_idx - 1
    - strategies: List of strategies to try for sampling
    - last_gate (bool): Will run sim and evaluate
    - check_traj_direction: Evaluate if velocity points towards gate
    - num_segments: num_segments_to_sample hard override if not None
    - use_curve (bool): Use the defined GV curve

    Returns:
    - list[PolynomialTrajectory]: List all of all trajectories that go from starting point to gate_idx
    """
    trajectories_new = []

    for trajectory in trajectories_prev:
        num_segments_to_sample = 25
        if gate_idx == 5:
            num_segments_to_sample = 15
        if num_segments is not None:
            num_segments_to_sample = num_segments

        for strategy in strategies:
            sampled_segment_lengths = np.linspace(time_bounds[0], time_bounds[1], num_segments_to_sample)
            # If the gate idx is the last one then we only need one boundary condition that works at the end
            if gate_idx == 5:
                random.shuffle(sampled_segment_lengths)

            for segment_length in sampled_segment_lengths:
                new_trajectory: PolynomialTrajectory = copy.deepcopy(trajectory)

                previous_velocity = np.array([
                    new_trajectory.states[-1].x.v,
                    new_trajectory.states[-1].y.v,
                    new_trajectory.states[-1].z.v,
                ])

                if strategy == BoundarySamplingStrategy.ZERO:
                    velocity = np.zeros(3)
                    acceleration = np.zeros(3)
                elif strategy == BoundarySamplingStrategy.NEAR_ZERO:
                    vel_mag = 0.25
                    vel_theta, vel_psi = random_vel_direction(gate_idx, course)
                    velocity = spherical_to_cartesian(vel_mag, vel_theta, vel_psi)
                    acceleration = np.zeros(3)
                elif strategy == BoundarySamplingStrategy.NEW_VELOCITY_ZERO_ACCEL:
                    vel_mag = np.random.uniform(vel_bounds[0], vel_bounds[1])
                    vel_theta, vel_psi = random_vel_direction(gate_idx, course)
                    velocity = spherical_to_cartesian(vel_mag, vel_theta, vel_psi)
                    acceleration = np.zeros(3)
                elif strategy == BoundarySamplingStrategy.NEW_VELOCITY_AND_ACCEL and np.linalg.norm(previous_velocity) < 0.1:
                    vel_mag = np.random.uniform(vel_bounds[0], vel_bounds[1])
                    vel_theta, vel_psi = random_vel_direction(gate_idx, course)
                    velocity = spherical_to_cartesian(vel_mag, vel_theta, vel_psi)

                    accel_mag = np.random.uniform(0.8, 1.2)
                    acceleration = spherical_to_cartesian(accel_mag, vel_theta, vel_psi)
                elif strategy == BoundarySamplingStrategy.NEW_NORMAL_VELOCITY_AND_ACCEL:
                    vel_mag = np.random.uniform(vel_bounds[0], vel_bounds[1])
                    vel_theta, vel_psi = random_vel_direction(
                        gate_idx, course, use_gate_normal_theta=True)
                    velocity = spherical_to_cartesian(vel_mag, vel_theta, vel_psi)

                    accel_mag = 0.0  # np.random.uniform(0.8, 1.2) if np.linalg.norm(previous_velocity) < 0.1 else 0.0
                    acceleration = spherical_to_cartesian(accel_mag, vel_theta, vel_psi)
                elif strategy == BoundarySamplingStrategy.USE_EXISTING_STATE:
                    current_state = copy.deepcopy(new_trajectory.states[gate_idx])
                    velocity = np.array([current_state.x.v, current_state.y.v, current_state.z.v])
                    acceleration = np.array([current_state.x.a, current_state.y.a, current_state.z.a])
                # If strategy is not defined or not met conditions of won't try
                else:
                    continue

                new_state = State3D(
                    x=State(course[gate_idx][0], velocity[0], acceleration[0]),
                    y=State(course[gate_idx][1], velocity[1], acceleration[1]),
                    z=State(course[gate_idx][2], velocity[2], acceleration[2])
                )

                new_trajectory.states.append(new_state)
                new_trajectory.segment_lengths.append(segment_length)

                boundary_conditions, segment_lengths = new_trajectory.as_boundaries()
                pos = get_positions_from_boundary_conditions(boundary_conditions, segment_lengths, 30)
                vel = derive_trajectory(pos, 30)
                acc = derive_trajectory(pos, 30, 2)

                if check_traj_direction:
                    gate_direction = R.from_euler('z', course[gate_idx][3]).as_matrix() @ INITIAL_GATE_EXIT
                else:
                    gate_direction = np.zeros(3)

                if evaluate_vel_accel_profile(vel, acc, gate_direction=gate_direction, use_curve=use_curve):
                    if not last_gate:
                        trajectories_new.append(new_trajectory)
                        continue

                    ref_pos = new_trajectory.as_ref_pos()
                    worked, _ = play_trajectory(ref_pos, use_gui=False)
                    if worked:
                        trajectories_new.append(new_trajectory)
                        break

    return trajectories_new
