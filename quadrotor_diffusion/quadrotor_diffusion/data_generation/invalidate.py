import enum
import copy

import numpy as np
import random

from quadrotor_diffusion.utils.dataset.boundary_condition import PolynomialTrajectory


class InvalidationStrategy(enum.Enum):
    CHANGE_ONE_HEIGHT = 0
    CHANGE_TWO_HEIGHT = 1
    SHIFT_ONE_GATE_XY = 2
    SHIFT_TWO_GATE_XY = 3
    SHIFT_COURSE = 4
    SWAP_TWO_GATES = 5
    REVERSE = 6
    RANDOMIZE = 7
    SKIP_GATE = 8


def invalidate_trajectory(trajectory: PolynomialTrajectory) -> tuple[np.ndarray, str]:
    """
    Takes a trajectory that traverses a course and creates one that doesn't

    Args:
    - trajectory (PolynomialTrajectory): Valid trajectory

    Returns:
    - tuple[np.ndarray, str]: invalid trajectory, invalidation strategy
    """
    LOWER_LATERAL_SHIFT = 0.25
    UPPER_LATERAL_SHIFT = 0.5

    strategy = random.choice(list(InvalidationStrategy))
    invalid_trajectory = copy.deepcopy(trajectory)

    # Swap one of the heights for a gate
    if strategy == InvalidationStrategy.CHANGE_ONE_HEIGHT:
        gate = random.randint(1, 4)
        current_z = invalid_trajectory.states[gate].z
        new_z = 0.3 if current_z == 0.525 else 0.525
        invalid_trajectory.states[gate].z.s = new_z
        return invalid_trajectory.as_ref_pos(), InvalidationStrategy.CHANGE_ONE_HEIGHT.name

    # Swap up to two of the heights for a gate
    elif strategy == InvalidationStrategy.CHANGE_TWO_HEIGHT:
        for _ in range(2):
            gate = random.randint(1, 4)
            current_z = invalid_trajectory.states[gate].z
            new_z = 0.3 if current_z == 0.525 else 0.525
            invalid_trajectory.states[gate].z.s = new_z
        return invalid_trajectory.as_ref_pos(), InvalidationStrategy.CHANGE_TWO_HEIGHT.name

    # Move a single gate laterally
    elif strategy == InvalidationStrategy.SHIFT_ONE_GATE_XY:
        gate = random.randint(1, 4)
        invalid_trajectory.states[gate].x.s += random.choice([-1, 1]) * \
            random.uniform(LOWER_LATERAL_SHIFT, UPPER_LATERAL_SHIFT)
        invalid_trajectory.states[gate].y.s += random.choice([-1, 1]) * \
            random.uniform(LOWER_LATERAL_SHIFT, UPPER_LATERAL_SHIFT)
        return invalid_trajectory.as_ref_pos(), InvalidationStrategy.SHIFT_ONE_GATE_XY.name

    # Move up to two gates laterally
    elif strategy == InvalidationStrategy.SHIFT_TWO_GATE_XY:
        for _ in range(2):
            gate = random.randint(1, 4)
            invalid_trajectory.states[gate].x.s += random.choice([-1, 1]) * \
                random.uniform(LOWER_LATERAL_SHIFT, UPPER_LATERAL_SHIFT)
            invalid_trajectory.states[gate].y.s += random.choice([-1, 1]) * \
                random.uniform(LOWER_LATERAL_SHIFT, UPPER_LATERAL_SHIFT)
        return invalid_trajectory.as_ref_pos(), InvalidationStrategy.SHIFT_TWO_GATE_XY.name

    # Move all gates by the same lateral shift
    elif strategy == InvalidationStrategy.SHIFT_COURSE:
        x_shift = random.choice([-1, 1]) * random.uniform(0.5 * LOWER_LATERAL_SHIFT, 0.5 * UPPER_LATERAL_SHIFT)
        y_shift = random.choice([-1, 1]) * random.uniform(0.5 * LOWER_LATERAL_SHIFT, 0.5 * UPPER_LATERAL_SHIFT)
        ref_pos = invalid_trajectory.as_ref_pos()
        ref_pos[:, 0] += x_shift
        ref_pos[:, 1] += y_shift
        return ref_pos, InvalidationStrategy.SHIFT_COURSE.name

    # Swap two gates so theres a double back
    elif strategy == InvalidationStrategy.SWAP_TWO_GATES:
        gate = random.randint(1, 3)
        invalid_trajectory.states[gate], invalid_trajectory.states[gate + 1] = \
            invalid_trajectory.states[gate + 1], invalid_trajectory.states[gate]
        return invalid_trajectory.as_ref_pos(), InvalidationStrategy.SWAP_TWO_GATES.name

    # Reverse the states
    elif strategy == InvalidationStrategy.REVERSE:
        ref_pos = invalid_trajectory.as_ref_pos()
        ref_pos = ref_pos[::-1]
        return ref_pos, InvalidationStrategy.REVERSE.name

    elif strategy == InvalidationStrategy.RANDOMIZE:
        random.shuffle(invalid_trajectory.states)
        return invalid_trajectory.as_ref_pos(), InvalidationStrategy.RANDOMIZE.name

    elif strategy == InvalidationStrategy.SKIP_GATE:
        gate = random.randint(1, 4)
        invalid_trajectory.states.pop(gate)
        invalid_trajectory.segment_lengths[gate - 1] = invalid_trajectory.segment_lengths[gate] + 2
        invalid_trajectory.segment_lengths.pop(gate - 1)
        return invalid_trajectory.as_ref_pos(), InvalidationStrategy.SKIP_GATE.name

    return np.array([]), "NONE"
