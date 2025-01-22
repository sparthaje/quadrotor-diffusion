from dataclasses import dataclass, field

import numpy as np

from quadrotor_diffusion.utils.trajectory import get_positions_from_boundary_conditions


@dataclass
class State:
    """
    State at one dimension
    s: pos
    v: vel
    a: acc
    j: jerk
    """
    s: float = 0
    v: float = 0
    a: float = 0
    j: float = 0


@dataclass
class State3D:
    """
    Set of three states
    """
    x: State = State()
    y: State = State()
    z: State = State()


@dataclass
class PolynomialTrajectory:
    """
    Set of states and time interval between each state
    """
    states: list[State3D] = field(default_factory=list)
    segment_lengths: list[float] = field(default_factory=list)

    def as_boundaries(self) -> tuple[list[list[np.array]], list[float]]:
        """
        Converts data class into appropriate numpy array formats for trajectory creation

        Returns:
            tuple[list[list[np.array]], list[float]]: Boundary Conditions, Segment Lengths
        """

        def state_1d_to_numpy(state_0: State, state_f: State):
            return np.array([state_0.s, state_f.s, state_0.v, state_f.v, state_0.a, state_f.a, state_0.j, state_f.j])

        def group_to_array(group: tuple[State3D, State3D]):
            state_0, state_f = group
            return [
                state_1d_to_numpy(state_0.x, state_f.x),
                state_1d_to_numpy(state_0.y, state_f.y),
                state_1d_to_numpy(state_0.z, state_f.z)
            ]

        grouped_states = [(self.states[i], self.states[i + 1]) for i in range(len(self.states) - 1)]
        boundary_conditions = [group_to_array(group) for group in grouped_states]

        return boundary_conditions, self.segment_lengths

    def as_ref_pos(self, ctrl_freq=30) -> np.ndarray:
        """
        Returns ref_pos for trajectory
        - ctrl_freq: Points per second

        Returns:
            np.ndarray: (nx3) array of positions xyz where time step is 1/ctrl_freq
        """
        return get_positions_from_boundary_conditions(*self.as_boundaries(), ctrl_freq)

    def __str__(self):
        return f"{sum(self.segment_lengths):.2f}"
