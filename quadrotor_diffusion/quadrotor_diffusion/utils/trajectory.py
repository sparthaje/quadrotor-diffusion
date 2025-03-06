import numpy as np

INITIAL_GATE_EXIT = np.array([0, 1, 0])


def spherical_to_cartesian(magnitude: float, theta: float, psi: float) -> np.array:
    """
    Converts in gym coordinates a spherical coordinate to xyz
    """
    x = magnitude * np.cos(psi) * np.sin(-theta)
    y = magnitude * np.cos(psi) * np.cos(-theta)
    z = magnitude * np.sin(psi)
    return np.array([x, y, z])


def compute_min_snap_segment(sigma, T):
    M = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1],
        [T**7, T**6, T**5, T**4, T**3, T**2, T**1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [7*T**6, 6*T**5, 5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
        [0, 0, 0, 0, 0, 2, 0, 0],
        [42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0],
        [0, 0, 0, 0, 6, 0, 0, 0],
        [210*T**4, 120*T**3, 60*T**2, 24*T, 6, 0, 0, 0],
    ])

    return np.linalg.inv(M) @ sigma


def get_positions_from_boundary_conditions(boundaries, segment_lengths, CTRL_FREQ) -> np.ndarray:
    """Generates a nx3 set of positions to track

    Args:
        boundaries: boundary condition to generate from (num_segments x dim(3) x boundary_condition(8))
        segment_length: time to complete each segment (num_segments)
    """
    ref_pos = []

    for segment, segment_length in zip(boundaries, segment_lengths):
        positions = []
        for sigma in segment:
            coeffs = compute_min_snap_segment(sigma, segment_length)
            t = np.linspace(0, segment_length, int(segment_length * CTRL_FREQ))
            seg_xyz = np.polyval(coeffs, t)[1:]
            positions.append(seg_xyz)

        # Stack all segments and transpose
        positions = np.vstack(positions).T
        ref_pos.append(positions)

    ref_pos = np.vstack(ref_pos)
    return ref_pos


def derive_trajectory(data: np.array, ctrl_freq: int, order: int = 1):
    """
    Calculate the derivative of trajectory data with respect to time.
    This function can be used to calculate velocities from positions,
    accelerations from velocities, jerk from accelerations, etc.

    Parameters:
    - data: np.array
    - ctrl_freq: Number of states per second
    - order: Order of derivatives
    """

    if order == 0:
        return data

    # Time step between measurements (in seconds)
    delta_t = 1/ctrl_freq
    deltas = np.diff(data, axis=0)
    derivatives = deltas / delta_t
    derivatives = np.row_stack((np.zeros(data.shape[1]), derivatives))

    return derive_trajectory(derivatives, ctrl_freq, order - 1)


def integrate_trajectory(data: np.ndarray, ctrl_freq: int, initial_conditions=None):
    """
    Integrate trajectory data with respect to time.
    This function can be used to calculate positions from velocities,
    velocities from accelerations, etc.

    Parameters:
    - data : numpy.ndarray [n x m]
    - ctrl_freq : float
    - initial_conditions : numpy.ndarray, optional, Initial values for the integral, shape should be (m,)
    """
    delta_t = 1/ctrl_freq

    # Set initial conditions to zeros if not provided
    if initial_conditions is None:
        initial_conditions = np.zeros(data.shape[1])
    else:
        initial_conditions = np.array(initial_conditions)
        if initial_conditions.shape != (data.shape[1],):
            raise ValueError(f"Initial conditions shape {initial_conditions.shape} "
                             f"does not match data dimensions {(data.shape[1],)}")

    integrated = np.zeros_like(data)
    integrated[0] = initial_conditions

    for i in range(1, len(data)):
        increment = (data[i] + data[i-1]) * delta_t / 2
        integrated[i] = integrated[i-1] + increment

    return integrated


def compute_tracking_error(ref_pos: np.ndarray, pos: np.ndarray):
    """
    Compute average tracking error per command

    Parameters:
    - ref_pos: [n x 3] Commanded states
    - pos: [n x 3] Actual states in simulation

    Returns:
    - tracking_error: [1 x 3] Average tracking error per state dimension
    """

    error = ref_pos - pos
    tracking_error = np.mean(np.abs(error), axis=0)

    return tracking_error


def has_knot(trajectory):
    n = len(trajectory)
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if line_segments_intersect(trajectory[i], trajectory[i + 1], trajectory[j], trajectory[j + 1]):
                return True
    return False


def line_segments_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


def evaluate_vel_accel_profile(vel: np.ndarray, acc: np.ndarray, gate_direction: np.array = np.zeros(3), use_curve=True) -> bool:
    """
    Returns True if velocity and acceleration values are reasonable based on IRL testing

    Args:
        vel (np.ndarray): Velocity profile (n x 3)
        acc (np.ndarray): Acceleration profile (n x 3)
        gate_direction (np.array): Vector in direction of gate to evaluate if moving in right direction
                                   If not provided, won't check if drone is moving in right direction

    Returns:
        bool: GV constraints met or not
    """
    for v, a in zip(vel, acc):
        if use_curve:
            limit = -0.3 * (np.linalg.norm(v) ** 3) + 2.0
        else:
            limit = 2.0
        acc_towards_vel = np.dot(v, a) > 0
        vel_backwards = np.dot(v, gate_direction) < 0.0

        if np.linalg.norm(a) > limit and acc_towards_vel:
            return False

        if vel_backwards:
            return False

    return True
