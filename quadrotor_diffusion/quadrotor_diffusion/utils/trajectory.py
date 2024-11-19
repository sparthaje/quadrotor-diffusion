import numpy as np

from collections import deque
from enum import Enum

INITIAL_GATE_EXIT = np.array([0, 1, 0])


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


def get_positions_from_boundary_conditions(boundaries, segment_lengths, CTRL_FREQ):
    """_summary_

    Args:
        boundaries (np.ndarray): np.array[][]
        segment_length: float[]
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


def derive_trajectory(data: np.array, ctrl_freq: int, order: int = 2):
    """
    Calculate the derivative of trajectory data with respect to time.
    This function can be used to calculate velocities from positions,
    accelerations from velocities, jerk from accelerations, etc.

    Parameters:
    - data: np.array
    - ctrl_freq: Number of states per second
    - order: Order of derivatives
    """

    if order == 1:
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


def smooth_columns(arr, window_size=5, threshold=0.5):
    """
    Smooths columns in a numpy array by detecting and handling outliers.

    Parameters:
    arr : numpy.ndarray
        Input array where each column will be smoothed
    window_size : int
        Size of the rolling window (default: 5)
    threshold : float
        Z-score threshold for outlier detection (default: 2)

    Returns:
    numpy.ndarray
        Smoothed array with same shape as input
    """
    # Make a copy to avoid modifying the original array
    smoothed = arr.copy()

    # Process each column
    for col in range(arr.shape[1]):
        data = arr[:, col]

        # Calculate rolling mean and std
        rolling_mean = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        rolling_std = np.array([np.std(data[max(0, i-window_size//2):min(len(data), i+window_size//2+1)])
                                for i in range(len(data))])

        # Calculate z-scores
        z_scores = np.abs((data - rolling_mean) / (rolling_std + 1e-10))  # Add small value to avoid division by zero

        # Identify outliers
        outliers = z_scores > threshold

        # Replace outliers with interpolated values
        if np.any(outliers):
            # Create indices for valid points
            valid_indices = np.where(~outliers)[0]
            outlier_indices = np.where(outliers)[0]

            # Interpolate outliers using neighboring points
            smoothed[outlier_indices, col] = np.interp(
                outlier_indices,
                valid_indices,
                data[valid_indices]
            )

        # Apply final smoothing
        smoothed[:, col] = np.convolve(smoothed[:, col],
                                       np.ones(window_size)/window_size,
                                       mode='same')

    return smoothed
