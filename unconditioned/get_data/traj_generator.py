import numpy as np

from collections import deque
from enum import Enum

from rich import print

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

REF_YAW_RATIO = 0.25

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

def derive_target_velocities(positions, CTRL_FREQ):
  # Time step between velocity measurements (in seconds)
  delta_t = 1/CTRL_FREQ
  
  # Compute the difference between consecutive positions for each dimension
  delta_x = np.diff(positions[:, 0])
  delta_y = np.diff(positions[:, 1])
  delta_z = np.diff(positions[:, 2])
  
  # Calculate velocity for each dimension: v = delta_position / delta_t
  vel_x = delta_x / delta_t
  vel_y = delta_y / delta_t
  vel_z = delta_z / delta_t
  
  # Combine the velocities into a single array
  velocities = np.column_stack((vel_x, vel_y, vel_z))
  
  # Add a row of zeros at the beginning to match the shape of the positions array
  velocities = np.row_stack((np.array([0, 0, 0]), velocities))
  
  return velocities

def derive_target_accelerations(velocities, CTRL_FREQ):
  # Time step between velocity measurements (in seconds)
  delta_t = 1/CTRL_FREQ
  
  # Compute the difference between consecutive velocities for each dimension
  delta_vx = np.diff(velocities[:, 0])
  delta_vy = np.diff(velocities[:, 1])
  delta_vz = np.diff(velocities[:, 2])
  
  # Calculate acceleration for each dimension: a = delta_velocity / delta_t
  accelerations_x = delta_vx / delta_t
  accelerations_y = delta_vy / delta_t
  accelerations_z = delta_vz / delta_t
  
  # Combine the accelerations into a single array
  accelerations = np.column_stack((accelerations_x, accelerations_y, accelerations_z))
  
  # Add a row of zeros at the beginning to match the shape of the velocities array
  accelerations = np.row_stack((np.array([0, 0, 0]), accelerations))
  
  return accelerations
