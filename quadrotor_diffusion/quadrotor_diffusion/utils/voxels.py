import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance

from quadrotor_diffusion.utils.trajectory import INITIAL_GATE_EXIT


def create_occupancy_map(course: list[np.array], voxel_size: float) -> np.ndarray:
    """
    Creates an occupancy map based on the gates

    Args:
        course (list[np.array]): List of [x, y, z, theta] for each gate
        voxel_size (float): Size of each voxel side

    Returns:
        np.ndarray: Occupancy Map
    """
    x_range = np.arange(-1.5, 1.5 + voxel_size, voxel_size)
    y_range = np.arange(-2, 2 + voxel_size, voxel_size)
    z_range = np.arange(0, 1 + voxel_size, voxel_size)
    occupancy_map = np.zeros((len(x_range), len(y_range), len(z_range)), dtype=bool)

    for x, y, z, theta in course[1:-1]:
        GATE_WIDTH_2 = 0.5 / 2
        center = np.array([x, y, z])
        left = R.from_euler('z', np.pi / 2).as_matrix() @ R.from_euler('z', theta).as_matrix() @ INITIAL_GATE_EXIT
        top = np.array([0, 0, 1])

        rectangle_vertices = [
            center + GATE_WIDTH_2 * left + GATE_WIDTH_2 * top,
            center + GATE_WIDTH_2 * left - GATE_WIDTH_2 * top,
            center - GATE_WIDTH_2 * left - GATE_WIDTH_2 * top,
            center - GATE_WIDTH_2 * left + GATE_WIDTH_2 * top,
            center + GATE_WIDTH_2 * left + GATE_WIDTH_2 * top,
        ]
        rectangle_vertices = np.array(rectangle_vertices)

        for i in range(len(rectangle_vertices) - 1):
            fill_line_voxels(occupancy_map, rectangle_vertices[i], rectangle_vertices[i + 1], voxel_size)

        post = [
            center - GATE_WIDTH_2 * top,
            center,
        ]
        post[-1][2] = 0.0
        post = np.array(post)
        fill_line_voxels(occupancy_map, post[0], post[1], voxel_size)

    return occupancy_map


def real_to_voxel(x: float, y: float, z: float, voxel_size: float) -> tuple[int, int, int]:
    """
    Converts real position to indicies
    """
    idx_x = int((x + 1.5) / voxel_size)
    idx_y = int((y + 2) / voxel_size)
    idx_z = int(z / voxel_size)
    idx_x = max(0, min(idx_x, int(3 / voxel_size) - 1))  # Clamp to valid range
    idx_y = max(0, min(idx_y, int(4 / voxel_size) - 1))  # Clamp to valid range
    idx_z = max(0, min(idx_z, int(1 / voxel_size) - 1))  # Clamp to valid range
    return idx_x, idx_y, idx_z


def fill_line_voxels(occupancy_map: np.ndarray, start: np.array, end: np.array, voxel_size: float):
    """
    Fills voxels along a line
    """
    line_length = distance.euclidean(start, end)
    steps = int(np.ceil(line_length / voxel_size))
    for i in range(steps + 1):
        point = start + i / steps * (end - start)
        idx_x, idx_y, idx_z = real_to_voxel(point[0], point[1], point[2], voxel_size)
        if 0 <= idx_x < occupancy_map.shape[0] and \
           0 <= idx_y < occupancy_map.shape[1] and \
           0 <= idx_z < occupancy_map.shape[2]:
            occupancy_map[idx_x, idx_y, idx_z] = True


def voxel_to_real(idx_x: int, idx_y: int, idx_z: int, voxel_size: float) -> tuple[float, float, float]:
    """
    Returns center of voxel in real world coordinates
    """
    x = idx_x * voxel_size - 1.5 + voxel_size / 2
    y = idx_y * voxel_size - 2 + voxel_size / 2
    z = idx_z * voxel_size + voxel_size / 2
    return x, y, z


def distance_to_nearest_obstacle(occupancy_map: np.ndarray, x: float, y: float, z: float, voxel_size: float) -> float:
    """
    Returns distance to the nearest occupied voxel's center from the given position
    """
    idx_x, idx_y, idx_z = real_to_voxel(x, y, z, voxel_size)
    occupied_voxels = np.argwhere(occupancy_map)
    if len(occupied_voxels) == 0:
        return float('inf')

    distances = np.linalg.norm(occupied_voxels * voxel_size - np.array([idx_x, idx_y, idx_z]) * voxel_size, axis=1)
    nearest_voxel_idx = np.argmin(distances)
    nearest_distance = distances[nearest_voxel_idx]

    return nearest_distance


def visualize_occupancy_map(occupancy_map: np.ndarray, voxel_size: float):
    """
    Shows the voxel map as a scatter plot in 3d
    """
    x, y, z = np.where(occupancy_map)
    x_real = x * voxel_size - 1.5 + voxel_size / 2
    y_real = y * voxel_size - 2 + voxel_size / 2
    z_real = z * voxel_size + voxel_size / 2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_real, y_real, z_real, c='blue', alpha=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 1)

    # Set equal aspect ratios for all axes
    ax.set_box_aspect([3, 4, 1])  # Match the physical size ratios (x, y, z)

    plt.show()


def collision_along_trajectory(ref_pos: np.ndarray, occupancy_map: np.ndarray, voxel_size: float) -> float:
    """
    Returns True if ref_pos is too close to an occupied voxel
    """
    DRONE_SIZE = 0.1
    SAFETY_MARGIN = 0.1

    for row in ref_pos:
        min_distance = distance_to_nearest_obstacle(occupancy_map, *row, voxel_size)
        if min_distance < DRONE_SIZE / 2 + SAFETY_MARGIN:
            return True

    return False
