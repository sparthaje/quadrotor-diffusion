import os
from typing import Union, List

import matplotlib.axes
import matplotlib.figure
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.legend_handler import HandlerBase
import numpy as np
from scipy.signal import savgol_filter

from quadrotor_diffusion.utils.dataset.boundary_condition import PolynomialTrajectory
from quadrotor_diffusion.utils.trajectory import derive_trajectory


def plot_reference_time_series(save_path: str, title: str, reference, sim_states=None):
    """
    Plots reference trajectory and optionally compares with simulated states.

    Parameters:
    - save_path: str, path where to save the plot
    - title: str, plot title
    - reference: numpy array [n x 3] where columns are [x, y, z] positions
    - sim_states: Optional, same format as reference
    """
    time = np.arange(len(reference)) * 1/30

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(title)

    dimensions = ['x', 'y', 'z']
    colors = ['blue', 'red', 'green']

    for dim_idx in range(3):
        ax = axs[dim_idx]

        ax.plot(time, reference[:, dim_idx],
                label='Reference',
                color=colors[dim_idx], linestyle='--' if sim_states is not None else '-')

        # Plot simulation states if provided
        if sim_states is not None:
            ax.plot(time, sim_states[:, dim_idx],
                    label='States in Sim',
                    color=colors[dim_idx], linestyle='-')

        ax.set_title(f'{dimensions[dim_idx]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{dimensions[dim_idx]}')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    return save_path


def view_trajectories_in_3d(save_path: str, title: str, reference, sim_states=None, smoothing_window=5):
    """
    Creates and saves a 3D visualization of reference trajectory and optionally simulation states.

    Parameters:
    - save_path: str, path where to save the plot
    - title: str, plot title
    - reference: numpy array [n x 3] containing reference trajectory points
    - sim_states: Optional numpy array [n x 3] containing simulation trajectory points
    - smoothing_window: Integer, window size for moving average smoothing (odd number recommended)
    """

    if len(reference.shape) > 2:
        reference = reference[:, 0, :]

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(reference[0, 0], reference[0, 1], reference[0, 2],
               color='red', s=20, label='Reference Start', marker='o')
    ax.scatter(reference[:, 0], reference[:, 1], reference[:, 2],
               color='blue', label='Reference Path', marker='.', alpha=0.7)

    if sim_states is not None:
        if len(sim_states.shape) > 2:
            sim_states = sim_states[:, 0, :]

        # Apply Savitzky-Golay smoothing to simulation trajectory
        smoothed_x = savgol_filter(sim_states[:, 0], smoothing_window, 3)
        smoothed_y = savgol_filter(sim_states[:, 1], smoothing_window, 3)
        smoothed_z = savgol_filter(sim_states[:, 2], smoothing_window, 3)

        ax.scatter(sim_states[0, 0], sim_states[0, 1], sim_states[0, 2],
                   color='green', s=20, label='Simulation Start', marker='o')
        ax.plot(smoothed_x, smoothed_y, smoothed_z,
                color='green', label='Simulation Path', alpha=0.7)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(title)

    ax.legend()
    ax.view_init(elev=20, azim=45)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_box_aspect([1, 1, 1])

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)

    plt.tight_layout()

    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    return save_path


def plot_loss_and_time(csv_file: str, losses: List[str]):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Get the directory of the CSV file
    dir_path = os.path.dirname(os.path.abspath(csv_file))

    plt.figure(figsize=(10, 5))
    for col in losses:
        plt.plot(df['epoch'], df[col], label=col, linewidth=3.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True)
    plt.legend()
    loss_file_path = os.path.join(dir_path, 'loss.pdf')
    plt.savefig(loss_file_path)  # Save the plot as a PDF
    plt.close()  # Close the plot to avoid overlap

    # Plot Histogram of Time
    plt.figure(figsize=(10, 5))
    plt.hist(df['time'], bins=20, color='green', edgecolor='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time Passed')
    time_file_path = os.path.join(dir_path, 'time.pdf')
    plt.savefig(time_file_path)  # Save the plot as a PDF
    plt.close()  # Close the plot to avoid overlap


def pcd_plot(trajectory: np.ndarray, file_path: str):
    """
    Plot a trajectory (set of points) to view in 3D

    Parameters:
    - trajectory: [n x 3]
    - file_path: export to either .xyz or .pcd
    """

    if file_path.endswith(".xyz"):
        np.savetxt(file_path, trajectory, fmt="%.6f")

    elif file_path.endswith(".pcd"):
        raise NotImplementedError("Haven't added PCD support yet")

    else:
        raise ValueError("Not supporting this file type")


def plot_states(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    plot_title: Union[str, None],
    filename: Union[str, None],
):
    """
    Plot all dimensions of trajectory in 3x3 figure

    Parameters:
    - pos: [n x 3]
    - vel: [n x 3]
    - acc: [n x 3]
    - plot_title: str
    - filename: str
    """

    # Labels for the rows and columns
    labels = ['Position', 'Velocity', 'Acceleration']
    dimensions = ['X', 'Y', 'Z']

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=16)

    for i in range(3):  # Rows: Position, Velocity, Acceleration
        for j in range(3):  # Columns: X, Y, Z dimensions
            ax = axes[i, j]
            if i == 0:
                data = pos[:, j]
            elif i == 1:
                data = vel[:, j]
            else:
                data = acc[:, j]

            # Plot with line and markers
            ax.plot(data, marker='.', linestyle='-', label=f'{dimensions[j]}')
            ax.set_title(f'{labels[i]} - {dimensions[j]}')
            ax.grid(True)
            ax.legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plot_ref_obs_states(
    ref_pos: np.ndarray,
    ref_vel: np.ndarray,
    ref_acc: np.ndarray,
    obs_pos: np.ndarray,
    obs_vel: np.ndarray,
    obs_acc: np.ndarray,
    plot_title: Union[str, None],
    filename: Union[str, None],
):
    """
    Plot all dimensions of trajectory in 3x3 figure with reference and observed data.

    Parameters:
    - ref_pos: [n x 3] Reference positions
    - ref_vel: [n x 3] Reference velocities
    - ref_acc: [n x 3] Reference accelerations
    - obs_pos: [n x 3] Observed positions
    - obs_vel: [n x 3] Observed velocities
    - obs_acc: [n x 3] Observed accelerations
    - plot_title: str, Title of the plot
    - filename: str, Path to save the plot
    """

    # Labels for the rows and columns
    labels = ['Position', 'Velocity', 'Acceleration']
    dimensions = ['X', 'Y', 'Z']

    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=16)

    for i in range(3):  # Rows: Position, Velocity, Acceleration
        for j in range(3):  # Columns: X, Y, Z dimensions
            ax = axes[i, j]
            if i == 0:
                ref_data, obs_data = ref_pos[:, j], obs_pos[:, j]
            elif i == 1:
                ref_data, obs_data = ref_vel[:, j], obs_vel[:, j]
            else:
                ref_data, obs_data = ref_acc[:, j], obs_acc[:, j]

            # Plot reference and observed data
            ax.plot(ref_data, marker='.', linestyle='-',
                    label=f'Ref {dimensions[j]}', alpha=0.7, linewidth=1, markersize=4)
            ax.plot(obs_data, marker='.', linestyle='--',
                    label=f'Obs {dimensions[j]}', alpha=0.7, linewidth=1, markersize=4)

            ax.set_title(f'{labels[i]} - {dimensions[j]}')
            ax.grid(True)
            ax.legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def course_base_plot() -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Creates base plot for course

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(-1.5, 1.6, 1))
    ax.set_yticks(np.arange(-2, 2.1, 1))
    ax.grid(which='both', linestyle='--', linewidth=0.5)

    return fig, ax


class GateLegendHandler(HandlerBase):
    def __init__(self, rect_color, left_circle_color, right_circle_color):
        super().__init__()
        self.rect_color = rect_color
        self.left_circle_color = left_circle_color
        self.right_circle_color = right_circle_color

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        rect = patches.Rectangle(
            [xdescent, ydescent + height / 4],
            width / 2,
            height / 1.5,
            edgecolor='black',
            facecolor=self.rect_color,
            transform=trans,
        )
        left_circle = patches.Circle(
            [xdescent - width / 8, ydescent + height / 2],
            width / 10,
            edgecolor='black',
            facecolor=self.left_circle_color,
            transform=trans,
        )
        right_circle = patches.Circle(
            [xdescent + width / 2 + width / 8, ydescent + height / 2],
            width / 10,
            edgecolor='black',
            facecolor=self.right_circle_color,
            transform=trans,
        )
        return [rect, left_circle, right_circle]


def add_gates_to_course(course: list[np.array], ax: plt.Axes):
    """
    Adds a list of gates to the plot

    Args:
        course (list[np.array]): Full course including the starting and ending point
        ax (plt.Axes)
    """
    for obj in course[1:-1]:
        x, y, z, yaw = obj
        rect = patches.Rectangle((-0.25, -0.05), 0.5, 0.1, edgecolor='black', facecolor='white', alpha=0.8)
        t = plt.matplotlib.transforms.Affine2D()
        t.rotate(yaw)
        t.translate(x, y)
        rect.set_transform(t + ax.transData)
        ax.add_patch(rect)

        circle_color = 'black' if z == 0.525 else 'white'
        left_circle = patches.Circle((-0.25 - 0.025, 0), 0.025, edgecolor='black',
                                     facecolor=circle_color, transform=t + ax.transData)
        right_circle = patches.Circle((0.25 + 0.025, 0), 0.025, edgecolor='black',
                                      facecolor=circle_color, transform=t + ax.transData)
        ax.add_patch(left_circle)
        ax.add_patch(right_circle)

        # arrow = patches.FancyArrow(0, 0, 0, 0.05, width=0.02, head_width=0.05,
        #                            head_length=0.05, color='black', transform=t + ax.transData)
        # ax.add_patch(arrow)

    starting_face_color = 'black' if course[0][2] == 0.525 else 'white'
    ending_face_color = 'black' if course[-1][2] == 0.525 else 'white'
    ax.scatter([course[0][0]], [course[0][1]], s=100, marker='*', facecolor=starting_face_color, edgecolor='black')
    ax.scatter([course[-1][0]], [course[-1][1]], s=100, marker='o', facecolor=ending_face_color, edgecolor='black')

    high_patch = patches.Patch(color='none', label="High Gate (0.525 m)")
    low_patch = patches.Patch(color='none', label="Low Gate (0.3 m)")

    ax.legend(
        handles=[high_patch, low_patch],
        handler_map={
            high_patch: GateLegendHandler('white', 'black', 'black'),
            low_patch: GateLegendHandler('white', 'white', 'white'),
        },
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=2,
        borderpad=0.7,
    )


def get_render_map(ref_vel: np.ndarray, max_point_size: float, min_point_size: float) -> tuple[list[float], list[str]]:
    """
    Returns point sizes and colors for trajectory rendering (2d/3d)

    Args:
        ref_vel (np.ndarray): Velocity profile
        max_point_size (float): Max point size for full velocity
        min_point_size (float): Min point size for zero velocity

    Returns:
        tuple[list[float], list[str]]: point_sizes, colors
    """
    MAX_VEL = 2.0  # m/s

    vel_mag = np.linalg.norm(ref_vel, axis=1)
    point_sizes = (max_point_size - min_point_size) * (vel_mag / MAX_VEL) + min_point_size

    timestamps = np.linspace(0, 1, ref_vel.shape[0])
    primary_colors = [matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][0],
                      matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][1]]
    colormap = LinearSegmentedColormap.from_list('custom_map', primary_colors)
    norm = Normalize(vmin=0, vmax=1)

    return point_sizes, colormap(norm(timestamps))


def add_trajectory_to_course(trajectory: Union[PolynomialTrajectory, np.ndarray], velocity_profile=None, reference=False):
    """
    Add trajectory to a BEV gate plot with plt

    Args:
        trajectory (Union[PolynomialTrajectory, np.ndarray]): Trajectory to pot
        velocity_profile (_type_, optional): Velocity profile for trajectory. Defaults to deriving the given trajectory.
        reference (bool, optional): Draw the trajectory as a red reference line. Defaults to False.
    """

    if isinstance(trajectory, PolynomialTrajectory):
        ref_pos = trajectory.as_ref_pos()
    else:
        ref_pos = trajectory
    ref_vel = derive_trajectory(ref_pos, ctrl_freq=30) if velocity_profile is None else velocity_profile

    # This code shows where a trajectory violates GV limits
    # ref_acc = derive_trajectory(ref_vel, ctrl_freq=30)
    # colors = []
    # for v, a in zip(ref_vel, ref_acc):
    #     limit = -0.3 * (np.linalg.norm(v) ** 3) + 2.0
    #     acc_towards_vel = np.dot(v, a) > 0

    #     if np.linalg.norm(a) > limit and acc_towards_vel:
    #         colors.append('red')
    #         continue

    #     colors.append('blue')

    MIN_POINT_SIZE = 5
    MAX_POINT_SIZE = 125
    point_sizes, colors = get_render_map(ref_vel, MAX_POINT_SIZE, MIN_POINT_SIZE)

    if reference:
        plt.plot(ref_pos[:, 0], ref_pos[:, 1], c='red', linewidth=2)
    else:
        plt.scatter(ref_pos[:, 0], ref_pos[:, 1], s=point_sizes, c=colors, marker='o', alpha=0.8)
