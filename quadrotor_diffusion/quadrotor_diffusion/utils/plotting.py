import os
from typing import Union, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


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
        plt.plot(df['epoch'], df[col], label=col)
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
    filename: str,
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
        fig.suptitle('plot_title', fontsize=16)

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
    plt.savefig(filename)


def plot_ref_obs_states(
    ref_pos: np.ndarray,
    ref_vel: np.ndarray,
    ref_acc: np.ndarray,
    obs_pos: np.ndarray,
    obs_vel: np.ndarray,
    obs_acc: np.ndarray,
    plot_title: Union[str, None],
    filename: str,
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
    plt.savefig(filename)
