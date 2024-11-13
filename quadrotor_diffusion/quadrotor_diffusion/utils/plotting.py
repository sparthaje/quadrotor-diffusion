import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import os


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


def plot_loss_and_time(csv_file):
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Get the directory of the CSV file
    dir_path = os.path.dirname(os.path.abspath(csv_file))

    # Plot Loss vs Epoch (Line Plot)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['loss'], label='Loss', color='blue')
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
