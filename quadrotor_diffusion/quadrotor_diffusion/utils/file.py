import os
import re
from typing import Tuple
import pickle

import numpy as np

from quadrotor_diffusion.utils.dataset.boundary_condition import PolynomialTrajectory


def get_experiment_folder(train_dir: str, experiment_number: int):
    """
    Get folder where all experiment data is stored (ie checkpoints, logs, etc)

    Parameters: 
    - train_dir: log directory where all training data is (e.g. logs/training)
    - experiment_nubmer: first number in the experiment folder number
    """

    folders = os.listdir(train_dir)
    training_folder = next((folder for folder in folders if folder.startswith(f"{experiment_number}.")), None)
    if training_folder is None:
        raise NameError(f"No folder found for experiment {experiment_number}")

    return training_folder


def get_checkpoint_file(train_dir: str, experiment_number: int, epoch=None):
    """
    Returns path to the checkpoint file

    Parameters: 
    - train_dir: log directory where all training data is (e.g. logs/training)
    - experiment_nubmer: first number in the experiment folder number
    - epoch: Optional, don't ues the latest epoch and specify a specific checkpoint
    """

    training_folder = get_experiment_folder(train_dir, experiment_number)
    checkpoint_folder = os.path.join(train_dir, training_folder, "checkpoints")
    pattern = r'epoch_(\d+)_loss_([\d.]+)'
    checkpoints = {int(re.match(pattern, file).group(1)): os.path.join(checkpoint_folder, file)
                   for file in os.listdir(checkpoint_folder) if re.match(pattern, file)}

    if not checkpoints:
        raise ValueError("No epoch files found.")

    if epoch is None:
        return checkpoints[max(checkpoints)]
    if epoch in checkpoints:
        return checkpoints[epoch]
    raise ValueError(f"Epoch {epoch} not found.")


def save_trajectory(filepath: str, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray):
    """
    Saves a trajectory (set of commands) to a .trajectory.npy file
    """
    assert filepath.endswith(".trajectory.npy") or filepath.endswith(".trajectory")
    traj = np.array([pos, vel, acc])
    np.save(filepath, traj)


def load_trajectory(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a .trajectory.npy file into numpy arrays for pos, vel, and accl
    """

    assert filepath.endswith(".trajectory.npy")
    traj = np.load(filepath)

    return traj[0], traj[1], traj[2]


def load_course_trajectory(course_type: str, course_number: int, sample_number: int) -> tuple[list[np.array], PolynomialTrajectory, str]:
    """
    Loads trajectory sample

    Args:
        course_type (str): course type: linear/u
        course_number (int): Course number
        sample_number (int): Sample number in valid folder

    Returns:
        tuple[list[np.array], PolynomialTrajectory, str]: course, sample trajectory, trajectory filename
    """

    base_dir = "data/courses"
    base_dir = os.path.join(base_dir, course_type, str(course_number), "valid")
    all_filenames = os.listdir(base_dir)
    filename = [f for f in all_filenames if f.startswith(f"{sample_number}_")][0]
    filename = os.path.join(base_dir, filename)

    with open(filename, "rb") as file:
        trajectory: PolynomialTrajectory = pickle.load(file)

    base_dir = "data/courses"
    course_filename = os.path.join(base_dir, course_type, str(course_number), "course.npy")
    course = np.load(course_filename)

    return course, trajectory, filename
