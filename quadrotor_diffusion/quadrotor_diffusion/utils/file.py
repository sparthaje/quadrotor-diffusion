import os
import re
from typing import Tuple

import numpy as np


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
