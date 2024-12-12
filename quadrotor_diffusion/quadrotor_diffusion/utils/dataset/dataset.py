import os

import numpy as np
import torch
from torch.utils.data import Dataset

from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.trajectory import derive_trajectory


class QuadrotorTrajectoryDataset(Dataset):
    def __init__(self, data_dir, normalizer: Normalizer, order: int = 0):
        self.data_dir = data_dir
        self.length = len([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.normalizer = normalizer
        self.order = order

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, f"{idx}.npy")
        data = np.load(filepath, allow_pickle=True)

        # Horizon should be divisible by 2^(channel_mults - 1) in unet
        data = data[:336, :]
        data = derive_trajectory(data, 30, order=self.order)
        data = self.normalizer(data)

        data = torch.tensor(data).float()  # [n x 3]
        return data

    def __str__(self):
        return "\n".join([
            "Quadrotor Trajectory Dataset: ",
            f"\torder={self.order}"
        ])


class QuadrotorFullStateDataset(Dataset):
    def __init__(self, data_dir, normalizer: Normalizer):
        self.data_dir = data_dir
        self.length = len([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.normalizer = normalizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, f"{idx}.npy")
        pos = np.load(filepath, allow_pickle=True)

        # Horizon should be divisible by 2^(channel_mults - 1) in unet
        pos = pos[:336, :]
        vel = derive_trajectory(pos, 30)
        acc = derive_trajectory(vel, 30)

        gap = np.zeros((pos.shape[0], 1))
        data = np.hstack((pos, gap, vel, gap, acc))
        data = self.normalizer(data)

        data = torch.tensor(data).float()  # [n x 11]
        return data


def evaluate_dataset(dataset: Dataset):
    """
    Get key stats about a dataset

    Returns:
    - mean, variance, min max
    """
    # Collect all data first
    all_data = [dataset[x].numpy() for x in range(len(dataset))]
    data_array = np.concatenate(all_data, axis=0)

    # Calculate statistics
    mean = np.mean(data_array, axis=0)
    variance = np.var(data_array, axis=0)
    min_values = np.min(data_array, axis=0)
    max_values = np.max(data_array, axis=0)

    return mean, variance, min_values, max_values
