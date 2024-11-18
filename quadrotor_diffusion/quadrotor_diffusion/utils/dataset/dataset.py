import os

import numpy as np
import torch
from torch.utils.data import Dataset

from quadrotor_diffusion.utils.dataset.normalizer import Normalizer
from quadrotor_diffusion.utils.trajectory import derive_target_velocities, derive_target_accelerations


class QuadrotorTrajectoryDataset(Dataset):
    def __init__(self, data_dir, normalizer: Normalizer):
        self.data_dir = data_dir
        self.length = len([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.normalizer = normalizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, f"{idx}.npy")
        data = np.load(filepath, allow_pickle=True)

        # Horizon should be divisible by 2^(channel_mults - 1) in unet
        data = data[:336, :]
        data = self.normalizer(data)

        data = torch.tensor(data).float()  # [n x 3]
        return data


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
        vel = derive_target_velocities(pos, 30)
        acc = derive_target_accelerations(vel, 30)

        data = np.hstack((pos, vel, acc))
        data = self.normalizer(data)

        data = torch.tensor(data).float()  # [n x 6]
        return data
