import os

import numpy as np
import torch
from torch.utils.data import Dataset

from quadrotor_diffusion.utils.dataset.normalizer import Normalizer


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
