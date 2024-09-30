import os

import numpy as np
import torch
from torch.utils.data import Dataset

class UnconditionedDataset(Dataset):
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self.length = len([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    
  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    filepath = os.path.join(self.data_dir, f"{idx}.npy")
    data = np.load(filepath, allow_pickle=True)
    # data ranges from 353-355 points, so we cap it at 348 (so its divisible by 2*3 and 3)
    data = data[:336, :]
    data = torch.tensor(data).float()
    data = data.unsqueeze(1)
    data = data.permute(1, 0, 2)
    return data
  
