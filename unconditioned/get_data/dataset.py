import numpy as np
import torch
from torch.utils.data import Dataset

class UnconditionedDataset(Dataset):
  def __init__(self):
    return
  
  def __getitem__(self, idx):
    return np.load(f"data/{idx}.npy")
