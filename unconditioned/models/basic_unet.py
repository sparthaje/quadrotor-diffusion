import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import random
import math
from torch.utils.data import DataLoader 
from timm.utils import ModelEmaV3 
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np

class SinusoidalEmbeddings(nn.Module):
  def __init__(self, time_steps: int, embedding_dimensions: int):
    super().__init__()
    position = torch.arange(time_steps).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, embedding_dimensions, 2)).float() * -
