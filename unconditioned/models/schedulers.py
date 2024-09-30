import math

import torch
import torch.nn as nn
import numpy as np

# Not an actual nn module
class SinusoidalEmbeddings(nn.Module):
  def __init__(self, time_steps: int, embed_dim: int, device: torch.device):
    super().__init__()
    position = torch.arange(time_steps).unsqueeze(1).float() # (time_steps, 1)
    # (embed_dim//2, 1)
    div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
    # (time_steps, embed_dim)
    embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
    embeddings[:, 0::2] = torch.sin(position * div)
    embeddings[:, 1::2] = torch.cos(position * div)
    self.embeddings = embeddings
    self.device = device

  def forward(self, t):
    # t: (batch_size, 1)
    # (batch_size, embed_size)
    embeds = self.embeddings[t].to(self.device)
    
    # (batch_size, embed_dim, 1, 1)
    return embeds[:, :, None, None]

class LinearScheduler:
  MIN = 1e-8
  MAX = 8e-4
  
  def __init__(self, timesteps: int):
    self.beta  = torch.linspace(self.MIN, self.MAX, timesteps)
    alpha      = 1 - self.beta
    self.alpha = torch.cumprod(alpha, dim=0)
  
  def get_vals(self, t):
    return self.beta[t], self.alpha[t]

class CosineScheduler:
    MIN = 1e-8
    MAX = 8e-4
    
    def __init__(self, timesteps: int):
        # Compute beta values using the cosine function
        self.beta = self._cosine_schedule(timesteps)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0)
    
    def _cosine_schedule(self, timesteps):
        # Generate the cosine schedule
        return self.MIN + (self.MAX - self.MIN) * 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, timesteps)))
    
    def get_vals(self, t):
        return self.beta[t], self.alpha[t]


class SigmoidScheduler:
    MIN = 1e-8
    MAX = 8e-4
    
    LEFT_BOUND = -5
    RIGHT_BOUND = 3
    
    SIGMOID_MIN = 1 / (1 + np.exp(-LEFT_BOUND))
    SIGMOID_MAX = 1 / (1 + np.exp(-RIGHT_BOUND))
    
    def __init__(self, timesteps: int):
        # Compute beta values using the cosine function
        self.beta = self._sigmoid_schedule(timesteps)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0)
    
    def _sigmoid_schedule(self, timesteps):
        # Generate the cosine schedule
        sigmoid_value = torch.sigmoid(torch.linspace(self.LEFT_BOUND, self.RIGHT_BOUND, timesteps))
        sigmoid_value_scaled = (sigmoid_value - self.SIGMOID_MIN) / (self.SIGMOID_MAX - self.SIGMOID_MIN)
        return self.MIN + (self.MAX - self.MIN) * sigmoid_value_scaled
    
    def get_vals(self, t):
        return self.beta[t], self.alpha[t]

