import torch

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
    
    LEFT_BOUND = -9
    RIGHT_BOUND = 7.5
    
    def __init__(self, timesteps: int):
        # Compute beta values using the cosine function
        self.beta = self._sigmoid_schedule(timesteps)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0)
    
    def _sigmoid_schedule(self, timesteps):
        # Generate the cosine schedule
        return self.MIN + (self.MAX - self.MIN) * torch.sigmoid(torch.linspace(self.LEFT_BOUND, self.RIGHT_BOUND, timesteps))
    
    def get_vals(self, t):
        return self.beta[t], self.alpha[t]

