import torch
import torch.nn as nn
import torch.nn.functional as F

from .VAE_blocks import (
  SequentialToSquareEncoder,
  VAE_AttentionBlock,
  VAE_ResidualBlock
)

class VAE_Encoder(nn.Sequential):
  def __init__(self):
    super().__init__(
      # (BatchSize, 324, 3) -> (BatchSize, H:60, W:60)
      SequentialToSquareEncoder(
                                29,  # 0.97 seconds
                                5,   # 0.17 seconds
                                324  # 10.8 seconds
                                ),
      
      # (Batch_Size, Channel, H:60, W:60) -> (Batch_Size, 32, H:60, W:60)
      nn.Conv2d(1, 32, kernel_size=3, padding=1),
      
        # (Batch_Size, 32, H:60, W:60) -> (Batch_Size, 32, H:60, W:60)
      VAE_ResidualBlock(32, 32),
      
      # (Batch_Size, 32, H:60, W:60) -> (Batch_Size, 32, H:60, W:60)
      VAE_ResidualBlock(32, 32),
      
      # (Batch_Size, 32, H:60, W:60) -> (Batch_Size, 32, H:60 / 2, W:60 / 2)
      nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=0),
      
      # (Batch_Size, 128, H:60 / 2, W:60 / 2) -> (Batch_Size, 256, H:60 / 2, W:60 / 2)
      VAE_ResidualBlock(32, 32), 
      
      # (Batch_Size, 64, H:60 / 2, W:60 / 2) -> (Batch_Size, 64, H:60 / 2, W:60 / 2)
      VAE_ResidualBlock(64, 64), 
      
      # (Batch_Size, 256, H:60 / 2, W:60 / 2) -> (Batch_Size, 256, H:60 / 4, W:60 / 4)
      nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0), 
      
      # (Batch_Size, 64, H:60 / 4, W:60 / 4) -> (Batch_Size, 64, H:60 / 4, W:60 / 4)
      VAE_ResidualBlock(64, 128), 
      
      # (Batch_Size, 128, H:60 / 4, W:60 / 4) -> (Batch_Size, 128, H:60 / 4, W:60 / 4)
      VAE_ResidualBlock(128, 128), 
      
      # (Batch_Size, 128, H:60 / 4, W:60 / 4) -> (Batch_Size, 128, H:60 / 4, W:60 / 4)
      VAE_ResidualBlock(128, 128), 
      
      # (Batch_Size, 128, H:60 / 4, W:60 / 4) -> (Batch_Size, 128, H:60 / 4, W:60 / 4)
      VAE_AttentionBlock(128), 
      
      # (Batch_Size, 128, H:60 / 4, W:60 / 4) -> (Batch_Size, 128, H:60 / 4, W:60 / 4)
      VAE_ResidualBlock(128, 128), 
      
      # (Batch_Size, 128, H:60 / 4, W:60 / 4) -> (Batch_Size, 128, H:60 / 4, W:60 / 4)
      nn.GroupNorm(32, 128), 
      
      # (Batch_Size, 128, H:60 / 4, W:60 / 4) -> (Batch_Size, 128, H:60 / 4, W:60 / 4)
      nn.SiLU(), 
      
      # (Batch_Size, 128, H:60 / 4, W:60 / 4) -> (Batch_Size, 8, H:60 / 4, W:60 / 4). 
      nn.Conv2d(128, 8, kernel_size=3, padding=1), 

      # (Batch_Size, 8, H:60 / 4, W:60 / 4) -> (Batch_Size, 8, H:60 / 4, W:60 / 4)
      nn.Conv2d(8, 8, kernel_size=1, padding=0), 
    )

  def forward(self, x, noise, sample_with_noise=True):
    # x: (Batch_Size, Channel, H:60, W:60)
    # noise: (Batch_Size, 4, H:60 / 4, W:60 / 4)
    for module in self:

        if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
            # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
            # Pad with zeros on the right and bottom.
            # (Batch_Size, Channel, H:60, W:60) -> (Batch_Size, Channel, H:60 + Padding_Top + Padding_Bottom, W:60 + Padding_Left + Padding_Right) = (Batch_Size, Channel, H:60 + 1, W:60 + 1)
            x = F.pad(x, (0, 1, 0, 1))
        
        x = module(x)
    # (Batch_Size, 8, H:60 / 4, W:60 / 4) -> two tensors of shape (Batch_Size, 4, H:60 / 4, W:60 / 4)
    mean, log_variance = torch.chunk(x, 2, dim=1)
    
    if not sample_with_noise:
      return mean * 0.18215
    
    # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
    # (Batch_Size, 4, H:60 / 4, W:60 / 4) -> (Batch_Size, 4, H:60 / 4, W:60 / 4)
    log_variance = torch.clamp(log_variance, -30, 20)
    # (Batch_Size, 4, H:60 / 4, W:60 / 4) -> (Batch_Size, 4, H:60 / 4, W:60 / 4)
    variance = log_variance.exp()
    # (Batch_Size, 4, H:60 / 4, W:60 / 4) -> (Batch_Size, 4, H:60 / 4, W:60 / 4)
    stdev = variance.sqrt()
    
    # Transform N(0, 1) -> N(mean, stdev) 
    # (Batch_Size, 4, H:60 / 4, W:60 / 4) -> (Batch_Size, 4, H:60 / 4, W:60 / 4)
    x = mean + stdev * noise
    
    # Scale by a constant
    # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
    x *= 0.18215
    
    return x

def compare_speed_and_memory(batch_size, traj_height, kernel, stride):
  # Check if MPS is available
  device = torch.device("cpu")
  print(f"Using device: {device}")
  SequentialToSquareEncoder.find_valid_kernel_stride_combinations(324)
  input()
  x = torch.randn(batch_size, traj_height, 3, requires_grad=True, device=device)    
  vectorized_model = SequentialToSquareEncoder(kernel, stride, traj_height).to(device)
  
  import time    
  with torch.no_grad():
    o = vectorized_model(x)
  
  start_time = time.time()  
  output = vectorized_model(x)
  loss = output.mean()  
  loss.backward()
  end_time = time.time()
  
  # Clear memory
  if device.type == "mps":
      torch.mps.empty_cache()
  return end_time - start_time

if __name__ == "__main__":
  t = compare_speed_and_memory(128, 324, 60, 6)
  print(f"Time taken for vectorized version: {t:.4f} seconds")
