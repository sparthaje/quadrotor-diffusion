import math
from typing import List

import torch
from torch import device
import torch.nn as nn

from .attention_blocks import Attention, TrajectoryAttention
from .trajectory_convs import TrajectoryConv, TrajectoryTransposeConv
from .schedulers import SinusoidalEmbeddings
  
class ResBlock(nn.Module):
  def __init__(self, channels: int, num_groups: int, dropout_prob: float, use_trajectory_conv: bool):
    super().__init__()
    self.relu = nn.ReLU(inplace=True)
    self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)

    if use_trajectory_conv:
      self.conv1 = TrajectoryConv(channels, channels, kernel_size=3, stride=1, padding=1)
      self.conv2 = TrajectoryConv(channels, channels, kernel_size=3, stride=1, padding=1)
    else:
      self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
      self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
  
  def forward(self, x, embeddings):
    # Only take embeddings up to the available number of channels
    # Due to broadcasting this expand to the dimensions of x
    x = x + embeddings[:, :x.shape[1], :, :]
    r = self.conv1(self.relu(self.gnorm1(x)))
    r = self.dropout(r)
    r = self.conv2(self.relu(self.gnorm2(r)))
    return r + x

class TrajectoryUNETLayer(nn.Module):
  def __init__(self,
             upscale: bool,
             num_groups: int,
             dropout_prob: float,
             num_heads: int,  # set to None for no attention
             channels: int,
             ):
    super().__init__()
    
    self.resblock = ResBlock(channels=channels, num_groups=num_groups, dropout_prob=dropout_prob, use_trajectory_conv=True)
    self.resblock_two = ResBlock(channels=channels, num_groups=num_groups, dropout_prob=dropout_prob, use_trajectory_conv=True)
    
    if num_heads:
      self.attention_layer = TrajectoryAttention(channels, num_heads, dropout_prob)
    
    if upscale:
      self.conv = TrajectoryTransposeConv(channels, 
                                          channels // 2,  # cut output channels in half
                                          # (4, 2, 1, 0) Kernel, Stride, Input padding, Output padding
                                          # 
                                          4, 2, 1, 0)
    else:
      self.conv = TrajectoryConv(channels,
                                 channels*2,
                                 # (3, 2, 1) kernel, stride, input padding
                                 3, 2, 1)

  def forward(self, x, embeddings):
    x = self.resblock(x, embeddings)
    if hasattr(self, "attention_layer"):
      x = self.attention_layer(x)
    x = self.resblock_two(x, embeddings)
    return self.conv(x), x
  
class UNETLayer(nn.Module):
  def __init__(self,
             upscale: bool,
             num_groups: int,
             dropout_prob: float,
             num_heads: int,  # set to None for no attention
             channels: int,
             ):
    super().__init__()
    
    self.resblock = ResBlock(channels=channels, num_groups=num_groups, dropout_prob=dropout_prob, use_trajectory_conv=False)
    self.resblock_two = ResBlock(channels=channels, num_groups=num_groups, dropout_prob=dropout_prob, use_trajectory_conv=False)
    
    if num_heads:
      self.attention_layer = Attention(channels, num_heads, dropout_prob)
    
    if upscale:
      # Doubles the third dimension (height)
      self.conv = nn.ConvTranspose2d(channels, 
                                     channels // 2,  # cut output channels in half
                                     # (4, 2, 1, 0) Kernel, Stride, Input padding, Output padding
                                     # 
                                     4, 2, 1, 0)
    else:
      # Halves the third dimension (height)
      self.conv = nn.Conv2d(channels,
                            channels*2,
                            # (3, 2, 1) kernel, stride, input padding
                            3, 2, 1)

  def forward(self, x, embeddings):
    x = self.resblock(x, embeddings)
    if hasattr(self, "attention_layer"):
      x = self.attention_layer(x)
    x = self.resblock_two(x, embeddings)
    return self.conv(x), x

class UNET(nn.Module):
  def __init__(
      self,
      trajectory_height: int,
      channels: List[int] = [64//4, 128//4, 256//4, 512//4, 512//4, 384//4],
      attentions: List[bool] = [0, 0.0, 0, 0, 0, 0.0],
      upscales: List[bool] = [False, False, False, True, True, True],
      num_groups: int = 8,
      dropout_prob: float = 0.0,
      input_channels: int = 1,
      output_channels: int = 1,
      device: device = 'cuda',
      time_steps: int = 1000,
  ):
    super().__init__()
    
    # Input validation
    assert len(channels) == len(attentions) == len(upscales), "Configuration lists must have equal length"
    self.num_layers = len(channels)
    
    # (b, 1, n, 3) => (b, 64*n/3, n, 3)g
    # Initial and final processing
    # (B, input_channels, H, W) => (B, (H/W)C[0], H, W)
    self.channels = channels
    self.make_square = TrajectoryConv(input_channels, trajectory_height//3, kernel_size=3, stride=1, padding=1)
    self.input_block = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1)
    
    out_channels = (channels[-1] // 2) + channels[0]
    self.last_latent_block = nn.Sequential(
      nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels // 2, output_channels, kernel_size=1)
    )
    
    latent_channels = trajectory_height // 3
    self.rtn_to_trajectory_shape = nn.Sequential(
      TrajectoryTransposeConv(latent_channels, latent_channels // 4, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      TrajectoryTransposeConv(latent_channels // 4, latent_channels // 8, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      TrajectoryTransposeConv(latent_channels // 8, latent_channels // 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      TrajectoryTransposeConv(latent_channels // 16, 1, kernel_size=3, stride=1, padding=1),
    )
    
    # Time embeddings
    self.embeddings = SinusoidalEmbeddings(
      time_steps=time_steps,
      embed_dim=max(channels),
      device=device
    )
    
    # Create encoder and decoder layers
    self.encoder_layers = nn.ModuleList([
      UNETLayer(
          upscale=upscales[i],
          num_groups=num_groups,
          dropout_prob=dropout_prob,
          channels=channels[i],
          num_heads=attentions[i]
      ) for i in range(self.num_layers // 2)
    ])
    
    self.decoder_layers = nn.ModuleList([
      UNETLayer(
          upscale=upscales[i],
          num_groups=num_groups,
          dropout_prob=dropout_prob,
          channels=channels[i],
          num_heads=attentions[i]
      ) for i in range(self.num_layers // 2, self.num_layers)
    ])

  def encode(self, x: torch.Tensor, embeddings: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor]]:
    residuals = []
    for layer in self.encoder_layers:
      x, r = layer(x, embeddings)
      residuals.append(r)
    return x, residuals

  def decode(self, x: torch.Tensor, residuals: List[torch.Tensor], embeddings: torch.Tensor) -> torch.Tensor:
    for layer, residual in zip(self.decoder_layers, reversed(residuals)):
      x = torch.cat([layer(x, embeddings)[0], residual], dim=1)
    return x

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    B, _, H, _ = x.shape
    x = self.make_square(x)
    # To make a square image:
    # 1. Merge the second dimension (H / W) with the last dimension (W): H/W * W = H
    # 2. This will give us a perfect square H x H
    x = x.permute(0, 2, 1, 3).reshape(B, 1, H, H)

    x = self.input_block(x)
    
    embeddings = self.embeddings(t)
    
    # Encoder path
    x, residuals = self.encode(x, embeddings)
    
    # Decoder path
    x = self.decode(x, residuals, embeddings)
    
    # Final processing
    x = self.last_latent_block(x)
    
    # Convert Back to Embedded Trajectory Shape 
    
    # (batch_size, 1, height, channels, width)
    x = x.view(B, 1, H, H // 3, 3)
    # (batch_size, 1, height, width, channels)
    x = x.permute(0, 1, 3, 2, 4)
    x = x.squeeze(1)
    
    # Convert back to trajectory
    x = self.rtn_to_trajectory_shape(x)

    return x

