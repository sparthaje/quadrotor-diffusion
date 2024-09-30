import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryAttention(nn.Module):
  def __init__(self, channels: int, num_heads: int, dropout_prob: float):
    super().__init__()
    self.x_proj_1 = nn.Linear(channels, channels * 3)
    self.x_proj_2 = nn.Linear(channels, channels)

    self.y_proj_1 = nn.Linear(channels, channels * 3)
    self.y_proj_2 = nn.Linear(channels, channels)

    self.z_proj_1 = nn.Linear(channels, channels * 3)
    self.z_proj_2 = nn.Linear(channels, channels)
    
    self.num_heads = num_heads
    self.dropout_prob = dropout_prob
    
  def apply_attention(self, component, proj1, proj2):
    # component shape: [batch_size, input_channels, points]
    b, c, p = component.shape
    
    # Reshape for projection: [B, P, C]
    component = component.permute(0, 2, 1)
    
    # Project to Q,K,V space: [B, P, 3C]
    component = proj1(component)
    
    # Reshape for multi-head attention
    head_dim = c // self.num_heads
    component = component.view(b, p, 3, self.num_heads, head_dim)
    # Permute to (QKV, B, num_heads, P, head_dim)
    component = component.permute(2, 0, 3, 1, 4)
    
    # Split into Q,K,V: each [B, num_heads, P, head_dim]
    q, k, v = component[0], component[1], component[2]
    
    # Apply attention
    attn_output = F.scaled_dot_product_attention(
        q, k, v, 
        is_causal=True, 
        dropout_p=self.dropout_prob
    )
    
    # Permute to [B, P, num_heads, head_dim]
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
    # Reshape back: [B, P, C]
    attn_output = attn_output.view(b, p, c)
    
    # Final projection
    attn_output = proj2(attn_output)
    
    # Return to original format [B, C, P]
    return attn_output.permute(0, 2, 1)
  
  def forward(self, traj):
    # traj shape: [batch_size, timesteps, points, 3]
    x = traj[:, :, :, 0]  # shape [batch_size, timesteps, points]
    y = traj[:, :, :, 1]
    z = traj[:, :, :, 2]

    # Process each component separately using attention
    x_attn = self.apply_attention(x, self.x_proj_1, self.x_proj_2)
    y_attn = self.apply_attention(y, self.y_proj_1, self.y_proj_2)
    z_attn = self.apply_attention(z, self.z_proj_1, self.z_proj_2)

    # Combine the attended components back into a trajectory
    traj_attn = torch.stack([x_attn, y_attn, z_attn], dim=-1)
    
    return traj_attn

class Attention(nn.Module):
  def __init__(self, channels: int, num_heads: int, dropout_prob: float):
    super().__init__()
    self.proj1 = nn.Linear(channels, channels * 3)
    self.proj2 = nn.Linear(channels, channels)
    self.num_heads = num_heads
    self.dropout_prob = dropout_prob
    
  def forward(self, x):
    # Shape: (B, C, H, W)
    b, c, h, w = x.shape
    
    # Shape: (B, HW, C)
    x = x.view(b, c, h * w).permute(0, 2, 1)
    # Shape: (B, HW, 3C)
    x = self.proj1(x)
    # Shape: B, HW, num_heads, 3, channels_per_head)
    x = x.view(b, h*w, self.num_heads, 3, c // self.num_heads)
    # Shape: (3, B, num_heads, HW, channels_per_head)
    x = x.permute(3, 0, 2, 1, 4)
    
    # Shape: (B, num_heads, HW, channels_per_head)
    q, k, v = x[0], x[1], x[2]
    
    # Apply attention
    x = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=self.dropout_prob)
    
    # Shape: (B, HW, C)
    x = x.permute(0, 2, 1, 3).contiguous()
    
    # Shape: (B, H, W, C)
    x = x.view(b, h, w, c)
    x = self.proj2(x)
    
    # Shape: (B, C, H, W)
    return x.permute(0, 3, 1, 2)