import torch
import torch.nn as nn
import torch.nn.functional as F


class SequentialToSquareEncoder(nn.Module):
  
  @staticmethod
  def find_valid_kernel_stride_combinations(traj_height):
    valid_combinations = []

    # Iterate through possible values for kernel and stride
    for kernel in range(1, 30 + 1):  # Don't consider any kernel that encodes more than 30s
      for stride in range(1, traj_height + 1):
        if kernel == stride:
          continue
        
        # Check the first condition
        if (traj_height - kernel) % stride == 0:  
          output_dim = (traj_height - kernel) // stride + 1
          
          # Check the second condition
          if output_dim % 3 == 0 and output_dim % 4 == 0 and output_dim > 32:  
            valid_combinations.append((kernel, stride, f"{kernel / 30:.2f}", output_dim))
    
    valid_combinations.sort(key=lambda x: x[-1])
    for v in valid_combinations:
      print(v)
  
  def __init__(self, 
               kernel: int,      # kernel size
               stride: int,      # overlap between chunks
               traj_height: int  # points in traj
               ):
    super(SequentialToSquareEncoder, self).__init__()
    
    assert (traj_height - kernel) % stride == 0, "Kernel, Stride doesn't fit evenly into the traj height"
    
    self.kernel     = kernel
    self.stride     = stride
    self.output_dim = (traj_height - kernel) // stride + 1
    
    assert self.output_dim % 3 == 0, "Output dimensions should be divisible by 3, revisit kernel / stride"
    
    col_out = self.output_dim // 3
    # Shared column encoder, goes from (kernel, 1) to (1, output_dim / 3)
    self.column_encoder = nn.Sequential(
        nn.Linear(kernel, kernel),
        nn.LeakyReLU(),
        nn.Linear(kernel, 2 * col_out),
        nn.ReLU(),
        nn.Linear(2 * col_out, col_out)
    )

  def forward(self, x):
    # x shape: (batch_size, traj_height, 3)
    batch_size = x.shape[0]
    
    encoded_columns = []
    for coord_idx in range(3):
      # Extract the coordinate column
      coord_data = x[:, :, coord_idx]  # shape: (batch_size, traj_height)
      
      # Create overlapping chunks using unfold
      # shape: (batch_size, num_chunks, kernel)
      chunks = coord_data.unfold(1, self.kernel, self.stride)
      
      # Reshape for batch processing
      # shape: (batch_size * num_chunks, kernel)
      chunks = chunks.reshape(-1, self.kernel)
      
      # Apply column encoder to all chunks at once
      # shape: (batch_size * num_chunks, col_out)
      encoded = self.column_encoder(chunks)
      
      # Reshape back to organize as a column of the final square
      # shape: (batch_size, output_dim, output_dim // 3)
      # time goes top to down
      encoded = encoded.reshape(batch_size, self.output_dim, self.output_dim // 3)
      encoded_columns.append(encoded)
    
    # Concatenate the encoded columns
    # Final shape: (batch_size, output_dim [time latent], output_dim [xyz latent])
    output = torch.cat(encoded_columns, dim=2)
    
    return output

# Modifying https://github.com/hkproj/pytorch-stable-diffusion/blob/main/sd/decoder.py
class VAE_AttentionBlock(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.groupnorm = nn.GroupNorm(32, channels)
    # Using nn.MultiheadAttention with 1 head and embedding size = channels
    self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=1, batch_first=True)

  def forward(self, x):
    # x: (Batch_Size, Features, Height, Width)
    residue = x  # Save the input for the residual connection

    # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
    x = self.groupnorm(x)  # Apply GroupNorm

    n, c, h, w = x.shape
    
    # (Batch_Size, Features, Height * Width)
    x = x.view((n, c, h * w))
    
    # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
    x = x.transpose(-1, -2)  # Transpose to make each pixel a feature

    # Perform self-attention
    # Query, Key, and Value are all the same (x) for self-attention
    # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
    x, _ = self.attention(x, x, x)

    # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
    x = x.transpose(-1, -2)
    
    # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
    x = x.view((n, c, h, w))
    
    # Residual connection
    x += residue

    return x

class VAE_ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.groupnorm_1 = nn.GroupNorm(32, in_channels)
    self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    self.groupnorm_2 = nn.GroupNorm(32, out_channels)
    self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    if in_channels == out_channels:
      self.residual_layer = nn.Identity()
    else:
      self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
  
  def forward(self, x):
    # x: (Batch_Size, In_Channels, Height, Width)

    residue = x

    # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
    x = self.groupnorm_1(x)
    
    # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
    x = F.silu(x)
    
    # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
    x = self.conv_1(x)
    
    # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
    x = self.groupnorm_2(x)
    
    # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
    x = F.silu(x)
    
    # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
    x = self.conv_2(x)
    
    # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
    return x + self.residual_layer(residue)

