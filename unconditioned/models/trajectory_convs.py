import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryConv(nn.Module):
  def __init__(self,
              input_channels: int,
              output_channels: int,
              kernel_size: int,
              stride: int,
              padding: int):
    super().__init__()
    
    self.conv_x = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.conv_y = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.conv_z = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
  
  def forward(self, trajectory):
    # trajectory: (batch_size, input_channels, points_in_traj, 3)
    x_out = self.conv_x(trajectory[:, :, :, 0])
    y_out = self.conv_y(trajectory[:, :, :, 1])
    z_out = self.conv_z(trajectory[:, :, :, 2])
    
    # Stack outputs so shape will be (batch_size, output_channels, points_in_traj, 3)
    output = torch.stack((x_out, y_out, z_out), dim=-1)
    return output
  
class TrajectoryTransposeConv(nn.Module):
  def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 output_padding: int = 0):
    super().__init__()
    self.conv_x = nn.ConvTranspose1d(in_channels=input_channels, out_channels=output_channels,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding, output_padding=output_padding)
    self.conv_y = nn.ConvTranspose1d(in_channels=input_channels, out_channels=output_channels,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding, output_padding=output_padding)
    self.conv_z = nn.ConvTranspose1d(in_channels=input_channels, out_channels=output_channels,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding, output_padding=output_padding)

  def forward(self, trajectory):
    # trajectory: (batch_size, input_channels, points_in_traj, 3)
    x_out = self.conv_x(trajectory[:, :, :, 0])
    y_out = self.conv_y(trajectory[:, :, :, 1])
    z_out = self.conv_z(trajectory[:, :, :, 2])
    
    # Stack outputs so shape will be (batch_size, output_channels, new_points_in_traj, 3)
    output = torch.stack((x_out, y_out, z_out), dim=-1)
    return output
