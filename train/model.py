import torch
from torch import nn

class BoundaryPredictor(nn.Module):
  def __init__(self, inp_dim):
    super().__init__()
    self.inp_dim = inp_dim
    self.hidden1 = nn.Linear(inp_dim, 512)
    self.dropout1 = nn.Dropout(0.1)
    self.activation1 = nn.LeakyReLU(negative_slope=0.1)
    
    self.hidden2 = nn.Linear(512, 256)
    self.dropout2 = nn.Dropout(0.1)
    self.activation2 = nn.LeakyReLU(negative_slope=0.04)
    
    self.hidden3 = nn.Linear(256, 32)
    self.dropout3 = nn.Dropout(0.1)
    self.activation3 = nn.LeakyReLU(negative_slope=0.02)
    
    self.hidden4 = nn.Linear(32, 24)
    self.activation4 = nn.LeakyReLU(negative_slope=0.02)
    
    self.output = nn.Linear(24, 2)
    self.output_activation = nn.ReLU()

  def forward(self, x):
    x = self.activation1(self.dropout1(self.hidden1(x)))
    x = self.activation2(self.dropout2(self.hidden2(x)))
    x = self.activation3(self.dropout3(self.hidden3(x)))
    x = self.activation4(self.hidden4(x))
    out = self.output_activation(self.output(x))
    return out
