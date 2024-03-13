from torch import nn

class BoundaryPredictor(nn.Module):
  def __init__(self, inp_dim):
    super().__init__()
    self.hidden1 = nn.Linear(inp_dim, 512)
    self.activation1 = nn.LeakyReLU(negative_slope=0.05)
    
    self.hidden2 = nn.Linear(512, 256)
    self.activation2 = nn.LeakyReLU(negative_slope=0.02)
    
    self.hidden3 = nn.Linear(256, 128)
    self.activation3 = nn.LeakyReLU(negative_slope=0.01)
    
    self.hidden4 = nn.Linear(128, 128)
    self.activation4 = nn.LeakyReLU(negative_slope=0.01)
    
    self.output = nn.Linear(128, 2)
    self.output_activation = nn.ReLU()

  def forward(self, x):
    x = self.activation1(self.hidden1(x))
    x = self.activation2(self.hidden2(x))
    x = self.activation3(self.hidden3(x))
    x = self.activation4(self.hidden4(x))
    out = self.output_activation(self.output(x))
    return out
