from torch import nn

class BoundaryPredictor(nn.Module):
  def __init__(self, inp_dim):
    super().__init__()
    self.hidden1 = nn.Linear(inp_dim, 256)
    self.activation1 = nn.LeakyReLU(negative_slope=0.01)
    
    self.hidden2 = nn.Linear(256, 256)
    self.activation2 = nn.LeakyReLU(negative_slope=0.01)
    
    self.hidden3 = nn.Linear(256, 128)
    self.activation3 = nn.ReLU()
    
    self.output = nn.Linear(128, 2)
    self.output_activation = nn.ReLU()

  def forward(self, x):
    x = self.activation1(self.hidden1(x))
    x = self.activation2(self.hidden2(x))
    x = self.activation3(self.hidden3(x))
    out = self.output_activation(self.output(x))
    return out
