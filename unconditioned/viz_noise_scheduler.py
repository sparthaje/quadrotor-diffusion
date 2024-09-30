SAMPLE_NUM = 4682

import torch
import numpy as np

from models.schedulers import LinearScheduler, CosineScheduler, SigmoidScheduler
from get_data.plotting_utils import view_references_in_3d

reference = np.load(f"data/{SAMPLE_NUM}.npy")
ref_torch = torch_tensor = torch.from_numpy(reference)
T         = 1000
scheduler = SigmoidScheduler(T)
num_graph = 10
splits    = [int(x) for x in np.linspace(0, T-1, num_graph)]

trajectories = [ref_torch]
for t in splits[1:]:
  beta, alpha = scheduler.get_vals(t)
  
  e = torch.randn_like(ref_torch, requires_grad=False)
  x = (torch.sqrt(alpha)*ref_torch) + (torch.sqrt(1-alpha)*e)
  trajectories.append(x)

view_references_in_3d(trajectories, splits)

