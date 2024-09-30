import random

import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3 

from models.unet import UNET
from models.schedulers import CosineScheduler
from get_data.dataset import UnconditionedDataset

CONFIG         = yaml.safe_load(open("models/hyperparameters.yaml", "r"))
LR             = CONFIG["training"]["learning_rate"]
EMA_DECAY      = CONFIG["training"]["ema_decay"]
NUM_EPOCHS     = CONFIG["training"]["num_epochs"]
BATCH_SIZE     = CONFIG["training"]["batch_size"]
NUM_TIME_STEPS = CONFIG["training"]["num_time_steps"]

def set_seed(seed):
    torch.manual_seed(seed)    
    np.random.seed(seed)
    random.seed(seed)

set_seed(random.randint(0, 2**32-1))

dataset      = UnconditionedDataset("data")
traj_height  = dataset[0].shape[1]
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

scheduler = CosineScheduler(NUM_TIME_STEPS)
device    = torch.device("cuda")
model     = UNET(traj_height).to(device)
optimizer = optim.Adam(model.parameters(), lr = LR)
ema       = ModelEmaV3(model, decay = EMA_DECAY)
criterion = nn.MSELoss(reduction='mean')

for epoch in range(NUM_EPOCHS):
  total_loss = 0
  for bidx, x in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")):
    batch_size = x.shape[0]  # last batch might be smaller than BATCH_SIZE
    
    x = x.to(device)
    t = torch.randint(0, NUM_TIME_STEPS, (batch_size,))
    e = torch.randn_like(x, requires_grad=False)
    a = scheduler.get_vals(t)[1].view(batch_size, 1, 1, 1).to(device)
    x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
    output = model(x, t)
    optimizer.zero_grad()
    loss = criterion(output, e)
    total_loss += loss.item()
    optimizer.step()
    ema.update(model)
  print(f'Epoch {epoch+1} | Loss {total_loss / (len(dataset) // BATCH_SIZE):.5f}')

  checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
  torch.save(checkpoint, f'models/checkpoints/ddpm_checkpoint{epoch}')
