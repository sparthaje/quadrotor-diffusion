import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torchview import draw_graph
import matplotlib.pyplot as plt
import datetime
import random

from model import BoundaryPredictor

# HYPERPARAMETERS

# Data Preprocessing
## If best_v is 0, replace it with a small value (OPTIONAL)
ZERO_EQ = 0.0

# Loss
## Epsilon value for the custom loss function (divide by for zero)
EPSILON = 0.1
## Weight for the higher speed loss
W_L = 2.0
## Weight for lower speed loss
W_R = 0.8

# Training
EPOCHS = 1000
BATCH_SIZE = 1280
## Patience for early stopping (higher value means more epochs)
PATIENCE = 100
## Random bound for adding noise to the input data
## RANDOM_BOUND * Unif(0,1) * N(0,1)
RANDOM_BOUND = 0.1


# Load the dataset
dataset = pd.read_csv('data.csv')

dataset.loc[dataset['best_v'] == 0, 'best_v'] = ZERO_EQ

normalizing_vals = {
  "best_v": dataset["best_v"].max(),
  "best_t": dataset["best_t"].max(),
  "v0": dataset["v0"].max(),
}

print(normalizing_vals)

# Normalize the dataset
dataset["best_v"] = dataset["best_v"] / normalizing_vals["best_v"]
dataset["best_t"] = dataset["best_t"] / normalizing_vals["best_t"]
dataset["v0"] = dataset["v0"] / normalizing_vals["v0"]

dataset["d1"] = (dataset["d1"] - 0.8) / (1.5 - 0.8)
dataset["d2"] = (dataset["d2"] - 0.8) / (1.5 - 0.8)

dataset["z0"] = (dataset["z0"] == 0.3) * 1.0
dataset["z1"] = (dataset["z1"] == 0.3) * 1.0
dataset["z2"] = (dataset["z2"] == 0.3) * 1.0

dataset["theta1"] = dataset["theta1"] / (np.pi / 4)
dataset["theta2"] = dataset["theta2"] / (np.pi / 4)

X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8:].values

# Convert the data to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.08, random_state=19)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

# Define the model
input_size = X.shape[1]
output_size = y.shape[1]
print("Input and output dimensions", input_size, output_size)
model = BoundaryPredictor(input_size)
print()
print(model)
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")
print()

# Define the loss function and optimizer
class CustomLoss(torch.nn.Module):
  def __init__(self):
    super(CustomLoss, self).__init__()

  def forward(self, outputs, targets):
    percent_errors = (outputs - targets) / (targets + EPSILON)
    percent_errors[:, 1] = -percent_errors[:, 1]  # this way -time indicates slower
    alpha = 10.0
    weights = torch.sigmoid(alpha * percent_errors)
    loss = torch.mean(weights * W_L * percent_errors + (1 - weights) * (-W_R * percent_errors))
    # loss = torch.mean(torch.where(percent_errors >= 0, 
    #                                    W_L * percent_errors,  # worse to go fast
    #                                   -W_R * percent_errors)) # better to go slow
    return loss

# Create an instance of the custom loss function
criterion = CustomLoss()
criterion_test = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8) 

train_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# Train the model
loss_list = []
test_loss_list = []
lowest_test_loss = float('inf')
counter = 0
best_model = None

for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    for batch_X, batch_y in train_loader:
        rbx = batch_X + RANDOM_BOUND * random.random() * torch.randn(batch_X.shape)
        optimizer.zero_grad()
        outputs = model(rbx)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    scheduler.step()  # Update learning rate scheduler

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        total_loss = 0
        for batch_X_test, batch_y_test in test_loader:
            test_outputs = model(batch_X_test)
            test_loss = criterion(test_outputs, batch_y_test)
            total_loss += test_loss.item() * len(batch_X_test)
        avg_test_loss = total_loss / len(X_test)
        
        if avg_test_loss < lowest_test_loss:
            lowest_test_loss = avg_test_loss
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Test Loss: {avg_test_loss:.4f}, Best Test Loss: {lowest_test_loss:.4f}')
        
    test_loss_list.append(avg_test_loss)
    
    if counter >= PATIENCE:
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Test Loss: {avg_test_loss:.4f}, Best Test Loss: {lowest_test_loss:.4f}')
        break

if best_model is not None:
    model.load_state_dict(best_model)

loss_list = loss_list[20:]
test_loss_list = test_loss_list[20:]

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
torch.save(model.state_dict(), f'../models/model-{timestamp}-{lowest_test_loss}.pth')

print()
err = []
a, b = 0, 0
for x, y in zip(X_test, y_test):
  pred = model(x).detach().numpy()
  y = y.detach().numpy()
  if np.isclose(y[0], ZERO_EQ, atol=1e-3):
    a += pred[0]
    b += 1
    continue
  
  err.append(np.array([
    (pred[0] - y[0]) / y[0] * 100,
    (pred[1] - y[1]) / y[1] * 100
  ]))

err_0 = [item[0] for item in err]
err_1 = [item[1] for item in err]
if b != 0:
  print(f"average velocity predicted when it should be {ZERO_EQ} in test: {a/b}, examples {b}")

# Plot Velocity Prediction % Error and Time Prediction % Error in the same figure
plt.figure(figsize=(10, 5))

# Plot Velocity Prediction % Error
plt.subplot(1, 2, 1)
plt.hist(err_0, bins=20, color='blue', alpha=0.7, label='Velocity Prediction')
plt.xlabel('Velocity Prediction % Error')
plt.ylabel('Frequency')
plt.yscale('log')  # Set y-axis to log scale

# Plot Time Prediction % Error
plt.subplot(1, 2, 2)
plt.hist(err_1, bins=20, color='green', alpha=0.7, label='Time Prediction')
plt.xlabel('Time Prediction % Error')
plt.ylabel('Frequency')
plt.yscale('log')  # Set y-axis to log scale

plt.suptitle('Histogram of Velocity and Time Prediction % Errors')
plt.legend()
plt.savefig('test_dataset_histogram.png')
plt.close()


# model_graph = draw_graph(model, input_size=(1,8), expand_nested=True)
# model_graph.visual_graph.render("computation_graph", format="png")

plt.plot(list(range(len(loss_list))), loss_list, label='Training Loss')
plt.plot(list(range(len(test_loss_list))), test_loss_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss and Test Loss vs Epoch')
plt.legend()
plt.savefig('loss.png')
plt.close()
