import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torchview import draw_graph
import matplotlib.pyplot as plt
import datetime

from model import BoundaryPredictor

# Replace 0.0 for optimal velocity with a small value
ZERO_EQ = 0.0

# Load the dataset
dataset = pd.read_csv('data-90k.csv')
dataset.loc[dataset['best_v'] == 0, 'best_v'] = ZERO_EQ
# Duplicate rows where dataset['best_v'] == 0
# zero_rows = dataset[dataset['best_v'] == 0]
# for i in range(10):
#   dataset = pd.concat([dataset, zero_rows], ignore_index=True)
print(sum(dataset['best_v'] == 0))

X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8:].values

# Convert the data to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

# Define the model
input_size = X.shape[1]
output_size = y.shape[1]
print(input_size, output_size)
model = BoundaryPredictor(input_size)

print(model)

# Define the loss function and optimizer
class CustomLoss(torch.nn.Module):
  def __init__(self):
    super(CustomLoss, self).__init__()

  def forward(self, outputs, targets):
    # Custom loss calculation
    loss = torch.mean((outputs - targets) ** 2)
    return loss

# Create an instance of the custom loss function
criterion = CustomLoss()
criterion_test = CustomLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
loss_list = []
test_loss_list = []
num_epochs = 1000
lowest_test_loss = float('inf')
best_model = None

for epoch in range(num_epochs):
  # Forward pass
  outputs = model(X_train)
  loss = criterion(outputs, y_train)

  # Backward and optimize
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  test_loss = criterion_test(model(X_test), y_test)
  if test_loss < lowest_test_loss:
        lowest_test_loss = test_loss
        best_model = model.state_dict()

  if (epoch+1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Best Test Loss: {lowest_test_loss.item():.4f}')

  loss_list.append(loss.item())
  test_loss_list.append(test_loss.item())

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

# dataset = pd.read_csv('data-90k.csv')
# dataset.loc[dataset['best_v'] == 0, 'best_v'] = ZERO_EQ

# X = dataset.iloc[:, 0:input_size].values
# y = dataset.iloc[:, input_size:].values

# # # Convert the data to tensors
# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32)
# err = []
# a, b = 0, 0
# for x, y in zip(X, y):
#   pred = model(x).detach().numpy()
#   y = y.detach().numpy()
#   if np.isclose(y[0], ZERO_EQ, atol=1e-3):
#     a += pred[0]
#     b += 1
#     continue
  
#   err.append(np.array([
#     (pred[0] - y[0]) / y[0] * 100,
#     (pred[1] - y[1]) / y[1] * 100
#   ]))

# err_0 = [item[0] for item in err]
# err_1 = [item[1] for item in err]
# if b != 0:
#   print(f"average velocity predicted when it should be {ZERO_EQ} in full dataset: {a/b}, examples {b}")

# # Plot Velocity Prediction % Error and Time Prediction % Error in the same figure
# plt.figure(figsize=(10, 5))

# # Plot Velocity Prediction % Error
# plt.subplot(1, 2, 1)
# plt.hist(err_0, bins=20, color='blue', alpha=0.7, label='Velocity Prediction')
# plt.xlabel('Velocity Prediction % Error')
# plt.ylabel('Frequency')
# plt.yscale('log')  # Set y-axis to log scale

# # Plot Time Prediction % Error
# plt.subplot(1, 2, 2)
# plt.hist(err_1, bins=20, color='green', alpha=0.7, label='Time Prediction')
# plt.xlabel('Time Prediction % Error')
# plt.ylabel('Frequency')
# plt.yscale('log')  # Set y-axis to log scale

# plt.suptitle('Histogram of Velocity and Time Prediction Errors')
# plt.legend()
# plt.savefig('fulldataset_histogram.png')
# plt.close()

# Create a 3D histogram of all the errors
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set the number of bins for err_0 and err_1
bins = 20

# Compute the histogram
hist, xedges, yedges = np.histogram2d(err_0, err_1, bins=bins)

# Convert the histogram to log scale
hist = np.log(hist)

# Create the x and y meshgrid
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])

# Flatten the histogram and convert to 1D array
zpos = 0
dx = dy = 0.1
dz = hist.flatten()

# Create the 3D bar plot
ax.bar3d(xpos.flatten(), ypos.flatten(), zpos, dx, dy, dz, zsort='average')

# Set the labels and title
ax.set_xlabel('Velocity Prediction % Error')
ax.set_ylabel('Time Prediction % Error')
ax.set_zlabel('Log Frequency')
ax.set_title('3D Histogram of Velocity and Time Prediction Errors')
plt.savefig('3d_histogram.png')
plt.close()

# look at the errors within 100%
err_0 = [item[0] for item in err if abs(item[0]) < 100]
err_1 = [item[1] for item in err if abs(item[0]) < 100]
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

plt.suptitle('Histogram of Velocity and Time Prediction Errors')
plt.legend()
plt.savefig('testdataset_histogram_within100.png')
plt.close()

plt.figure(figsize=(10, 5))
# Plot Velocity Prediction % Error
plt.subplot(1, 2, 1)
plt.hist(err_0, bins=20, color='blue', alpha=0.7, label='Velocity Prediction')
plt.xlabel('Velocity Prediction % Error')
plt.ylabel('Frequency')

# Plot Time Prediction % Error
plt.subplot(1, 2, 2)
plt.hist(err_1, bins=20, color='green', alpha=0.7, label='Time Prediction')
plt.xlabel('Time Prediction % Error')
plt.ylabel('Frequency')

plt.suptitle('Histogram of Velocity and Time Prediction Errors')
plt.legend()
plt.savefig('testdataset_histogram_within100-no-log.png')
plt.close()

