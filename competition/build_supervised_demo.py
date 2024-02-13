import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sys import argv

from data_collection_ctrls import yaw_rot, INITIAL_GATE_EXIT

def plot_points(x_coords, y_coords, z_coords, angles):
  fig, ax = plt.subplots()
  
  # Plot points
  values = [0.2 * (yaw_rot(x) @ yaw_rot(np.pi/2) @ INITIAL_GATE_EXIT) for x in angles]
  description_x = []
  description_y = []
  description_z = []
  
  for x, y, z, d in zip(x_coords, y_coords, z_coords, values):
    point = np.array([x, y])
    d = np.array([d[0], d[1]])
    right = point + d
    left = point - d
    description_x.append([right[0], left[0]])
    description_y.append([right[1], left[1]])
    description_z.append([z, z])
    
  scatter = ax.scatter(x_coords, y_coords, c=z_coords, cmap='viridis')
  scatter = ax.scatter(description_x, description_y, c=description_z, cmap='viridis', s=10, marker='x')

  # Add rectangle with center at (0, 0) and dimensions 3x4
  rectangle = patches.Rectangle((-1.5, -2), 3, 4, linewidth=1.2, edgecolor='blue', facecolor='none')
  ax.add_patch(rectangle)

  ax.set_title('Picked Points with Rectangle')
  ax.set_xlabel('X-axis')
  ax.set_ylabel('Y-axis')
  ax.grid(True)
  ax.set_aspect('equal', adjustable='box')
  plt.colorbar(scatter, ax=ax, label='Z-values')
  plt.pause(0.1)  # Add a small pause to update the plot
  plt.clf()  # Clear the plot for the next iteration

def main():
  x_coords = [float(argv[1])]
  y_coords = [float(argv[2])]
  z_coords = [0.525 if int(argv[3]) == 1 else 0.3]
  distances = [0]
  relative_angles = [float(argv[4]) * np.pi]
  angles = [float(argv[4]) * np.pi]
      
  print("Command-line Gate Setup Tool")
  print("Type 'exit' to finish picking points.")
  
  plt.ion()
  
  plot_points(x_coords, y_coords, z_coords, angles)

  while True:
    d_input = input("Enter d to next gate: ")
    if d_input.lower() == 'exit':
        print("------ INPUTS ------")
        print(f"gate_x = {x_coords}")
        print(f"gate_y = {y_coords}")
        print(f"gate_z = {[1 if z == 0.3 else 0 for z in z_coords]}")
        print(f"heights = {z_coords}")
        print(f"gate_theta = {angles}")
        print("d_vals =", distances)
        print("rel_angles =", relative_angles)
        print("--------------------")
        break
    
    distances.append(float(d_input))
    angle = float(input("Relative angle as a multiple of pi to next gate: ")) * np.pi
    
    relative_angles.append(angle)
    angle = angle + angles[-1]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    angles.append(angle)
    
    d_vec = yaw_rot(angles[-2]) @ (distances[-1] * INITIAL_GATE_EXIT)
    new_pos = np.array([x_coords[-1], y_coords[-1], 0]) + d_vec
    x_coords.append(new_pos[0])
    y_coords.append(new_pos[1])
    
    z = 0.525 if float(input("0 for low 1 for high: ")) == 1 else 0.3
    z_coords.append(z)

    plot_points(x_coords, y_coords, z_coords, angles)

if __name__ == "__main__":
  main()
