import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

def plot_reference_time_series(reference):
  # criteria:
  # tracking error below, accel / vel limits met, 
  
  time = np.arange(reference.shape[0])

  # Create subplots
  fig, axs = plt.subplots(3, 3, figsize=(15, 10))

  # Plotting Position, Velocity, and Acceleration
  labels = ['X', 'Y', 'Z']
  for i in range(3):
      # Position plots
      axs[0, i].plot(time, reference[:, 0, i], label='Position')
      axs[0, i].set_title(f'Position - {labels[i]}')
      axs[0, i].set_xlabel('Time')
      axs[0, i].set_ylabel('Position')
      axs[0, i].legend()
      
      # Velocity plots
      axs[1, i].plot(time, reference[:, 1, i], label='Velocity', color='orange')
      axs[1, i].set_title(f'Velocity - {labels[i]}')
      axs[1, i].set_xlabel('Time')
      axs[1, i].set_ylabel('Velocity')
      axs[1, i].legend()

      # Acceleration plots
      axs[2, i].plot(time, reference[:, 2, i], label='Acceleration', color='green')
      axs[2, i].set_title(f'Acceleration - {labels[i]}')
      axs[2, i].set_xlabel('Time')
      axs[2, i].set_ylabel('Acceleration')
      axs[2, i].legend()

  # Adjust layout
  plt.tight_layout()
  plt.show()

def view_reference_in_3d(reference):
  # Extract position data (all x, y, z positions)
  positions = reference[:, 0, :]  # Get all position data

  # Create a 3D plot
  fig = plt.figure(figsize=(10, 7))
  ax = fig.add_subplot(111, projection='3d')

  # Plot the first point with a unique marker
  ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
            color='red', s=100, label='Start Point', marker='o')  # Unique marker for the first point

  # Plot the remaining positions
  ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
          marker='o', color='blue', label='Path')

  # Set labels
  ax.set_xlabel('X Position')
  ax.set_ylabel('Y Position')
  ax.set_zlabel('Z Position')
  ax.set_title('Trajectory')
  ax.legend()

  # Show the plot
  plt.show()
  
def view_references_in_3d(positions_list, titles):
    # Calculate global min and max for axis limits
    all_positions = np.vstack(positions_list)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

    # Set up the figure and axes for subplots
    num_positions = len(positions_list)
    fig = plt.figure(figsize=(5 * num_positions, 7))  # Adjust size for all subplots

    for i, positions in enumerate(positions_list):
        ax = fig.add_subplot(1, num_positions, i + 1, projection='3d')

        # Plot the first point with a unique marker
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   color='red', s=100, label='Start Point', marker='o')  # Unique marker for the first point

        # Plot the remaining positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                marker='o', color='blue', label='Path')

        # Set labels and title for each subplot
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'Trajectory at t={titles[i]}')
        ax.legend()

        # Set the same axis limits for all subplots
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def draw_trajectory_on_pybullet(initial_info,
                    ref_x,
                    ref_y,
                    ref_z
                  ):
  """Draw a trajectory in PyBullet's GUI.

  """
  
  step = min(int(ref_x.shape[0]/50), ref_x.shape[0])
  if step == 0:
      step = 1
  for i in range(step, ref_x.shape[0], step):
      p.addUserDebugLine(lineFromXYZ=[ref_x[i-step], ref_y[i-step], ref_z[i-step]],
                          lineToXYZ=[ref_x[i], ref_y[i], ref_z[i]],
                          lineColorRGB=[1, 0, 0],
                          physicsClientId=initial_info["pyb_client"])
  p.addUserDebugLine(lineFromXYZ=[ref_x[i], ref_y[i], ref_z[i]],
                      lineToXYZ=[ref_x[-1], ref_y[-1], ref_z[-1]],
                      lineColorRGB=[1, 0, 0],
                      physicsClientId=initial_info["pyb_client"])