try:
  import matplotlib.pyplot as plt
  import numpy as np
  import pybullet as p
  import torch
  from scipy import stats
  from scipy.stats import chi2
except:
  print("Missing packages")

plt.rcParams['font.family'] = 'Avenir'

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
  
def view_references_in_3d(positions_list, titles, fig_name=None):
    # Calculate global min and max for axis limits
    all_positions = np.vstack(positions_list)
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
    z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

    # Set up the figure and axes for subplots
    num_positions = len(positions_list)
    fig = plt.figure(figsize=(5 * num_positions, 7))  # Adjust size for all subplots

    handles = []
    labels = []
    for i, positions in enumerate(positions_list):
        ax = fig.add_subplot(1, num_positions, i + 1, projection='3d')

        # Plot the remaining positions
        path_handle = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                marker='o', color=ColorPalletteThree.B, label='Path')
        
        # Plot the first point with a unique marker
        start_point_handle = ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   color=ColorPalletteTwo.A, s=100, label='Start Point', marker='*')  # Unique marker for the first point
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   color=ColorPalletteTwo.A, s=300, label='Start Point', marker='*')  # Unique marker for the first point

        # Set labels and title for each subplot
        ax.set_xlabel('x (meters)')
        ax.set_ylabel('y (meters)')
        ax.set_zlabel('')
        ax.set_title(titles[i])
        if i == 0:
          handles = [start_point_handle, path_handle]
          labels = ['Start Point', 'Path']

        # Set the same axis limits for all subplots
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        x_ticks = [-2, 0, 2]
        y_ticks = [-2, 0, 2]
        z_ticks = [-1, 0, 1]
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_zticks(z_ticks)
        
        ax.view_init(elev=90, azim=45)
        
          # Enhance grid and axis styling
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)
        
    # Adjust layout for better spacing
    plt.tight_layout()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12, frameon=False)
    if fig_name:
      plt.savefig(f"{fig_name}.png", format="png")
    else:
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
  
def test_multivariate_normality(data):
    """
    source chatgpt :) , so use with a grain of salt. I'm most interested in the marginal distribution plot
    
    Test for multivariate normality using multiple approaches.
    Handles both PyTorch tensors and numpy arrays. 
    
    Parameters:
    data : torch.Tensor or numpy array of shape (n_samples, 3) containing x, y, z coordinates
    
    Returns:
    dict containing test results and p-values
    """
    # Convert PyTorch tensor to numpy if needed
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    n_samples = data.shape[0]
    
    # Calculate mean and covariance
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    
    # Calculate Mahalanobis distances
    inv_cov = np.linalg.inv(cov)
    diff = data - mean
    mahal_dist = np.array([np.sqrt(np.dot(np.dot(d, inv_cov), d)) for d in diff])
    
    # Mardia's test
    # Skewness
    b1 = np.mean([np.dot(np.dot(d, inv_cov), d)**3 for d in diff])
    skew_stat = (n_samples/6) * b1
    skew_df = (data.shape[1] * (data.shape[1] + 1) * (data.shape[1] + 2)) / 6
    skew_p_value = 1 - chi2.cdf(skew_stat, df=skew_df)
    
    # Kurtosis
    b2 = np.mean([np.dot(np.dot(d, inv_cov), d)**2 for d in diff])
    kurt_stat = (b2 - data.shape[1] * (data.shape[1] + 2)) / np.sqrt(8 * data.shape[1] * (data.shape[1] + 2) / n_samples)
    kurt_p_value = 2 * (1 - stats.norm.cdf(abs(kurt_stat)))
    
    # Shapiro-Wilk test for each dimension
    sw_tests = [stats.shapiro(data[:, i]) for i in range(data.shape[1])]
    
    # Create visualizations
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.5)
    ax1.set_title('3D Scatter Plot')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Q-Q plot of Mahalanobis distances
    ax2 = fig.add_subplot(132)
    stats.probplot(mahal_dist, dist="chi2", sparams=(3,), plot=ax2)
    ax2.set_title('Q-Q Plot of Mahalanobis Distances')
    
    # Marginal distributions
    ax3 = fig.add_subplot(133)
    ax3.boxplot([data[:, i] for i in range(data.shape[1])])
    ax3.set_xticklabels(['X', 'Y', 'Z'])
    ax3.set_title('Marginal Distributions')
    
    plt.tight_layout()
    
    return {
        'mardia_skewness': {'statistic': skew_stat, 'p_value': skew_p_value},
        'mardia_kurtosis': {'statistic': kurt_stat, 'p_value': kurt_p_value},
        'shapiro_wilk_tests': {
            'x': {'statistic': sw_tests[0][0], 'p_value': sw_tests[0][1]},
            'y': {'statistic': sw_tests[1][0], 'p_value': sw_tests[1][1]},
            'z': {'statistic': sw_tests[2][0], 'p_value': sw_tests[2][1]}
        }
    }
  
class ColorPalletteTwo:
  A = "#ff595e"
  B = "#ffca3a"
  C = "#8ac926"
  D = "#1982c4"
  E = "#6a4c93"

class ColorPalletteOne:
  A = "#003f5c"
  B = "#58508d"
  C = "#bc5090"
  D = "#ff6361"
  E = "#ffa600"  
  
class ColorPalletteThree:
  A = "#de324c"
  B = "#f4895f"
  C = "#f8e16f"
  D = "#95cf92"
  E = "#369acc"
  F = "#9656a2"
  
class ColorPalletteFour: 
  A = "#42826c"
  B = "#38705d"
  C = "#2f5f4e"
  D = "#254e40"
  E = "#1c3e32"
  F = "#142f25"
  G = "#0b2019"
  H = "#05120d"  

