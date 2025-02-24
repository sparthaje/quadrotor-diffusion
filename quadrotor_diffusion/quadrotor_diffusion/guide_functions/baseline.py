import torch
import torch.nn.functional as F

from quadrotor_diffusion.models.nn_blocks import soft_argmax, soft_argmin


def point_line_distance(points, A, B):
    """
    Compute the Euclidean distance from each point to the line segment from A to B.

    Args:
        points: tensor of shape (..., 3)
        A, B: tensors of shape (..., 3) representing the endpoints of the segment.

    Returns:
        A tensor of shape (...,) with the distances from each point to the line segment.
    """
    AB = B - A                   # vector along the segment
    AP = points - A              # vector from A to the point
    # Projection factor t along the segment (clamped to [0,1] to stay on the segment)
    AB_norm_sq = (AB * AB).sum(dim=-1, keepdim=True) + 1e-8  # add epsilon for safety
    t = (AP * AB).sum(dim=-1, keepdim=True) / AB_norm_sq
    t = t.clamp(0, 1)
    # Closest point on the segment to the grid point
    projection = A + t * AB
    return (points - projection).norm(dim=-1)


def create_distance_field(waypoints, cell_size):
    """
    Creates a distance grid for each batch in 'waypoints'.

    waypoints: tensor of shape (B, N_waypoints, 3)
    cell_size: scalar (in meters) for each grid cell's side length.

    Returns:
      - dist_grid: a tensor of shape (B, 1, D, H, W) with the minimum distances.
      - vol_bounds: a tuple with the volume bounds (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    B, N, _ = waypoints.shape
    # Define volume bounds
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -2.0, 2.0
    z_min, z_max = 0.0, 1.0

    # Create 1D arrays of cell centers for each axis.
    x_centers = torch.arange(x_min + cell_size/2, x_max, cell_size, device=waypoints.device)
    y_centers = torch.arange(y_min + cell_size/2, y_max, cell_size, device=waypoints.device)
    z_centers = torch.arange(z_min + cell_size/2, z_max, cell_size, device=waypoints.device)
    nx, ny, nz = len(x_centers), len(y_centers), len(z_centers)

    # Create a meshgrid of cell center coordinates.
    # The resulting grid has shape (nx, ny, nz, 3)
    grid_x, grid_y, grid_z = torch.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    grid_centers = torch.stack([grid_x, grid_y, grid_z], dim=-1)

    # Expand the grid to have a batch dimension: (B, nx, ny, nz, 3)
    grid_centers = grid_centers.unsqueeze(0).expand(B, -1, -1, -1, -1)

    # For each grid cell, we want the minimum distance to any line segment connecting consecutive waypoints.
    # Start with a large initial value.
    dist_grid = torch.full((B, nx, ny, nz), float('inf'), device=waypoints.device)

    # Loop over each segment in the waypoint trajectory.
    for i in range(N - 1):
        A = waypoints[:, i, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)      # shape: (B, 1, 1, 1, 3)
        B_point = waypoints[:, i+1, :].unsqueeze(1).unsqueeze(1).unsqueeze(1)  # shape: (B, 1, 1, 1, 3)
        d = point_line_distance(grid_centers, A, B_point)  # shape: (B, nx, ny, nz)
        dist_grid = torch.min(dist_grid, d)

    # Rearrange dimensions so that the grid is in (D, H, W) order as required by grid_sample:
    # Our computed grid is (B, nx, ny, nz) with nx = number of x cells, etc.
    # For 3D grid_sample, we need the shape to be (B, C, D, H, W) where:
    #   D corresponds to the z dimension, H to y, and W to x.
    dist_grid = dist_grid.permute(0, 3, 2, 1).unsqueeze(1)  # now shape: (B, 1, nz, ny, nx)

    vol_bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
    return dist_grid, vol_bounds


def compute_trajectory_alignment(trajectory: torch.Tensor, course: torch.Tensor) -> torch.Tensor:
    """
    Compute trajectory alignment with a course

    Args:
        trajectory (torch.Tensor): [B, H, 3] xyz trajectories
        course (torch.Tensor): [B, N_gates, 4]: (x, y, z, theta)

    Returns:
        torch.Tensor: [B] A probability for how well the trajectory aligns with the course
    """
    # 1) drop the last column in course tensor to make it [B, N_gates, 3]
    course = course[..., :3]

    # 2) Compute a field where high values indicate being on "course"
    distance_field, vol_bounds = create_distance_field(course, cell_size=0.05)
    distance_field = torch.exp(-distance_field)

    # 3) Adjust trajectory to use grid_sample
    x_min, x_max, y_min, y_max, z_min, z_max = vol_bounds
    traj_normalized = trajectory.clone()
    traj_normalized[..., 0] = 2 * (traj_normalized[..., 0] - x_min) / (x_max - x_min) - 1
    traj_normalized[..., 1] = 2 * (traj_normalized[..., 1] - y_min) / (y_max - y_min) - 1
    traj_normalized[..., 2] = 2 * (traj_normalized[..., 2] - z_min) / (z_max - z_min) - 1
    grid = traj_normalized.view(trajectory.shape[0], trajectory.shape[1], 1, 1, 3)

    # 3) use F.grid_sample to compute a (B, H) tensor of distances from each point on the horizon to the course
    sampled = F.grid_sample(distance_field, grid, align_corners=True, mode='bilinear')
    probability_of_alignment = sampled.squeeze(1).squeeze(-1).squeeze(-1)

    # 4) Return mean scores
    return torch.mean(probability_of_alignment, dim=-1)
