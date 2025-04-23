import enum
import torch

# Threshold between projected x_{i-1} and x_0 in planned trajectory to be acceptable
THRESHOLD_FOR_INITIAL_STATE = 0.1  # [m]

# Threshold to be considered passing gate safely
THRESHOLD_FOR_GATE = 0.15  # [m]


class ScoringMethod(enum.Enum):
    # Chooses the fastest trajectory
    FAST = "fast"

    # Chooses the slowest trajectory
    SLOW = "slow"

    # Chooses trajectory with highest curvature
    CURVATURE = "curvature"

    # Chooses trajectory with lowest curvature
    STRAIGHT = "straight"

    # Chooses trajectory with the point closest to the gate position
    CENTER = "center"


def filter_valid_trajectories(
    trajectories: torch.Tensor,
    x_0: torch.Tensor,
    g_i: torch.Tensor,
    start: bool,
) -> torch.Tensor:
    """
    Filters trajectories to ones which have a valid starting state and cross the next gate

    Args:
        trajectories (torch.Tensor): [B, N, 3] (trajectory length N)
        x_0 (torch.Tensor): Expected next state [n, 3] (i.e. x_{i-1} + v_{i-1} * t + a_{i-1} * t^2 / 2)
        g_i (torch.Tensor): Next gate [x, y, z]

    Returns:
        torch.Tensor: Indexes in trajectories which are valid
    """
    dists0 = torch.norm(trajectories[:, 0, :] - x_0, dim=-1)
    threshold = 2 * THRESHOLD_FOR_INITIAL_STATE if start else THRESHOLD_FOR_INITIAL_STATE
    filtered_idxs = torch.nonzero(dists0 < threshold, as_tuple=True)[0]
    if len(filtered_idxs) == 0:
        raise ValueError("Planner Failed Initial State")

    filtered_trajectories = trajectories[filtered_idxs]
    dists_gate = torch.norm(filtered_trajectories - g_i, dim=-1)
    min_dists_gate, _ = torch.min(dists_gate, dim=1)

    filtered_idxs = filtered_idxs[min_dists_gate <= THRESHOLD_FOR_GATE]
    if len(filtered_idxs) == 0:
        raise ValueError("Planner Failed Goal")

    return filtered_idxs


def fastest(
    trajectories: torch.Tensor,
) -> torch.Tensor:
    """

    Args:
        trajectories (torch.Tensor): [B, N, 3]

    Returns:
        torch.Tensor: Idx of the fastest trajectory
    """
    distances = torch.norm(trajectories[:, 1:] - trajectories[:, :-1], dim=2).sum(dim=1)
    return torch.argmax(distances)


def slowest(
    trajectories: torch.Tensor,
) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: Idx of the slowest trajectory
    """
    distances = torch.norm(trajectories[:, 1:] - trajectories[:, :-1], dim=2).sum(dim=1)
    return torch.argmin(distances)


def min_curvature(trajectories: torch.Tensor) -> torch.Tensor:
    """
    Args:
        trajectories (torch.Tensor): [B, N, 3]

    Returns:
        torch.Tensor: The trajectory index with the least curvature
    """
    segments = trajectories[:, 1:] - trajectories[:, :-1]
    norms = segments.norm(dim=2, keepdim=True)
    unit_segments = segments / (norms + 1e-8)
    dots = (unit_segments[:, 1:] * unit_segments[:, :-1]).sum(dim=2).clamp(-1, 1)
    angles = torch.acos(dots)
    avg_curvature = angles.mean(dim=1)
    return torch.argmin(avg_curvature)


def center_gate(
    trajectories: torch.Tensor,
    g_i: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        tra (torch.Tensor): Expected next state [n, 3] (i.e. x_{i-1} + v_{i-1} * t + a_{i-1} * t^2 / 2)
        g_i (torch.Tensor): Next gate [x, y, z]

    Returns:
        torch.Tensor: Indexes in trajectories which are valid
    """
    dists_gate = torch.norm(trajectories - g_i, dim=-1)
    min_dists, _ = dists_gate.min(dim=1)
    return torch.argmin(min_dists)
