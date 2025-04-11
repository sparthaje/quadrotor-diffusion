import enum


class SamplerType(enum.Enum):
    DDPM = "ddpm"
    DDIM = "ddim"
    CONSISTENCY = "consistency"


class ScoringMethod(enum.Enum):
    # Chooses the fastest trajectory
    FAST = "fast"

    # Chooses the slowest trajectory
    SLOW = "slow"

    # Chooses trajectory with highest curvature
    CURVATURE = "curvature"

    # Chooses trajectory with lowest curvature
    STRAIGHT = "straight"
