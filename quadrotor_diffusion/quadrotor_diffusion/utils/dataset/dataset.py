import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from quadrotor_diffusion.utils.dataset.normalizer import Normalizer, NormalizerTuple
from quadrotor_diffusion.utils.trajectory import derive_trajectory
from quadrotor_diffusion.utils.dataset.boundary_condition import PolynomialTrajectory


class ContrastiveEmbeddingDataset(Dataset):
    def __init__(self, data_dir: str, course_types: list[str], traj_len: int, normalizer: NormalizerTuple):
        """
        Args:
            data_dir (str): Root dir where course/linear, course/u, etc exists
            course_types (list[str]): linear, u, etc.
            traj_len (int): Trajectory length to pad to (i.e. 12s = 360 points)
            normalizer (NormalizerTuple): Normalizer for course and trajectory data
        """

        super().__init__()
        self.data_dir = data_dir
        self.course_types = course_types
        self.normalizer = normalizer
        self.traj_len = traj_len
        self.data: list[tuple[str, str]] = []
        self._load_data()

    def _load_data(self):
        for course_type in self.course_types:
            course_dir = os.path.join(self.data_dir, "courses", course_type)
            if not os.path.isdir(course_dir):
                continue

            for sample in os.listdir(course_dir):
                sample_dir = os.path.join(course_dir, sample)
                if not os.path.isdir(sample_dir):
                    continue

                course_filename = os.path.join(sample_dir, "course.npy")
                if not os.path.exists(course_filename):
                    continue

                # Find valid trajectories
                valid_dir = os.path.join(sample_dir, "valid")
                if os.path.exists(valid_dir):
                    for valid_file in os.listdir(valid_dir):
                        if valid_file.endswith(".pkl"):
                            traj_filename = os.path.join(valid_dir, valid_file)

                            # 1 for valid trajectory
                            self.data.append((course_filename, traj_filename))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        course_filename, trajectory_filename = self.data[idx]
        assert trajectory_filename.endswith(".pkl")

        course = np.array(np.load(course_filename))

        with open(trajectory_filename, "rb") as trajectory_file:
            trajectory: PolynomialTrajectory = pickle.load(trajectory_file)
        trajectory = trajectory.as_ref_pos(pad_to=self.traj_len)

        # Find positions along the trajectory where each gate is passed
        gate_positions = []
        for gate in course:
            gate_xyz = gate[:3]
            idx = np.linalg.norm(trajectory - gate_xyz, axis=1).argmin(0)
            gate_positions.append(idx)

        course, trajectory = self.normalizer(course, trajectory)
        return {
            "course": torch.tensor(course, dtype=torch.float32),
            "trajectory": torch.tensor(trajectory, dtype=torch.float32),
            "gate_positions": torch.tensor(gate_positions, dtype=torch.int64),
        }


class QuadrotorTrajectoryDataset(Dataset):
    def __init__(self, data_dir, normalizer: Normalizer, order: int = 0):
        self.data_dir = data_dir
        self.length = len([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.normalizer = normalizer
        self.order = order

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, f"{idx}.npy")
        data = np.load(filepath, allow_pickle=True)

        # Horizon should be divisible by 2^(channel_mults - 1) in unet
        data = data[:336, :]
        data = derive_trajectory(data, 30, order=self.order)
        data = self.normalizer(data)

        data = torch.tensor(data).float()  # [n x 3]
        return data

    def __str__(self):
        return "\n".join([
            "Quadrotor Trajectory Dataset: ",
            f"\torder={self.order}"
        ])


class QuadrotorFullStateDataset(Dataset):
    def __init__(self, data_dir, normalizer: Normalizer):
        self.data_dir = data_dir
        self.length = len([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.normalizer = normalizer

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, f"{idx}.npy")
        pos = np.load(filepath, allow_pickle=True)

        # Horizon should be divisible by 2^(channel_mults - 1) in unet
        pos = pos[:336, :]
        vel = derive_trajectory(pos, 30)
        acc = derive_trajectory(vel, 30)

        gap = np.zeros((pos.shape[0], 1))
        data = np.hstack((pos, gap, vel, gap, acc))
        data = self.normalizer(data)

        data = torch.tensor(data).float()  # [n x 11]
        return data


class QuadrotorRaceTrajectoryDataset(Dataset):
    def __init__(self, data_dir: str, course_types: list[str], traj_len: int, normalizer: Normalizer):
        """
        Args:
            data_dir (str): Root dir where course/linear, course/u, etc exists
            course_types (list[str]): linear, u, etc.
            traj_len (int): Trajectory length to pad to (i.e. 12s = 360 points)
            normalizer (Normalizer): Normalizer for trajectory data
        """

        super().__init__()
        self.data_dir = data_dir
        self.course_types = course_types
        self.normalizer = normalizer
        self.traj_len = traj_len
        self.data: list[str] = []
        self._load_data()

    def _load_data(self):
        for course_type in self.course_types:
            course_dir = os.path.join(self.data_dir, "courses", course_type)
            if not os.path.isdir(course_dir):
                continue

            for sample in os.listdir(course_dir):
                sample_dir = os.path.join(course_dir, sample)
                if not os.path.isdir(sample_dir):
                    continue

                course_filename = os.path.join(sample_dir, "course.npy")
                if not os.path.exists(course_filename):
                    continue

                # Find valid trajectories
                valid_dir = os.path.join(sample_dir, "valid")
                if os.path.exists(valid_dir):
                    for valid_file in os.listdir(valid_dir):
                        if valid_file.endswith(".pkl"):
                            traj_filename = os.path.join(valid_dir, valid_file)

                            # 1 for valid trajectory
                            self.data.append(traj_filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_filename = self.data[idx]
        assert trajectory_filename.endswith(".pkl")

        with open(trajectory_filename, "rb") as trajectory_file:
            trajectory: PolynomialTrajectory = pickle.load(trajectory_file)
        trajectory = trajectory.as_ref_pos(pad_to=self.traj_len)
        trajectory = self.normalizer(trajectory)
        return torch.tensor(trajectory, dtype=torch.float32)


def evaluate_dataset(dataset: Dataset):
    """
    Get key stats about a dataset

    Returns:
    - mean, variance, min max
    """
    # Collect all data first
    all_data = [dataset[x].numpy() for x in range(len(dataset))]
    data_array = np.concatenate(all_data, axis=0)

    # Calculate statistics
    mean = np.mean(data_array, axis=0)
    variance = np.var(data_array, axis=0)
    min_values = np.min(data_array, axis=0)
    max_values = np.max(data_array, axis=0)

    return mean, variance, min_values, max_values
