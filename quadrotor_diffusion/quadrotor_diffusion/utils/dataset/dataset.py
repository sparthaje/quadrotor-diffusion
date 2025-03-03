import os
import pickle
import random
import copy
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from quadrotor_diffusion.utils.dataset.normalizer import Normalizer, NormalizerTuple
from quadrotor_diffusion.utils.trajectory import derive_trajectory
from quadrotor_diffusion.utils.dataset.boundary_condition import PolynomialTrajectory
from quadrotor_diffusion.utils.logging import iprint as print


class CourseTrajectoryCrossEmbedding(Dataset):
    def __init__(self, data_dir: str, course_types: list[str], traj_len: int, mini_batch_size: int, normalizer: NormalizerTuple):
        """
        Args:
            data_dir (str): Root dir where course/linear, course/u, etc exists
            course_types (list[str]): linear, u, etc.
            traj_len (int): Trajectory length to pad to (i.e. 12s = 360 points)
            mini_batch_size (int): How many pairs to include in each mini-batch
            normalizer (NormalizerTuple): Normalizer for course and trajectory data
        """
        super().__init__()
        self.data_dir = data_dir
        self.course_types = course_types
        self.normalizer = normalizer
        self.traj_len = traj_len
        self.data: dict[str, list[str]] = defaultdict(list)
        self._load_data()
        self.mini_batches: list[list[tuple[str, str]]] = []
        self.mini_batch_size = mini_batch_size
        self._create_mini_batches(copy.deepcopy(self.data))
        self.length = len(self.mini_batches)
        self._current_idx = 0

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
                            self.data[course_filename].append(traj_filename)

    def _create_mini_batches(self, data):
        self.mini_batches = []
        courses = list(data.keys())

        # Randomize the data order
        random.shuffle(courses)
        for c in courses:
            random.shuffle(data[c])

        while len(courses) > self.mini_batch_size:
            mini_batch = []

            mini_batch_courses = random.sample(courses, self.mini_batch_size)
            for mini_batch_course in mini_batch_courses:
                traj_idx = random.randint(0, len(data[mini_batch_course]) - 1)
                trajectory = data[mini_batch_course][traj_idx]
                del data[mini_batch_course][traj_idx]
                mini_batch.append((mini_batch_course, trajectory))

            self.mini_batches.append(mini_batch)
            courses = [c for c in courses if len(data[c]) > 0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mini_batch = self.mini_batches[idx]
        courses = [
            self.normalizer.normalizer_a(torch.tensor(np.load(pair[0]), dtype=torch.float32).unsqueeze(0))
            for pair in mini_batch
        ]
        courses = torch.concat(courses)

        trajectories = []
        for _, traj_filename in mini_batch:
            with open(traj_filename, "rb") as trajectory_file:
                trajectory: PolynomialTrajectory = pickle.load(trajectory_file)
            trajectory = trajectory.as_ref_pos(pad_to=self.traj_len)
            trajectories.append(self.normalizer.normalizer_b(
                torch.tensor(trajectory, dtype=torch.float32).unsqueeze(0)))
        trajectories = torch.concat(trajectories)

        sample = {
            "courses": courses,
            "trajectories": trajectories,
        }

        self._current_idx += 1
        if self._current_idx >= len(self.mini_batches):
            self._current_idx = 0
            self.mini_batches = []
            self._create_mini_batches(copy.deepcopy(self.data))
            print("Resetting Mini Batches")

        return sample


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
