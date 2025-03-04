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


class QuadrotorRaceTrajectoryDataset(Dataset):
    def __init__(self, data_dir: str, course_types: list[str], traj_len: int, normalizer: Normalizer, includes_course: bool):
        """
        Args:
            data_dir (str): Root dir where course/linear, course/u, etc exists
            course_types (list[str]): linear, u, etc.
            traj_len (int): Trajectory length to pad to (i.e. 12s = 360 points)
            normalizer (Normalizer): Normalizer for trajectory data
            includes_course (bool): Returns a dictionary with the course as well
        """

        super().__init__()
        self.data_dir = data_dir
        self.course_types = course_types
        self.normalizer = normalizer
        self.traj_len = traj_len
        self.includes_course = includes_course
        if self.includes_course:
            self.data: list[tuple[str, str]] = []
        else:
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

                            if self.includes_course:
                                self.data.append((course_filename, traj_filename))
                            else:
                                self.data.append(traj_filename)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.includes_course:
            course_filename, trajectory_filename = self.data[idx]
        else:
            trajectory_filename = self.data[idx]

        assert trajectory_filename.endswith(".pkl")
        assert course_filename.endswith(".npy")

        with open(trajectory_filename, "rb") as trajectory_file:
            trajectory: PolynomialTrajectory = pickle.load(trajectory_file)
        trajectory = trajectory.as_ref_pos(pad_to=self.traj_len)
        trajectory = self.normalizer(trajectory)

        if not self.includes_course:
            return torch.tensor(trajectory, dtype=torch.float32)

        course = np.load(course_filename)
        return {
            "trajectory": torch.tensor(trajectory, dtype=torch.float32),
            "course": torch.tensor(course, dtype=torch.float32)
        }


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
