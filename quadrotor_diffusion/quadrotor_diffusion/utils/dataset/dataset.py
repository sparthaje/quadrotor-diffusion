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
from quadrotor_diffusion.utils.quad_logging import iprint as print


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
            assert course_filename.endswith(".npy")
        else:
            trajectory_filename = self.data[idx]

        assert trajectory_filename.endswith(".pkl")

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


class QuadrotorRaceSegmentDataset(Dataset):
    def __init__(self, data_dir: str, course_types: list[str], traj_len: int, padding: int, normalizer: Normalizer):
        """
        Args:
            data_dir (str): Root dir where course/linear, course/u, etc exists
            course_types (list[str]): linear, u, etc.
            traj_len (int): Trajectory length to pad to (i.e. 12s = 360 points)
            normalizer (Normalizer): Normalizer for trajectory data
            padding (int):  How much to pad trajectory with the consecutive / previous states
        """

        super().__init__()
        self.data_dir = data_dir
        self.course_types = course_types
        self.normalizer = normalizer
        self.traj_len = traj_len
        self.padding = padding
        self.data: list[tuple[str, int, int]] = []  # trajectory filename, start index, end index
        self._load_data()

    def _load_data(self):
        n_gates = {
            "linear": 4,
            "u": 4,
            "triangle": 4,
            "pill": 5,
            "square": 5,
        }

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

                            with open(traj_filename, "rb") as trajectory_file:
                                trajectory: PolynomialTrajectory = pickle.load(trajectory_file)

                            trajectory.states.append(trajectory.states[2])
                            trajectory.segment_lengths.append(trajectory.segment_lengths[1])

                            trajectory.states.append(trajectory.states[3])
                            trajectory.segment_lengths.append(trajectory.segment_lengths[2])

                            trajectory_np = trajectory.as_ref_pos(30)
                            ref_pos_idx = 0

                            for gate_idx in range(n_gates[course_type]):
                                ending_gate = trajectory.states[gate_idx+1]
                                ending_idx = np.argmin(
                                    np.linalg.norm(
                                        trajectory_np[ref_pos_idx:ref_pos_idx+self.traj_len] -
                                        np.array([ending_gate.x.s, ending_gate.y.s, ending_gate.z.s]),
                                        axis=1,
                                    )
                                )
                                self.data.append((traj_filename, ref_pos_idx, ref_pos_idx+self.traj_len))

                                ref_pos_idx += ending_idx

                            total_trajectory_length = int(sum(trajectory.segment_lengths) * 30)
                            for _ in range(6):
                                start = random.randint(0, total_trajectory_length - self.traj_len)
                                end = start + self.traj_len
                                self.data.append((traj_filename, start, end))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        trajectory_filename, start, end = self.data[idx]

        assert trajectory_filename.endswith(".pkl")

        with open(trajectory_filename, "rb") as trajectory_file:
            trajectory: PolynomialTrajectory = pickle.load(trajectory_file)

            # Re-append the segment from gate 1 to gate 2, so if the traj_len exceeds the last part it can loop back around
            trajectory.states.append(trajectory.states[2])
            trajectory.segment_lengths.append(trajectory.segment_lengths[1])

            trajectory.states.append(trajectory.states[3])
            trajectory.segment_lengths.append(trajectory.segment_lengths[2])

            trajectory.states.append(trajectory.states[4])
            trajectory.segment_lengths.append(trajectory.segment_lengths[2])

        trajectory: np.ndarray = trajectory.as_ref_pos()

        trajectory = self.normalizer(trajectory)

        # Add the data around the starting and ending position of the trajectory as padding
        if self.padding > 0:
            trajectory = np.vstack((
                np.tile(trajectory[0], (self.padding, 1)),
                trajectory
            ))

            end += 2 * self.padding

        return torch.tensor(trajectory[start:end], dtype=torch.float32)

    def __str__(self):
        return f"QuadrotorRaceSegmentDataset({self.course_types}, {self.traj_len}, {self.padding})"


class DiffusionDataset(Dataset):
    def __init__(self, data_dir: str, traj_len: int, normalizer: Normalizer):
        """
        Args:
            data_dir (str): Root dir where course/linear, course/u, etc exists
            traj_len (int): Trajectory length to pad to (i.e. 12s = 360 points)
            padding (int):  How much to pad trajectory with the consecutive / previous states
        """

        super().__init__()
        self.data_dir = data_dir
        self.normalizer = normalizer
        self.traj_len = traj_len
        self.data: list[str] = []
        self._load_data()

    def _load_data(self):
        course_dir = os.path.join(self.data_dir, "courses", "diffusion2")

        for sample in os.listdir(course_dir):
            self.data.append(os.path.join(course_dir, sample))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_filename = self.data[idx]

        assert data_filename.endswith(".pkl")

        with open(data_filename, "rb") as data_file:
            sample = pickle.load(data_file)

        x0 = sample["trajectory_slice"][:self.traj_len]
        x0 = torch.tensor(x0, dtype=torch.float32)

        c_global = sample["global_context"]
        GLOBAL_CONTEXT_SIZE = 6
        null_tokens = np.tile(np.array(5 * np.ones((1, 4))), (GLOBAL_CONTEXT_SIZE - len(c_global), 1))
        c_global = np.vstack((c_global, null_tokens))

        LOCAL_CONTEXT_SIZE = 6
        c_local = sample["local_context"]
        c_local = np.hstack((c_local[-LOCAL_CONTEXT_SIZE:], np.zeros((LOCAL_CONTEXT_SIZE, 1))))

        # c_global = c_global[:2]
        # context = np.vstack((c_local, c_global))
        # context = torch.tensor(context, dtype=torch.float32)

        return {
            "x_0": x0,
            "local_conditioning": torch.tensor(c_local, dtype=torch.float32),
            "global_conditioning": torch.tensor(c_global, dtype=torch.float32),
        }

    def __str__(self):
        return f"DiffusionDataset({self.traj_len})"


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
