import os
import pickle
from dataclasses import dataclass, fields

import numpy as np

from quadrotor_diffusion.utils.logging import dataclass_to_table


# Path to the data directory
data_dir = "data/courses/"


@dataclass
class CourseStats:
    total_samples: int = 0
    average_valid_count: float = 0.0
    min_valid_count: int = 0
    max_valid_count: int = 0
    average_max_ttc: float = 0.0
    average_min_ttc: float = 0.0
    average_mean_ttc: float = 0.0


def round_decimals(dataclass_instance):
    for field in fields(dataclass_instance):
        value = getattr(dataclass_instance, field.name)
        if isinstance(value, float):
            setattr(dataclass_instance, field.name, round(value, 2))


# Statistics containers for each course type
course_type_stats = {}

# Iterate over each course type (e.g., linear, u, zig-zag)
for course_type in os.listdir(data_dir):
    course_type_path = os.path.join(data_dir, course_type)
    if not os.path.isdir(course_type_path):
        continue

    # Initialize stats for the current course type
    course_stats = {
        "valid_counts": [],
        "ttc_max": [],
        "ttc_min": [],
        "ttc_mean": [],
        "total_samples": 0
    }

    # Iterate over each sample course (e.g., 1, 2, 3)
    for sample in os.listdir(course_type_path):
        sample_path = os.path.join(course_type_path, sample)
        if not os.path.isdir(sample_path):
            continue

        valid_path = os.path.join(sample_path, "valid")

        if not os.path.exists(valid_path):
            continue

        # Collect TTC values and count valid trajectories
        ttc_values = []
        valid_count = 0

        for traj_file in os.listdir(valid_path):
            course_stats["total_samples"] += 2
            if traj_file.endswith(".pkl"):
                valid_count += 1
                ttc = float(traj_file.split("_(")[-1].rstrip(").pkl"))
                ttc_values.append(ttc)

        if valid_count > 0:
            course_stats["valid_counts"].append(valid_count)
            course_stats["ttc_max"].append(max(ttc_values))
            course_stats["ttc_min"].append(min(ttc_values))
            course_stats["ttc_mean"].append(np.mean(ttc_values))

    # Store stats for the current course type
    course_type_stats[course_type] = CourseStats(
        total_samples=course_stats["total_samples"],
        average_valid_count=np.mean(course_stats["valid_counts"]) if course_stats["valid_counts"] else 0,
        min_valid_count=np.min(course_stats["valid_counts"]) if course_stats["valid_counts"] else 0,
        max_valid_count=np.max(course_stats["valid_counts"]) if course_stats["valid_counts"] else 0,
        average_max_ttc=np.mean(course_stats["ttc_max"]) if course_stats["ttc_max"] else 0,
        average_min_ttc=np.mean(course_stats["ttc_min"]) if course_stats["ttc_min"] else 0,
        average_mean_ttc=np.mean(course_stats["ttc_mean"]) if course_stats["ttc_mean"] else 0
    )

# Print results for each course type
print("Data Statistics by Course Type:")
for course_type, stats in course_type_stats.items():
    round_decimals(stats)
    print(dataclass_to_table(stats, f"Stats for {course_type} race course"))
