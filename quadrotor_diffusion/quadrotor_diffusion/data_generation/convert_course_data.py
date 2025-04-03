import os
import argparse
import pickle

import numpy as np
import tqdm

from quadrotor_diffusion.utils.dataset.boundary_condition import PolynomialTrajectory

LOCAL_CONTEXT_SIZE = 20
TRAJECTORY_SLICE_LENGTH = 128

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--courses', nargs='+', type=str, help='Course types to create [linear, u, zigzag]')
parser.add_argument('-b', '--base-dir', type=str, help='Data directory', default="data/courses")
parser.add_argument('-d', '--debug', action='store_true', help="Turn on debug mode.")
args = parser.parse_args()


def process_one_example(filename: str, trajectory: PolynomialTrajectory, course: np.ndarray, number_gates: int, cyclic: bool):
    """
    Takes trajectory and course and converts them to diffusion examples
    """

    trajectory.states.append(trajectory.states[2])
    trajectory.segment_lengths.append(trajectory.segment_lengths[1])

    trajectory.states.append(trajectory.states[3])
    trajectory.segment_lengths.append(trajectory.segment_lengths[2])

    trajectory_np = trajectory.as_ref_pos(30)

    row_0 = np.tile(trajectory_np[0], (LOCAL_CONTEXT_SIZE, 1))
    trajectory_np = np.vstack((row_0, trajectory_np))

    ref_pos_idx = LOCAL_CONTEXT_SIZE

    for gate_idx in range(number_gates):
        # All following gates up to the current gate (i.e. don't include current gate as part of global context)
        if cyclic:
            global_context = np.vstack((course[gate_idx+1:], course[1:gate_idx]))
        else:
            global_context = course[gate_idx+1:-1]

        ending_gate = trajectory.states[gate_idx+1]
        ending_idx = np.argmin(
            np.linalg.norm(
                trajectory_np[ref_pos_idx:ref_pos_idx+TRAJECTORY_SLICE_LENGTH] -
                np.array([ending_gate.x.s, ending_gate.y.s, ending_gate.z.s]),
                axis=1,
            )
        )

        local_context = trajectory_np[ref_pos_idx - LOCAL_CONTEXT_SIZE: ref_pos_idx+LOCAL_CONTEXT_SIZE]
        trajectory_slice = trajectory_np[ref_pos_idx:ref_pos_idx+TRAJECTORY_SLICE_LENGTH]

        with open(f"{filename}_{gate_idx}.pkl", 'wb') as file:
            pickle.dump({
                "global_context": global_context,
                "local_context": local_context,
                "trajectory_slice": trajectory_slice
            }, file)

        ref_pos_idx += ending_idx


sample_num = 0
n_gates = {
    "linear": 4,
    "u": 4,
    "triangle": 4,
    "pill": 5,
    "square": 5,
}
cyclic = {
    "linear": False,
    "u": False,
    "triangle": True,
    "pill": True,
    "square": True,
}

for course_type in args.courses:
    course_dir = os.path.join(args.base_dir, course_type)
    if not os.path.isdir(course_dir):
        continue

    for sample in tqdm.tqdm(os.listdir(course_dir)):
        sample_dir = os.path.join(course_dir, sample)
        if not os.path.isdir(sample_dir):
            continue

        course_filename = os.path.join(sample_dir, "course.npy")
        if not os.path.exists(course_filename):
            continue
        course = np.load(course_filename)

        # Find valid trajectories
        valid_dir = os.path.join(sample_dir, "valid")
        num_trajectories_in_course = 0
        if os.path.exists(valid_dir):
            for valid_file in os.listdir(valid_dir):
                if valid_file.endswith(".pkl"):
                    traj_filename = os.path.join(valid_dir, valid_file)

                    with open(traj_filename, "rb") as trajectory_file:
                        trajectory: PolynomialTrajectory = pickle.load(trajectory_file)

                    filename = os.path.join(args.base_dir, "diffusion2", f"{sample_num}")
                    process_one_example(filename, trajectory, course, n_gates[course_type], cyclic[course_type])
                    sample_num += 1
                    num_trajectories_in_course += 1

                    if not cyclic[course_type] and num_trajectories_in_course >= 10:
                        break
