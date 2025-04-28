import os
import argparse
import random
import pickle

from scipy.spatial.transform import Rotation as R
import numpy as np
import tqdm

from quadrotor_diffusion.utils.dataset.boundary_condition import PolynomialTrajectory
from quadrotor_diffusion.utils.plotting import plot_states, add_gates_to_course, add_trajectory_to_course, course_base_plot

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

    trajectory.states.append(trajectory.states[4])
    trajectory.segment_lengths.append(trajectory.segment_lengths[3])

    trajectory_np = trajectory.as_ref_pos(30)

    row_0 = np.tile(trajectory_np[0], (LOCAL_CONTEXT_SIZE, 1))
    trajectory_np = np.vstack((row_0, trajectory_np))

    ref_pos_idx = LOCAL_CONTEXT_SIZE

    for gate_idx in range(number_gates):
        # All following gates up to the current gate (i.e. don't include current gate as part of global context)
        if cyclic:
            global_context = np.vstack((course[gate_idx+1:], course[1:gate_idx])).copy()
        else:
            global_context = course[gate_idx+1:-1].copy()

        ending_gate = trajectory.states[gate_idx+1]
        ending_idx = np.argmin(
            np.linalg.norm(
                trajectory_np[ref_pos_idx:ref_pos_idx+TRAJECTORY_SLICE_LENGTH] -
                np.array([ending_gate.x.s, ending_gate.y.s, ending_gate.z.s]),
                axis=1,
            )
        )

        local_context = trajectory_np[ref_pos_idx - LOCAL_CONTEXT_SIZE: ref_pos_idx+LOCAL_CONTEXT_SIZE].copy()
        trajectory_slice = trajectory_np[ref_pos_idx:ref_pos_idx+TRAJECTORY_SLICE_LENGTH].copy()

        if gate_idx == 0:
            yaw = global_context[0][3] + np.pi/2
        else:
            p2 = trajectory_np[ref_pos_idx]
            p1 = trajectory_np[ref_pos_idx - 1]

            delta = p2 - p1
            yaw = np.arctan2(delta[1], delta[0])

        rotation = R.from_euler('z', -yaw).as_matrix()

        rot_xy = rotation[:2, :2]
        trans_xy = trajectory_np[ref_pos_idx, :2]

        xy_shifted = trajectory_slice[:, :2] - trans_xy
        xy_ego = xy_shifted @ rot_xy.T
        trajectory_slice[:, :2] = xy_ego

        local_context_xy = local_context[:, :2]
        local_context_xy = local_context_xy - trans_xy
        local_context_xy = local_context_xy @ rot_xy.T
        local_context[:, :2] = local_context_xy

        global_context_xy = global_context[:, :2]
        global_context_xy = global_context_xy - trans_xy
        global_context_xy = global_context_xy @ rot_xy.T
        global_context[:, :2] = global_context_xy

        global_context[:, 3] -= yaw
        global_context[:, 3] = np.arctan2(np.sin(global_context[:, 3]), np.cos(global_context[:, 3]))

        with open(f"{filename}_{gate_idx}.pkl", 'wb') as file:
            pickle.dump({
                "global_context": global_context,
                "local_context": local_context,
                "trajectory_slice": trajectory_slice
            }, file)

        # I just added this part do you see any bugs (within the forloop)
        for i in range(2):
            if ending_idx < 50:
                continue

            offset = random.randint(25, 45)
            ref_pos_idx += offset

            if cyclic:
                global_context = np.vstack((course[gate_idx+1:], course[1:gate_idx])).copy()
            else:
                global_context = course[gate_idx+1:-1].copy()

            local_context = trajectory_np[ref_pos_idx - LOCAL_CONTEXT_SIZE: ref_pos_idx+LOCAL_CONTEXT_SIZE].copy()
            trajectory_slice = trajectory_np[ref_pos_idx:ref_pos_idx+TRAJECTORY_SLICE_LENGTH].copy()

            p2 = trajectory_np[ref_pos_idx]
            p1 = trajectory_np[ref_pos_idx - 1]

            delta = p2 - p1
            yaw = np.arctan2(delta[1], delta[0])

            rotation = R.from_euler('z', -yaw).as_matrix()

            rot_xy = rotation[:2, :2]
            trans_xy = trajectory_np[ref_pos_idx, :2]

            xy_shifted = trajectory_slice[:, :2] - trans_xy
            xy_ego = xy_shifted @ rot_xy.T
            trajectory_slice[:, :2] = xy_ego

            local_context_xy = local_context[:, :2]
            local_context_xy = local_context_xy - trans_xy
            local_context_xy = local_context_xy @ rot_xy.T
            local_context[:, :2] = local_context_xy

            global_context_xy = global_context[:, :2]
            global_context_xy = global_context_xy - trans_xy
            global_context_xy = global_context_xy @ rot_xy.T
            global_context[:, :2] = global_context_xy

            global_context[:, 3] -= yaw
            global_context[:, 3] = np.arctan2(np.sin(global_context[:, 3]), np.cos(global_context[:, 3]))

            with open(f"{filename}_{gate_idx}_{i}.pkl", 'wb') as file:
                pickle.dump({
                    "global_context": global_context,
                    "local_context": local_context,
                    "trajectory_slice": trajectory_slice
                }, file)

            ref_pos_idx -= offset

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

                    filename = os.path.join(args.base_dir, "diffusion4", f"{sample_num}")
                    process_one_example(filename, trajectory, course, n_gates[course_type], cyclic[course_type])
                    sample_num += 1
                    num_trajectories_in_course += 1

                    if not cyclic[course_type] and num_trajectories_in_course >= 10:
                        break
