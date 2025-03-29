import os
import argparse
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

from quadrotor_diffusion.utils.simulator import play_trajectory, course_list_to_gate_list, create_perspective_rendering
from quadrotor_diffusion.utils.dataset.boundary_condition import PolynomialTrajectory
from quadrotor_diffusion.utils.trajectory import compute_tracking_error, derive_trajectory
from quadrotor_diffusion.utils.plotting import (
    plot_states,
    plot_ref_obs_states,
    course_base_plot,
    add_gates_to_course,
    add_trajectory_to_course
)
from quadrotor_diffusion.utils.quad_logging import iprint as print

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--course_type', type=str, help='Course types [linear, u, zigzag]', required=True)
parser.add_argument('-s', '--sample_number', type=str, help="Sample number", required=True)
parser.add_argument('-t', '--trajectory_number', type=str, help="Trajectory number", required=True)
parser.add_argument('-b', '--base-dir', type=str, help='Data directory', default="data/courses")
parser.add_argument('-g', '--gui', action='store_true', help="Use gui on playback in sim.")
parser.add_argument('-i', '--invalid', action='store_true', help="Examining invalid trajectory")

args = parser.parse_args()

base_dir = os.path.join(os.path.join(args.base_dir, args.course_type), args.sample_number)
course_filename = os.path.join(base_dir, 'course.npy')
course = np.load(course_filename)
course = np.vstack((course, course[1]))

if args.invalid:
    invalid_dir = os.path.join(base_dir, "invalid")
    invalid_trajectory_filename = [x for x in os.listdir(invalid_dir) if x.startswith(f"{args.trajectory_number}_")]
    if len(invalid_trajectory_filename) != 1:
        raise RuntimeError("Trajectory number doesn't exist or has duplicates?")
    invalid_trajectory_filename = os.path.join(invalid_dir, invalid_trajectory_filename[0])
    invalid_trajectory: np.ndarray = np.load(invalid_trajectory_filename)

    _, ax = course_base_plot()
    add_gates_to_course(course, ax)
    add_trajectory_to_course(invalid_trajectory)
    plt.show()
    sys.exit(0)

valid_dir = os.path.join(base_dir, "valid")
trajectory_filename = [x for x in os.listdir(valid_dir) if x.startswith(f"{args.trajectory_number}_")]
if len(trajectory_filename) != 1:
    raise RuntimeError("Trajectory number doesn't exist or has duplicates?")
trajectory_filename = os.path.join(valid_dir, trajectory_filename[0])
with open(trajectory_filename, "rb") as trajectory_file:
    trajectory: PolynomialTrajectory = pickle.load(trajectory_file)
    trajectory.states.append(trajectory.states[2])
    trajectory.segment_lengths.append(trajectory.segment_lengths[1])
    trajectory.states.append(trajectory.states[3])
    trajectory.segment_lengths.append(trajectory.segment_lengths[2])
    trajectory.states.append(trajectory.states[4])
    trajectory.segment_lengths.append(trajectory.segment_lengths[3])
    trajectory.states.append(trajectory.states[5])

_, ax = course_base_plot()
add_gates_to_course(course, ax)
add_trajectory_to_course(trajectory)
plt.show()


ref_pos = trajectory.as_ref_pos()
gates = course_list_to_gate_list(course[1:-1])

worked, drone_states = play_trajectory(ref_pos, use_gui=args.gui)

if not worked:
    print("Crashed")
else:
    print("Avg errors: ", compute_tracking_error(ref_pos, drone_states[0]))

# plot_states(
#     ref_pos,
#     derive_trajectory(ref_pos, 30),
#     derive_trajectory(ref_pos, 30, 2),
#     f"Reference Trajectory {args.course_type}-{args.sample_number}-{args.trajectory_number}",
#     None,
# )

# plot_ref_obs_states(
#     ref_pos,
#     derive_trajectory(ref_pos, 30),
#     derive_trajectory(ref_pos, 30, 2),
#     drone_states[0],
#     drone_states[1],
#     derive_trajectory(drone_states[1], 30),
#     f"Reference vs observed for {args.course_type}-{args.sample_number}-{args.trajectory_number}",
#     None
# )

_, ax = course_base_plot()
add_gates_to_course(course, ax)
add_trajectory_to_course(drone_states[0], velocity_profile=drone_states[1])
add_trajectory_to_course(trajectory, reference=True)
plt.show()


# Example data
create_perspective_rendering(drone_states, course, "test.mp4", trajectory.as_ref_pos())
