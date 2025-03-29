import os
import pickle
import argparse
import time

import numpy as np

from quadrotor_diffusion.data_generation.linear_course import generate_linear
from quadrotor_diffusion.data_generation.u_course import generate_u
from quadrotor_diffusion.data_generation.zig_zag import generate_zig_zag
from quadrotor_diffusion.data_generation.triangle import generate_triangle
from quadrotor_diffusion.data_generation.triangle import generate_triangle
from quadrotor_diffusion.data_generation.pill import generate_pill
from quadrotor_diffusion.data_generation.square import generate_square
from quadrotor_diffusion.utils.quad_logging import iprint as print

MIN_VALID_TRAJECTORIES = 20

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--courses', nargs='+', type=str, help='Course types to create [linear, u, zigzag]')
parser.add_argument('-b', '--base-dir', type=str, help='Data directory', default="data/courses")
parser.add_argument('-d', '--debug', action='store_true', help="Turn on debug mode.")
args = parser.parse_args()

os.environ['DEBUG'] = 'True' if args.debug else 'False'

counter = 0

# for each course type
current_course_idx = dict()
for course_type in args.courses:
    base_dir = os.path.join(args.base_dir, course_type)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        current_course_idx[course_type] = 0
        continue

    existing_indices = [int(dir_name) for dir_name in os.listdir(base_dir) if dir_name.isdigit()]
    current_course_idx[course_type] = max(existing_indices, default=0) + 1

while True:
    course_type = args.courses[counter % len(args.courses)]

    start = time.time()
    if course_type == "linear":
        course, trajectories = generate_linear()
    elif course_type == "u":
        course, trajectories = generate_u()
    elif course_type == "zig_zag":
        # Right now zig zag trajectories are >12s so don't use this because no valid trajectories will be made
        course, trajectories = generate_zig_zag()
    elif course_type == "triangle":
        course, trajectories = generate_triangle()
    elif course_type == "pill":
        course, trajectories = generate_pill()
    elif course_type == "square":
        course, trajectories = generate_square()
    else:
        raise ValueError("Invalid course")
    total = time.time() - start
    print(f"[{current_course_idx[course_type]}] Generating {course_type}: {len(trajectories)} samples in {total:.2f}s")

    trajectories.sort(key=lambda trajectory: sum(trajectory.segment_lengths))
    if len(trajectories) < MIN_VALID_TRAJECTORIES:
        continue

    base_dir = os.path.join(os.path.join(args.base_dir, course_type), f"{current_course_idx[course_type]}")
    os.makedirs(base_dir)

    course_file_name = os.path.join(base_dir, "course.npy")
    np.save(course_file_name, course)

    valid_dir = os.path.join(base_dir, "valid")

    os.makedirs(valid_dir)

    for idx, trajectory in enumerate(trajectories):
        valid_file_name = os.path.join(valid_dir, f"{idx}_({trajectory}).pkl")
        with open(valid_file_name, 'wb') as valid_file:
            pickle.dump(trajectory, valid_file)

    counter += 1
    current_course_idx[course_type] += 1
