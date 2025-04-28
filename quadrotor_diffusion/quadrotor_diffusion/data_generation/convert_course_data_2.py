import os
import argparse
import pickle
import random

import numpy as np
import tqdm

from quadrotor_diffusion.utils.dataset.boundary_condition import PolynomialTrajectory
from quadrotor_diffusion.utils.dataset.dataset import QuadrotorRaceSegmentDataset
from quadrotor_diffusion.utils.dataset.normalizer import NoNormalizer

TRAJECTORY_SLICE_LENGTH = 128

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--courses', nargs='+', type=str, help='Course types to create [linear, u, zigzag]')
parser.add_argument('-b', '--base-dir', type=str, help='Data directory', default="data/courses")
parser.add_argument('-d', '--debug', action='store_true', help="Turn on debug mode.")
args = parser.parse_args()


write_to = os.path.join(args.base_dir, "vae3")
# Only normalize in the torch dataset if needed
dataset = QuadrotorRaceSegmentDataset('data', args.courses, 128, 0, NoNormalizer())
for idx, d in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
    with open(os.path.join(write_to, f"{idx}.pkl"), 'wb') as f:
        pickle.dump(d, f)
