import multiprocessing
from functools import partial

import os
import numpy as np
import yaml
import time
from itertools import count
import argparse

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from traj_generator import (
  get_positions_from_boundary_conditions,
  derive_target_velocities,
  derive_target_accelerations
)
from plotting_utils import (
  plot_reference_time_series,
  view_reference_in_3d,
  draw_trajectory_on_pybullet
)

parser = argparse.ArgumentParser(description='Generate unconditioned diffusion data.')
parser.add_argument('--overrides', type=str, help='Config file')
args = parser.parse_args()

with open(args.overrides, 'r') as file:
  CONFIG      = yaml.safe_load(file)

DATA_CONFIG = CONFIG["data"]
CTRL_FREQ   = CONFIG["quadrotor_config"]["ctrl_freq"]
USING_GUI   = CONFIG["quadrotor_config"]["gui"]
NUM_SEGMENTS = DATA_CONFIG["num_segments"]
LIMS         = DATA_CONFIG["lims"]
TOTAL_TIME   = DATA_CONFIG["total_time"]
NUM_SAMPLES  = DATA_CONFIG["num_samples"]
MAX_AVG_DEV  = DATA_CONFIG["max_avg_dev"] * 0.01

def sample_random_trajectory():
  pos_bounds = np.column_stack((
    np.random.uniform(LIMS["x"][0], LIMS["x"][1], size=NUM_SEGMENTS + 1),   # X
    np.random.uniform(LIMS["y"][0], LIMS["y"][1], size=NUM_SEGMENTS + 1),   # Y
    np.random.choice(LIMS["z"], size=NUM_SEGMENTS + 1)                      # Z
  ))

  # time between positions
  xy_diffs      = np.diff(pos_bounds, axis=0)[:, :2]
  xy_diffs_norm = np.linalg.norm(xy_diffs, axis=1)
  ratios        = xy_diffs_norm / sum(xy_diffs_norm)
  segment_lengths = ratios * TOTAL_TIME
  
  vel_bounds = np.column_stack((
    np.random.uniform(LIMS["vx"][0], LIMS["vx"][1], size=NUM_SEGMENTS),    # VX
    np.random.uniform(LIMS["vy"][0], LIMS["vy"][1], size=NUM_SEGMENTS),    # VY
    np.random.uniform(LIMS["vz"][0], LIMS["vz"][1], size=NUM_SEGMENTS),    # VZ
  ))
  vel_bounds = np.row_stack((np.array([0.0, 0.0, 0.0]), vel_bounds))
  
  acc_bounds = np.column_stack((
    np.random.uniform(LIMS["ax"][0], LIMS["ax"][1], size=NUM_SEGMENTS),    # AX
    np.random.uniform(LIMS["ay"][0], LIMS["ay"][1], size=NUM_SEGMENTS),    # AY
    np.random.uniform(LIMS["az"][0], LIMS["az"][1], size=NUM_SEGMENTS),    # AZ
  ))
  acc_bounds = np.row_stack((np.array([0.0, 0.0, 0.0]), acc_bounds))
  
  get_sigma = lambda seg, xyz: np.array([pos_bounds[seg][xyz], pos_bounds[seg+1][xyz], vel_bounds[seg][xyz], vel_bounds[seg+1][xyz], acc_bounds[seg][xyz], acc_bounds[seg+1][xyz], 0.0, 0.0])
  boundary_conditions = [[get_sigma(seg, xyz) for xyz in range(3)] for seg in range(0, NUM_SEGMENTS)]
  
  ref_pos   = get_positions_from_boundary_conditions(boundary_conditions, segment_lengths, CTRL_FREQ)
  ref_vel   = derive_target_velocities(ref_pos, CTRL_FREQ)
  ref_acc   = derive_target_accelerations(ref_vel, CTRL_FREQ)

  reference = np.stack((ref_pos, ref_vel, ref_acc), axis=1)
  return reference

def trajectory_is_valid(reference):
  for _, vel, accel in reference:
    if np.linalg.norm(accel) > LIMS["gv"]["shape"] * np.linalg.norm(vel) ** LIMS["gv"]["exp"] + LIMS["gv"]["intercept"]:
      # print(np.linalg.norm(vel), np.linalg.norm(accel))
      return False

  if USING_GUI:
    plot_reference_time_series(reference)
    view_reference_in_3d(reference)
  
  config = ConfigFactory().merge()
  config["quadrotor_config"]["seed"]                            = int(time.time())
  config["quadrotor_config"]["init_state"]["init_x"]            = reference[0][0][0]
  config["quadrotor_config"]["init_state"]["init_y"]            = reference[0][0][1]
  config["quadrotor_config"]["init_state"]["init_z"]            = reference[0][0][2]
  config["quadrotor_config"]["init_state"]["init_psi"]          = 0.0
  config["quadrotor_config"]["task_info"]["stabilization_goal"] = reference[-1][0]
  
  CTRL_DT = 1 / CTRL_FREQ
  FIRMWARE_FREQ = 500
  assert(config.quadrotor_config['pyb_freq'] % FIRMWARE_FREQ == 0), "pyb_freq must be a multiple of firmware freq"
  config.quadrotor_config['ctrl_freq'] = FIRMWARE_FREQ
  
  env_func = partial(make, 'quadrotor', **config.quadrotor_config)
  firmware_wrapper = make('firmware',
              env_func, FIRMWARE_FREQ, CTRL_FREQ
              )

  obs, info             = firmware_wrapper.reset()
  info['ctrl_timestep'] = CTRL_DT
  info['ctrl_freq']     =  CTRL_FREQ
  env                   = firmware_wrapper.env
  action                = np.zeros(4)
  
  if USING_GUI:
    draw_trajectory_on_pybullet(info, reference[:, 0, 0], reference[:, 0, 1], reference[:, 0, 2])
  
  total_pos_tracking_error = 0.0
  for step in range(reference.shape[0]):
    curr_time = step * CTRL_DT
    args      = [reference[step][0], reference[step][1], reference[step][2], 0.0, np.zeros(3)]
    
    firmware_wrapper.sendFullStateCmd(*args, curr_time)
    obs, reward, _, info, action = firmware_wrapper.step(curr_time, action)
    
    if step > 0:
      current_position         = np.array([obs[0], obs[2], obs[4]])
      total_pos_tracking_error += np.linalg.norm(current_position - reference[step - 1][0])
    
    if reward < 0:
      print("Failed via crash")
      env.close()
      return False
  
  average_error = total_pos_tracking_error / (step - 1)
  if average_error > MAX_AVG_DEV:
    print("Failed via tracking")
    return False
  
  env.close()
  return True

def get_valid_random_trajectory():
  for attempt in count(1):
    trajectory = sample_random_trajectory()
    if trajectory_is_valid(trajectory):
        return trajectory, attempt

def main():
  existing_samples  = [int(f.split('.')[0]) for f in os.listdir("data") if f.endswith('.npy')]
  last_sample_saved = 0 if len(existing_samples) == 0 else max(existing_samples)
  # Number of total attempts used to create `num_samples` valid trajectories
  total_attempts    = 0
  
  start_time = time.time()
  for i in range(NUM_SAMPLES):
    filename = f"data/{last_sample_saved + i}.npy"
    trajectory, attempts = get_valid_random_trajectory()
    # np.save(filename, trajectory)
    total_attempts += attempts
  total_time = time.time() - start_time
  
  average_attempts = total_attempts / NUM_SAMPLES
  print(f"Created {NUM_SAMPLES} samples using {average_attempts} attempts on average in {total_time}s")

if __name__ == "__main__":
  main()
