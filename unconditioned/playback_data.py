# Sample to visualize
SAMPLE_NUM = 13543

from functools import partial

import numpy as np
import yaml
import time
import argparse

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

from get_data.traj_generator import (
  derive_target_velocities,
  derive_target_accelerations
)
from get_data.plotting_utils import (
  plot_reference_time_series,
  view_reference_in_3d,
  draw_trajectory_on_pybullet
)

parser = argparse.ArgumentParser(description='Generate unconditioned diffusion data.')
parser.add_argument('--overrides', type=str, help='Config file')
args = parser.parse_args()

with open(args.overrides, 'r') as file:
  CONFIG = yaml.safe_load(file)

DATA_CONFIG = CONFIG["data"]
CTRL_FREQ   = CONFIG["quadrotor_config"]["ctrl_freq"]

ref_pos   = np.load(f"data/{SAMPLE_NUM}.npy")
ref_vel   = derive_target_velocities(ref_pos, CTRL_FREQ)
ref_acc   = derive_target_accelerations(ref_vel, CTRL_FREQ)
reference = np.stack((ref_pos, ref_vel, ref_acc), axis=1)

plot_reference_time_series(reference)
view_reference_in_3d(reference)

config = ConfigFactory().merge()
config["quadrotor_config"]["seed"]                            = int(time.time())
config["quadrotor_config"]["gui"]                             = True
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
    print("Crashed")
    env.close()

env.close()
average_error = total_pos_tracking_error / (step - 1)
print("=" * 100 + "\n")
print(f"Observed average tracking error of {average_error:.2f}")
