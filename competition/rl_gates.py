import time
import inspect
import numpy as np
import pybullet as p

from functools import partial
from rich.tree import Tree
from rich import print

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger

try:
    from competition_utils import Command, thrusts
    from edit_this import Controller, yaw_rot, INITIAL_GATE_EXIT
except ImportError:
    # Test import.
    from .competition_utils import Command, thrusts
    from .edit_this import Controller, yaw_rot, INITIAL_GATE_EXIT

import pycffirmware
print("Module 'cffirmware' available")

if __name__ == "__main__":
  START = time.time()

  CONFIG_FACTORY = ConfigFactory()
  config = CONFIG_FACTORY.merge()
  config["quadrotor_config"]["gates"] = [[0.5, -2.5, 0, 0, 0, -1.57, 0], [5, -2.5, 0, 0, 0, -1.57, 1]]
  # TODO(shreepa): get gate dimensions / modify gate dimensions to be shorter to account for vicon
  final_exit_vector = yaw_rot(config["quadrotor_config"]["gates"][-1][-2]) @ INITIAL_GATE_EXIT

  last_gate = np.array([config["quadrotor_config"]["gates"][-1][i] for i in range(3)])
  config["quadrotor_config"]["task_info"]["stabilization_goal"] = last_gate + 0.1 * final_exit_vector

  CTRL_FREQ = config.quadrotor_config['ctrl_freq']
  CTRL_DT = 1/CTRL_FREQ

  FIRMWARE_FREQ = 500
  assert(config.quadrotor_config['pyb_freq'] % FIRMWARE_FREQ == 0), "pyb_freq must be a multiple of firmware freq"
  # The env.step is called at a firmware_freq rate, but this is not as intuitive to the end user, and so 
  # we abstract the difference. This allows ctrl_freq to be the rate at which the user sends ctrl signals, 
  # not the firmware. 
  config.quadrotor_config['ctrl_freq'] = FIRMWARE_FREQ
  env_func = partial(make, 'quadrotor', **config.quadrotor_config)
  firmware_wrapper = make('firmware',
              env_func, FIRMWARE_FREQ, CTRL_FREQ
              ) 
  obs, info = firmware_wrapper.reset()
  info['ctrl_timestep'] = CTRL_DT
  info['ctrl_freq'] = CTRL_FREQ
  print(info)
  env = firmware_wrapper.env

  vicon_obs = [obs[0], 0, obs[2], 0, obs[4], 0, obs[6], obs[7], obs[8], 0, 0, 0]
  ctrl = Controller(vicon_obs, info, config.use_firmware, verbose=config.verbose)

  action = np.zeros(4)
  cumulative_reward = 0
  done = False
  info = {}
  ep_start = time.time()
  for i in range(CTRL_FREQ*env.EPISODE_LEN_SEC):
    curr_time = i * CTRL_DT
    vicon_obs = [obs[0], 0, obs[2], 0, obs[4], 0, obs[6], obs[7], obs[8], 0, 0, 0]
    command_type, args = ctrl.cmdFirmware(curr_time, vicon_obs, cumulative_reward, done, info)

    if command_type == Command.FULLSTATE:
        firmware_wrapper.sendFullStateCmd(*args, curr_time)
    elif command_type == Command.TAKEOFF:
        firmware_wrapper.sendTakeoffCmd(*args)
    elif command_type == Command.LAND:
        firmware_wrapper.sendLandCmd(*args)
    elif command_type == Command.STOP:
        firmware_wrapper.sendStopCmd()
    elif command_type == Command.GOTO:
        firmware_wrapper.sendGotoCmd(*args)
    elif command_type == Command.NOTIFYSETPOINTSTOP:
        firmware_wrapper.notifySetpointStop()
    elif command_type == Command.NONE:
        pass
    else:
        raise ValueError("[ERROR] Invalid command_type.")
    
    obs, reward, done, info, action = firmware_wrapper.step(curr_time, action)
    ctrl.interStepLearn(action, obs, reward, done, info)
    cumulative_reward += reward

    """
    pos = [obs[0],obs[2],obs[4]]
    rpy = [obs[6],obs[7],obs[8]]
    vel = [obs[1],obs[3],obs[5]]
    bf_rates = [obs[9],obs[10],obs[11]]
    """

    if config.quadrotor_config.gui:
      sync(i, ep_start, CTRL_DT)

  env.close()

  elapsed_sec = time.time() - START
  print(f"\n{i} iterations (@{env.CTRL_FREQ}Hz) and {elapsed_sec} seconds i.e. {i/elapsed_sec} steps/sec for a {(i*CTRL_DT)/elapsed_sec}x speedup. Total Reward from Simulation {cumulative_reward}\n")
  print(f"Reward from trajectory: {ctrl.trajectory_reward}")

