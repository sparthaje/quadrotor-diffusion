from functools import partial

import numpy as np
import yaml
import time
import argparse
import sys

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from quadrotor_diffusion.utils.trajectory import (
    derive_target_velocities,
    derive_target_accelerations,
)


def play_trajectory(ref_pos: np.ndarray):
    """
    Plays a trajectory sample in simulator

    Parameters:
    - ref_pos: nx3 trajectory matrix

    Returns: No crash (bool), drone states (np.ndarray)
    """
    sys.argv.extend(["--overrides", "quadrotor_diffusion/quadrotor_diffusion/utils/play_trajectory.yaml"])
    parser = argparse.ArgumentParser(description='Generate unconditioned diffusion data.')
    parser.add_argument('--overrides', type=str, help='Config file')
    args = parser.parse_args()

    with open(args.overrides, 'r') as file:
        CONFIG = yaml.safe_load(file)

    CTRL_FREQ = CONFIG["quadrotor_config"]["ctrl_freq"]

    ref_vel = derive_target_velocities(ref_pos, CTRL_FREQ)
    ref_acc = derive_target_accelerations(ref_vel, CTRL_FREQ)
    reference = np.stack((ref_pos, ref_vel, ref_acc), axis=1)

    config = ConfigFactory().merge()
    config["quadrotor_config"]["seed"] = int(time.time())
    config["quadrotor_config"]["init_state"]["init_x"] = reference[0][0][0]
    config["quadrotor_config"]["init_state"]["init_y"] = reference[0][0][1]
    config["quadrotor_config"]["init_state"]["init_z"] = reference[0][0][2]
    config["quadrotor_config"]["init_state"]["init_psi"] = 0.0
    config["quadrotor_config"]["task_info"]["stabilization_goal"] = reference[-1][0]

    CTRL_DT = 1 / CTRL_FREQ
    FIRMWARE_FREQ = 500
    assert (config.quadrotor_config['pyb_freq'] % FIRMWARE_FREQ == 0), "pyb_freq must be a multiple of firmware freq"
    config.quadrotor_config['ctrl_freq'] = FIRMWARE_FREQ

    env_func = partial(make, 'quadrotor', **config.quadrotor_config)
    firmware_wrapper = make('firmware',
                            env_func, FIRMWARE_FREQ, CTRL_FREQ
                            )

    obs, info = firmware_wrapper.reset()
    info['ctrl_timestep'] = CTRL_DT
    info['ctrl_freq'] = CTRL_FREQ
    env = firmware_wrapper.env
    action = np.zeros(4)

    drone_states = [[obs[0], obs[2], obs[4]]]
    for step in range(reference.shape[0]):
        curr_time = step * CTRL_DT
        args = [reference[step][0], reference[step][1], reference[step][2], 0.0, np.zeros(3)]

        firmware_wrapper.sendFullStateCmd(*args, curr_time)
        obs, reward, _, info, action = firmware_wrapper.step(curr_time, action)

        if step > 0:
            drone_states.append([obs[0], obs[2], obs[4]])

        if reward < 0:
            env.close()
            return False, np.array(drone_states)

    env.close()
    return True, np.array(drone_states)
