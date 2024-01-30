import time
import inspect
import numpy as np
import pybullet as p
import random
import multiprocessing
from copy import deepcopy

from functools import partial
from rich.tree import Tree
from rich import print

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync
from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger

try:
    from competition_utils import Command, thrusts
    from edit_this import Controller, yaw_rot, INITIAL_GATE_EXIT, States
except ImportError:
    # Test import.
    from .competition_utils import Command, thrusts
    from .edit_this import Controller, yaw_rot, INITIAL_GATE_EXIT, States

import pycffirmware
print("Module 'cffirmware' available")

class ProcessInformation:

    def __init__(self, id, gates, reward):
        self.id = id
        self.gates = gates
        self.reward = reward

def run_one_environment(config, id, queue):
    # https://www.notion.so/Drone-Racing-429476c02eba498b9ba04e24b4a0f967?pvs=4#a625193707624b3e8a717e53655a934d
    # Visual of the gate spawn bounds
    gate_spawn_bounds = [
        [(-1, -1.8), (1.3, -0.8)],  # lower-left and upper-right bounds for each spawn region
        [(-1.2, -1), (0, 0.3)],
        [(-1.3, 0.7), (1.2, 1)],
        [(0.1, 1.4), (1, 1.5)]
    ]

    for i in range(4):
        bounds = gate_spawn_bounds[i]
        config["quadrotor_config"]["gates"].append([
            random.uniform(bounds[0][0], bounds[1][0]),  # x
            random.uniform(bounds[0][1], bounds[1][1]),  # y
            0,
            0,
            0,
            random.uniform(-np.pi/2, np.pi/2),  # gate orientation
            random.randint(0, 1)  # short or tall gate
        ])

    final_exit_vector = yaw_rot(config["quadrotor_config"]["gates"][-1][-2]) @ INITIAL_GATE_EXIT

    last_gate = np.array([config["quadrotor_config"]["gates"][-1][i] for i in range(3)])
    last_gate[2] = 0.525 if config["quadrotor_config"]["gates"][-1][6] == 0 else 0.3
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

    env = firmware_wrapper.env

    vicon_obs = [obs[0], 0, obs[2], 0, obs[4], 0, obs[6], obs[7], obs[8], 0, 0, 0]
    ctrl = Controller(vicon_obs, info, config.use_firmware, verbose=config.verbose, gui=False)

    action = np.zeros(4)
    cumulative_reward = 0
    done = False
    info = {}
    prev_state = ctrl.state
    for i in range(CTRL_FREQ*(ctrl.total_time + 2)):
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

        # whatever rewards accumulated before following trajectory should be ignored
        # as we are just utilizing out of box crazyflie tools (i.e. takeoff)
        if ctrl.state == States.FOLLOWING_TRAJ and prev_state == States.TAKEOFF:
            cumulative_reward = 0
        
        # preventing clutter in multiprocess
        # if prev_state != ctrl.state:
        #     print(f"Switched from {prev_state} to {ctrl.state}")

        prev_state = ctrl.state

    env.close()

    cumulative_reward -= 10 * np.linalg.norm(sum(ctrl.pos_errors) / len(ctrl.pos_errors))

    queue_item = ProcessInformation(id, config["quadrotor_config"]["gates"], cumulative_reward + ctrl.trajectory_reward)
    queue.put(queue_item)

if __name__ == "__main__":
    CONFIG_FACTORY = ConfigFactory()
    config = CONFIG_FACTORY.merge()
    config["quadrotor_config"]["gui"] = False
    config["quadrotor_config"]["gates"] = []

    START = time.time()

    num_processes = multiprocessing.cpu_count()
    processes = []
    data_queue = multiprocessing.Queue()

    for i in range(num_processes):
        process = multiprocessing.Process(target=run_one_environment, args=(deepcopy(config), i, data_queue,))
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    while not data_queue.empty():
        data: ProcessInformation = data_queue.get()
        print(f"Process {data.id} finished with {data.reward} reward")

    elapsed_sec = time.time() - START
    print(f"\nFinished Running {num_processes} processes in {round(elapsed_sec,2)} s")
