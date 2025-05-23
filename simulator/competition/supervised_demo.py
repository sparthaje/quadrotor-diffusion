from functools import partial
import random

import os
import numpy as np
from time import sleep
import torch
from old_model_training.model import BoundaryPredictor
from sys import argv
import pandas as pd

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

try:
    from competition_utils import Command, thrusts
    from data_collection_ctrls import Controller, yaw_rot, INITIAL_GATE_EXIT, States
except ImportError:
    # Test import.
    from .competition_utils import Command, thrusts
    from .data_collection_ctrls import Controller, yaw_rot, INITIAL_GATE_EXIT, States
    
def newest_file_in_directory(directory):
    # Get list of files in the directory
    files = os.listdir(directory)
    if not files:
        return None  # Return None if directory is empty

    # Get file paths along with their modification times
    files_with_times = [(os.path.join(directory, file), os.path.getmtime(os.path.join(directory, file))) for file in files if ".DS_Store" not in file]

    # Sort files based on modification time, newest first
    newest_file = max(files_with_times, key=lambda x: x[1])

    return newest_file[0]  # Return the path of the newest file

RENDER_GATES = "gates" in argv
GENERATE_CSV = "gen_csv" in argv

# Copied from `build_supervised_demo.py`
# NOTE(shreepa): MAKE SURE THAT THE FIRST GATE IS TAKEOFF POSITION AND LAST TWO GATES ARE DUMMY GATES WHEN RUNNING BUILD DEMO
gate_x = [1.0, 0.29289321881345254, 0.29289321881345254, 1.0, 0.9999999999999999, 0.2928932188134523, -0.41421356237309526]
gate_y = [-1.7, -0.9928932188134524, 0.007106781186547617, 0.7142135623730953, 1.7142135623730952, 2.4213203435596427, 3.12842712474619]
gate_z = [1, 0, 0, 1, 1, 1, 1]
heights = [0.3, 0.525, 0.525, 0.3, 0.3, 0.3, 0.3]
gate_theta = [0.7853981633974483, 0.0, -0.7853981633974482, 1.1102230246251565e-16, 0.7853981633974484, 0.7853981633974484, 0.7853981633974484]
d_vals = [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
rel_angles = [0.7853981633974483, -0.7853981633974483, -0.7853981633974483, 0.7853981633974483, 0.7853981633974483, 0.0, 0.0]

gate_x = gate_x[:-2]
gate_y = gate_y[:-2]
gate_z = gate_z[:-2]
gate_theta = gate_theta[:-2]
d_vals = [(d_vals[i], d_vals[i+1], d_vals[i+2]) for i in range(len(gate_x))]
rel_angles = [(rel_angles[i], rel_angles[i+1], rel_angles[i+2]) for i in range(len(gate_x))]
heights = [(heights[i], heights[i+1], heights[i+2]) for i in range(len(gate_x))]

# assert that all the lists have the same length
assert len(gate_x) == len(gate_y) == len(gate_z) == len(gate_theta) == len(d_vals) == len(rel_angles) == len(heights), "All lists must have the same length"
optimal_vals = None

model = BoundaryPredictor(8)
model.eval()
print(newest_file_in_directory("../models"))
model.load_state_dict(torch.load(newest_file_in_directory("../models")))
model.eval()
normalized_vals = {'best_v': 1.5555555555555554, 'best_t': 2.25, 'v0': 2.0}

boundary_conditions = []
boundary_conditions.append([
  0, 
  np.array([gate_x[0], gate_y[0], heights[0][0]]),
  np.zeros(3),
  np.zeros(3),
  np.zeros(3),
  gate_theta[0],
])

i = 0
print("Inputs normalized, outputs not normalized")
for gx, gy, gz, h, gt, d, ra in list(zip(gate_x, gate_y, gate_z, heights, gate_theta, d_vals, rel_angles))[:-1]:
  # v0,z0,d1,theta1,z1,d2,theta2,z2,best_v,best_t
  inputs = np.array([
    np.linalg.norm(boundary_conditions[-1][2]) / normalized_vals["v0"],  # v_0
    (h[0] == 0.3) * 1.0,  # z_0
    
    (d[1] - 0.8) / (1.5 - 0.8),  # d_1
    ra[1] / (np.pi /4),  # ra_1
    (h[1] == 0.3) * 1.0,  # z_1
    
    (d[2] - 0.8) / (1.5 - 0.8),  # d_2,
    ra[2] / (np.pi /4),  # ra_2,
    (h[2] == 0.3) * 1.0,  # z_2
  ])

  x = list(inputs)
  inputs = torch.tensor(inputs, dtype=torch.float32)
  v, t = model(inputs).detach().numpy()
  v = v * normalized_vals["best_v"]
  t = t * normalized_vals["best_t"]
  if optimal_vals is not None:
    v, t = optimal_vals[i]
  print(x, v, t)
  print('-----')
  
  theta = np.arctan2(np.sin(ra[1] + gt), np.cos(ra[1] + gt))
  boundary_conditions.append([
    t,
    np.array([gate_x[i+1], gate_y[i+1], heights[i+1][0]]),
    yaw_rot(theta) @ (v * INITIAL_GATE_EXIT),
    np.zeros(3),
    np.zeros(3),
    theta,
  ])
  i += 1
  
STAB_DIST = 0.5
END = boundary_conditions[-1][1] + STAB_DIST * (yaw_rot(gate_theta[-1]) @ INITIAL_GATE_EXIT)

boundary_conditions.append([
  1, 
  END,
  np.zeros(3),
  np.zeros(3),
  np.zeros(3),
  gate_theta[-1],
])

print("--------------------")
for b in boundary_conditions:
  for x in b:
    print(x)
    
  print("--------------------")
  print()

CONFIG_FACTORY = ConfigFactory()
config = CONFIG_FACTORY.merge()
config["quadrotor_config"]["gui"] = not GENERATE_CSV
config["quadrotor_config"]["gates"] = []
config["quadrotor_config"]["obstacles"] = []

if RENDER_GATES:
  config["quadrotor_config"]["gates"] = [
    [gx, gy, 0, 0, 0, gt, gz] for gx, gy, gz, gt in list(zip(gate_x, gate_y, gate_z, gate_theta))[1:]
  ]
else:
  config["quadrotor_config"]["gates"] = []

config["quadrotor_config"]["init_state"]["init_x"] = boundary_conditions[0][1][0]
config["quadrotor_config"]["init_state"]["init_y"] = boundary_conditions[0][1][1]
config["quadrotor_config"]["init_state"]["init_z"] = boundary_conditions[0][1][2]
velocity_vec = np.zeros(3)
config["quadrotor_config"]["init_state"]["init_x_dot"] = velocity_vec[0]
config["quadrotor_config"]["init_state"]["init_y_dot"] = velocity_vec[1]
config["quadrotor_config"]["init_state"]["init_z_dot"] = velocity_vec[2]

config["quadrotor_config"]["init_state"]["init_psi"] = 0.1  # gate_theta[0]
  
config["quadrotor_config"]["task_info"]["stabilization_goal"] = END
CTRL_FREQ = config.quadrotor_config['ctrl_freq']
CTRL_DT = 1/CTRL_FREQ

FIRMWARE_FREQ = 500
assert(config.quadrotor_config['pyb_freq'] % FIRMWARE_FREQ == 0), "pyb_freq must be a multiple of firmware freq"
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
ctrl = Controller(vicon_obs, info, config.use_firmware, verbose=config.verbose)
total_time, waypoints, coeffs, Ts, Yaws = ctrl.build_traj_with_boundaries(boundary_conditions)

if GENERATE_CSV:
  filename = input("csv name? ")
  f = open("../sim-data/" + filename + ".csv", "w")
  f2 = open("../sim-data/" + filename + "_cmds.csv", "w")
  f3 = open("../sim-data/" + filename + "_gates.csv", "w")
  f.write("t,x,y,z\n")
  f2.write("t,x,y,z\n")
  df = pd.DataFrame()
  df["gate_x"] = gate_x
  df["gate_y"] = gate_y
  df.to_csv(f3, index=False)
  ctrl.file = f
  ctrl.cmd_file = f2

## Print stuff for IRL testing
print("---------")
print("coeff_x =", coeffs[0])
print()
print("coeff_y =", coeffs[1])
print()
print("coeff_z =", coeffs[2])
print()
print("Ts =", Ts)
print()
print("Yaws =", Yaws)
print("---------")

visited = set()
bases = []

for theta in gate_theta:
  y = yaw_rot(theta) @ INITIAL_GATE_EXIT
  x = yaw_rot(theta - np.pi/2) @ INITIAL_GATE_EXIT
  z = np.cross(x, y)
  matrix = np.array([x, y, z]).T
  bases.append(np.linalg.inv(matrix))  

action = np.zeros(4)
cumulative_reward = 0
info = {}
prev_state = ctrl.state
prev_args = None
total_deviation = 0
total_dist = 0

for i in range(int(CTRL_FREQ*(ctrl.total_time + 2))):
    curr_time = i * CTRL_DT
    vicon_obs = [obs[0], 0, obs[2], 0, obs[4], 0, obs[6], obs[7], obs[8], 0, 0, 0]
    command_type, args = ctrl.cmdFirmware(curr_time, obs=vicon_obs)

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

    obs, reward, _, info, action = firmware_wrapper.step(curr_time, action)
    if command_type == Command.FULLSTATE and prev_args is not None:
      total_deviation += np.linalg.norm(args[0] - np.array([obs[0], obs[2], obs[4]]))
      total_dist += np.linalg.norm(args[0] - prev_args[0])
    
    for waypoint, matrix in zip(waypoints, bases):
      error = waypoint - np.array([obs[0], obs[2], obs[4]])
      error = matrix @ error  # convert to frame with respect to gate
      if np.abs(error[0]) < 0.2 and np.abs(error[2]) < 0.2 and np.abs(error[1]) < 0.1:
        visited.add(",".join(str(x) for x in waypoint))
    
    # whatever rewards accumulated before following trajectory should be ignored
    # as we are just utilizing out of box crazyflie tools (i.e. takeoff)
    if ctrl.state == States.FOLLOWING_TRAJ and prev_state == States.TAKEOFF:
        cumulative_reward = 0
        
    cumulative_reward += reward

    prev_state = ctrl.state
    prev_args = args

env.close()

skipped_waypoints = len(waypoints) - len(visited)
average_error = total_deviation / int(CTRL_FREQ*(ctrl.total_time))
error_reward = 0 if average_error < 0.25 else -10

if optimal_vals is not None:
  print("Optimal values were used")
print("Crash cost: ", cumulative_reward)
print("Time to complete course: ", ctrl.total_time) 
print("Skpped waypoints: ", -100*skipped_waypoints)
print("Cost from tracking error: ", average_error)
print("--------------------")

if GENERATE_CSV:
  ctrl.file.close()
  ctrl.cmd_file.close()
