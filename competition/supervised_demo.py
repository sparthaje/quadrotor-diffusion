from functools import partial
import random

import os
import numpy as np
from time import sleep
import torch
from train.model import BoundaryPredictor

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
    files_with_times = [(os.path.join(directory, file), os.path.getmtime(os.path.join(directory, file))) for file in files]

    # Sort files based on modification time, newest first
    newest_file = max(files_with_times, key=lambda x: x[1])

    return newest_file[0]  # Return the path of the newest file

# Copied from `build_supervised_demo.py`
# NOTE(shreepa): MAKE SURE THAT THE FIRST GATE IS TAKEOFF POSITION AND LAST TWO GATES ARE DUMMY GATES WHEN RUNNING BUILD DEMO
gate_x = [0.0, 0.7071067811865475, 0.9417584787468938, 0.05075195455852588, -1.0099082172212954, -0.7008912228463483, -0.3918742284714012]
gate_y = [-1.75, -1.0428932188134525, 0.4386392920792541, 0.892629791818801, -0.16803037996102022, -1.119086896256174, -2.0701434125513276]
gate_z = [0, 0, 1, 1, 0, 0, 0]
heights = [0.525, 0.525, 0.3, 0.3, 0.525, 0.525, 0.525]
gate_theta = [-0.7853981633974483, -0.15707963267948966, 1.0995574287564276, 2.356194490192345, -2.8274333882308142, -2.8274333882308142, -2.8274333882308142]
d_vals = [0, 1.0, 1.5, 1.0, 1.5, 1.0, 1.0]
rel_angles = [-0.7853981633974483, 0.6283185307179586, 1.2566370614359172, 1.2566370614359172, 1.0995574287564276, 0.0, 0.0]

gate_x = [-1.0, 0.0, 0.7071067811865475, 0.7071067811865474, 0.7071067811865472, 0.7071067811865471]
gate_y = [-1.75, -1.75, -1.0428932188134523, -0.042893218813452316, 0.7571067811865477, 1.5571067811865478]
gate_z = [0, 0, 0, 0, 0, 0]
heights = [0.525, 0.525, 0.525, 0.525, 0.525, 0.525]
gate_theta = [-1.5707963267948966, -0.7853981633974482, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16]
d_vals = [0, 1.0, 1.0, 1.0, 0.8, 0.8]
rel_angles = [-1.5707963267948966, 0.7853981633974483, 0.7853981633974483, 0.0, 0.0, 0.0]

gate_x = gate_x[:-2]
gate_y = gate_y[:-2]
gate_z = gate_z[:-2]
gate_theta = gate_theta[:-2]
d_vals = [(d_vals[i], d_vals[i+1], d_vals[i+2]) for i in range(len(gate_x))]
rel_angles = [(rel_angles[i], rel_angles[i+1], rel_angles[i+2]) for i in range(len(gate_x))]
heights = [(heights[i], heights[i+1], heights[i+2]) for i in range(len(gate_x))]

# assert that all the lists have the same length
assert len(gate_x) == len(gate_y) == len(gate_z) == len(gate_theta) == len(d_vals) == len(rel_angles) == len(heights), "All lists must have the same length"

model = BoundaryPredictor()
# model.load_state_dict(torch.load('../models/model.pth'))
# model.load_state_dict(torch.load('../models/model-20240202-211827.pth'))
model.load_state_dict(torch.load(newest_file_in_directory("../models")))
model.eval()

boundary_conditions = []
boundary_conditions.append([
  0, 
  np.array([gate_x[0], gate_y[0], heights[0][0]]),
  np.zeros(3),
  np.zeros(3),
  np.zeros(3)
])

i = 0
for gx, gy, gz, h, gt, d, ra in list(zip(gate_x, gate_y, gate_z, heights, gate_theta, d_vals, rel_angles))[:-1]:
  inputs = np.array([
    np.linalg.norm(boundary_conditions[-1][2]),  # v_0
    gt,  # theta_0,
    h[0],  # z_0
    
    d[1],  # d_1
    ra[1],  # ra_1
    h[1],  # z_1
    
    d[2],  # d_2,
    ra[2],  # ra_2,
    h[2],  # z_2
  ])

  inputs = torch.tensor(inputs, dtype=torch.float32)
  v, t = model(inputs).detach().numpy()
  print(inputs, v, t)
  print('-----')
  
  boundary_conditions.append([
    t,
    np.array([gate_x[i+1], gate_y[i+1], heights[i+1][0]]),
    yaw_rot(gt) @ (v * INITIAL_GATE_EXIT),
    np.zeros(3),
    np.zeros(3),
  ])
  i += 1
  
STAB_DIST = 0.5
END = boundary_conditions[-1][1] + STAB_DIST * (yaw_rot(gate_theta[-1]) @ INITIAL_GATE_EXIT)

boundary_conditions.append([
  1, 
  END,
  np.zeros(3),
  np.zeros(3),
  np.zeros(3)
])

print("--------------------")
for b in boundary_conditions:
  for x in b:
    print(x)
    
  print("--------------------")
  print()

for i in range(1):
  CONFIG_FACTORY = ConfigFactory()
  config = CONFIG_FACTORY.merge()
  config["quadrotor_config"]["gui"] = True
  config["quadrotor_config"]["gates"] = []
  config["quadrotor_config"]["obstacles"] = []

  config["quadrotor_config"]["gates"] = [ #]
    [gx, gy, 0, 0, 0, gt, gz] for gx, gy, gz, gt in list(zip(gate_x, gate_y, gate_z, gate_theta))[1:]
  ]

  config["quadrotor_config"]["init_state"]["init_x"] = boundary_conditions[0][1][0]
  config["quadrotor_config"]["init_state"]["init_y"] = boundary_conditions[0][1][1]
  config["quadrotor_config"]["init_state"]["init_z"] = boundary_conditions[0][1][2]
  velocity_vec = np.zeros(3)
  config["quadrotor_config"]["init_state"]["init_x_dot"] = velocity_vec[0]
  config["quadrotor_config"]["init_state"]["init_y_dot"] = velocity_vec[1]
  config["quadrotor_config"]["init_state"]["init_z_dot"] = velocity_vec[2]

  config["quadrotor_config"]["init_state"]["init_psi"] = gate_theta[0]
    
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
  total_time, waypoints = ctrl.build_traj_with_boundaries(boundary_conditions)
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
      command_type, args = ctrl.cmdFirmware(curr_time)

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
      if cumulative_reward < 0:
        break

      prev_state = ctrl.state
      prev_args = args

  env.close()

  skipped_waypoints = len(waypoints) - len(visited)
  average_error = total_deviation / int(CTRL_FREQ*(ctrl.total_time))
  error_reward = 0 if average_error < 0.25 else -10

  print(cumulative_reward)
  print(ctrl.total_time) 
  print(-100*skipped_waypoints)
  print(error_reward)
  print("--------------------")
