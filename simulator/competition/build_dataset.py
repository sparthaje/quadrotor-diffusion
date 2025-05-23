import multiprocessing
from functools import partial

import numpy as np
import random
import datetime
from sys import argv

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

try:
    from competition_utils import Command, thrusts
    from data_collection_ctrls import Controller, yaw_rot, INITIAL_GATE_EXIT, States
except ImportError:
    # Test import.
    from .competition_utils import Command, thrusts
    from .data_collection_ctrls import Controller, yaw_rot, INITIAL_GATE_EXIT, States
    
from data_gen_costs import *
from test_case import TestCase

MIN_VELOCITY = 0.0
MAX_VELOCITY = 2.0
DV = 10

# Used to calculate next gate
MIN_DISTANCE = 0.8
MAX_DISTANCE = 1.50
D_D = 6

MIN_GATE_THETA = -np.pi / 4
MAX_GATE_THETA = np.pi / 4
D_GATE_THETA = 15

# Secondary Gate
MIN_END_DISTANCE = 0.8
MAX_END_DISTANCE = 1.50
D_END = 6

MIN_END_THETA = -np.pi / 4
MAX_END_THETA = np.pi / 4
D_END_THETA = 3

TEST_CASES = 2 * DV * \
             2 * D_D * D_GATE_THETA * \
             2 * D_END * D_END_THETA
# 129,600

# input current state
# output boundary

EXIT_VELOCITY_MIN = 0.0
EXIT_VELOCITY_MAX = 2.0
EXIT_VELOCITY_STEP = 10

DATA_FILE = "../train/data.csv"
LOG_FILE = "logs.txt"

def log_important_info(info: str):
  # only use this for crtical information
  with open(LOG_FILE, "a") as f:
    f.write(f"[{datetime.datetime.now()}] {info}\n")

def run_env(test_case, bv, bt, gui=False, print_accel_limits=False, return_vals=False):
  # bv: velocity to test at gate 1
  # bt: time to test at gate 1
  
  CONFIG_FACTORY = ConfigFactory()
  config = CONFIG_FACTORY.merge()
  config["quadrotor_config"]["gui"] = gui
  config["quadrotor_config"]["gates"] = []
  config["quadrotor_config"]["obstacles"] = []
  
  # add gates to the map convert local frame data in test_case to global frame
  second_gate_pos = np.array([test_case.x, test_case.y, test_case.z]) + test_case.dist * (yaw_rot(test_case.theta) @ INITIAL_GATE_EXIT)
  second_gate_pos[2] = test_case.g1z
  third_gate_pos = second_gate_pos + test_case.end_dist * (yaw_rot(test_case.g_theta) @ INITIAL_GATE_EXIT)
  third_gate_pos[2] = test_case.g2z
  config["quadrotor_config"]["gates"] = [
      [test_case.x, test_case.y, 0, 0, 0, test_case.theta, 1 if test_case.z == 0.3 else 0],
      [second_gate_pos[0], second_gate_pos[1], 0, 0, 0, test_case.g_theta, 1 if test_case.g1z == 0.3 else 0],
      [third_gate_pos[0], third_gate_pos[1], 0, 0, 0, test_case.end_theta, 1 if test_case.g2z == 0.3 else 0]
  ]
  
  # set initial state from gate 1
  config["quadrotor_config"]["init_state"]["init_x"] = test_case.x
  config["quadrotor_config"]["init_state"]["init_y"] = test_case.y
  config["quadrotor_config"]["init_state"]["init_z"] = test_case.z
  velocity_vec = yaw_rot(test_case.theta) @ (test_case.v * INITIAL_GATE_EXIT)
  config["quadrotor_config"]["init_state"]["init_x_dot"] = velocity_vec[0]
  config["quadrotor_config"]["init_state"]["init_y_dot"] = velocity_vec[1]
  config["quadrotor_config"]["init_state"]["init_z_dot"] = velocity_vec[2]
  
  # this will be zero as we ignore global rotation of gate system
  config["quadrotor_config"]["init_state"]["init_psi"] = test_case.theta
  
  # goal is to stabilize at the third gate
  config["quadrotor_config"]["task_info"]["stabilization_goal"] = third_gate_pos
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
  
  # build trajectory such that the quadrotor will stop at the third gate
  total_time, waypoints = ctrl.build_traj(test_case, bv, bt, gui, print_accel_limits=print_accel_limits)
  if print_accel_limits:
    print(f"Acceleration limits: ({min(ctrl.ref_acc[0])}, {max(ctrl.ref_acc[0])}), ({min(ctrl.ref_acc[1])}, {max(ctrl.ref_acc[1])}), ({min(ctrl.ref_acc[2])}, {max(ctrl.ref_acc[2])})")
    print(f"Velocity limits: ({min(ctrl.ref_vel[0])}, {max(ctrl.ref_vel[0])}), ({min(ctrl.ref_vel[1])}, {max(ctrl.ref_vel[1])}), ({min(ctrl.ref_vel[2])}, {max(ctrl.ref_vel[2])})")
  visited = set()
  
  # transformation to get the position relative to each gate
  bases = []
  
  for theta in [test_case.theta, test_case.g_theta, test_case.end_theta]:
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
  
  if ACCEL_VECTOR and not return_vals:
    for a1, a2, a3, v1, v2, v3 in zip(ctrl.ref_acc[0], ctrl.ref_acc[1], ctrl.ref_acc[2], ctrl.ref_vel[0], ctrl.ref_vel[1], ctrl.ref_vel[2]):
      a = np.linalg.norm(np.array([a1, a2, a3]))
      v = np.linalg.norm(np.array([v1, v2, v3]))
      if v > VEL_LIMIT_COST:
        return -VEL_COST
      
      if abs(a) > PARABOLA_SHAPE * (v**2) + PARABOLA_INTERCEPT:
        return -ACCEL_COST
  elif not return_vals:  
    for xyz in range(3):
      for a, v in zip(ctrl.ref_acc[xyz], ctrl.ref_vel[xyz]):
        if v > VEL_LIMIT_COST:
          return -VEL_COST
        
        if abs(a) > PARABOLA_SHAPE * (v**2) + PARABOLA_INTERCEPT:
          return -ACCEL_COST
    
  for i in range(int(CTRL_FREQ*(ctrl.total_time))):
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
      
      # if the quadrotor is in the gate / flies through then add that to the visited set
      # we do this by putting it in gate coordinate for each gate where the xy axis are aligned
      # so y axis aligns with the gate theta, this way if the y coord is < 0.1 we know its in the 
      # gate path, and if x and z are < 0.2 we know its close to the center of the gate
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

      # this if only triggers on crash, no need to continue if this is the case
      if cumulative_reward < 0:
        break

      prev_state = ctrl.state
      prev_args = args

  env.close()
  
  skipped_waypoints = len(waypoints) - len(visited)
  average_error = total_deviation / int(CTRL_FREQ*(ctrl.total_time))
  
  ## cost function
  ## cumulative_reward: -number of crashes
  ## skipped_waypoints: number of waypoints not visited
  ## average_error: average deviation from trajectory
  ## bt: time taken to complete trajectory
  ## accel_range: range of acceleration normalized by max absolute acceleration
  
  accel_range = (np.max(ctrl.ref_acc) - np.min(ctrl.ref_acc)) / np.max(np.abs(ctrl.ref_acc))

  if return_vals:
    return abs(cumulative_reward), skipped_waypoints, average_error, accel_range
  
  return -CRASH_COST * abs(cumulative_reward) - \
          SKIPPED_WAYPOINT_COST*skipped_waypoints - \
            TRACKING_COST * average_error - TIME_COST * bt - \
              ACCEL_RANGE_COST * accel_range

def get_relevant_vals(z, v, dist, g_theta, g1z, end_dist, end_theta, g2z):
  """
    Returns a list of relevant values for the test case
  """
  test_cases = []
  for bv in np.linspace(EXIT_VELOCITY_MIN, EXIT_VELOCITY_MAX, EXIT_VELOCITY_STEP):
    # calculates time to linearly accelerate/decelerate between gate 1 and 2
    t_linear = 2 * dist / (bv + v)
    # if we are testing starting at stopping at 0 m/s let's just use 
    # 1.5 seconds as can't divide by zero
    if bv + v == 0:
      t_linear = 1.5
    # sample half the linear time to 1.1 times the linear time
    # favors going faster than linear acceleration/deceleration
    for bt in np.linspace(0.5 * t_linear, 1.1 * t_linear, 7):
      test_case = TestCase(z, v, dist, g_theta, g1z, end_dist, end_theta, g2z)
      cumulative_reward, skipped_waypoints, average_error, accel_range = run_env(test_case, bv, bt, return_vals=True)
      test_case.optimal_velocity = bv
      test_case.optimal_time = bt
      test_case.crashes = cumulative_reward
      test_case.skipped_waypoints = skipped_waypoints
      test_case.average_error = average_error
      test_case.accel_range = accel_range
      test_cases.append(test_case)
  return test_cases

def get_optimal_vals(test_case):
  """
  Returns the optimal velocity and time for the test_case
  """
  best = (-float('inf'), -1, -1)
  for v in np.linspace(EXIT_VELOCITY_MIN, EXIT_VELOCITY_MAX, EXIT_VELOCITY_STEP):
    # calculates time to linearly accelerate/decelerate between gate 1 and 2
    t_linear = 2 * test_case.dist / (v + test_case.v)
    # if we are testing starting at stopping at 0 m/s let's just use 
    # 1.5 seconds as can't divide by zero
    if v + test_case.v == 0:
      t_linear = 1.5
    # sample half the linear time to 1.1 times the linear time
    # favors going faster than linear acceleration/deceleration
    for t in np.linspace(0.5 * t_linear, 1.1 * t_linear, 7):
      reward = run_env(test_case, v, t, False)
      if reward > best[0]:
        best = (reward, v, t)

  return best[1], best[2]

def split_list(lst, n):
    """Split a list into n approximately equal-sized chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def run_process(id, num_processes):
  test_case_list = []
  # loop through starting velocities: v0 in header
  for v in np.linspace(MIN_VELOCITY, MAX_VELOCITY, DV):
    # loop through second gate arrangements: d1 and theta1 in header
    for dist in np.linspace(MIN_DISTANCE, MAX_DISTANCE, D_D):
      for g_theta in np.linspace(MIN_GATE_THETA, MAX_GATE_THETA, D_GATE_THETA):
        # loop through third gate arrangements: d2 and theta2 in header
        for end_dist in np.linspace(MIN_END_DISTANCE, MAX_END_DISTANCE, D_END):
          for end_theta in np.linspace(MIN_END_THETA, MAX_END_THETA, D_END_THETA):
            # loop through z0, z1, and z2 in header
            for z in [0.3, 0.525]:
              for g1z in [0.3, 0.525]:
                for g2z in [0.3, 0.525]:
                  test_case = [z, v, dist, g_theta, g1z, end_dist, end_theta, g2z]
                  test_case_list.append(test_case)
    
  random.seed(42)
  random.shuffle(test_case_list)
  
  # shuffle both lists with the same random seed across all processes
  # split both lists into num_processes chunks and only process the chunk
  
  # important thing here is we put the zero velocity stuff in the beginning of every chunk
  # seen by the concatenated chunks. this is because if we opt to use a minimal dataset
  # empirically the neural network performs worse with initial velocities of zero, so I want
  # all that data to be included int he data set

  to_process = split_list(test_case_list, num_processes)[id]
  
  for test_case in to_process:
    output = get_relevant_vals(*test_case)
    # with open has a built in lock / unlock system to be process and thread safe
    with open(DATA_FILE, "a") as f:
      for tc in output:
        f.write(str(tc) + "\n")

def main():  
  with open(DATA_FILE, "w") as f:
    f.write(TestCase.get_header() + "\n")
  
  num_processes = multiprocessing.cpu_count()
  processes = []
  
  for i in range(num_processes):
    process = multiprocessing.Process(target=run_process, args=(i, num_processes,))
    processes.append(process)

  for process in processes:
    process.start()
  
  for process in processes:
    process.join()
  
  print("Data collection finished")
  

# if testing is included as an arg then run these tests instead of generating all test data
if "testing" in ''.join(argv):
  # Copied from `build_supervised_demo.py`
  # NOTE(shreepa): MAKE SURE THAT THE FIRST GATE IS TAKEOFF POSITION AND LAST TWO GATES ARE DUMMY GATES WHEN RUNNING BUILD DEMO
  gate_x = [-1.0, -1.0, -0.42465867716842287, 0.4186388189488832, 1.0747105836281534, 1.2620918982138778, 1.4494732127996022]
  gate_y = [-1.5, -0.7, -0.04740336942150014, 0.05912987910815866, 0.6752222744439786, 1.6575095251726673, 2.639796775901356]
  gate_z = [1, 1, 0, 0, 0, 0, 0]
  heights = [0.3, 0.3, 0.525, 0.525, 0.525, 0.525, 0.525]
  gate_theta = [0.0, -0.7225663103256523, -1.4451326206513047, -0.816814089933346, -0.1884955592153874, -0.1884955592153874, -0.1884955592153874]
  d_vals = [0, 0.8, 0.87, 0.85, 0.9, 1.0, 1.0]
  rel_angles = [0.0, -0.7225663103256524, -0.7225663103256524, 0.6283185307179586, 0.6283185307179586, 0.0, 0.0]
  
  gate_x = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  gate_y = [-1.7, -0.7, 0.30000000000000004, 1.3, 2.3, 3.3]
  gate_z = [1, 1, 1, 1, 1, 1]
  heights = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
  gate_theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  d_vals = [0, 1.0, 1.0, 1.0, 1.0, 1.0]
  rel_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  
  gate_x = gate_x[:-2]
  gate_y = gate_y[:-2]
  gate_z = gate_z[:-2]
  gate_theta = gate_theta[:-2]
  d_vals = [(d_vals[i], d_vals[i+1], d_vals[i+2]) for i in range(len(gate_x))]
  rel_angles = [(rel_angles[i], rel_angles[i+1], rel_angles[i+2]) for i in range(len(gate_x))]
  heights = [(heights[i], heights[i+1], heights[i+2]) for i in range(len(gate_x))]

  """ 
    z: Any,
    v: Any,
    dist: Any,
    g_theta: Any,
    g1z: Any,
    end_dist: Any,
    end_theta: Any,
    g2z: Any
  """
  inps = [
    [vals[3][0], 0.0, vals[5][1], vals[6][1], vals[3][1], vals[5][2], vals[6][2], vals[3][2]] for vals in zip(gate_x, gate_y, gate_z, heights, gate_theta, d_vals, rel_angles)
  ]
  
  outputs = []
  get_tc = lambda inp: TestCase(inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6], inp[7])
  
  i = 0
  for inp in inps:    
    v,t = get_optimal_vals(get_tc(inp))
    outputs.append((v, t))
    
    if i == len(inps) - 1:
      continue
    
    inps[i + 1][1] = v
    i += 1
  
  print("---------" * 4)
  for i, o in zip(inps, outputs):
    print(i, o)
    run_env(get_tc(i), o[0], o[1], False, print_accel_limits=True)
    print("---------" * 4)
  print("Total course time: ", sum([x[1] for x in outputs]))
  print("optimal_vals =", outputs)
      
elif __name__ == "__main__":
  main()
