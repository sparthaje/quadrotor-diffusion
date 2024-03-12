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
D_END = 3

MIN_END_THETA = -np.pi / 4
MAX_END_THETA = np.pi / 4
D_END_THETA = 3

TEST_CASES = 2 * DV * \
             2 * D_D * D_GATE_THETA * \
             2 * D_END * D_END_THETA
# 64,800

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
class TestCase:
  def __init__(self, z, v, dist, g_theta, g1z, end_dist, end_theta, g2z):
    # first gate position (z is the only one that really matters xy are there from legacy cocde)
    self.x = 0
    self.y = -1.5
    self.z = z
    
    # first gate velocity
    self.v = v
    
    # theta relative to each previous gate, in this implementation theta should always be zero (relative to nothing)
    self.theta_rel = [0, g_theta, end_theta]
    
    self.theta = 0
    
    # (dist, g_theta) is the polar coordinates from first gate to second gate
    self.dist = dist
    # here we modify g_theta to be second gate angle in global frame
    # note that this is sort of legacy since self.theta is fixed to be zero
    g_theta += self.theta
    g_theta = np.arctan2(np.sin(g_theta), np.cos(g_theta)) 
    self.g_theta = g_theta
    # z position of second gate
    self.g1z = g1z
    
    # (end_dst, end_theta) is the polar coordinates from second gate to third gate
    self.end_dist = end_dist
    # modify end_theta to be third gate angle in global frame
    end_theta += g_theta
    end_theta = np.arctan2(np.sin(end_theta), np.cos(end_theta))
    self.end_theta = end_theta
    # z position of third gate
    self.g2z = g2z
    
    # optimal values that needed to be calculated for this test_case
    self.optimal_velocity = None
    self.optimal_time = None
  
  @staticmethod
  def get_header():
    # header for csv file
    # v0: velocity @ gate 1
    # z0: z position @ gate 1
    # d1: distance from gate 1 to gate 2
    # theta1: angle from gate 1 to gate 2
    # z1: z position @ gate 2
    # d2: distance from gate 2 to gate 3
    # theta2: angle from gate 2 to gate 3
    # z2: z position @ gate 3
    # best_v: optimal velocity for gate 1
    # best_t: optimal time for gate 1
    return "v0,z0,d1,theta1,z1,d2,theta2,z2,best_v,best_t"
  
  def __str__(self):
    # returns all fields in class as a comma separated string ordered by the get_header output
    return f"{self.v},{self.z},{self.dist},{self.theta_rel[1]},{self.g1z},{self.end_dist},{self.theta_rel[2]},{self.g2z},{self.optimal_velocity},{self.optimal_time}"

def run_env(test_case, bv, bt, gui=False):
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
  total_time, waypoints = ctrl.build_traj(test_case, bv, bt, gui)
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

  return 10000 * cumulative_reward - 100*skipped_waypoints - 4 * average_error - 1 * bt

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
                  test_case = TestCase(z, v, dist, g_theta, g1z, end_dist, end_theta, g2z)
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
    output = get_optimal_vals(test_case)
    optimal_v, optimal_t = output
    if optimal_t < 0 or optimal_v < 0:
      log_important_info(f"This test case {str(test_case)} had an invalid output")
      continue
    test_case.optimal_velocity = optimal_v
    test_case.optimal_time = optimal_t
    # with open has a built in lock / unlock system to be process and thread safe
    with open(DATA_FILE, "a") as f:
      f.write(str(test_case) + "\n")

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
  
  # circular course
  inps = [
    [0.0000, -0.7854,  0.5250,  1.0000,  0.6283,  0.5250,  1.5000,  1.2566, 0.3000],
    [1.77, -0.1571,  0.5250,  1.5000,  1.2566,  0.3000,  1.0000,  1.2566, 0.3000],
    [0.44, 1.0996, 0.3000, 1.0000, 1.2566, 0.3000, 1.5000, 1.0996, 0.5250],
    [1.11, 2.3562, 0.3000, 1.5000, 1.0996, 0.5250, 1.0000, 0.0000, 0.5250]
  ]
  
  # straight line course
  # inps = [
  #   [0.0, 0.0, 0.525, 0.8, 0.0, 0.525, 0.8, 0.0, 0.525],
  #   [1.7777777777777777, 0.0, 0.525, 0.8, 0.0, 0.525, 0.8, 0.0, 0.525],
  #   [0.4444444444444444, 0.0, 0.525, 0.8, 0.0, 0.525, 0.8, 0.0, 0.525]
  # ]
  
  outputs = []
  get_tc = lambda inp: TestCase(inp[2], inp[0], inp[3], inp[4], inp[5], inp[6], inp[7], inp[8])
  
  i = 0
  for inp in inps:
    inp[1] = 0
    
    v,t = get_optimal_vals(get_tc(inp))
    outputs.append((v, t))
    
    if i == len(inps) - 1:
      continue
    inps[i + 1][0] = v
    i += 1
  
  print(inps)
  print(outputs)
  print("Total course time: ", sum([x[1] for x in outputs]))
      
elif __name__ == "__main__":
  main()
