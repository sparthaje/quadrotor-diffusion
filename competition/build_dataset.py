import multiprocessing
from functools import partial

import numpy as np
from time import sleep

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

# Current heading
MIN_THETA = -np.pi/2
MAX_THETA = np.pi/2
D_THETA = 16

# Used to calculate next gate
MIN_DISTANCE = 0.8
MAX_DISTANCE = 2.00
D_D = 4

MIN_GATE_THETA = -np.pi / 4
MAX_GATE_THETA = np.pi / 4
D_GATE_THETA = 8

# Secondary Gate
MIN_END_DISTANCE = 0.8
MAX_END_DISTANCE = 2.00
D_D = 4

MIN_END_THETA = -np.pi / 4
MAX_END_THETA = np.pi / 4
D_END_THETA = 5

TEST_CASES = DV*D_THETA*D_D*D_GATE_THETA*D_D*D_END_THETA*2*2*2
# 819200

# input current state
# output boundary

EXIT_VELOCITY_MIN = 0.0
EXIT_VELOCITY_MAX = 2.0
STEP = 10

class TestCase:
  def __init__(self, z, v, theta, dist, g_theta, g1z, end_dist, end_theta, g2z):
    self.x = 0
    self.y = -1.5
    self.z = z
    self.v = v
    self.theta_rel = [theta, g_theta, end_theta]
    self.theta = theta
    self.dist = dist
    g_theta += theta
    g_theta = np.arctan2(np.sin(g_theta), np.cos(g_theta)) 
    self.g_theta = g_theta
    self.g1z = g1z
    self.end_dist = end_dist
    end_theta += g_theta
    end_theta = np.arctan2(np.sin(end_theta), np.cos(end_theta))
    self.end_theta = end_theta
    self.g2z = g2z
    self.optimal_velocity = None
    self.optimal_time = None
  
  def __str__(self):
    # returns all fields in class as a comma separated stringv 
    return f"{self.v},{self.theta_rel[0]},{self.z},{self.dist},{self.theta_rel[1]},{self.g1z},{self.end_dist},{self.theta_rel[2]},{self.g2z},{self.optimal_velocity},{self.optimal_time}"

def run_env(test_case, bv, bt, gui=False):
  CONFIG_FACTORY = ConfigFactory()
  config = CONFIG_FACTORY.merge()
  config["quadrotor_config"]["gui"] = gui
  config["quadrotor_config"]["gates"] = []
  config["quadrotor_config"]["obstacles"] = []
  
  second_gate_pos = np.array([test_case.x, test_case.y, test_case.z]) + test_case.dist * (yaw_rot(test_case.theta) @ INITIAL_GATE_EXIT)
  second_gate_pos[2] = test_case.g1z
  third_gate_pos = second_gate_pos + test_case.end_dist * (yaw_rot(test_case.g_theta) @ INITIAL_GATE_EXIT)
  third_gate_pos[2] = test_case.g2z
  config["quadrotor_config"]["gates"] = [
      [test_case.x, test_case.y, 0, 0, 0, test_case.theta, 1 if test_case.z == 0.3 else 0],
      [second_gate_pos[0], second_gate_pos[1], 0, 0, 0, test_case.g_theta, 1 if test_case.g1z == 0.3 else 0],
      [third_gate_pos[0], third_gate_pos[1], 0, 0, 0, test_case.end_theta, 1 if test_case.g2z == 0.3 else 0]
  ]
  
  # for x, y, _, _, _, _, _ in config["quadrotor_config"]["gates"]:
  #   if not (-1.3 < x < 1.3) or not (-1.8 < y < 1.8):
  #     return "Bad test case"
  
  config["quadrotor_config"]["init_state"]["init_x"] = test_case.x
  config["quadrotor_config"]["init_state"]["init_y"] = test_case.y
  config["quadrotor_config"]["init_state"]["init_z"] = test_case.z
  velocity_vec = yaw_rot(test_case.theta) @ (test_case.v * INITIAL_GATE_EXIT)
  config["quadrotor_config"]["init_state"]["init_x_dot"] = velocity_vec[0]
  config["quadrotor_config"]["init_state"]["init_y_dot"] = velocity_vec[1]
  config["quadrotor_config"]["init_state"]["init_z_dot"] = velocity_vec[2]
  
  config["quadrotor_config"]["init_state"]["init_psi"] = test_case.theta
   
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
  total_time, waypoints = ctrl.build_traj(test_case, bv, bt, gui)
  visited = set()
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

  return cumulative_reward - bt - 100*skipped_waypoints + error_reward

def get_optimal_vals(test_case):
  best = (-float('inf'), -1, -1)
  for v in np.linspace(EXIT_VELOCITY_MIN, EXIT_VELOCITY_MAX, STEP):
    t_opt = 2 * test_case.dist / (v + test_case.v)
    if v + test_case.v == 0:
      t_opt = 1.3
    for t in np.linspace(0.5 * t_opt, 1.1 * t_opt, 7):
      reward = run_env(test_case, v, t, False)
      if reward == "Bad test case":
        return "Bad test case"
      if reward > best[0]:
        best = (reward, v, t)
  return best[1], best[2]

def run_process(id, num_processes, iterations):
  i = 0
  gap = TEST_CASES / num_processes
  # loop through starting velocities
  for v in np.linspace(MIN_VELOCITY, MAX_VELOCITY, DV):
    for theta in np.linspace(MIN_THETA, MAX_THETA, D_THETA):
      # loop through second gate arrangements
      for dist in np.linspace(MIN_DISTANCE, MAX_DISTANCE, D_D):
        for g_theta in np.linspace(MIN_GATE_THETA, MAX_GATE_THETA, D_GATE_THETA):
          # loop through secondary gate
          for end_dist in np.linspace(MIN_END_DISTANCE, MAX_END_DISTANCE, D_D):
            for end_theta in np.linspace(MIN_END_THETA, MAX_END_THETA, D_END_THETA):
              for z in [0.3, 0.525]:
                for g1z in [0.3, 0.525]:
                  for g2z in [0.3, 0.525]:
                    i += 1
                    iterations.value += 1
                    if not (gap * id <= i < gap * (id + 1)):
                      continue
                    test_case = TestCase(z, v, theta, dist, g_theta, g1z, end_dist, end_theta, g2z)
                    output = get_optimal_vals(test_case)
                    if output == "Bad test case":
                      continue
                    optimal_v, optimal_t = output
                    # invalid gate configuration
                    if (optimal_v, optimal_t) == (-1, -1):
                      continue
                    test_case.optimal_velocity = optimal_v
                    test_case.optimal_time = optimal_t
                    with open("../train/data.csv", "a") as f:
                      f.write(str(test_case) + "\n")

def main():  
  with open("../train/data.csv", "w") as f:
    f.write("v0,theta0,z0,d1,theta1,z1,d2,theta2,z2,best_v,best_t\n")
  num_processes = multiprocessing.cpu_count()
  # num_processes = 1
  training_data_size = multiprocessing.Value('i', 0)
  processes = []
  
  for i in range(num_processes):
    process = multiprocessing.Process(target=run_process, args=(i, num_processes, training_data_size,))
    processes.append(process)

  for process in processes:
    process.start()
  
  for process in processes:
    process.join()
  
  print("Data collection finished")
  
if __name__ == "__main__":
  # main()
  inps = [
    # [0.0000, -0.7854,  0.5250,  1.0000,  0.6283,  0.5250,  1.5000,  1.2566, 0.3000],
    # [1.8162, -0.1571,  0.5250,  1.5000,  1.2566,  0.3000,  1.0000,  1.2566, 0.3000],
    # [0.5544, 1.0996, 0.3000, 1.0000, 1.2566, 0.3000, 1.5000, 1.0996, 0.5250],
    # [1.4816, 2.3562, 0.3000, 1.5000, 1.0996, 0.5250, 1.0000, 0.0000, 0.5250]
  ]
  inps = [
    #  [0.0000, -1.5708,  0.5250,  1.0000,  0.7854,  0.5250,  1.0000,  0.7854,
    #      0.5250],
    #  [ 1.4135, -0.7854,  0.5250,  1.0000,  0.7854,  0.5250,  1.0000,  0.0000,
    #      0.5250],
     [1.4725e+00, 1.1102e-16, 5.2500e-01, 1.0000e+00, 0.0000e+00, 5.2500e-01,
        8.0000e-01, 0.0000e+00, 5.2500e-01]
  ]
  for inp in inps:
    print(inp)
    tc = TestCase(inp[2], inp[0], inp[1], inp[3], inp[4], inp[5], inp[6], inp[7], inp[8])
    v,t = get_optimal_vals(tc)
    # print("Search here: ", v, t)
    run_env(tc, v, t, True)

