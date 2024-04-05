"""Write your control strategy.

Then run:

    $ python3 getting_started.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS:` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) __init__
        2) cmdFirmware
        3) interStepLearn (optional)
        4) interEpisodeLearn (optional)

"""
import numpy as np

from collections import deque
from enum import Enum

try:
    from competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory
except ImportError:
    # PyTest import.
    from .competition_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory

from safe_control_gym.envs.gym_pybullet_drones.Logger import Logger

#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
try:
    import example_custom_utils as ecu
except ImportError:
    # PyTest import.
    from . import example_custom_utils as ecu

from rich import print

#########################
# REPLACE THIS (END) ####
#########################

INITIAL_GATE_EXIT = np.array([0, 1, 0])

def yaw_rot(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

def calc_traj(sigma, T):
    M = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1],
        [T**7, T**6, T**5, T**4, T**3, T**2, T**1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [7*T**6, 6*T**5, 5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
        [0, 0, 0, 0, 0, 2, 0, 0],
        [42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0],
        [0, 0, 0, 0, 6, 0, 0, 0],
        [210*T**4, 120*T**3, 60*T**2, 24*T, 6, 0, 0, 0],
    ])
    return np.linalg.inv(M) @ sigma

class States(Enum):
    PRE_LAUNCH = 1
    TAKEOFF = 2
    FOLLOWING_TRAJ = 3
    FINISHED = 4

REF_YAW_RATIO = 0.25
class Controller():
    """Template controller class.

    """
    
    def build_traj_with_boundaries(self, boundaries, gui=True):
        waypoints = [b[1] for b in boundaries]
        self.ref_pos = [[], [], []]
        self.ref_vel = [[], [], []]
        self.ref_acc = [[], [], []]
        self.ref_yaw = []
        self.T = []
        
        coeffs = [[], [], []]
        Ts = []
        Yaws = []

        for index in range(len(boundaries) - 1):
            b_0 = boundaries[index]
            b_f = boundaries[index + 1]
            for xyz in range(3):
                sigma = np.array([b_0[1][xyz], b_f[1][xyz], b_0[2][xyz], b_f[2][xyz], b_0[3][xyz], b_f[3][xyz], b_0[4][xyz], b_f[4][xyz]])
                coeff = calc_traj(sigma, b_f[0])  
                coeffs[xyz].append(list(coeff))

                t = np.linspace(0, b_f[0], int(b_f[0] * self.CTRL_FREQ))
                self.ref_pos[xyz] = np.append(self.ref_pos[xyz], np.polyval(coeff, t))
                self.ref_vel[xyz] = np.append(self.ref_vel[xyz], np.polyval(np.polyder(coeff, 1), t))
                self.ref_acc[xyz] = np.append(self.ref_acc[xyz], np.polyval(np.polyder(coeff, 2), t))
                
            # print("Max vels:", max(self.ref_vel[0]), max(self.ref_vel[1]), max(self.ref_vel[2]))
            # print("Min vels:", min(self.ref_vel[0]), min(self.ref_vel[1]), min(self.ref_vel[2]))

            # print("Max accels:", max(self.ref_acc[0]), max(self.ref_acc[1]), max(self.ref_acc[2]))
            # print("Min accels:", min(self.ref_acc[0]), min(self.ref_acc[1]), min(self.ref_acc[2]))
            # print("--------")
            
            total_steps = int(b_f[0] * self.CTRL_FREQ)
            self.ref_yaw = np.append(self.ref_yaw, np.linspace(b_0[5], b_f[5], int(total_steps * REF_YAW_RATIO)))
            self.ref_yaw = np.append(self.ref_yaw, np.full(total_steps - int(total_steps * REF_YAW_RATIO), b_f[5]))
            self.T.append(b_f[0])
            Ts.append(b_f[0])
            # in vicon 0 degrees is along the x-axis, so im just gonna rotate everything
            target_yaw_vicon_map = b_f[5] + np.pi / 4
            target_yaw_vicon_map = np.arctan2(np.sin(target_yaw_vicon_map), np.cos(target_yaw_vicon_map))
            Yaws.append(target_yaw_vicon_map)

        for xyz in range(3):
            assert len(self.ref_pos[xyz]) == len(self.ref_vel[xyz]) and len(self.ref_vel[xyz]) == len(self.ref_acc[xyz]) and len(self.ref_acc[xyz]) == len(self.ref_yaw)
        
        assert len(self.ref_pos[0]) == len(self.ref_pos[1]) and len(self.ref_pos[1]) == len(self.ref_pos[2])
        assert len(self.ref_vel[0]) == len(self.ref_vel[1]) and len(self.ref_vel[1]) == len(self.ref_vel[2])
        assert len(self.ref_acc[0]) == len(self.ref_acc[1]) and len(self.ref_acc[1]) == len(self.ref_acc[2])

        print("Max vels:", max(self.ref_vel[0]), max(self.ref_vel[1]), max(self.ref_vel[2]))
        print("Min vels:", min(self.ref_vel[0]), min(self.ref_vel[1]), min(self.ref_vel[2]))

        print("Max accels:", max(self.ref_acc[0]), max(self.ref_acc[1]), max(self.ref_acc[2]))
        print("Min accels:", min(self.ref_acc[0]), min(self.ref_acc[1]), min(self.ref_acc[2]))

        self.total_time = sum(self.T)

        if gui:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(self.initial_info, waypoints, self.ref_pos[0], self.ref_pos[1], self.ref_pos[2])
            
        return self.total_time, waypoints, coeffs, Ts, Yaws
    
    def build_traj(self, test_case, v, t, gui=False, print_accel_limits=False):
        heights = [test_case.z, test_case.g1z, test_case.g2z]
        waypoints = [np.array([x[0], x[1], z]) for x, z in zip(self.NOMINAL_GATES, heights)]
        boundaries = []
        boundaries.append([
            0,
            waypoints[0],  # pos
            yaw_rot(test_case.theta) @ (test_case.v * INITIAL_GATE_EXIT),  # vel
            np.zeros(3),  # acc
            np.zeros(3),  # jerk
            test_case.theta,  # yaw
        ])
        
        boundaries.append([
            t,
            waypoints[1],
            yaw_rot(test_case.g_theta) @ (v * INITIAL_GATE_EXIT),
            np.zeros(3),
            np.zeros(3),
            test_case.g_theta,
        ])
        
        """
        vf^2 = v0^2 + 2*a*d
        -v0^2 = 2ad
        a = -v0^2 / 2d
        vf = v0 + at
        t = (vf - v0) / a
        t = -v0 / a
        t = -v0 / (-v0^2 / 2d)
        t = v0 / (v0^2 / 2d)
        t = 2d / v0  // same as what i did for mixnet LMAO
        """
        t_f = 2 * test_case.end_dist / v if v != 0 else 2 * test_case.end_dist
        # t_f = 2 * test_case.end_dist / (v + 0.5)
        boundaries.append([
            t_f,
            waypoints[2],
            yaw_rot(test_case.end_theta) @ (0.1 * INITIAL_GATE_EXIT),
            np.zeros(3),
            np.zeros(3),
            test_case.end_theta,
        ])
        
        self.ref_pos = [[], [], []]
        self.ref_vel = [[], [], []]
        self.ref_acc = [[], [], []]
        self.ref_yaw = []
        self.T = []

        for index in range(len(boundaries) - 1):
            b_0 = boundaries[index]
            b_f = boundaries[index + 1]
            for xyz in range(3):
                sigma = np.array([b_0[1][xyz], b_f[1][xyz], b_0[2][xyz], b_f[2][xyz], b_0[3][xyz], b_f[3][xyz], b_0[4][xyz], b_f[4][xyz]])
                coeff = calc_traj(sigma, b_f[0])
                t = np.linspace(0, b_f[0], int(b_f[0] * self.CTRL_FREQ))
                self.ref_pos[xyz] = np.append(self.ref_pos[xyz], np.polyval(coeff, t))
                self.ref_vel[xyz] = np.append(self.ref_vel[xyz], np.polyval(np.polyder(coeff, 1), t))
                self.ref_acc[xyz] = np.append(self.ref_acc[xyz], np.polyval(np.polyder(coeff, 2), t))
            total_steps = int(b_f[0] * self.CTRL_FREQ)
            self.ref_yaw = np.append(self.ref_yaw, np.linspace(b_0[5], b_f[5], int(total_steps * REF_YAW_RATIO)))
            self.ref_yaw = np.append(self.ref_yaw, np.full(total_steps - int(total_steps * REF_YAW_RATIO), b_f[5]))
            self.T.append(b_f[0])
            
        if print_accel_limits:
            print("--------------------")
            for b in boundaries:
                for x in b:
                    print(x)
                print("--------------------")
            print()

        for xyz in range(3):
            assert len(self.ref_pos[xyz]) == len(self.ref_vel[xyz]) and len(self.ref_vel[xyz]) == len(self.ref_acc[xyz]) and len(self.ref_acc[xyz]) == len(self.ref_yaw)
        
        assert len(self.ref_pos[0]) == len(self.ref_pos[1]) and len(self.ref_pos[1]) == len(self.ref_pos[2])
        assert len(self.ref_vel[0]) == len(self.ref_vel[1]) and len(self.ref_vel[1]) == len(self.ref_vel[2])
        assert len(self.ref_acc[0]) == len(self.ref_acc[1]) and len(self.ref_acc[1]) == len(self.ref_acc[2])
        
        # x = [np.array([x, y, z]) for x, y, z in zip(self.ref_pos[0], self.ref_pos[1], self.ref_pos[2])]
        # print("Traj Length", sum([np.linalg.norm(x[i+1] - x[i]) for i in range(len(x) - 1)]))
        
        self.total_time = sum(self.T)

        if gui:
            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(self.initial_info, waypoints, self.ref_pos[0], self.ref_pos[1], self.ref_pos[2])
            
        return self.total_time, waypoints


    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False,
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size
        self.initial_info = initial_info
        self.file = None
        self.cmd_file = None

        # Store a priori scenario information.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition. 
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        #########################
        # REPLACE THIS (START) ##
        #########################
        
        self.state = States.PRE_LAUNCH
        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self, time, obs=[0 for _ in range(10)]):
        """
        Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.
        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        iteration = int(time*self.CTRL_FREQ)
        if iteration < self.CTRL_FREQ*self.CTRL_FREQ:
            step = min(iteration, len(self.ref_pos[0]) - 1)
            target_pos = np.array([self.ref_pos[0][step], self.ref_pos[1][step], self.ref_pos[2][step]])
            target_vel = np.array([self.ref_vel[0][step], self.ref_vel[1][step], self.ref_vel[2][step]])
            target_acc = np.array([self.ref_acc[0][step], self.ref_acc[1][step], self.ref_acc[2][step]])            
            # target_yaw = self.initial_obs[8] #+ np.pi/2
            # target_yaw = np.arctan2(np.sin(target_yaw), np.cos(target_yaw))
            target_yaw = self.ref_yaw[step]
            target_rpy_rates = np.zeros(3)
            
            if self.file:
                self.file.write(f"{time},{obs[0]},{obs[2]},{obs[4]}\n")
                
            if self.cmd_file:
                self.cmd_file.write(f"{time},{target_pos[0]},{target_pos[1]},{target_pos[2]}\n")

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
            self.prev_args = args   
            self.state = States.FOLLOWING_TRAJ
        elif iteration >= self.total_time*self.CTRL_FREQ:
            x = self.ref_pos[0][-1]
            y = self.ref_pos[1][-1]
            z = self.ref_pos[2][-1]
            yaw = 0.
            duration = 2.5

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]
            self.state = States.FINISHED
        else:
            command_type = Command(0)  # None.
            args = []
        
        return command_type, args

    
    @timing_step
    def interStepLearn(self,
                       action,
                       obs,
                       reward,
                       done,
                       info):
        """Learning and controller updates called between control steps.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        Args:
            action (List): Most recent applied action.
            obs (List): Most recent observation of the quadrotor state.
            reward (float): Most recent reward.
            done (bool): Most recent done flag.
            info (dict): Most recent information dictionary.

        """
        self.interstep_counter += 1

        # Store the last step's events.
        self.action_buffer.append(action)
        self.obs_buffer.append(obs)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)
        self.info_buffer.append(info)

        #########################
        # REPLACE THIS (START) ##
        #########################
        
        # not full state
        if len(self.prev_args) == 0:
            return
        
        self.pos_errors.append(self.prev_args[0] - np.array([obs[i] for i in [0,2,4]]))
        self.vel_errors.append(self.prev_args[1] - np.array([obs[i] for i in [1, 3, 5]]))

        #########################
        # REPLACE THIS (END) ####
        #########################

    @timing_ep
    def interEpisodeLearn(self):
        """Learning and controller updates called between episodes.

        INSTRUCTIONS:
            Use the historically collected information in the five data buffers of actions, observations,
            rewards, done flags, and information dictionaries to learn, adapt, and/or re-plan.

        """
        self.interepisode_counter += 1

        #########################
        # REPLACE THIS (START) ##
        #########################

        _ = self.action_buffer
        _ = self.obs_buffer
        _ = self.reward_buffer
        _ = self.done_buffer
        _ = self.info_buffer
        
        # import matplotlib.pyplot as plt

        # # Unpack the tuples into separate lists for positions and velocities
        # try:
        #     positions, velocities = zip(*self.errors[1:])
        # except:
        #     return
    
        # # Create subplots
        # plt.figure(figsize=(12, 6))

        # # Plot Position RMSE
        # plt.subplot(1, 2, 1)
        # plt.plot(positions, label='Position RMSE', marker='o')
        # plt.xlabel('Time Step')
        # plt.ylabel('Position RMSE')
        # plt.title('Position RMSE over Time')
        # plt.legend(['x', 'y', 'z'])

        # # Plot Velocity RMSE
        # plt.subplot(1, 2, 2)
        # plt.plot(velocities, label='Velocity RMSE', marker='o')
        # plt.xlabel('Time Step')
        # plt.ylabel('Velocity RMSE')
        # plt.title('Velocity RMSE over Time')
        # plt.legend(['x', 'y', 'z'])

        # # Adjust layout for better spacing
        # plt.tight_layout()

        # # Display the plot
        # plt.savefig('results/rmse_plot.png')

        #########################
        # REPLACE THIS (END) ####
        #########################

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
