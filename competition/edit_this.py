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
    return np.linalg.inv(M) @ sigma(T)

class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
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

        # Call a function in module `example_custom_utils`.
        ecu.exampleFunction()

        # Example: hardcode waypoints through the gates.
        if use_firmware:
            # this is the waypoint after takeoff
            waypoints = [(self.initial_obs[0], self.initial_obs[2], 0.525)]
            # waypoints = [(self.initial_obs[0], self.initial_obs[2], initial_info["gate_dimensions"]["tall"]["height"])]  # Height is hardcoded scenario knowledge.
        else:
            waypoints = [(self.initial_obs[0], self.initial_obs[2], self.initial_obs[4])]
        
        for idx, g in enumerate(self.NOMINAL_GATES):
            height = initial_info["gate_dimensions"]["tall"]["height"] if g[6] == 0 else initial_info["gate_dimensions"]["low"]["height"]
            waypoints.append((g[0]+0.1, g[1]-0.1, height))

        # this adds the last position to hit
        waypoints.append([initial_info["x_reference"][0], initial_info["x_reference"][2], initial_info["x_reference"][4]])

        #TODO(shreepa): find the real centers of the gates + the exit velocities to build this waypoints list
        #TODO(shreepa): formalize a way to compute the net trajectory using some kind of policy π(current_pos, next_gate, following_gate)
        print(waypoints)
        # input()
        # if len(waypoints) == 2:
        #     waypoints.insert(1, (0.6, -2.6, 1.0))
        # print(waypoints)
        ### WAYPOINT 1 TO GATE

        x_0 = waypoints[0][0]
        x_f = waypoints[1][0]
        vx_0 = 0
        vx_f = 0.5 # the way the gate is positioned the exit velocity should be in the +x direction
        ax_0 = 0
        ax_f = vx_f - vx_0  # need to divide by T after its calculated
        jx_0 = 0
        jx_f = ax_f - ax_0  # need to divide by T after its calculated
        sx_0 = 0
        sx_f = jx_f - jx_0  # need to divide by T after its calculated

        T_one = abs(2 * (x_f - x_0) / (vx_f + vx_0))
        # T_one = 2.5
        sigma_x = lambda T: np.array([x_0, x_f, vx_0, vx_f, ax_0, ax_f / T, jx_0, jx_f / T]).T

        y_0 = waypoints[0][1]
        y_f = waypoints[1][1]
        vy_0 = 0
        vy_f = 0
        ay_0 = 0
        ay_f = 0
        jy_0 = 0
        jy_f = 0
        sy_0 = 0
        sy_f = 0

        sigma_y = lambda T: np.array([y_0, y_f, vy_0, vy_f, ay_0, ay_f, jy_0, jy_f]).T

        z_0 = waypoints[0][2]
        z_f = waypoints[1][2]
        vz_0 = 0
        vz_f = 0
        az_0 = 0
        az_f = 0
        jz_0 = 0
        jz_f = 0
        sz_0 = 0
        sz_f = 0

        sigma_z = lambda T: np.array([z_0, z_f, vz_0, vz_f, az_0, az_f, jz_0, jz_f]).T

        coeff_x = calc_traj(sigma_x, T_one)
        coeff_y = calc_traj(sigma_y, T_one)
        coeff_z = calc_traj(sigma_z, T_one)

        t_one = np.linspace(0, T_one, int(T_one * self.CTRL_FREQ))
        x_one = np.polyval(coeff_x, t_one)
        x_vel_one = np.polyval(np.polyder(coeff_x, 1), t_one) 
        x_acc_one = np.polyval(np.polyder(coeff_x, 2), t_one) 

        y_one = np.polyval(coeff_y, t_one)
        y_vel_one = np.polyval(np.polyder(coeff_y, 1), t_one) 
        y_acc_one = np.polyval(np.polyder(coeff_y, 2), t_one) 

        z_one = np.polyval(coeff_z, t_one)
        z_vel_one = np.polyval(np.polyder(coeff_z, 1), t_one) 
        z_acc_one = np.polyval(np.polyder(coeff_z, 2), t_one) 

        ###

        ### WAYPOINT 1 TO 2

        x_0 = waypoints[1][0]
        x_f = waypoints[2][0]
        vx_0 = vx_f
        vx_f = 1
        ax_0 = ax_f / T_one
        ax_f = vx_f - vx_0  # need to divide by T after its calculated
        jx_0 = jx_f / T_one
        jx_f = ax_f - ax_0  # need to divide by T after its calculated
        sx_0 = sx_f / T_one
        sx_f = jx_f - jx_0  # need to divide by T after its calculated

        sigma_x = lambda T: np.array([x_0, x_f, vx_0, vx_f, ax_0, ax_f / T, jx_0, jx_f / T]).T

        y_0 = waypoints[1][1]
        y_f = waypoints[2][1]
        vy_0 = 0
        vy_f = 0
        ay_0 = 0
        ay_f = 0
        jy_0 = 0
        jy_f = 0
        sy_0 = 0
        sy_f = 0

        sigma_y = lambda T: np.array([y_0, y_f, vy_0, vy_f, ay_0, ay_f, jy_0, jy_f]).T

        z_0 = waypoints[1][2]
        z_f = waypoints[2][2]
        vz_0 = 0
        vz_f = 0
        az_0 = 0
        az_f = 0
        jz_0 = 0
        jz_f = 0
        sz_0 = 0
        sz_f = 0

        sigma_z = lambda T: np.array([z_0, z_f, vz_0, vz_f, az_0, az_f, jz_0, jz_f]).T
        T_two = abs(2 * (x_f - x_0) / (vx_f + vx_0))

        coeff_x = calc_traj(sigma_x, T_two)
        coeff_y = calc_traj(sigma_y, T_two)
        coeff_z = calc_traj(sigma_z, T_two)

        t_two = np.linspace(0, T_two, int(T_two * self.CTRL_FREQ))
        x_two = np.polyval(coeff_x, t_two)
        x_vel_two = np.polyval(np.polyder(coeff_x, 1), t_two) 
        x_acc_two = np.polyval(np.polyder(coeff_x, 2), t_two) 

        y_two = np.polyval(coeff_y, t_two)
        y_vel_two = np.polyval(np.polyder(coeff_y, 1), t_two) 
        y_acc_two = np.polyval(np.polyder(coeff_y, 2), t_two) 

        z_two = np.polyval(coeff_z, t_two)
        z_vel_two = np.polyval(np.polyder(coeff_z, 1), t_two) 
        z_acc_two = np.polyval(np.polyder(coeff_z, 2), t_two) 

        ###

        # Polynomial fit.
        self.ref_x = np.append(x_one, x_two)
        self.ref_y = np.append(y_one, y_two)
        self.ref_z = np.append(z_one, z_two)

        self.ref_vel_x = np.append(x_vel_one, x_vel_two)
        self.ref_vel_y = np.append(y_vel_one, y_vel_two)
        self.ref_vel_z = np.append(z_vel_one, z_vel_two)    

        self.ref_acc_x = np.append(x_acc_one, x_acc_two)
        self.ref_acc_y = np.append(y_acc_one, y_acc_two)
        self.ref_acc_z = np.append(z_acc_one, z_acc_two)
        # print(self.ref_x)
        self.total_time = T_one + T_two

        self.trajectory_reward = 0

        log = np.vstack((np.append(t_one, t_two + T_one), self.ref_x, self.ref_y, self.ref_z, self.ref_vel_x, self.ref_vel_y, self.ref_vel_z, self.ref_acc_x, self.ref_acc_y, self.ref_acc_z)).T
        np.savetxt("results/trajectory.csv", log, delimiter=',')

        self.waypoints = np.array(waypoints)
        self.errors = []

        if self.VERBOSE:
            # Plot trajectory in each dimension and 3D.
            plot_trajectory(np.append(t_one, t_two + T_one), self.waypoints, self.ref_x, self.ref_y, self.ref_z, self.ref_vel_x, self.ref_vel_y, self.ref_vel_z)

            # Draw the trajectory on PyBullet's GUI.
            draw_trajectory(initial_info, self.waypoints, self.ref_x, self.ref_y, self.ref_z)
        self.prev_args = []
        #########################
        # REPLACE THIS (END) ####
        #########################

    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        iteration = int(time*self.CTRL_FREQ)
        #########################
        # REPLACE THIS (START) ##
        #########################

        # Handwritten solution for GitHub's getting_stated scenario.

        TAKEOFF_TIME = 3
        # print(iteration)
        if iteration == 0:
            height = 0.525
            duration = 1.5

            command_type = Command(2)  # Take-off.
            args = [height, duration]
            # print("takeoff")
            self.prev_args = []
        elif iteration >= TAKEOFF_TIME*self.CTRL_FREQ and iteration < (TAKEOFF_TIME + self.total_time)*self.CTRL_FREQ:
            # print("stepping")
            step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel = np.array([self.ref_vel_x[step], self.ref_vel_y[step], self.ref_vel_z[step]])
            target_acc = np.array([self.ref_acc_x[step], self.ref_acc_y[step], self.ref_acc_z[step]])
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
            self.prev_args = args   

        elif iteration == (TAKEOFF_TIME + self.total_time)*self.CTRL_FREQ:
            command_type = Command(6)  # Notify setpoint stop.
            args = []

        elif iteration == (TAKEOFF_TIME + self.total_time)*self.CTRL_FREQ+1:
            x = self.ref_x[-1]
            y = self.ref_y[-1]
            z = 1.5 
            yaw = 0.
            duration = 2.5

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]

        elif iteration == (TAKEOFF_TIME + self.total_time)*self.CTRL_FREQ+1+3*self.CTRL_FREQ:
            x = self.initial_obs[0]
            y = self.initial_obs[2]
            z = 1.5
            yaw = 0.
            duration = 6

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]

        else:
            command_type = Command(0)  # None.
            args = []

        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the IROS 2022 Safe Robot Learning competition.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

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
        
        ref_pos_error = self.prev_args[0] - np.array([obs[i] for i in [0,2,4]])
        ref_vel_error = self.prev_args[1] - np.array([obs[i] for i in [1, 3, 5]])
        self.errors.append((ref_pos_error, ref_vel_error))

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
