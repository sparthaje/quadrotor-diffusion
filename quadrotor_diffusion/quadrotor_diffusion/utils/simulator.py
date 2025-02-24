import enum
import time
import argparse
import sys
from typing import Tuple
from functools import partial
import sys
from contextlib import contextmanager
import os
import subprocess

import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R


from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from quadrotor_diffusion.utils.trajectory import derive_trajectory, INITIAL_GATE_EXIT
from quadrotor_diffusion.utils.plotting import get_render_map


@contextmanager
def suppress_output():
    """
    Sim has this annoying log statement, this is used to suppress that
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def course_list_to_gate_list(course: list[np.array]) -> list[list[float]]:
    """
    - course: list of [x, y, z, yaw]

    Returns:
    - list[list[float]] -> Format of gates for simulator
    """
    return [
        [gate[0], gate[1], 0, 0, 0, gate[3], 1 if gate[3] == 0.3 else 0]
        for gate in course
    ]


def play_trajectory(ref_pos: np.ndarray,
                    ref_vel: np.ndarray = None,
                    ref_acc: np.ndarray = None,
                    use_gui: bool = False,
                    gates: list[np.array] = []) -> Tuple[bool, np.ndarray]:
    """
    Plays a trajectory sample in simulator

    Parameters:
    - ref_pos: nx3 trajectory matrix

    Returns: No crash (bool), drone states (np.ndarray)
    """
    with suppress_output():
        sys.argv.extend(["--overrides", "quadrotor_diffusion/quadrotor_diffusion/utils/play_trajectory.yaml"])
        parser = argparse.ArgumentParser(description='Generate unconditioned diffusion data.')
        parser.add_argument('--overrides', type=str, help='Config file')
        args, unknown = parser.parse_known_args()

        with open(args.overrides, 'r') as file:
            CONFIG = yaml.safe_load(file)

        CTRL_FREQ = CONFIG["quadrotor_config"]["ctrl_freq"]

        # Give it 1/3 second of just hover to stabilize similar to how I would run real experiment
        padding = 10
        ref_pos = np.vstack([np.vstack([ref_pos[0]] * padding), ref_pos])
        if ref_vel is not None:
            ref_vel = np.vstack([np.vstack([ref_vel[0]] * padding), ref_vel])
        if ref_acc is not None:
            ref_acc = np.vstack([np.vstack([ref_acc[0]] * padding), ref_acc])

        if ref_vel is None or ref_acc is None:
            ref_vel = derive_trajectory(ref_pos, CTRL_FREQ)
            ref_acc = derive_trajectory(ref_vel, CTRL_FREQ)
        reference = np.stack((ref_pos, ref_vel, ref_acc), axis=1)

        config = ConfigFactory().merge()
        config["quadrotor_config"]["gui"] = use_gui
        config["quadrotor_config"]["seed"] = int(time.time())
        config["quadrotor_config"]["gates"] = gates
        config["quadrotor_config"]["init_state"]["init_x"] = reference[0][0][0]
        config["quadrotor_config"]["init_state"]["init_y"] = reference[0][0][1]
        config["quadrotor_config"]["init_state"]["init_z"] = reference[0][0][2]
        config["quadrotor_config"]["init_state"]["init_psi"] = 0.0
        config["quadrotor_config"]["task_info"]["stabilization_goal"] = reference[-1][0]

        CTRL_DT = 1 / CTRL_FREQ
        FIRMWARE_FREQ = 500
        assert (config.quadrotor_config['pyb_freq'] % FIRMWARE_FREQ ==
                0), "pyb_freq must be a multiple of firmware freq"
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

        drone_states = [[[obs[0], obs[2], obs[4]], [obs[1], obs[3], obs[5]], [obs[6], obs[7], obs[8]]]]
        for step in range(reference.shape[0]):
            curr_time = step * CTRL_DT
            args = [reference[step][0], reference[step][1], reference[step][2], 0.0, np.zeros(3)]

            firmware_wrapper.sendFullStateCmd(*args, curr_time)
            obs, reward, _, info, action = firmware_wrapper.step(curr_time, action)

            if step > 0:
                drone_states.append([[obs[0], obs[2], obs[4]], [obs[1], obs[3], obs[5]], [obs[6], obs[7], obs[8]]])

            if reward < 0:
                env.close()
                states = np.transpose(np.array(drone_states), (1, 0, 2))
                states = states[:, padding:, :]
                return False, states

        env.close()
        states = np.transpose(np.array(drone_states), (1, 0, 2))
        states = states[:, padding:, :]
        return True, states


class SimulatorViewingAngle(enum.Enum):
    """
    Viewing angles for the simulator
    """
    BEV = 0
    PERSPECTIVE = 1
    YZ = 2


def render_simulation(drone_states: np.ndarray, course: list[np.array], reference: np.ndarray = None, viewing_angle: SimulatorViewingAngle = SimulatorViewingAngle.PERSPECTIVE, filename: str = None):
    """
    Renders simulation data

    Args:
        drone_states (np.ndarray): [position, velocity, orientations]
        course (list[np.array]): list[x, y, z, theta]
        reference (np.ndarray, optional): Reference trajectory drone is trying to track. Defaults to None.
        viewing_angle (SimulatorViewingAngle, optional): Viewing angle to render. Defaults to SimulatorViewingAngle.PERSPECTIVE
        filename (str, optional): Filename to save mp4 to if nothing passed in will plt.show. Defaults to None.
    """
    fig = plt.figure(figsize=(4, 4))
    fig.subplots_adjust(0, 0, 1, 1)
    ax = fig.add_subplot(111, projection='3d')

    ax.axis('off')
    ax.grid(False)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 1.0)

    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        z_range = z_limits[1] - z_limits[0]

        max_range = max(x_range, y_range, z_range)
        x_mid = (x_limits[0] + x_limits[1]) / 2
        y_mid = (y_limits[0] + y_limits[1]) / 2
        z_mid = (z_limits[0] + z_limits[1]) / 2

        ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
        ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
        ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])

    set_axes_equal(ax)

    # Draw base rectangle
    rectangle_vertices = np.array([
        [-1.5, -2, 0],
        [1.5, -2, 0],
        [1.5, 2, 0],
        [-1.5, 2, 0],
        [-1.5, -2, 0]
    ])
    line = Line3DCollection([rectangle_vertices[:, :3]], colors='black', linewidths=1)
    ax.add_collection3d(line)

    GATE_WIDTH_2 = 0.5 / 2
    for x, y, z, theta in course[1:-1]:
        center = np.array([x, y, z])
        left = R.from_euler('z', np.pi / 2).as_matrix() @ R.from_euler('z', theta).as_matrix() @ INITIAL_GATE_EXIT
        top = np.array([0, 0, 1])
        rectangle_vertices = [
            center + GATE_WIDTH_2 * left + GATE_WIDTH_2 * top,
            center + GATE_WIDTH_2 * left - GATE_WIDTH_2 * top,
            center - GATE_WIDTH_2 * left - GATE_WIDTH_2 * top,
            center - GATE_WIDTH_2 * left + GATE_WIDTH_2 * top,
            center + GATE_WIDTH_2 * left + GATE_WIDTH_2 * top,
        ]
        rectangle_vertices = np.array(rectangle_vertices)
        line = Line3DCollection([rectangle_vertices[:, :3]], colors='black', linewidths=2)
        ax.add_collection3d(line)

        post = [
            center - GATE_WIDTH_2 * top,
            center,
        ]
        post[-1][2] = 0.0
        post = np.array(post)
        ax.plot(post[:, 0], post[:, 1], post[:, 2], 'black', linewidth=2)

    # Prepare for reference trail (if needed)
    trail_scatter = None
    if reference is not None:
        point_sizes, colors = get_render_map(drone_states[1], 8, 1)
        # Initialize an empty scatter plot for the trail
        trail_scatter = ax.scatter([], [], [], s=[], c=[], alpha=0.7)

    if viewing_angle == SimulatorViewingAngle.YZ:
        ax.view_init(elev=5, azim=0)
    elif viewing_angle == SimulatorViewingAngle.BEV:
        ax.view_init(elev=90, azim=0)
    elif viewing_angle == SimulatorViewingAngle.PERSPECTIVE:
        ax.view_init(elev=25, azim=-145)
    else:
        raise ValueError("Unknown viewing angle")

    ax.dist = 2

    quadrotor_lines = []

    def draw_quadrotor(x, y, z, roll, pitch):
        LENGTH = 0.1 / 2
        HEIGHT = LENGTH / 2

        for line in quadrotor_lines:
            line.remove()
        quadrotor_lines.clear()
        c, s = np.cos(roll), np.sin(roll)
        rotation_roll = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        c, s = np.cos(pitch), np.sin(pitch)
        rotation_pitch = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        rotation_matrix = rotation_pitch @ rotation_roll

        arms = np.array([
            [LENGTH, 0, 0],
            [-LENGTH, 0, 0],
            [0, LENGTH, 0],
            [0, -LENGTH, 0]
        ])
        stubs = np.array([
            [LENGTH, 0, HEIGHT],
            [-LENGTH, 0, HEIGHT],
            [0, LENGTH, HEIGHT],
            [0, -LENGTH, HEIGHT]
        ])
        rotated_arms = arms @ rotation_matrix.T + np.array([x, y, z])
        rotated_stubs = stubs @ rotation_matrix.T + np.array([x, y, z])
        quadrotor_lines.append(ax.plot([rotated_arms[0, 0], rotated_arms[1, 0]],
                                       [rotated_arms[0, 1], rotated_arms[1, 1]],
                                       [rotated_arms[0, 2], rotated_arms[1, 2]], 'k-', lw=2)[0])
        quadrotor_lines.append(ax.plot([rotated_arms[2, 0], rotated_arms[3, 0]],
                                       [rotated_arms[2, 1], rotated_arms[3, 1]],
                                       [rotated_arms[2, 2], rotated_arms[3, 2]], 'k-', lw=2)[0])
        for i in range(4):
            quadrotor_lines.append(ax.plot([rotated_arms[i, 0], rotated_stubs[i, 0]],
                                           [rotated_arms[i, 1], rotated_stubs[i, 1]],
                                           [rotated_arms[i, 2], rotated_stubs[i, 2]], 'k-', lw=2)[0])

    position = drone_states[0]
    rpy_orientations = drone_states[2]

    # For trail optimization, only add new points to existing trail
    current_trail_length = 0

    def update(frame):
        nonlocal current_trail_length
        x, y, z = position[frame, 0], position[frame, 1], position[frame, 2]
        roll, pitch = rpy_orientations[frame, 0], rpy_orientations[frame, 1]
        draw_quadrotor(x, y, z, roll, pitch)

        # Update the reference trail efficiently - only add new points
        if reference is not None and trail_scatter is not None:
            # Only update if we have new points to add
            if frame > current_trail_length:
                # Get only the new points since last update
                new_x = reference[current_trail_length:frame, 0]
                new_y = reference[current_trail_length:frame, 1]
                new_z = reference[current_trail_length:frame, 2]
                new_sizes = point_sizes[current_trail_length:frame]
                new_colors = colors[current_trail_length:frame]

                # If this is the first update, create the scatter plot
                if current_trail_length == 0:
                    trail_scatter._offsets3d = (new_x, new_y, new_z)
                    trail_scatter.set_sizes(new_sizes)
                    trail_scatter.set_color(new_colors)
                else:
                    # Add new points to existing trail
                    current_x, current_y, current_z = trail_scatter._offsets3d
                    trail_scatter._offsets3d = (
                        np.append(current_x, new_x),
                        np.append(current_y, new_y),
                        np.append(current_z, new_z)
                    )
                    trail_scatter.set_sizes(np.append(trail_scatter.get_sizes(), new_sizes))

                    # For colors, we need to handle different color representations
                    if isinstance(trail_scatter.get_facecolor(), np.ndarray):
                        current_colors = trail_scatter.get_facecolor()
                        new_colors_array = np.array([colors[i] for i in range(current_trail_length, frame)])
                        trail_scatter.set_color(np.vstack((current_colors, new_colors_array)))
                    else:
                        # If not array, probably using a colormap
                        trail_scatter.set_array(np.append(trail_scatter.get_array(),
                                                          np.array(new_colors)))

                # Update the current trail length
                current_trail_length = frame

    ani = FuncAnimation(fig, update, frames=len(position), interval=1000 / 30, blit=False)
    if filename is not None:
        ani.save(filename, writer="ffmpeg")
    else:
        plt.show()
    plt.close()


def create_perspective_rendering(drone_states: np.ndarray, course: list[np.array], filename: str, reference: np.ndarray = None):
    """
    Creates three views of the simulation data

    Args:
        drone_states (np.ndarray): [position, _velocity, _orientation]
        course (list[np.array]): [[x, y, z, theta]]
        filename (str): Filename to save to
        reference (np.ndarray, optional): Reference trajectory to follow. Defaults to None.
    """
    # extract directory of filename
    base_dir = os.path.dirname(filename)

    bev = os.path.join(base_dir, "bev.mp4")
    perspective = os.path.join(base_dir, "perspective.mp4")
    yz = os.path.join(base_dir, "yz.mp4")

    render_simulation(drone_states, course, None, SimulatorViewingAngle.BEV, filename=bev)
    render_simulation(drone_states, course, None, SimulatorViewingAngle.YZ, filename=yz)
    render_simulation(drone_states, course, reference, SimulatorViewingAngle.PERSPECTIVE, filename=perspective)

    with suppress_output():
        subprocess.run(["ffmpeg", "-i", bev, "-i", perspective, "-i", yz,
                        "-filter_complex", "hstack=inputs=3", filename], check=True, stdout=subprocess.DEVNULL)

    os.remove(bev)
    os.remove(perspective)
    os.remove(yz)
