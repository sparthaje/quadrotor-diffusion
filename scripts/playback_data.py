import os

import numpy as np

from quadrotor_diffusion.utils.simulator import play_trajectory
from quadrotor_diffusion.utils.trajectory import (
    compute_tracking_error,
    derive_trajectory
)
from quadrotor_diffusion.utils.plotting import (
    plot_states,
    plot_ref_obs_states,
    pcd_plot,
)

SAMPLE_NUM = 29
ref_pos = np.load(f"data/quadrotor_random/{SAMPLE_NUM}.npy")
ref_vel = derive_trajectory(ref_pos, 30)
ref_acc = derive_trajectory(ref_vel, 30, order=3)

worked, obs_pos = play_trajectory(ref_pos)
obs_vel = derive_trajectory(obs_pos, 30)
obs_acc = derive_trajectory(obs_vel, 30, order=3)

if not worked:
    print("Crashed")
else:
    print("Avg errors: ", compute_tracking_error(ref_pos, obs_pos))

base_dir = f"logs/plots/sample_plots/{SAMPLE_NUM}"

if not os.path.exists(base_dir):
    os.mkdir(base_dir)

plot_states(
    ref_pos,
    ref_vel,
    ref_acc,
    f"Reference Trajectory {SAMPLE_NUM}",
    os.path.join(base_dir, f"reference_{SAMPLE_NUM}.pdf"),
)

plot_ref_obs_states(
    ref_pos,
    ref_vel,
    ref_acc,
    obs_pos,
    obs_vel,
    obs_acc,
    f"Reference vs Observed {SAMPLE_NUM}",
    os.path.join(base_dir, f"reference_vs_observed_{SAMPLE_NUM}.pdf"),
)

pcd_plot(ref_pos, os.path.join(base_dir, f"reference_{SAMPLE_NUM}.xyz"))
pcd_plot(obs_pos, os.path.join(base_dir, f"observed_{SAMPLE_NUM}.xyz"))
