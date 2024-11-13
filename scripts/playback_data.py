import os

import numpy as np

from quadrotor_diffusion.utils.simulator import play_trajectory
from quadrotor_diffusion.utils.trajectory import compute_tracking_error
from quadrotor_diffusion.utils.plotting import (
    plot_reference_time_series,
    view_trajectories_in_3d,
)

SAMPLE_NUM = 25
ref_pos = np.load(f"data/quadrotor_random/{SAMPLE_NUM}.npy")

worked, states = play_trajectory(ref_pos)

if not worked:
    print("Crashed")
else:
    print("Avg errors: ", compute_tracking_error(ref_pos, states))

base_dir = f"logs/plots/sample_plots/{SAMPLE_NUM}"

os.mkdir(base_dir)

plot_reference_time_series(os.path.join(base_dir, "ref.pdf"), "reference trajectory {SAMPLE_NUM}", ref_pos)
plot_reference_time_series(os.path.join(base_dir, "ref_states.pdf"),
                           "reference trajectory comparison {SAMPLE_NUM}", ref_pos, states)

view_trajectories_in_3d(os.path.join(base_dir, "ref3d.pdf"), "reference trajectory {SAMPLE_NUM}", ref_pos)
view_trajectories_in_3d(os.path.join(base_dir, "ref3d_states.pdf"),
                        "reference trajectory comparison {SAMPLE_NUM}", ref_pos, states)
