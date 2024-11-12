import numpy as np

from quadrotor_diffusion.utils.simulator import play_trajectory
from quadrotor_diffusion.utils.trajectory import compute_tracking_error
from quadrotor_diffusion.utils.plotting import (
    plot_reference_time_series,
    view_trajectories_in_3d,
)

SAMPLE_NUM = 4682
ref_pos = np.load(f"data/quadrotor_random/{SAMPLE_NUM}.npy")

worked, states = play_trajectory(ref_pos)

if not worked:
    print("Crashed")
else:
    print("Avg errors: ", compute_tracking_error(ref_pos, states))

plot_reference_time_series("ref.svg", "reference trajectory {SAMPLE_NUM}", ref_pos)
plot_reference_time_series("ref__st.svg", "reference trajectory comparison {SAMPLE_NUM}", ref_pos, states)

view_trajectories_in_3d("ref3d.svg", "reference trajectory {SAMPLE_NUM}", ref_pos)
view_trajectories_in_3d("ref__st3d.svg", "reference trajectory comparison {SAMPLE_NUM}", ref_pos, states)
