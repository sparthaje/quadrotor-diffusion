import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

from quadrotor_diffusion.utils.trajectory import derive_trajectory


def fit_to_recon(recon_pos: np.ndarray, ctrl_freq: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit the reconstructed position to a smooth trajectory

    Args:
        recon_pos (np.ndarray): nx3 position array
        ctrl_freq (int): The frequency of the data in the array

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Fitted position, velocity, and accel data as function of time
    """

    x = recon_pos[:, 0]
    y = recon_pos[:, 1]
    z = recon_pos[:, 2]

    n = len(recon_pos)
    t = np.linspace(0, (n - 1) / ctrl_freq, n)

    spline_x = UnivariateSpline(t, x, s=1e-1, k=5)
    spline_y = UnivariateSpline(t, y, s=1e-1, k=5)
    spline_z = UnivariateSpline(t, z, s=5e-2, k=5)
    pos_fitted = np.column_stack((spline_x(t), spline_y(t), spline_z(t)))

    derived_vel = derive_trajectory(recon_pos, 30)
    derived_vel_smooth = savgol_filter(derived_vel, window_length=30, polyorder=4, axis=0)

    spline_vx = UnivariateSpline(t, derived_vel_smooth[:, 0], s=0.3, k=4)
    spline_vy = UnivariateSpline(t, derived_vel_smooth[:, 1], s=0.3, k=4)
    spline_vz = UnivariateSpline(t, derived_vel_smooth[:, 2], s=0.3, k=4)

    vel_fitted = np.column_stack((spline_vx(t), spline_vy(t), spline_vz(t)))
    # vel_fitted[0] = np.array([0.0, 0.0, 0.0])
    # vel_fitted[1] = np.array([0.0, 0.0, 0.0])

    derived_accel = derive_trajectory(vel_fitted, 30, order=1)
    derived_accel_smooth = savgol_filter(derived_accel, window_length=50, polyorder=3, axis=0)
    spline_ax = UnivariateSpline(t, derived_accel_smooth[:, 0], s=0.2, k=3)
    spline_ay = UnivariateSpline(t, derived_accel_smooth[:, 1], s=0.2, k=3)
    spline_az = UnivariateSpline(t, derived_accel_smooth[:, 2], s=0.2, k=3)
    accel_fitted = np.column_stack((spline_ax(t), spline_ay(t), spline_az(t)))
    # accel_fitted[0] = np.array([0.0, 0.0, 0.0])
    # accel_fitted[1] = np.array([0.0, 0.0, 0.0])

    return pos_fitted, vel_fitted, accel_fitted
