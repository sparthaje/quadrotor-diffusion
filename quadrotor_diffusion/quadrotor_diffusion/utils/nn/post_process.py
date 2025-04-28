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

    n = len(recon_pos)
    t = np.linspace(0, (n - 1) / ctrl_freq, n)

    derived_vel = derive_trajectory(recon_pos, 30)

    t = np.linspace(0, (n - 2) / ctrl_freq, n-1)
    spline_vx = UnivariateSpline(t, derived_vel[:, 0], s=0.3, k=4)
    spline_vy = UnivariateSpline(t, derived_vel[:, 1], s=0.3, k=4)
    spline_vz = UnivariateSpline(t, derived_vel[:, 2], s=0.3, k=4)

    vel_fitted = np.column_stack((spline_vx(t), spline_vy(t), spline_vz(t)))

    t = np.linspace(0, (n - 3) / ctrl_freq, n-2)
    derived_accel = derive_trajectory(vel_fitted, 30, order=1)
    spline_ax = UnivariateSpline(t, derived_accel[:, 0], s=0.2, k=3)
    spline_ay = UnivariateSpline(t, derived_accel[:, 1], s=0.2, k=3)
    spline_az = UnivariateSpline(t, derived_accel[:, 2], s=0.2, k=3)
    accel_fitted = np.column_stack((spline_ax(t), spline_ay(t), spline_az(t)))

    return recon_pos, derived_vel, accel_fitted
