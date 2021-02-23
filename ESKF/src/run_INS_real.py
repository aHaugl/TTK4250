# %% imports
from matplotlib import use
from optimization import optimize, cost_function_NIS
import scipy
import scipy.io
import scipy.stats

import matplotlib.pyplot as plt
import numpy as np
import os
import time
from plotter import plot_path, plot_estimate, state_error_plots, plot_NIS
from eskf_runner import run_eskf

from eskf import (
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
)
# Done
# %% plot config check and style setup

from plott_setup import setup_plot
setup_plot()

# %% load data and plot
folder = os.path.dirname(__file__)
# filename_to_load = f"{folder}/../data/task_simulation.mat"
filename_to_load = f"{folder}/../data/task_real.mat"
cache_folder = os.path.join(folder, '..', 'cache')
loaded_data = scipy.io.loadmat(filename_to_load)

timeIMU = loaded_data["timeIMU"].ravel()
if 'xtrue' in loaded_data:
    x_true = loaded_data["xtrue"].T
else:
    x_true = None
z_GNSS = loaded_data["zGNSS"].T
dt = np.mean(np.diff(timeIMU))


# %% Measurement noise
# IMU noise values for STIM300, based on datasheet and simulation sample rate
# Continous noise
# cont_gyro_noise_std = 4.36e-5  # (rad/s)/sqrt(Hz)
# cont_acc_noise_std = 1.167e-3  # (m/s**2)/sqrt(Hz)

# Discrete sample noise at simulation rate used
# Hvorfor gange med en halv? (eq. 10.70)
# acc_std = 0.5 * cont_acc_noise_std * np.sqrt(1 / dt)
# rate_std = 0.5 * cont_gyro_noise_std * np.sqrt(1 / dt)


acc_std = 3.92364040e-03 * np.sqrt(0.01/dt)
rate_std = 2.74409187e-04 * np.sqrt(0.01/dt)
# Bias values
# acc_bias_driving_noise_std = 4e-3
# cont_acc_bias_driving_noise_std = 6 * \
#     acc_bias_driving_noise_std / np.sqrt(1 / dt)

# rate_bias_driving_noise_std = 5e-5
# cont_rate_bias_driving_noise_std = (
#     (1/3) * rate_bias_driving_noise_std / np.sqrt(1 / dt)
# )
cont_acc_bias_driving_noise_std = 1.48610933e-03 / np.sqrt(0.01/dt)
cont_rate_bias_driving_noise_std = 5.62468004e-04 / np.sqrt(0.01/dt)
# Position and velocity measurement
p_std = np.array([0.10614235, 0.10614235, 0.1061424])  # Measurement noise


p_acc = 1e-9
p_gyro = 1e-9

eskf_parameters = [acc_std,
                   rate_std,
                   cont_acc_bias_driving_noise_std,
                   cont_rate_bias_driving_noise_std,
                   p_acc,
                   p_gyro]
eskf_parameters = [1.59887024e-01, 1.16727905e-03, 6.70062772e-04, 1.80454915e-02,
                   1.42293717e-09, 1.93957163e-09]
# %% Initialise
x_pred_init = np.zeros(16)
x_pred_init[POS_IDX] = np.array([0, 0, 0])
x_pred_init[VEL_IDX] = np.array([0, 0, 0])
x_pred_init[ATT_IDX] = np.array([
    np.cos(45 * np.pi / 180),
    0, 0,
    np.sin(45 * np.pi / 180)
])

# no initial rotation: nose to North, right to East, and belly down
x_pred_init[6] = 1

# These have to be set reasonably to get good results

P_pred_init_pos = 0.584566
P_pred_init_vel = 0.78118225
P_pred_init_err_att = 0.00114295
P_pred_init_err_acc_bias = 0.02167314
P_pred_init_err_gyro_bias = 0.00939913
P_pred_init_list = [P_pred_init_pos,
                    P_pred_init_vel,
                    P_pred_init_err_att,
                    P_pred_init_err_acc_bias,
                    P_pred_init_err_gyro_bias]

P_pred_init_list = [3.10976674, 3.82838086,
                    0.37167027, -0.00466379, -0.00782738]

init_parameters = [x_pred_init, P_pred_init_list]

# %% Run estimation

N: int = int(300/dt)
# N: int = timeIMU.size
offset = 207.
doGNSS: bool = True


use_cache = True
parameters = eskf_parameters + init_parameters

"""
To find good parameters we used the Nelder-Mead algorithm.
"""
if input("Do you want to run the optimizer (takes several hours)? [y/n]: ") == 'y':
    optimize(cost_function_NIS, eskf_parameters, p_std,
             x_pred_init, P_pred_init_list, loaded_data, N, offset,
             use_GNSSaccuracy=True)

(x_pred,
    x_est,
    P_est,
    delta_x,
    NIS,
    NEES_all,
    NEES_pos,
    NEES_vel,
    NEES_att,
    NEES_accbias,
    NEES_gyrobias,
    GNSSk) = run_eskf(eskf_parameters, x_pred_init, P_pred_init_list, loaded_data,
                      p_std, N,
                      use_GNSSaccuracy=True, doGNSS=doGNSS, offset=offset)


t = np.linspace(0, dt * (N - 1), N)
plot_path(N, GNSSk, x_est, z_GNSS, x_true)
plot_estimate(t, N, x_est)
plot_NIS(NIS, 0.9)
plt.show()
