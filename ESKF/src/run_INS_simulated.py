# %% imports
import scipy
import scipy.io
import scipy.stats
from optimization import optimize, cost_function_SIM
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from plotter import (plot_path, plot_estimate, state_error_plots,
                     plot_NIS, plot_NEES)
from eskf_runner import run_eskf

from eskf import (
    POS_IDX,
    VEL_IDX,
)

# %% plot config check and style setup

from plott_setup import setup_plot
setup_plot()
np.seterr(all='raise')
scipy.special.seterr(all='raise')
# %% load data and plot
folder = os.path.dirname(__file__)
# filename_to_load = f"{folder}/../data/task_simulation.mat"
filename_to_load = f"{folder}/../data/task_simulation.mat"
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

# acc_std = 0.5 * cont_acc_noise_std * np.sqrt(1 / dt)
# rate_std = 0.5 * cont_gyro_noise_std * np.sqrt(1 / dt)
acc_std = 3.92364040e-03
rate_std = 2.74409187e-04
# Bias values
# acc_bias_driving_noise_std = 4e-3
# cont_acc_bias_driving_noise_std = 6 * \
#     acc_bias_driving_noise_std / np.sqrt(1 / dt)

# rate_bias_driving_noise_std = 5e-5
# cont_rate_bias_driving_noise_std = (
#     (1/3) * rate_bias_driving_noise_std / np.sqrt(1 / dt)
# )
cont_acc_bias_driving_noise_std = 1.48610933e-03
cont_rate_bias_driving_noise_std = 5.62468004e-04
# Position and velocity measurement
p_std = np.array([0.3887816, 0.3887816, 0.51122025])  # Measurement noise

p_acc = 1e-9
p_gyro = 1e-9
# [-2.71073648e-02  1.97296299e-04 -7.88136014e-04  5.62588030e-05
#  -7.26577980e-04  1.43524292e-01]
eskf_parameters = [acc_std,
                   rate_std,
                   cont_acc_bias_driving_noise_std,
                   cont_rate_bias_driving_noise_std,
                   p_acc,
                   p_gyro]
# %% Initialise
x_pred_init = np.zeros(16)
x_pred_init[POS_IDX] = np.array([0, 0, -5])  # starting 5 metres above ground
x_pred_init[VEL_IDX] = np.array([20, 0, 0])  # starting at 20 m/s due north
# no initial rotation: nose to North, right to East, and belly down
x_pred_init[6] = 1

# These have to be set reasonably to get good results

# [241.94198986 319.04325528   2.06011741   0.77419239   0.53211694] best result so far with simplex
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


init_parameters = [x_pred_init, P_pred_init_list]

# %% Run estimation

N: int = int(300/dt)
N: int = len(timeIMU)
offset = 0.
doGNSS: bool = True
# TODO: Set this to False if you want to check that the predictions make sense over reasonable time lenghts


parameters = eskf_parameters + init_parameters
# %% plotting


"""
To find good parameters we used the Nelder-Mead algorithm. 
"""
if input("Do you want to run the optimizer (takes several hours)? [y/n]: ") == 'y':
    optimize(cost_function_SIM, eskf_parameters, p_std,
             x_pred_init, P_pred_init_list, loaded_data, N, offset,
             use_GNSSaccuracy=False)
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
    GNSSk) = run_eskf(eskf_parameters, x_pred_init, P_pred_init_list,
                      loaded_data, p_std, N, doGNSS=doGNSS, offset=offset,
                      use_GNSSaccuracy=False)


t = np.linspace(0, dt * (N - 1), N)
plot_path(N, GNSSk, x_est, z_GNSS, x_true)
plot_estimate(t, N, x_est)
state_error_plots(t, N, x_est, x_true, delta_x)
plot_NIS(NIS)
plot_NEES(t, N, dt,
          NEES_all, NEES_pos, NEES_vel, NEES_att, NEES_accbias,
          NEES_gyrobias, confprob=0.95)
plt.show()
