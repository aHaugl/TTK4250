# %% imports
from typing import List

import scipy
import scipy.io
import scipy.stats

import matplotlib.pyplot as plt
import numpy as np

import dynamicmodels
import measurementmodels
import ekf
import imm
import pda
from gaussparams import GaussParams
from mixturedata import MixtureParameters
import estimationstatistics as estats

from plotting_utils import apply_settings
from plotting import (plot_measurements, plot_traj, plot_NEES_CI, plot_errors,
                      plot_NIS_NEES_model_specific, get_rotation_variance,
                      get_measurements_variance, get_acceleration_std)

# %% plot config check and style setup

# to see your plot config
apply_settings()


# %% load data and plot
filename_to_load = "data_joyride.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item()
Ts = loaded_data["Ts"].squeeze()
Xgt = loaded_data["Xgt"].T
Z = [zk.T for zk in loaded_data["Z"].ravel()]

# %% setup and track

# %% IMM-PDA with CV/CT-models copied from run_im_pda.py

# sensor
sigma_z = 22.7
clutter_intensity = 0.00005
PD = 0.95
gate_size = 4

# dynamic models
sigma_a_CV = 1.2
sigma_a_CT = 0.3
sigma_omega = 0.02*np.pi

# markov chain
PI11 = 0.9
PI22 = 0.9

p10 = 0.9  # initvalue for mode probabilities

PI = np.array([[PI11, (1 - PI11)], [(1 - PI22), PI22]])
assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"

# init values
mean_init = np.array([7200, 3700, 0, 0, 0])
cov_init = np.diag([100, 100, 10, 10, 0.1]) ** 2
mode_probabilities_init = np.array([p10, (1 - p10)])
mode_states_init = GaussParams(mean_init, cov_init)
init_imm_state = MixtureParameters(
    mode_probabilities_init, [mode_states_init] * 2)

assert np.allclose(
    np.sum(mode_probabilities_init), 1
), "initial mode probabilities must sum to 1"

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))
imm_filter = imm.IMM(ekf_filters, PI)

tracker = pda.PDA(imm_filter, clutter_intensity, PD, gate_size)

NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

NIS_CV_list = []
NIS_CT_list = []
NEES_CV_list = []
NEES_CT_list = []

gated_list = []
tracker_update = init_imm_state
tracker_update_list = []
tracker_predict_list = []
tracker_estimate_list = []

# First measurement is time 0 -> don't predict before first update.
Ts = [0, *Ts]

# estimate
for k, (Zk, x_true_k) in enumerate(zip(Z, Xgt)):

    tracker_predict = tracker.predict(tracker_update, Ts[k])
    tracker_update = tracker.update(Zk, tracker_predict)
    tracker_estimate = tracker.estimate(tracker_update)

    gated = tracker.gate(Zk, tracker_predict)
    Z_accepted = Zk[gated]
    for z_accepted in Z_accepted:
        NIS_CV_list.append([k, ekf_filters[0].NIS(
            z_accepted, tracker_predict.components[0])])

        NIS_CT_list.append([k, ekf_filters[1].NIS(
            z_accepted, tracker_predict.components[1])])

        cv_update = tracker_update.components[0]
        cv_update = GaussParams(cv_update.mean[:4], cv_update.cov[:4, :4])
        ct_update = tracker_update.components[1]
        ct_update = GaussParams(ct_update.mean[:4], ct_update.cov[:4, :4])
        NEES_CV_list.append([k, ekf_filters[0].NEES(
            cv_update,
            x_true_k,
            NEES_idx=np.arange(4))])

        NEES_CT_list.append([k, ekf_filters[1].NEES(
            ct_update,
            x_true_k,
            NEES_idx=np.arange(4))])

    NEES[k] = estats.NEES_indexed(
        tracker_estimate.mean, tracker_estimate.cov, x_true_k, idxs=np.arange(
            4)
    )

    NEESpos[k] = estats.NEES_indexed(
        tracker_estimate.mean, tracker_estimate.cov, x_true_k, idxs=np.arange(
            2)
    )
    NEESvel[k] = estats.NEES_indexed(
        tracker_estimate.mean, tracker_estimate.cov, x_true_k, idxs=np.arange(
            2, 4)
    )

    gated_list.append(gated)
    tracker_predict_list.append(tracker_predict)
    tracker_update_list.append(tracker_update)
    tracker_estimate_list.append(tracker_estimate)


x_hat = np.array([est.mean for est in tracker_estimate_list])
prob_hat = np.array([upd.weights for upd in tracker_update_list])

poserr = np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=0)
velerr = np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=0)
posRMSE = np.sqrt(
    np.mean(poserr ** 2)
)
velRMSE = np.sqrt(np.mean(velerr ** 2))
peak_pos_deviation = poserr.max()
peak_vel_deviation = velerr.max()


# consistency
confprob = 0.9
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K
ANEESpos = np.mean(NEESpos)
ANEESvel = np.mean(NEESvel)
ANEES = np.mean(NEES)

if 1:
    plot_measurements(K, Ts, Xgt, Z)
    plot_traj(Ts, Xgt, x_hat, Z, gated_list, posRMSE, velRMSE, prob_hat,
              peak_pos_deviation, peak_vel_deviation
              )
    plot_NEES_CI(Ts, NEESpos, ANEESpos, NEESvel, ANEESvel, NEES, ANEES,
                 CI2, CI4, CI2K, CI4K, confprob)
    plot_errors(Ts, Xgt, x_hat, CI2, CI4, CI2K, CI4K, confprob)
    plot_NIS_NEES_model_specific(Ts,
                                 NIS_CV_list, NIS_CT_list,
                                 NEES_CV_list, NEES_CT_list,
                                 confprob)

    print('Some data analysis for better guess')
    print(get_rotation_variance(Xgt))
    print(get_measurements_variance(Xgt, Z, gated_list))
    print(get_acceleration_std(Xgt, Z, gated_list))
    plt.show()
