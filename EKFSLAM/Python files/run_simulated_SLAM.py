# %% Imports
from scipy.io import loadmat
import numpy as np
from typing import List, Optional
from matplotlib import pyplot as plt
from EKFSLAM import EKFSLAM
from tqdm import tqdm
from scipy.stats import chi2
import matplotlib.pyplot as plt

from plott_setup import setup_plot
from plotting import *
setup_plot()
# %% plot config check and style setup

# to see your plot config

# %% Load data
simSLAM_ws = loadmat("simulatedSLAM")


z = [zk.T for zk in simSLAM_ws["z"].ravel()]

landmarks = simSLAM_ws["landmarks"].T
odometry = simSLAM_ws["odometry"].T
poseGT = simSLAM_ws["poseGT"].T

K = len(z)
# M = len(landmarks)

# %% Initilize
Q = np.diag([0.2, 0.2, 0.001])  # TODO Best
Q = np.diag([0.2, 0.2, 0.001])  # TODO Best
Q = np.diag([0.0012, 0.0012, 0.00007])  # TODO Best
R = np.diag([0.0025, 0.01])  # TODO Best
R = np.diag([0.0025, 0.01])  # TODO Best
R = np.diag([0.002, 0.0005])  # TODO Best
# assert 0
do_asso = True

JCBBalphas = np.array(
    [10e-4, 10e-6]  # TODO,
)   # first is for joint compatibility, second is individual
JCBBalphas = np.array(
    [10e-6, 10e-8]  # TODO,
)   # first is for joint compatibility, second is individual
# these can have a large effect on runtime either through the number of landmarks created
# or by the size of the association search space.

alpha = 0.95
N = 1000

eta_pred_init = poseGT[0]  # we start at the correct position for reference
P_pred_init = np.zeros((3, 3))
# allocate

# %% Set up plotting
# plotting

doAssoPlot = False
playMovie = True

K = len(z)
slam = EKFSLAM(Q, R, do_asso=do_asso, alphas=JCBBalphas)

axAsso = None
if doAssoPlot:
    figAsso, axAsso = plt.subplots()

eta_pred: List[Optional[np.ndarray]] = [None] * K
P_pred: List[Optional[np.ndarray]] = [None] * K
eta_hat: List[Optional[np.ndarray]] = [None] * K
P_hat: List[Optional[np.ndarray]] = [None] * K
a: List[Optional[np.ndarray]] = [None] * K
NIS = np.zeros(K)
NISnorm = np.zeros(K)
CI = np.zeros((K, 2))
CInorm = np.zeros((K, 2))
NEES = np.zeros((K, 3))

eta_pred[0] = eta_pred_init
P_pred[0] = P_pred_init
# %% Run simulation

for k, z_k in tqdm(enumerate(z[:N])):
    eta_hat[k], P_hat[k], NIS[k], a[k] = slam.update(
        eta_pred[k], P_pred[k], z_k)  # TODO update
    if k < K - 1:
        # TODO predict
        eta_pred[k + 1], P_pred[k + 1] = slam.predict(
            eta_hat[k], P_hat[k], odometry[k])

    # assert (
    #     eta_hat[k].shape[0] == P_hat[k].shape[0]
    # ), "dimensions of mean and covariance do not match"

    num_asso = np.count_nonzero(a[k] > -1)

    CI[k] = chi2.interval(alpha, 2 * num_asso)

    if num_asso > 0:
        NISnorm[k] = NIS[k] / (2 * num_asso)
        CInorm[k] = CI[k] / (2 * num_asso)
    else:
        NISnorm[k] = 1
        CInorm[k].fill(1)

    NEES[k] = EKFSLAM.NEESes(eta_hat[k][:3], P_hat[k][:3, :3], poseGT[k])

    if doAssoPlot and k > 0:
        axAsso.clear()
        axAsso.grid()
        zpred = slam.h(eta_pred[k]).reshape(-1, 2)
        axAsso.scatter(z_k[:, 0], z_k[:, 1], label="z")
        axAsso.scatter(zpred[:, 0], zpred[:, 1], label="zpred")
        xcoords = np.block(
            [[z_k[a[k] > -1, 0]], [zpred[a[k][a[k] > -1], 0]]]).T
        ycoords = np.block(
            [[z_k[a[k] > -1, 1]], [zpred[a[k][a[k] > -1], 1]]]).T
        for x, y in zip(xcoords, ycoords):
            axAsso.plot(x, y, lw=3, c="r")
        axAsso.legend()
        axAsso.set_title(
            f"k = {k}, {np.count_nonzero(a[k] > -1)} associations")
        plt.draw()
        plt.pause(0.001)

pose_est = np.array([x[:3] for x in eta_hat[:N]])
lmk_est = [eta_hat_k[3:].reshape(-1, 2) for eta_hat_k in eta_hat[:N]]
lmk_est_final = lmk_est[N - 1]

np.set_printoptions(precision=4, linewidth=100)

mins = np.amin(landmarks, axis=0)
maxs = np.amax(landmarks, axis=0)

ranges = maxs - mins
offsets = ranges * 0.2

mins -= offsets
maxs += offsets

# %% Plotting of results
# movie = play_movie(pose_est, poseGT, lmk_est, landmarks, P_hat, N)
plot_trajectory(pose_est, poseGT, P_hat, N)
plot_error(pose_est, poseGT, P_hat, N)
plot_path(pose_est, poseGT, lmk_est_final, landmarks, P_hat, N)
plot_NIS(NISnorm, CInorm, alpha, N)
plot_NEES(NEES, alpha, N)
plt.show()
