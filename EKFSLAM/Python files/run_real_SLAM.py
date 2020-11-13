# %% Imports
from numpy.lib.twodim_base import eye
from scipy.io import loadmat
from scipy.stats import chi2

try:
    from tqdm import tqdm
except ImportError as e:
    print(e)
    print("install tqdm for progress bar")

    # def tqdm as dummy
    def tqdm(*args, **kwargs):
        return args[0]


import numpy as np
from EKFSLAM import EKFSLAM
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from plotting import ellipse
from vp_utils import detectTrees, odometry, Car
from utils import rotmat2d

from tqdm import tqdm
from plott_setup import setup_plot
from plotting import *
from plotting import *
setup_plot()

# %% Load data
VICTORIA_PARK_PATH = "./victoria_park/"
realSLAM_ws = {
    **loadmat(VICTORIA_PARK_PATH + "aa3_dr"),
    **loadmat(VICTORIA_PARK_PATH + "aa3_lsr2"),
    **loadmat(VICTORIA_PARK_PATH + "aa3_gpsx"),
}

timeOdo = (realSLAM_ws["time"] / 1000).ravel()
timeLsr = (realSLAM_ws["TLsr"] / 1000).ravel()
timeGps = (realSLAM_ws["timeGps"] / 1000).ravel()

steering = realSLAM_ws["steering"].ravel()
speed = realSLAM_ws["speed"].ravel()
LASER = (
    realSLAM_ws["LASER"] / 100
)  # Divide by 100 to be compatible with Python implementation of detectTrees
La_m = realSLAM_ws["La_m"].ravel()
Lo_m = realSLAM_ws["Lo_m"].ravel()

K = timeOdo.size
mK = timeLsr.size
Kgps = timeGps.size

# %% Parameters
L = 2.83  # axel distance
H = 0.76  # center to wheel encoder
a = 0.95  # laser distance in front of first axel
b = 0.5  # laser distance to the left of center

car = Car(L, H, a, b)
# odometry = get_odometry(speed, steering, dt, car)
# sigmas = np.array([1, 1, 0.12]) * 1e-3 # TODO
# CorrCoeff = np.array([[1, 0, 0], [0, 1, 0.9], [0, 0.9, 1]])
# Q = np.diag(sigmas) @ CorrCoeff @ np.diag(sigmas)

# R = np.diag([0.08 ** 2, 0.02 ** 2]) # TODO

# Q = np.diag([0.01, 0.01, 0.0001])  # TODO Best
Q = np.diag([0.0012, 0.0012, 0.00007])
# R = np.diag([0.25, 0.0025])
R = np.diag([0.002, 0.0005])  # TODO Best

JCBBalphas = np.array([1e-6, 1e-8])
# JCBBalphas = np.array([10e-6, 10e-8]) 

sensorOffset = np.array([car.a + car.L, car.b])
doAsso = True

slam = EKFSLAM(Q, R, do_asso=doAsso, alphas=JCBBalphas,
               sensor_offset=sensorOffset)

# For consistency testing
alpha = 0.05
confidence_prob = 1 - alpha

xupd = np.zeros((mK, 3))
P_all = []
a = [None] * mK
NIS = np.zeros(mK)
NISnorm = np.zeros(mK)
CI = np.zeros((mK, 2))
CInorm = np.zeros((mK, 2))

# Initialize state
# you might want to tweak these for a good reference
eta = np.array([Lo_m[0], La_m[1], 35 * np.pi / 180])
# eta = np.array([Lo_m[0], La_m[1], 126 * np.pi / 180])
P = np.zeros((3, 3))

mk_first = 1  # first seems to be a bit off in timing
mk = mk_first
t = timeOdo[0]

# %%  run
N = K  # K = 61945 is max?

doPlot = False

lh_pose = None

ax = None
if doPlot:
    fig, ax = plt.subplots(num=1, clear=True)

    # lh_pose = ax.plot(eta[0], eta[1], "k", lw=3)[0]
    # sh_lmk = ax.scatter(np.nan, np.nan, c="r", marker="x")
    # sh_Z = ax.scatter(np.nan, np.nan, c="b", marker=".")

do_raw_prediction = True
if do_raw_prediction:  # TODO: further processing such as plotting
    odos = np.zeros((K, 3))
    odox = np.zeros((K, 3))
    odox[0] = eta.copy()

    for k in range(min(N, K - 1)):
        odos[k + 1] = odometry(speed[k + 1], steering[k + 1], 0.025, car)
        odox[k + 1], _ = slam.predict(odox[k], P.copy(), odos[k + 1])

for k in tqdm(range(N)):
    if mk < mK - 1 and timeLsr[mk] <= timeOdo[k + 1]:
        # Force P to symmetric: there are issues with long runs (>10000 steps)
        # seem like the prediction might be introducing some minor asymetries,
        # so best to force P symetric before update (where chol etc. is used).
        # TODO: remove this for short debug runs in order to see if there are small errors
        P = (P + P.T) / 2
        dt = timeLsr[mk] - t
        if dt < 0:  # avoid assertions as they can be optimized avay?
            raise ValueError("negative time increment")

        # ? reset time to this laser time for next post predict
        t = timeLsr[mk]
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        eta, P = slam.predict(eta, P, odo)  # TODO predict

        z = detectTrees(LASER[mk])
        eta, P, NIS[mk], a[mk] = slam.update(eta, P, z)  # TODO update

        num_asso = np.count_nonzero(a[mk] > -1)

        if num_asso > 0:
            NISnorm[mk] = NIS[mk] / (2 * num_asso)
            CInorm[mk] = np.array(chi2.interval(confidence_prob, 2 * num_asso)) / (
                2 * num_asso
            )
        else:
            NISnorm[mk] = 1
            CInorm[mk].fill(1)

        xupd[mk] = eta[:3]

        if doPlot:
            ax.clear()

            ax.scatter(*eta[3:].reshape(-1, 2).T, c="r", marker="x")
            if len(z) > 0:
                zinmap = (
                    rotmat2d(eta[2])
                    @ (
                        z[:, 0] * np.array([np.cos(z[:, 1]), np.sin(z[:, 1])])
                        + slam.sensor_offset[:, None]
                    )
                    + eta[0:2, None]
                )
                # sh_Z.set_offsets(zinmap.T)
                ax.scatter(*zinmap, c="b", marker=".")
            ax.plot(*xupd[mk_first:mk, :2].T, "k", lw=3)
            # el = ellipse(eta[:2], P[:2, :2], 2, 40)
            # ax.plot(*el.T, "g", lw=3)
            for i in range(eta[3::2].size):
                i *= 2
                el = ellipse(eta[3:][i:i+2], P[3:, 3:][i:i+2, i:i+2], 2, 40)
                ax.plot(*el.T, c="b")

            ax.set(
                xlim=[-200, 200],
                ylim=[-200, 200],
                title=f"step {k}, laser scan {mk}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}",
            )
            plt.draw()
            plt.pause(0.1)
            # input('press enter')

        mk += 1

    if k < K - 1:
        dt = timeOdo[k + 1] - t
        t = timeOdo[k + 1]
        odo = odometry(speed[k + 1], steering[k + 1], dt, car)
        eta, P = slam.predict(eta, P, odo)

# %% Consistency

# NIS
insideCI = (CInorm[:mk, 0] <= NISnorm[:mk]) * (NISnorm[:mk] <= CInorm[:mk, 1])

fig3, ax3 = plt.subplots(num=3, clear=True)
ax3.plot(CInorm[:mk, 0], "--")
ax3.plot(CInorm[:mk, 1], "--")
ax3.plot(NISnorm[:mk], lw=0.5)

ax3.set_title(f"NIS, {insideCI.mean()*100:.2f}% inside CI")

# %% slam

if do_raw_prediction:
    fig5, ax5 = plt.subplots(num=5, clear=True)
    ax5.scatter(
        Lo_m[timeGps < timeOdo[N - 1]],
        La_m[timeGps < timeOdo[N - 1]],
        c="r",
        marker=".",
        label="GPS",
    )
    ax5.plot(*odox[:N, :2].T, label="odom")
    ax5.grid()
    ax5.set_title("GPS vs odometry integration")
    ax5.legend()

# %%
fig6, ax6 = plt.subplots(num=6, clear=True)
ax6.scatter(*eta[3:].reshape(-1, 2).T, color="r", marker="x")
ax6.scatter(
    Lo_m[timeGps < timeOdo[N - 1]],
    La_m[timeGps < timeOdo[N - 1]],
    c="b",
    marker=".",
    label="GPS",
)
ax6.plot(*xupd[mk_first:mk, :2].T)
ax6.set(
    title=f"Steps {k}, laser scans {mk-1}, landmarks {len(eta[3:])//2},\nmeasurements {z.shape[0]}, num new = {np.sum(a[mk] == -1)}"
)


plt.show()

# %%