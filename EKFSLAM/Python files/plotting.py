import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.stats import chi2
from celluloid import Camera
from tqdm import tqdm
import utils
import plott_setup

from scipy.stats import chi2


def ellipse(mu, P, s, n):
    thetas = np.linspace(0, 2*np.pi, n)
    ell = mu + s * (la.cholesky(P).T @
                    np.array([np.cos(thetas), np.sin(thetas)])).T
    return ell


def plot_trajectory(pose_est, poseGT, P_hat,  N):

    fig, axs = plt.subplots(3)
    time = np.arange(N)
    names = ['x', 'y', 'heading']
    ylabels = ['m', 'm', 'rad']

    for ax, state, GT, name, ylabel in zip(
            axs, pose_est[:, :3].T, poseGT[:, :3].T, names, ylabels):

        ax.plot(GT, c='r', label='gt')
        ax.plot(state, c='g', label='estimate')
        ax.set_title(f'Estmatated {name}')
        ax.set_ylabel(f'[{ylabel}])')
        ax.legend()


def plot_path(pose_est, poseGT, lmk_est_final, landmarks, P_hat, N):
    mins = np.amin(landmarks, axis=0)
    maxs = np.amax(landmarks, axis=0)

    ranges = maxs - mins
    offsets = ranges * 0.2

    mins -= offsets
    maxs += offsets

    fig2, ax2 = plt.subplots()
    # landmarks
    ax2.scatter(*landmarks.T, c="r", marker="^")
    ax2.scatter(*lmk_est_final.T, c="b", marker=".")
    # Draw covariance ellipsis of measurements
    for l, lmk_l in enumerate(lmk_est_final):
        idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
        rI = P_hat[N - 1][idxs, idxs]
        el = ellipse(lmk_l, rI, 2, 200)
        ax2.plot(*el.T, "b")

    ax2.plot(*poseGT.T[:2, :N], c="r", label="gt")
    ax2.plot(*pose_est.T[:2, :N], c="g", label="est")
    ax2.plot(*ellipse(pose_est[-1, :2], P_hat[N - 1][:2, :2], 1, 30).T, c="g")
    ax2.set(title="results", xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
    ax2.axis("equal")
    ax2.grid()
    ax2.legend()

# %% Consistency


def plot_NIS(NISnorm, CInorm, alpha, N):
    # NIS
    insideCI = (CInorm[:N, 0] <= NISnorm[:N]) * (NISnorm[:N] <= CInorm[:N, 1])

    fig3, ax3 = plt.subplots()
    ax3.plot(CInorm[:N, 0], '--')
    ax3.plot(CInorm[:N, 1], '--')
    ax3.plot(NISnorm[:N], lw=0.5)
    ax3.set_yscale('log')
    ax3.set_title(
        f'NIS, {insideCI[:N].mean()*100}% inside {(alpha)*100:.2f}% CI, '
        f'(ANIS={np.mean(NISnorm[:N]):.3f})')


def plot_NEES(NEESes, alpha, N):
    # NEES
    fig4, ax4 = plt.subplots(nrows=3, ncols=1, sharex=True)
    tags = ['all', 'pos', 'heading']
    dfs = [3, 2, 1]

    for ax, tag, NEES, df in zip(ax4, tags, NEESes.T, dfs):
        CI_NEES = chi2.interval(alpha, df)
        ax.plot(np.full(N, CI_NEES[0]), '--')
        ax.plot(np.full(N, CI_NEES[1]), '--')
        ax.plot(NEES[:N], lw=0.5)
        insideCI = (CI_NEES[0] <= NEES) * (NEES <= CI_NEES[1])
        ax.set_title(
            f'NEES {tag}: '
            f'{insideCI[:N].mean()*100}% inside {(alpha)*100:.2f}% CI, '
            f'(ANEES={np.mean(NEES[:N]):.3f})')
        ax.set_yscale('log')

        CI_ANEES = np.array(chi2.interval(alpha, df*N)) / N
        print(f"CI ANEES {tag}: {CI_ANEES}")
        print(f"ANEES {tag}: {NEES.mean()}")

    fig4.tight_layout()

# %% RMSE


def plot_error(pose_est, poseGT, P_hat, N):
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

    pose_err = pose_est[:N, :3] - poseGT[:N, :3]
    pose_err[:, 2] *= 180/np.pi
    ylabels = ['m', 'm', 'deg']
    tags = ['x error', 'y error', 'heading error']
    std = np.sqrt(np.vstack([P[np.diag_indices(3)] for P in P_hat[:N]]))
    std[:, 2] *= 180/np.pi
    for ax, err, std, tag, ylabel, in zip(ax, pose_err.T, std.T, tags, ylabels):
        ax.plot(err, label='error')
        ax.fill_between(
            np.arange(std.size),
            -std,
            std,
            color='g', alpha=0.2, label='estimated STD')
        ax.set_title(
            f"{tag} (RMSE={np.sqrt((err**2).mean()):.3f}{ylabel})")
        ax.set_ylabel(f"[{ylabel}]")
        ax.grid()
        ax.legend()
    fig.tight_layout()


def plot_RMSE(pose_est, poseGT, N):
    fig5, ax5 = plt.subplots(nrows=2, ncols=1, sharex=True)

    pos_err = np.linalg.norm(pose_est[:N, :2] - poseGT[:N, :2], axis=1)
    heading_err = np.abs(utils.wrapToPi(pose_est[:N, 2] - poseGT[:N, 2]))

    errs = np.vstack((pos_err, heading_err))

    ylabels = ['m', 'deg']
    scalings = np.array([1, 180/np.pi])
    tags = ['total error', 'position error', 'heading error']
    for ax, err, tag, ylabel, scaling in zip(ax5, errs, tags[1:], ylabels, scalings):
        ax.plot(err*scaling)
        ax.set_title(
            f"{tag}: RMSE {np.sqrt((err**2).mean())*scaling:.3f} {ylabel}")
        ax.set_ylabel(f"[{ylabel}]")
        ax.grid()

    fig5.tight_layout()

# %% Movie time


def play_movie(pose_est, poseGT, lmk_est, landmarks, P_hat, N):
    print("recording movie...")
    mins = np.amin(landmarks, axis=0)
    maxs = np.amax(landmarks, axis=0)

    ranges = maxs - mins
    offsets = ranges * 0.2

    mins -= offsets
    maxs += offsets

    pauseTime = 0.05
    fig_movie, ax_movie = plt.subplots()

    camera = Camera(fig_movie)

    ax_movie.grid()
    ax_movie.set(xlim=(mins[0], maxs[0]), ylim=(mins[1], maxs[1]))
    camera.snap()

    for k in tqdm(range(N)):
        ax_movie.scatter(*landmarks.T, c="r", marker="^")
        ax_movie.plot(*poseGT[:k, :2].T, "r-")
        ax_movie.plot(*pose_est[:k, :2].T, "g-")
        ax_movie.scatter(*lmk_est[k].T, c="b", marker=".")

        if k > 0:
            el = ellipse(pose_est[k, :2], P_hat[k][:2, :2], 2, 40)
            ax_movie.plot(*el.T, "g")

        numLmk = lmk_est[k].shape[0]
        for l, lmk_l in enumerate(lmk_est[k]):
            idxs = slice(3 + 2 * l, 3 + 2 * l + 2)
            rI = P_hat[k][idxs, idxs]
            el = ellipse(lmk_l, rI, 2, 200)
            ax_movie.plot(*el.T, "b")

        camera.snap()

    animation = camera.animate(interval=100, blit=True, repeat=True)
    print("playing movie")
    return animation
