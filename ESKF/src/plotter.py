import numpy as np
from matplotlib import pyplot as plt
from quaternion import quaternion_product, quaternion_to_euler
from cat_slice import CatSlice
import scipy.stats

POS_IDX = CatSlice(start=0, stop=3)
VEL_IDX = CatSlice(start=3, stop=6)
ATT_IDX = CatSlice(start=6, stop=10)
ACC_BIAS_IDX = CatSlice(start=10, stop=13)
GYRO_BIAS_IDX = CatSlice(start=13, stop=16)
ERR_ATT_IDX = CatSlice(start=6, stop=9)
ERR_ACC_BIAS_IDX = CatSlice(start=9, stop=12)
ERR_GYRO_BIAS_IDX = CatSlice(start=12, stop=15)


def plot_path(N, GNSSk, x_est, z_GNSS, x_true=None):
    # 3d position plot
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(1, 1, 1, projection='3d')

    ax.plot3D(x_est[:N, 1], x_est[:N, 0], -x_est[:N, 2])
    if x_true is not None:
        ax.plot3D(x_true[:N, 1], x_true[:N, 0], -x_true[:N, 2], c='k')

    ax.scatter3D(z_GNSS[:GNSSk, 1], z_GNSS[:GNSSk, 0], -z_GNSS[:GNSSk, 2])
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Altitude [m]")
    fig1.tight_layout()
    # if dosavefigures:
    #     fig1.savefig(figdir+"ned.pdf")

    # state estimation


def plot_estimate(t, N, x_est):
    fig2, axs2 = plt.subplots(5, 1, num=2, clear=True)

    eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])

    axs2[0].plot(t, x_est[:N, POS_IDX])
    axs2[0].set(ylabel="NED position [m]")
    axs2[0].legend(["North", "East", "Down"])

    axs2[1].plot(t, x_est[:N, VEL_IDX])
    axs2[1].set(ylabel="Velocities [m/s]")
    axs2[1].legend(["North", "East", "Down"])

    axs2[2].plot(t, eul[:N] * 180 / np.pi)
    axs2[2].set(ylabel="Euler angles [deg]")
    axs2[2].legend([r"$\phi$", r"$\theta$", r"$\psi$"])

    axs2[3].plot(t, x_est[:N, ACC_BIAS_IDX])
    axs2[3].set(ylabel="Accl bias [m/s^2]")
    axs2[3].legend(["$x$", "$y$", "$z$"])

    axs2[4].plot(t, x_est[:N, GYRO_BIAS_IDX] * 180 / np.pi * 3600)
    axs2[4].set(ylabel="Gyro bias [deg/h]")
    axs2[4].legend(["$x$", "$y$", "$z$"])

    fig2.suptitle("States estimates")
    fig2.tight_layout()
    # if dosavefigures:
    #     fig2.savefig(figdir+"state_estimates.pdf")


def state_error_plots(t, N, x_est, x_true, delta_x):
    if x_true is None:
        print('coud not plot error as xtrue is None')
        return
    fig3, axs3 = plt.subplots(5, 1, num=3, clear=True)
    eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])
    eul_true = np.apply_along_axis(quaternion_to_euler, 1, x_true[:N, ATT_IDX])

    # TODO use this in legends
    delta_x_RMSE = np.sqrt(np.mean(delta_x[:N] ** 2, axis=0))
    axs3[0].plot(t, delta_x[:N, POS_IDX])
    axs3[0].set(ylabel="NED position error [m]")
    axs3[0].legend(
        [
            f"North ({np.sqrt(np.mean(delta_x[:N, POS_IDX[0]]**2)):.2e})",
            f"East ({np.sqrt(np.mean(delta_x[:N, POS_IDX[1]]**2)):.2e})",
            f"Down ({np.sqrt(np.mean(delta_x[:N, POS_IDX[2]]**2)):.2e})",
        ]
    )

    axs3[1].plot(t, delta_x[:N, VEL_IDX])
    axs3[1].set(ylabel="Velocities error [m]")
    axs3[1].legend(
        [
            f"North ({np.sqrt(np.mean(delta_x[:N, VEL_IDX[0]]**2)):.2e})",
            f"East ({np.sqrt(np.mean(delta_x[:N, VEL_IDX[1]]**2)):.2e})",
            f"Down ({np.sqrt(np.mean(delta_x[:N, VEL_IDX[2]]**2)):.2e})",
        ]
    )

    # quick wrap func
    def wrap_to_pi(rads): return (rads + np.pi) % (2 * np.pi) - np.pi
    eul_error = wrap_to_pi(eul[:N] - eul_true[:N]) * 180 / np.pi
    axs3[2].plot(t, eul_error)
    axs3[2].set(ylabel="Euler angles error [deg]")
    axs3[2].legend(
        [
            rf"$\phi$ ({np.sqrt(np.mean((eul_error[:N, 0])**2)):.2e})",
            rf"$\theta$ ({np.sqrt(np.mean((eul_error[:N, 1])**2)):.2e})",
            rf"$\psi$ ({np.sqrt(np.mean((eul_error[:N, 2])**2)):.2e})",
        ]
    )

    axs3[3].plot(t, delta_x[:N, ERR_ACC_BIAS_IDX])
    axs3[3].set(ylabel="Accl bias error [m/s^2]")
    axs3[3].legend(
        [
            f"$x$ ({np.sqrt(np.mean(delta_x[:N, ERR_ACC_BIAS_IDX[0]]**2)):.2e})",
            f"$y$ ({np.sqrt(np.mean(delta_x[:N, ERR_ACC_BIAS_IDX[1]]**2)):.2e})",
            f"$z$ ({np.sqrt(np.mean(delta_x[:N, ERR_ACC_BIAS_IDX[2]]**2)):.2e})",
        ]
    )

    axs3[4].plot(t, delta_x[:N, ERR_GYRO_BIAS_IDX] * 180 / np.pi)
    axs3[4].set(ylabel="Gyro bias error [deg/s]")
    axs3[4].legend(
        [
            f"$x$ ({np.sqrt(np.mean((delta_x[:N, ERR_GYRO_BIAS_IDX[0]]* 180 / np.pi)**2)):.2e})",
            f"$y$ ({np.sqrt(np.mean((delta_x[:N, ERR_GYRO_BIAS_IDX[1]]* 180 / np.pi)**2)):.2e})",
            f"$z$ ({np.sqrt(np.mean((delta_x[:N, ERR_GYRO_BIAS_IDX[2]]* 180 / np.pi)**2)):.2e})",
        ]
    )

    fig3.suptitle("States estimate errors")
    fig3.tight_layout()
    # if dosavefigures:
    #     fig3.savefig(figdir+"state_estimate_errors.pdf")


def error_distance_plot(t, N, dt, GNSSk, x_true, delta_x, z_GNSS):
    # 3d position plot
    fig4, axs4 = plt.subplots(2, 1, num=4, clear=True)

    pos_err = np.linalg.norm(delta_x[:N, POS_IDX], axis=1)
    meas_err = np.linalg.norm(
        x_true[99:N:100, POS_IDX] - z_GNSS[:GNSSk], axis=1)
    axs4[0].plot(t, pos_err)
    axs4[0].plot(np.arange(0, N, 100) * dt, meas_err)

    axs4[0].set(ylabel="Position error [m]")
    axs4[0].legend(
        [
            f"Estimation error ({np.sqrt(np.mean(np.sum(delta_x[:N, POS_IDX]**2, axis=1)))})",
            f"Measurement error ({np.sqrt(np.mean(np.sum((x_true[99:N:100, POS_IDX] - z_GNSS[:GNSSk])**2, axis=1)))})",
        ]
    )

    axs4[1].plot(t, np.linalg.norm(delta_x[:N, VEL_IDX], axis=1))
    axs4[1].set(ylabel="Speed error [m/s]")
    axs4[1].legend(
        [f"RMSE: {np.sqrt(np.mean(np.sum(delta_x[:N, VEL_IDX]**2, axis=0)))}"])

    fig4.tight_layout()
    # if dosavefigures:
    #     fig4.savefig(figdir+"error_distance_plot.pdf")

    # %% Consistency


def plot_NEES(t, N, dt,
              NEES_all, NEES_pos, NEES_vel, NEES_att, NEES_accbias,
              NEES_gyrobias, confprob=0.95):
    fig5, axs5 = plt.subplots(6, 1, num=5, clear=True)
    for ax in axs5:
        ax.set_yscale('log')
    CI15 = np.array(scipy.stats.chi2.interval(confprob, 15)).reshape((2, 1))
    CI3 = np.array(scipy.stats.chi2.interval(confprob, 3)).reshape((2, 1))

    axs5[0].plot(t, (NEES_all[:N]).T)
    axs5[0].plot(np.array([0, N - 1]) * dt, (CI15 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI15[0] <= NEES_all[:N]) * (NEES_all[:N] <= CI15[1]))
    axs5[0].set(
        title=f"Total NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    axs5[1].plot(t, (NEES_pos[0:N]).T)
    axs5[1].plot(np.array([0, N - 1]) * dt, (CI3 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI3[0] <= NEES_pos[:N]) * (NEES_pos[:N] <= CI3[1]))
    axs5[1].set(
        title=f"Position NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    axs5[2].plot(t, (NEES_vel[0:N]).T)
    axs5[2].plot(np.array([0, N - 1]) * dt, (CI3 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI3[0] <= NEES_vel[:N]) * (NEES_vel[:N] <= CI3[1]))
    axs5[2].set(
        title=f"Velocity NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    axs5[3].plot(t, (NEES_att[0:N]).T)
    axs5[3].plot(np.array([0, N - 1]) * dt, (CI3 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI3[0] <= NEES_att[:N]) * (NEES_att[:N] <= CI3[1]))
    axs5[3].set(
        title=f"Attitude NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    axs5[4].plot(t, (NEES_accbias[0:N]).T)
    axs5[4].plot(np.array([0, N - 1]) * dt, (CI3 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI3[0] <= NEES_accbias[:N])
                       * (NEES_accbias[:N] <= CI3[1]))
    axs5[4].set(
        title=f"Accelerometer bias NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    axs5[5].plot(t, (NEES_gyrobias[0:N]).T)
    axs5[5].plot(np.array([0, N - 1]) * dt, (CI3 @ np.ones((1, 2))).T)
    insideCI = np.mean((CI3[0] <= NEES_gyrobias[:N])
                       * (NEES_gyrobias[:N] <= CI3[1]))
    axs5[5].set(
        title=f"Gyro bias NEES ({100 *  insideCI:.2f} inside {100 * confprob} confidence interval)"
    )

    fig5.tight_layout()
    # if dosavefigures:
    #     fig5.savefig(figdir+"nees_nis.pdf")

    # boxplot


def plot_NIS(
        NIS,
        confprob=0.95):

    fig, ax0 = plt.subplots(1, sharex=True, num=6, clear=True)
    ax0.set_yscale('log')
    Ts_list = NIS[:, 0]
    NIS_data = NIS[:, 1]

    CI3 = np.array(scipy.stats.chi2.interval(confprob, 3))

    ax0.plot(Ts_list, NIS_data)
    ax0.plot([0, Ts_list[-1]], np.repeat(CI3[None], 2, 0), "--r")
    ax0.set_ylabel("NIS CV")
    inCIpos = np.mean((CI3[0] <= NIS_data) * (NIS_data <= CI3[1]))
    ax0.set_title(
        f"NIS CV, {inCIpos*100:.2f}% inside {confprob*100:.1f}% CI")


def box_plot(N, GNSSk,
             NEES_all, NEES_pos, NEES_vel, NEES_att, NEES_accbias,
             NEES_gyrobias, NIS):
    fig6, axs6 = plt.subplots(1, 3)

    gauss_compare = np.sum(np.random.randn(3, GNSSk)**2, axis=0)
    axs6[0].boxplot([NIS[0:GNSSk], gauss_compare], notch=True)
    axs6[0].legend(['NIS', 'gauss'])
    plt.grid()

    gauss_compare_15 = np.sum(np.random.randn(15, N)**2, axis=0)
    axs6[1].boxplot([NEES_all[0:N].T, gauss_compare_15], notch=True)
    axs6[1].legend(['NEES', 'gauss (15 dim)'])
    plt.grid()

    gauss_compare_3 = np.sum(np.random.randn(3, N)**2, axis=0)
    axs6[2].boxplot([NEES_pos[0:N].T, NEES_vel[0:N].T, NEES_att[0:N].T,
                     NEES_accbias[0:N].T, NEES_gyrobias[0:N].T, gauss_compare_3], notch=True)
    axs6[2].legend(['NEES pos', 'NEES vel', 'NEES att',
                    'NEES accbias', 'NEES gyrobias', 'gauss (3 dim)'])
    plt.grid()

    fig6.tight_layout()
    # if dosavefigures:
    #     fig6.savefig(figdir+"boxplot.pdf")
