# %%
from matplotlib import pyplot as plt
import scipy
import scipy.io
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_measurements(
        K,
        Ts,
        Xgt,
        Z
):

    fig1, ax1 = plt.subplots(num=1, clear=True)

    Z_plot_data = np.empty((0, 2), dtype=float)
    plot_measurement_distance = np.inf
    for Zk, xgtk in zip(Z, Xgt):
        to_plot = np.linalg.norm(
            Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
        Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

    ax1.scatter(*Z_plot_data.T, color="C1")
    ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
    ax1.set_title("True trajectory and the nearby measurements")

# trajectory


def plot_traj(
        Ts,
        Xgt,
        x_hat,
        Z,
        gated_list,
        posRMSE,
        velRMSE,
        prob_hat,
        peak_pos_deviation,
        peak_vel_deviation
):

    fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
    gated_list = [
        gated for sublist in gated_list for gated in sublist]
    accepted_measurements = np.array([
        measurement for measurement, gate in zip(np.vstack(Z), gated_list)
        if gate])
    refused_measurements = np.array([
        measurement for measurement, gate in zip(np.vstack(Z), gated_list)
        if not gate])
    axs3[0].scatter(*accepted_measurements.T, color='g', s=14,
                    label='accepted measurements')
    axs3[0].scatter(*refused_measurements.T, color='r', s=2,
                    label='refused measurements')
    for i in range(x_hat.shape[0]-1):
        slice_tmp = slice(i, i+2)
        axs3[0].plot(*(x_hat[slice_tmp, :2].T),
                     color=cm.cool(prob_hat[i, 0]), linewidth=3)
        axs3[1].plot(np.cumsum(Ts)[slice_tmp], prob_hat[slice_tmp, 0],
                     color=cm.cool(prob_hat[i, 0]))

    sm = plt.cm.ScalarMappable(cmap=cm.cool, norm=Normalize(vmin=0, vmax=1))
    fig3.colorbar(sm, aspect=30)

    axs3[0].plot(*Xgt.T[:2], label="$Xgt$", color=[0, 0, 0, 0.7], linewidth=1)
    axs3[0].set_title(
        f"RMSE(pos, vel) = ({posRMSE:.3f}, {velRMSE:.3f})\npeak_dev(pos, vel)"
        f"= ({peak_pos_deviation:.3f}, {peak_vel_deviation:.3f})"
    )
    axs3[0].axis("equal")
    axs3[0].legend()

    # axs3[0].legend()
    # probabilities
    axs3[1].set_ylim([0, 1])
    axs3[1].set_title('Mode Probability')
    axs3[1].set_ylabel("mode probability")
    axs3[1].set_xlabel("time")


# %% NEES
def plot_NEES_CI(
        Ts,
        NEESpos,
        ANEESpos,
        NEESvel,
        ANEESvel,
        NEES,
        ANEES,
        CI2,
        CI4,
        CI2K,
        CI4K,
        confprob):

    fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
    for ax in axs4:
        ax.set_yscale('log')
    axs4[0].plot(np.cumsum(Ts), NEESpos)
    axs4[0].plot([0, sum(Ts)], np.repeat(CI2[None], 2, 0), "--r")
    axs4[0].set_ylabel("NEES pos")
    inCIpos = np.mean((CI2[0] <= NEESpos) * (NEESpos <= CI2[1]))
    axs4[0].set_title(f"{inCIpos*100:.2f}% inside {confprob*100:.1f}% CI")

    axs4[1].plot(np.cumsum(Ts), NEESvel)
    axs4[1].plot([0, sum(Ts)], np.repeat(CI2[None], 2, 0), "--r")
    axs4[1].set_ylabel("NEES vel")
    inCIvel = np.mean((CI2[0] <= NEESvel) * (NEESvel <= CI2[1]))
    axs4[1].set_title(f"{inCIvel*100:.2f}% inside {confprob*100:.1f}% CI")

    axs4[2].plot(np.cumsum(Ts), NEES)
    axs4[2].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    axs4[2].set_ylabel("NEES")
    inCI = np.mean((CI2[0] <= NEES) * (NEES <= CI2[1]))
    axs4[2].set_title(f"{inCI*100:.2f}% inside {confprob*100:.1f}% CI")

    print(
        f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
    print(
        f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
    print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")


# %% errors
def plot_errors(
        Ts,
        Xgt,
        x_hat,
        CI2,
        CI4,
        CI2K,
        CI4K,
        confprob,
):

    fig5, axs5 = plt.subplots(2, num=5, clear=True)
    axs5[0].plot(np.cumsum(Ts), np.linalg.norm(
        x_hat[:, :2] - Xgt[:, :2], axis=1))
    axs5[0].set_ylabel("position error")

    axs5[1].plot(np.cumsum(Ts), np.linalg.norm(
        x_hat[:, 2:4] - Xgt[:, 2:4], axis=1))
    axs5[1].set_ylabel("velocity error")


def plot_NIS_NEES_model_specific(
        Ts,
        NIS_CV_list,
        NIS_CT_list,
        NEES_CV_list,
        NEES_CT_list,
        confprob):

    fig6, axs6 = plt.subplots(4, sharex=True, num=6, clear=True)
    for ax in axs6:
        ax.set_yscale('log')
    Ts_list = [np.cumsum(Ts)[k] for (k, data) in NIS_CV_list]
    NIS_data = [data for (k, data) in NIS_CV_list]

    CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

    axs6[0].plot(Ts_list, NIS_data)
    axs6[0].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    axs6[0].set_ylabel("NIS CV")
    inCIpos = np.mean((CI4[0] <= NIS_data) * (NIS_data <= CI4[1]))
    axs6[0].set_title(
        f"NIS CV, {inCIpos*100:.2f}% inside {confprob*100:.1f}% CI")

    Ts_list = [np.cumsum(Ts)[k] for (k, data) in NIS_CT_list]
    NIS_data = [data for (k, data) in NIS_CT_list]

    axs6[1].plot(Ts_list, NIS_data)
    axs6[1].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    axs6[1].set_ylabel("NIS CT")
    inCIpos = np.mean((CI4[0] <= NIS_data) * (NIS_data <= CI4[1]))
    axs6[1].set_title(
        f"NIS CT, {inCIpos*100:.2f}% inside {confprob*100:.1f}% CI")

    Ts_list = [np.cumsum(Ts)[k] for (k, data) in NEES_CV_list]
    NEES_data = [data for (k, data) in NEES_CV_list]

    axs6[2].plot(Ts_list, NEES_data)
    axs6[2].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    axs6[2].set_ylabel("NEES CV")
    inCIpos = np.mean((CI4[0] <= NEES_data) * (NEES_data <= CI4[1]))
    axs6[2].set_title(
        f"NEES CV, {inCIpos*100:.2f}% inside {confprob*100:.1f}% CI")

    Ts_list = [np.cumsum(Ts)[k] for (k, data) in NEES_CT_list]
    NEES_data = [data for (k, data) in NEES_CT_list]

    axs6[3].plot(Ts_list, NEES_data)
    axs6[3].plot([0, sum(Ts)], np.repeat(CI4[None], 2, 0), "--r")
    axs6[3].set_ylabel("NEES CT")
    inCIpos = np.mean((CI4[0] <= NEES_data) * (NEES_data <= CI4[1]))
    axs6[3].set_title(
        f"NEES CT, {inCIpos*100:.2f}% inside {confprob*100:.1f}% CI")


def get_rotation_variance(Xgt):
    vel = Xgt[:, 2:4]
    unit_vectors = vel / np.linalg.norm(vel, axis=1)[:, None]
    dot_product = np.sum(unit_vectors[1:] * unit_vectors[:-1], axis=1)
    angle = np.arccos(dot_product)
    return (f"Rotation std is "
            f"{np.std(angle[np.where(np.isfinite(angle))])/np.pi}*pi")


def get_measurements_variance(Xgt, Z, gated_list,):
    pos = Xgt[:, :2]
    errors = []
    for p, z, g in zip(pos, Z, gated_list):
        error = p[None, :] - z[g]
        errors.append(error)
    errors = np.ravel(np.vstack(errors))
    return (f"Measurement std for gated measurements is "
            f"{np.std(errors)}")


def get_acceleration_std(Xgt, Z, gated_list):
    vel = Xgt[1:, 2:4] - Xgt[:-1, 2:4]
    return (f"Acceleration std  is "
            f"{np.std(np.ravel(vel))}")
