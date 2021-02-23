import numpy as np
from tqdm import trange
from tqdm import tqdm_notebook
from eskf import (
    ESKF,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)


def run_eskf(eskf_parameters, x_pred_init, P_pred_init_list, loaded_data,
             p_std, N, use_GNSSaccuracy=False, doGNSS=True,
             debug=False, offset=0.):

    S_a = loaded_data["S_a"]
    S_g = loaded_data["S_g"]

    # S_a = np.round(S_a)
    # S_g = np.round(S_g)

    lever_arm = loaded_data["leverarm"].ravel()
    timeGNSS = loaded_data["timeGNSS"].ravel()
    timeIMU = loaded_data["timeIMU"].ravel()
    if 'xtrue' in loaded_data:
        x_true = loaded_data["xtrue"].T
    else:
        x_true = None
    z_acceleration = loaded_data["zAcc"].T
    z_GNSS = loaded_data["zGNSS"].T
    z_gyroscope = loaded_data["zGyro"].T

    if use_GNSSaccuracy:
        GNSSaccuracy = loaded_data['GNSSaccuracy'].T
    else:
        GNSSaccuracy = None

    Ts_IMU = [0, *np.diff(timeIMU)]

    steps = len(z_acceleration)
    gnss_steps = len(z_GNSS)

    x_pred = np.zeros((steps, 16))
    x_pred[0] = x_pred_init

    P_pred_init = np.zeros(15)
    P_pred_init[POS_IDX] = P_pred_init_list[0]**2
    P_pred_init[VEL_IDX] = P_pred_init_list[1]**2
    P_pred_init[ERR_ATT_IDX] = P_pred_init_list[2]**2
    P_pred_init[ERR_ACC_BIAS_IDX] = P_pred_init_list[3]**2
    P_pred_init[ERR_GYRO_BIAS_IDX] = P_pred_init_list[4]**2
    P_pred_init = np.diag(P_pred_init)
    P_pred = np.zeros((steps, 15, 15))
    P_pred[0] = P_pred_init

    eskf = ESKF(
        *eskf_parameters,
        S_a=S_a,  # set the accelerometer correction matrix
        S_g=S_g,  # set the gyro correction matrix,
        debug=debug
    )
    R_GNSS = np.diag(p_std ** 2)

    x_est = np.zeros((N, 16))
    P_est = np.zeros((N, 15, 15))

    NIS = np.zeros(gnss_steps)
    delta_x = np.zeros((N, 15))
    NEES_all = np.zeros(N)
    NEES_pos = np.zeros(N)
    NEES_vel = np.zeros(N)
    NEES_att = np.zeros(N)
    NEES_accbias = np.zeros(N)
    NEES_gyrobias = np.zeros(N)

    # keep track of current step in GNSS measurements
    offset += timeIMU[0]
    GNSSk_init = np.searchsorted(timeGNSS, offset)
    GNSSk = GNSSk_init
    offset_idx = np.searchsorted(timeIMU, offset)
    timeIMU = timeIMU[offset_idx:]
    z_acceleration = z_acceleration[offset_idx:]
    z_gyroscope = z_gyroscope[offset_idx:]
    Ts_IMU = Ts_IMU[offset_idx:]
    k = 0
    for k in trange(N):
        if doGNSS and timeIMU[k] >= timeGNSS[GNSSk]:
            if use_GNSSaccuracy:
                R_GNSS_scaled = R_GNSS * GNSSaccuracy[GNSSk]
            else:
                R_GNSS_scaled = R_GNSS
            NIS[GNSSk] = eskf.NIS_GNSS_position(
                x_pred[k], P_pred[k], z_GNSS[GNSSk], R_GNSS_scaled, lever_arm)

            x_est[k], P_est[k] = eskf.update_GNSS_position(
                x_pred[k], P_pred[k], z_GNSS[GNSSk], R_GNSS_scaled, lever_arm)
            assert np.all(np.isfinite(P_est[k])
                          ), f"Not finite P_pred at index {k}"

            GNSSk += 1
        else:
            # no updates, so let us take estimate = prediction
            x_est[k] = x_pred[k]
            P_est[k] = P_pred[k]
        if x_true is not None:
            delta_x[k] = eskf.delta_x(x_est[k], x_true[k])
            (
                NEES_all[k],
                NEES_pos[k],
                NEES_vel[k],
                NEES_att[k],
                NEES_accbias[k],
                NEES_gyrobias[k],
            ) = eskf.NEESes(x_est[k], P_est[k], x_true[k])

        if k < N - 1:
            x_pred[k + 1], P_pred[k + 1] = eskf.predict(
                x_est[k], P_est[k], z_acceleration[k+1],
                z_gyroscope[k+1], Ts_IMU[k+1])

        if eskf.debug:
            assert np.all(np.isfinite(P_pred[k])
                          ), f"Not finite P_pred at index {k + 1}"
    result = (x_pred,
              x_est,
              P_est,
              delta_x,
              np.hstack((timeGNSS[:, None] - timeGNSS[GNSSk_init],
                         NIS[:, None]))[GNSSk_init:GNSSk],
              NEES_all,
              NEES_pos,
              NEES_vel,
              NEES_att,
              NEES_accbias,
              NEES_gyrobias,
              GNSSk)
    return result
