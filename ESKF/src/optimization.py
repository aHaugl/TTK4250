import numpy as np
import scipy.stats
from eskf_runner import run_eskf
from scipy.optimize import minimize
import time
import functools


def cost_function_NIS(tunables, *args):
    tunables[7] = max(tunables[6]*(1+np.random.random()*1e-6), tunables[7])
    tunables = tunables.copy()
    eskf_parameters = tunables[:6]
    # eskf_parameters = np.append(eskf_parameters, args[0])
    eskf_parameters[-2:] = 10**eskf_parameters[-2:]
    eskf_parameters = np.abs(eskf_parameters)
    p_std = np.abs(np.repeat(tunables[6:8], [2, 1]))
    P_pred_init_list = tunables[8:]
    x_init, loaded_data, N, offset, use_GNSSaccuracy = args
    try:
        result = run_eskf(eskf_parameters, x_init, P_pred_init_list, loaded_data,
                          p_std, N, offset=offset,
                          use_GNSSaccuracy=use_GNSSaccuracy)
        # delta_x = result[3]
        x_pred = result[0]
        bias = x_pred[:, -6:]
        NIS = result[4]
        P_est = result[4]
        P_pos = P_est[:, :3]

        cost_NIS = np.mean(np.log(NIS[:, 1])**2)
        cost_covarianve_pos = 1e-4 * np.mean(np.sum(P_pos * P_pos, axis=1))
        cost_bias = 1e4 * np.mean(np.sum(bias * bias, axis=1))
        cost = cost_NIS + cost_covarianve_pos + cost_bias

        with open('optimization.txt', 'a') as file:
            text = (f'eskf_parameters: {eskf_parameters}\n'
                    f'gps_parameters: {p_std}\n'
                    f'init_P_parameters: {P_pred_init_list}\n'
                    f'{cost_NIS}, {cost_covarianve_pos}, {cost_bias}\n'
                    f'{cost}\n\n')
            file.write(text)
            print(text)
            time.sleep(0.2)
        return cost
    except Exception as e:
        print(e)
        return np.inf


def cost_function_SIM(tunables, *args):
    tunables = tunables.copy()

    eskf_parameters = tunables[:6]
    eskf_parameters[-2:] = 10**eskf_parameters[-2:]

    eskf_parameters = np.abs(eskf_parameters)
    p_std = np.abs(np.repeat(tunables[6:8], [2, 1]))
    P_pred_init_list = tunables[8:]
    x_init, loaded_data, N, offset, use_GNSSaccuracy = args
    try:
        result = run_eskf(eskf_parameters, x_init, P_pred_init_list, loaded_data,
                          p_std, N, offset=offset,
                          use_GNSSaccuracy=use_GNSSaccuracy)
        # delta_x = result[3]
        NIS = result[4]
        NEES_list = result[6:11]

        def log_error_func(x):
            return np.mean(np.log(x)**2)
        # cost = np.mean(np.log(NIS[:, 1])**2)
        delta = result[3]
        cost_delta = np.mean(np.sum(delta**2, axis=1))
        nees_all = result[5]
        cost_nees = sum([log_error_func(i) for i in NEES_list])
        cost_nis = np.mean(np.log(NIS[:, 1])**2)
        cost = 20*cost_delta + cost_nees + cost_nis

        with open('optimization.txt', 'a') as file:
            text = (f'eskf_parameters: {eskf_parameters}\n'
                    f'gps_parameters: {p_std}\n'
                    f'init_P_parameters: {P_pred_init_list}\n'
                    f'cost {cost_delta}, {cost_nees}, {cost_nis}\n'
                    f'{cost}\n\n')
            file.write(text)
            print(text)
            time.sleep(0.2)
        return cost

    except:
        return np.inf


def optimize(cost_function, eskf_parameters, p_std,
             x_pred_init, P_pred_init_list, loaded_data, N, offset,
             use_GNSSaccuracy=True):
    with open('optimization.txt', 'a') as file:
        file.write(f'\n\nNew session\n')
    eskf_parameters[-2:] = np.log10(eskf_parameters[-2:])
    tunables_init = eskf_parameters + list(p_std[1:]) + P_pred_init_list
    extra_args = (
        [x_pred_init] + [loaded_data, N, offset, use_GNSSaccuracy])
    minimize(cost_function, tunables_init, tuple(
        extra_args), options={'disp': True})
