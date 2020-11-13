from matplotlib import pyplot as plt
from multiprocessing import Process, Queue
import time
from runSLAM import run_slam_simulated
import numpy as np


def get_output(func, q, *args):
    retval = func(*args)
    q.put(retval)
    return 0


def timeout(func, args, timeout):
    q = Queue()
    p = Process(target=get_output, args=[func, q, *args])
    time.sleep(0.1)
    p.start()
    start_time = time.time()
    while q.empty() and time.time() < start_time + timeout:
        time.sleep(0.01)
    data = None if q.empty() else q.get()
    p.terminate()
    return data


def cost_function(*args):
    (Q, R, JCBBalphas, eta_pred_init, P_pred_init,
        odometry, z, poseGT, N, alpha, do_asso, doAssoPlot) = args

    retval = run_slam_simulated(*args)
    (eta_pred, P_pred, eta_hat, P_hat, a, NIS, NISnorm, CI, CInorm, NEES) = retval
    posehat = np.vstack([i[:3] for i in eta_hat[:N]])
    RMSE_pos = np.linalg.norm(
        (posehat - poseGT[:N])[:, :2], axis=1)

    RMSE_hed = np.linalg.norm(
        (posehat - poseGT[:N])[:, :2], axis=1)

    return RMSE_pos


def f(a):
    return [1, 'emil', a]


if __name__ == '__main__':
    data = timeout(f, [123], 1.5)
    print(data)
