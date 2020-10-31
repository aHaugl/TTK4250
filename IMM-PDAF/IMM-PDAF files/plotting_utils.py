import matplotlib
from matplotlib import pyplot as plt
import numpy as np


def apply_settings():
    print(f"matplotlib backend: {matplotlib.get_backend()}")
    print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
    print(f"matplotlib config dir: {matplotlib.get_configdir()}")
    plt.close("all")

    # try to set separate window ploting
    if "inline" in matplotlib.get_backend():
        print("Plotting is set to inline at the moment:", end=" ")

        if "ipykernel" in matplotlib.get_backend():
            print("backend is ipykernel (IPython?)")
            print("Trying to set backend to separate window:", end=" ")
            import IPython

            IPython.get_ipython().run_line_magic("matplotlib", "")
        else:
            print("unknown inline backend")

    print("continuing with this plotting backend", end="\n\n\n")

    # set styles
    try:
        # installed with "pip install SciencePLots"
        # (https://github.com/garrettj403/SciencePlots.git)
        # gives quite nice plots
        plt_styles = ["science", "grid", "no-latex"]
        # plt.style.use(plt_styles)
        # plt.style.use("science")
        print(f"pyplot using style set {plt_styles}")
    except Exception as e:
        print(e)
        print("setting grid and only grid and legend manually")
        plt.rcParams.update(
            {
                # setgrid
                "axes.grid": True,
                "grid.linestyle": ":",
                "grid.color": "k",
                "grid.alpha": 0.5,
                "grid.linewidth": 0.5,
                # Legend
                "legend.frameon": True,
                "legend.framealpha": 1.0,
                "legend.fancybox": True,
                "legend.numpoints": 1,
            }
        )


def plot_cov_ellipse2d(
    ax: plt.Axes,
    mean: np.ndarray = np.zeros(2),
    cov: np.ndarray = np.eye(2),
    n_sigma: float = 1,
    *,
    edgecolor: "Color" = "C0",
    facecolor: "Color" = "none",
    **kwargs,  # extra Ellipse keyword arguments
) -> matplotlib.patches.Ellipse:
    """Plot a n_sigma covariance ellipse centered in mean into ax."""
    ell_trans_mat = np.zeros((3, 3))
    ell_trans_mat[:2, :2] = np.linalg.cholesky(cov)
    ell_trans_mat[:2, 2] = mean
    ell_trans_mat[2, 2] = 1

    ell = matplotlib.patches.Ellipse(
        (0.0, 0.0),
        2.0 * n_sigma,
        2.0 * n_sigma,
        edgecolor=edgecolor,
        facecolor=facecolor,
        **kwargs,
    )
    trans = matplotlib.transforms.Affine2D(ell_trans_mat)
    ell.set_transform(trans + ax.transData)
    return ax.add_patch(ell)
