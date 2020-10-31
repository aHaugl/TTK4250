from typing import Tuple

import numpy as np


def discrete_bayes(
    # the prior: shape=(n,)
    pr: np.ndarray,
    # the conditional/likelihood: shape=(n, m)
    cond_pr: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray
]:  # the new marginal and conditional: shapes=((m,), (m, n))
    """Swap which discrete variable is the marginal and conditional."""

    joint = cond_pr * np.array(pr)[None, :]

    marginal = cond_pr @ np.array(pr)

    conditional = joint / marginal[:, None]

    # flip axes

    # DEBUG
    assert np.all(
        np.isfinite(conditional)
    ), "NaN or inf in conditional in discrete bayes"
    assert np.all(
        np.less_equal(0, conditional)
    ), "Negative values for conditional in discrete bayes"
    assert np.all(
        np.less_equal(conditional, 1)
    ), "Value more than on in discrete bayes"
    assert np.all(
        np.isfinite(marginal)), "NaN or inf in marginal in discrete bayes"

    return marginal, conditional
