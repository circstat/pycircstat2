from typing import Union

import numpy as np
from scipy.stats import norm, wilcoxon

from .descriptive import circ_mean, circ_mean_ci

###########################################
# Testing Significance for the mean angle #
###########################################


def rayleigh_test(
    r: Union[float, None] = None,
    n: Union[int, None] = None,
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
) -> tuple:

    """
    Rayleigh's Test for Circular Uniformity.

    Parameters
    ----------

    r: float or None
        Resultant vector length from `descriptive.circ_mean()`.

    n: int or None
        Sample size

    alpha: np.array or None
        Angles in radian.

    w: np.array or None.
        Frequencies of angles

    Returns
    -------
    z: float
        Z-score of Rayleigh's Test. Or Rayleigh's Z.

    p: float
        P value from Rayleigh's Test.
    """

    if r is None:
        assert isinstance(
            alpha, np.ndarray
        ), "If `r` is None, then `alpha` (and `w`) is needed."
        if w is None:
            w = np.ones_like(alpha)
        n = np.sum(w)
        _, r = circ_mean(alpha, w)

    if n is None:
        raise ValueError("Sample size `n` is missing.")

    R = n * r
    z = R**2 / n  # eq(27.2)
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))  # eq(27.4)

    return z, p


def V_test(
    mean: float = None,
    r: float = None,
    n: int = None,
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    angle: Union[int, float] = 0,
    unit: str = "degree",
) -> tuple:

    """
    Modified Rayleigh Test for Uniformity versus a Specified Mean Angle.

    Parameters
    ----------
    mean: float or None
        Circular mean from `descriptive.circ_mean()`.

    r: float or None
        Resultant vector length from `descriptive.circ_mean()`.

    n: int or None
        Sample size

    alpha: np.array or None
        Angles in radian.

    w: np.array or None.
        Frequencies of angles

    angle: float or int
        Angle (in radian or degree) to be compared with mean angle.

    unit: str
        Radian or degree. Default is degree,
        which will be converted to radian.

    Returns
    -------

    V: float
        V value from modified Rayleigh's test.
    u: float
        U value from modified Rayleigh's test.
    p: float
        P value from modified Rayleigh's test.
    """

    if mean is None or r is None or n is None:
        assert isinstance(
            alpha, np.ndarray
        ), "If `mean`, `r` or `n` is None, then `alpha` (and `w`) is needed."
        if w is None:
            w = np.ones_like(alpha)
        n = np.sum(w)
        mean, r = circ_mean(alpha, w)

    if unit == "radian":
        angle = angle
    elif unit == "degree":
        angle = np.deg2rad(angle)

    R = n * r
    V = R * np.cos(mean - angle)  # eq(27.5)
    u = V * np.sqrt(2 / n)  # eq(27.6)
    p = 1 - norm().cdf(u)

    return V, u, p


def one_sample_test(
    lb: float = None,
    ub: float = None,
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    angle: Union[int, float] = 0,
    unit: str = "degree",
) -> bool:

    """
    To test wheter the population mean agle is equal to a specified value.

    Parameters
    ----------

    lb: float
        Lower bound of circular mean from `descriptive.circ_mean_ci()`.

    ub: float
        Upper bound of circular mean from `descriptive.circ_mean_ci()`.

    alpha: np.array or None
        Angles in radian.

    w: np.array or None.
        Frequencies of angles

    angle: float or int
        Angle (in radian or degree) to be compared with mean angle.

    unit: str
        Radian or degree. Default is degree,
        which will be converted to radian.

    Return
    ------
    reject: bool
        Reject or not reject the null hypothesis.
    """

    if lb is None or ub is None:
        assert isinstance(
            alpha, np.ndarray
        ), "If `ub` or `lb` is None, then `alpha` (and `w`) is needed."
        if w is None:
            w = np.ones_like(alpha)
        _, lb, ub = circ_mean_ci(alpha=alpha, w=w)

    if unit == "radian":
        angle = angle
    elif unit == "degree":
        angle = np.deg2rad(angle)

    if lb < angle < ub:
        reject = False  # not able reject null (mean angle == angle)
    else:
        reject = True  # reject null (mean angle == angle)

    return reject
