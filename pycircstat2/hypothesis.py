from typing import Union

import numpy as np
from scipy.stats import norm, wilcoxon

from .descriptive import circ_mean, circ_mean_ci, circ_median
from .utils import angrange


def rayleigh_test(
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    r: Union[float, None] = None,
    n: Union[int, None] = None,
) -> tuple:

    """
    Rayleigh's Test for Circular Uniformity.

    For method is for ungrouped data. For testing uniformity with
    grouped data, use scipy.stats.chisquare().

    H0: The data in the population are distributed uniformly around the circle.
    H1: THe data in the population are not disbutrited uniformly around the circle.

    Parameters
    ----------

    alpha: np.array or None
        Angles in radian.

    w: np.array or None.
        Frequencies of angles

    r: float or None
        Resultant vector length from `descriptive.circ_mean()`.

    n: int or None
        Sample size

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


def chisquare_test(w: np.ndarray):

    """Chi-Square Goodness of Fit for Circular data.

    For method is for grouped data.

    H0: The data in the population are distributed uniformly around the circle.
    H1: THe data in the population are not disbutrited uniformly around the circle.

    Parameter
    ---------
    w: np.ndarray
        Frequencies of angles

    Returns
    -------
    chi2: float
        The chi-squared test statistic.
    pval: float
        The p-value of the test.

    Note
    ----
    It's a wrapper of scipy.stats.chisquare()
    """
    from scipy.stats import chisquare

    res = chisquare(w)
    chi2 = res.statistic
    pval = res.pvalue

    return chi2, pval


def V_test(
    angle: Union[int, float],
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    mean: float = None,
    r: float = None,
    n: int = None,
    unit: str = "degree",
) -> tuple:

    """
    Modified Rayleigh Test for Uniformity versus a Specified Angle.

    Parameters
    ----------
    angle: float or int
        Angle (in radian or degree) to be compared with mean angle.

    alpha: np.array or None
        Angles in radian.

    w: np.array or None.
        Frequencies of angles

    mean: float or None
        Circular mean from `descriptive.circ_mean()`. Needed if `alpha` is None.

    r: float or None
        Resultant vector length from `descriptive.circ_mean()`. Needed if `alpha` is None.

    n: int or None
        Sample size. Needed if `alpha` is None.

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
    angle: Union[int, float] = 0,
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    lb: float = None,
    ub: float = None,
    unit: str = "degree",
) -> bool:

    """
    To test wheter the population mean angle is equal to a specified value.

    Parameters
    ----------

    angle: float or int
        Angle (in radian or degree) to be compared with mean angle.

    alpha: np.array or None
        Angles in radian.

    w: np.array or None.
        Frequencies of angles

    lb: float
        Lower bound of circular mean from `descriptive.circ_mean_ci()`.

    ub: float
        Upper bound of circular mean from `descriptive.circ_mean_ci()`.

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
        lb, ub = circ_mean_ci(alpha=alpha, w=w)

    if unit == "radian":
        angle = angle
    elif unit == "degree":
        angle = np.deg2rad(angle)

    if lb < angle < ub:
        reject = False  # not able reject null (mean angle == angle)
    else:
        reject = True  # reject null (mean angle == angle)

    return reject


def omnibus_test(
    alpha: np.ndarray, precision: float = 1.0, unit: str = "degree"
) -> float:

    """
    A simple alternative to the Rayleigh test, aka Hodge-Ajne test,
    which does notassume sampling from a specific distribution. This
    is called an "omnibus test" because it works well for unimodal,
    bimodal, and multimodal distributions (for ungrouped data).

    H0: The population is uniformly distributed around the circle
    H1: The population is not uniformly distributed.

    Parameters
    ----------
    alpha: np.array or None
        Angles in radian.

    precision: float
        lines to be tested in degree.

    degree: str
        `radian` or `degree`. Default is `degree`

    Return
    ------
    pval: float
        p-value.
    """

    if unit == "radian":
        lines = np.arange(0, np.pi, precision)
    elif unit == "degree":
        lines = np.deg2rad(np.arange(0, 180.0, precision))

    n = len(alpha)

    lines_rotated = angrange((lines[:, None] - alpha)).round(5)

    # # count number of points on the right half circle, excluding the boundaries
    right = n - np.logical_and(
        lines_rotated > 0.0, lines_rotated < np.round(np.pi, 5)
    ).sum(1)
    m = np.min(right)
    pval = (
        (n - 2 * m)
        * np.math.factorial(n)
        / (np.math.factorial(m) * np.math.factorial(n - m))
        / 2 ** (n - 1)
    )

    return pval


def batschelet_test(
    angle: float,
    alpha: np.ndarray,
    unit: str = "degree",
) -> float:

    """Modified Hodges-Ajne Test for Uniformity versus a specified Angle
    (for ungrouped data).

    A nonparametric test for circular uniformity against a specified angle
    by Batschelet (1981)

    Parameters
    ----------
    angle: np.array
        A specified angle.

    alpha: np.array or None
        Angles in radian.

    Return
    ------
    pval: float
        p-value
    """

    from scipy.stats import binom_test

    if unit == "radian":
        angle = angle
    elif unit == "degree":
        angle = np.deg2rad(angle)

    n = len(alpha)
    angle_diff = angrange(((angle + 0.5 * np.pi) - alpha)).round(5)
    m = np.logical_and(angle_diff > 0.0, angle_diff < np.round(np.pi, 5)).sum()
    C = n - m

    return binom_test(C, n=n, p=0.5)


def symmetry_test(
    alpha: np.ndarray,
    median: Union[int, float, None] = None,
) -> float:

    """Non-parametric test for symmetry around the median. Works by performing a
    Wilcoxon sign rank test on the differences to the median.

    H0: the population is symmetrical around the median
    HA: the population is not symmetrical around the median

    Parameters
    ----------
    alpha: np.array
        Angles in radian.

    median: float or None.
        Median computed by `descriptive.median()`.

    Return
    ------
    pval: float
        p-value
    """

    from scipy.stats import wilcoxon

    if median is None:
        median = circ_median(alpha=alpha)

    d = (alpha - median).round(5)

    return wilcoxon(d, alternative="two-sided").pvalue
