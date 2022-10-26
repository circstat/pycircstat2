from typing import Union

import numpy as np
from scipy.stats import f, norm, rankdata, wilcoxon

from .descriptive import circ_kappa, circ_mean, circ_mean_ci, circ_median
from .utils import angrange, angular_distance

#####################
## One-Sample Test ##
#####################


def rayleigh_test(
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    r: Union[float, None] = None,
    n: Union[int, None] = None,
) -> tuple:

    """
    Rayleigh's Test for Circular Uniformity.

    H0: The data in the population are distributed uniformly around the circle.
    H1: THe data in the population are not disbutrited uniformly around the circle.

    For method is for ungrouped data. For testing uniformity with
    grouped data, use scipy.stats.chisquare().

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

    Reference
    ---------
    P625, Section 27.1, Example 27.1 of Zar, 2010
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

    H0: The data in the population are distributed uniformly around the circle.
    H1: THe data in the population are not disbutrited uniformly around the circle.

    For method is for grouped data.

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

    Reference
    ---------
    P662-663, Section 27.17, Example 27.23 of Zar, 2010
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
) -> tuple:

    """
    Modified Rayleigh Test for Uniformity versus a Specified Angle.

    H0: The population is uniformly distributed around the circle (i.e., H0: ρ=0)
    H1: The population is not uniformly distributed around the circle (i.e., H1: ρ!=0),
        but has a mean of 90 degree.

    Parameters
    ----------
    angle: float or int
        Angle in radian to be compared with mean angle.

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

    Returns
    -------

    V: float
        V value from modified Rayleigh's test.
    u: float
        U value from modified Rayleigh's test.
    p: float
        P value from modified Rayleigh's test.

    Reference
    ---------
    P627, Section 27.1, Example 27.2 of Zar, 2010
    """

    if mean is None or r is None or n is None:
        assert isinstance(
            alpha, np.ndarray
        ), "If `mean`, `r` or `n` is None, then `alpha` (and `w`) is needed."
        if w is None:
            w = np.ones_like(alpha)
        n = np.sum(w)
        mean, r = circ_mean(alpha, w)

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
) -> bool:

    """
    To test wheter the population mean angle is equal to a specified value,
    which is achieved by observing whether the angle lies within the 95% CI.

    H0: The population has a mean of μ
    H1: The population mean is not μ

    Parameters
    ----------

    angle: float or int
        Angle in radian to be compared with mean angle.

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

    Reference
    ---------
    P628, Section 27.1, Example 27.3 of Zar, 2010
    """

    if lb is None or ub is None:
        assert isinstance(
            alpha, np.ndarray
        ), "If `ub` or `lb` is None, then `alpha` (and `w`) is needed."
        if w is None:
            w = np.ones_like(alpha)
        lb, ub = circ_mean_ci(alpha=alpha, w=w)

    if lb < angle < ub:
        reject = False  # not able reject null (mean angle == angle)
    else:
        reject = True  # reject null (mean angle == angle)

    return reject


def omnibus_test(
    alpha: np.ndarray,
    scale: int = 1,
) -> float:

    """
    A simple alternative to the Rayleigh test, aka Hodges-Ajne test,
    which does not assume sampling from a specific distribution. This
    is called an "omnibus test" because it works well for unimodal,
    bimodal, and multimodal distributions (for ungrouped data).

    H0: The population is uniformly distributed around the circle
    H1: The population is not uniformly distributed.

    Parameters
    ----------
    alpha: np.array or None
        Angles in radian.

    scale: int
        Scale factor for the number of lines to be tested.

    Return
    ------
    pval: float
        p-value.

    Reference
    ---------
    P629-630, Section 27.2, Example 27.4 of Zar, 2010
    """

    lines = np.linspace(0, np.pi, scale * 360)
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
) -> float:

    """Modified Hodges-Ajne Test for Uniformity versus a specified Angle
    (for ungrouped data).

    H0: The population is uniformly distributed around the circle.
    H1: The population is not uniformly distributed around the circle, but
        is concentrated around a specified angle.

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

    Reference
    ---------
    P630-631, Section 27.2, Example 27.5 of Zar, 2010
    """

    from scipy.stats import binom_test

    angle = angle

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
    Wilcoxon sign rank test on the differences to the median. Also known as
    Wilcoxon paired-sample test.

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

    Reference
    ---------
    P631-632, Section 27.3, Example 27.6 of Zar, 2010
    """

    if median is None:
        median = circ_median(alpha=alpha)

    d = (alpha - median).round(5)
    pval = wilcoxon(d, alternative="two-sided").pvalue

    return pval


###########################
## Two/Multi-Sample Test ##
###########################


def watson_williams_test(circs: list) -> tuple:

    """The Watson-Williams Test for multiple samples.

    H0: All samples are from populations with the same mean angle
    H1: All samples are not from populations with the same mean angle

    Parameter
    ---------
    circs: list (k, )
        A list of Circular objects.

    Returns
    -------
    F: float
        F value

    pval: float
        p-value

    Reference
    ---------
    P632-636, Section 27.4, Example 27.7/8 of Zar, 2010
    """

    k = len(circs)
    N = np.sum([circ.n for circ in circs])
    rw = np.mean([circ.r for circ in circs])

    K = 1 + 3 / 8 / circ_kappa(rw)

    Rs = [circ.R for circ in circs]
    R = (
        N
        * circ_mean(
            alpha=np.hstack([circ.alpha for circ in circs]),
            w=np.hstack([circ.w for circ in circs]),
        )[1]
    )
    F = K * (N - k) * (np.sum(Rs) - R) / (N - np.sum(Rs)) / (k - 1)
    pval = f.sf(F, k - 1, N - k)

    return F, pval


def watson_u2_test(circs: list) -> tuple:

    """Watson's U2 Test for nonparametric two-sample testing
    (with or without ties).

    H0: The two samples came from the same population,
        or from two populations having the same direction.
    H1: The two samples did not come from the same population,
        or from two populations having the same directions.

    Use this instead of Watson-Williams two-sample test when at
    least one of the sampled populations is not unimodal or when
    there are other considerable departures from the assumptions
    of the latter test. It may be used on grouped data if the
    grouping interval is no greater than 5 degree.

    Parameter
    ---------
    circs: list
        A list of Circular objects.

    Returns
    -------
    U2: float
        U2 value
    pval: float
        p value

    Reference
    ---------
    P637-638, Section 27.5, Example 27.9 of Zar, 2010
    P639-640, Section 27.5, Example 27.10 of Zar, 2010
    """

    from scipy.stats import rankdata

    def cumfreq(alpha, circ):

        indices = np.squeeze(
            [np.where(alpha == a)[0] for a in np.repeat(circ.alpha, circ.w)]
        )
        indices = np.hstack([0, indices, len(alpha)])
        freq_cumsum = rankdata(np.repeat(circ.alpha, circ.w), method="max") / circ.n
        freq_cumsum = np.hstack([0, freq_cumsum])

        tiles = np.diff(indices)
        cf = np.repeat(freq_cumsum, tiles)

        return cf

    a, t = np.unique(
        np.hstack([np.repeat(c.alpha, c.w) for c in circs]), return_counts=True
    )
    cfs = [cumfreq(a, c) for c in circs]
    d = np.diff(cfs, axis=0)

    N = np.sum([c.n for c in circs])
    U2 = (
        np.prod([c.n for c in circs])
        / N**2
        * (np.sum(t * d**2) - np.sum(t * d) ** 2 / N)
    )
    pval = 2 * np.exp(-19.74 * U2)
    # Approximated P-value from Watson (1961)
    # https://github.com/pierremegevand/watsons_u2/blob/master/watsons_U2_approx_p.m

    return U2, pval


def wheeler_watson_test(circs):
    """The Wheeler and Watson Two/Multi-Sample Test.

    H0: The two samples came from the same population,
        or from two populations having the same direction.
    H1: The two samples did not come from the same population,
        or from two populations having the same directions.

    Parameter
    ---------
    circs: list
        A list of Circular objects.

    Returns
    -------
    W: float
        W value
    pval: float
        p value

    Reference
    ---------
    P640-642, Section 27.5, Example 27.11 of Zar, 2010

    Note
    ----
    The current implementation doesn't consider ties in the data.
    Can be improved with P144, Pewsey et al. (2013)
    """
    from scipy.stats import chi2

    def get_circrank(alpha, circ, N):

        rank_of_direction = (
            np.squeeze([np.where(alpha == a)[0] for a in np.repeat(circ.alpha, circ.w)])
            + 1
        )
        circ_rank = 2 * np.pi / N * rank_of_direction
        return circ_rank

    N = np.sum([c.n for c in circs])
    a, t = np.unique(
        np.hstack([np.repeat(c.alpha, c.w) for c in circs]), return_counts=True
    )

    circ_ranks = [get_circrank(a, c, N) for c in circs]

    k = len(circ_ranks)

    if k == 2:
        C = np.sum(np.cos(circ_ranks[0]))
        S = np.sum(np.sin(circ_ranks[0]))
        W = 2 * (N - 1) * (C**2 + S**2) / np.prod([c.n for c in circs])

    elif k > 3:
        W = 0
        for i in range(k):
            circ_rank = circ_ranks[i]
            C = np.sum(np.cos(circ_rank))
            S = np.sum(np.sin(circ_rank))
            W += (C**2 + S**2) / circs[i].n
        W *= 2

    pval = chi2.sf(W, df=2 * (k - 1))

    return W, pval


def wallraff_test(circs: list, angle=float):

    """Wallraff test of angular distances against a specified angle."""

    assert (
        len(circs) == 2
    ), "Current implementation only supports two-sample comparision."

    angles = np.ones(len(circs)) * angle if isinstance(angle, float) else angle
    ns = [c.n for c in circs]
    ad = [angular_distance(a=c.alpha, b=angles[i]) for (i, c) in enumerate(circs)]

    rs = rankdata(np.hstack(ad))

    N = np.sum(ns)

    # mann-whitney
    R1 = np.sum(rs[: ns[0]])
    U1 = np.prod(ns) + ns[0] * (ns[0] + 1) / 2 - R1
    U2 = np.prod(ns) - U1
    U = np.min([U1, U2])

    z = (U - np.prod(ns) / 2 + 0.5) / np.sqrt(np.prod(ns) * (N + 1) / 12)
    pval = 2 * norm.cdf(z)

    return U, pval
