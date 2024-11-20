import math
from typing import Union

import numpy as np
from scipy.stats import f, norm, rankdata, vonmises, wilcoxon

from .descriptive import circ_kappa, circ_mean, circ_mean_ci, circ_median, circ_r
from .utils import angrange, angular_distance, significance_code

#####################
## One-Sample Test ##
#####################


def rayleigh_test(
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    r: Union[float, None] = None,
    n: Union[int, None] = None,
    verbose: bool = False,
) -> tuple:
    """
    Rayleigh's Test for Circular Uniformity.

    H0: The data in the population are distributed uniformly around the circle.
    H1: THe data in the population are not disbutrited uniformly around the circle.

    This method is for ungrouped data. For testing uniformity with
    grouped data, use chisquare_test() or scipy.stats.chisquare().

    Parameters
    ----------

    alpha: np.array or None
        Angles in radian.

    w: np.array or None.
        Frequencies of angles.

    r: float or None
        Resultant vector length from `descriptive.circ_mean()`.

    n: int or None
        Sample size.

    verbose: bool
        Print formatted results.

    Returns
    -------
    z: float
        Test Statistics (Rayleigh's Z).

    p: float
        P-value.

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
        r = circ_r(alpha, w)

    if n is None:
        raise ValueError("Sample size `n` is missing.")

    R = n * r
    z = R**2 / n  # eq(27.2)
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))  # eq(27.4)

    if verbose:
        print("Rayleigh's Test of Uniformity")
        print("-----------------------------")
        print("H0: ρ = 0")
        print("HA: ρ ≠ 0")
        print("")
        print(f"Test Statistics: {z:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return z, pval


def chisquare_test(w: np.ndarray, verbose=False):
    """Chi-Square Goodness of Fit for Circular data.

    H0: The data in the population are distributed uniformly around the circle.
    H1: THe data in the population are not disbutrited uniformly around the circle.

    For method is for grouped data.

    Parameter
    ---------
    w: np.ndarray
        Frequencies of angles

    verbose: bool
        Print formatted results.

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

    if verbose:
        print("Chi-Square Test of Uniformity")
        print("-----------------------------")
        print("H0: uniform")
        print("HA: not uniform")
        print("")
        print(f"Test Statistics: {chi2:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return chi2, pval


def V_test(
    angle: Union[int, float],
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    mean: float = None,
    r: float = None,
    n: int = None,
    verbose: bool = False,
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
        Frequencies of angles.

    mean: float or None
        Circular mean from `descriptive.circ_mean()`. Needed if `alpha` is None.

    r: float or None
        Resultant vector length from `descriptive.circ_mean()`. Needed if `alpha` is None.

    n: int or None
        Sample size. Needed if `alpha` is None.

    verbose: bool
        Print formatted results.

    Returns
    -------

    V: float
        Test Statistics.
    u: float
        circular mean.
    p: float
        P-value.

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
        mean, r = circ_mean(alpha, w, return_r=True)

    R = n * r
    V = R * np.cos(mean - angle)  # eq(27.5)
    u = V * np.sqrt(2 / n)  # eq(27.6)
    pval = 1 - norm().cdf(u)

    if verbose:
        print("Modified Rayleigh's Test of Uniformity")
        print("--------------------------------------")
        print(f"H0: ρ = 0")
        print(f"HA: ρ ≠ 0 and μ = {angle:.5f} rad")
        print("")
        print(f"Test Statistics: {V:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return V, u, pval


def one_sample_test(
    angle: Union[int, float] = 0,
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    lb: float = None,
    ub: float = None,
    verbose: bool = False,
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

    verbose: bool
        Print formatted results.

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

    if verbose:
        print("One-Sample Test for the Mean Angle")
        print("----------------------------------")
        print(f"H0: μ = μ0")
        print(f"HA: μ ≠ μ0 and μ0 = {angle:.5f} rad")
        print("")
        if reject:
            print(
                f"Reject H0:\nμ0 = {angle:.5f} lies outside the 95% CI of μ ({np.array([lb, ub]).round(5)})"
            )
        else:
            print(
                f"Failed to reject H0:\nμ0 = {angle:.5f} lies within the 95% CI of μ ({np.array([lb, ub]).round(5)})"
            )

    return reject


def omnibus_test(
    alpha: np.ndarray,
    scale: int = 1,
    verbose: bool = False,
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

    verbose: bool
        Print formatted results.

    Return
    ------
    A: float
        Test statistics

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
    m = int(np.min(right))
    pval = (
        (n - 2 * m)
        * math.factorial(n)
        / (math.factorial(m) * math.factorial(n - m))
        / 2 ** (n - 1)
    )
    A = np.pi * np.sqrt(n) / (2 * (n - 2 * m))

    if verbose:
        print('Hodges-Ajne ("omnibus") Test for Uniformity')
        print("-------------------------------------------")
        print(f"H0: uniform")
        print(f"HA: not unifrom")
        print("")
        print(f"Test Statistics: {A:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")
    return A, pval


def batschelet_test(
    angle: float,
    alpha: np.ndarray,
    verbose: bool = False,
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

    verbose: bool
        Print formatted results.

    Return
    ------
    pval: float
        p-value

    Reference
    ---------
    P630-631, Section 27.2, Example 27.5 of Zar, 2010
    """

    from scipy.stats import binomtest

    n = len(alpha)
    angle_diff = angrange(((angle + 0.5 * np.pi) - alpha)).round(5)
    m = np.logical_and(angle_diff > 0.0, angle_diff < np.round(np.pi, 5)).sum()
    C = n - m
    pval = binomtest(C, n=n, p=0.5).pvalue

    if verbose:
        print("Batschelet Test for Uniformity")
        print("------------------------------")
        print(f"H0: uniform")
        print(f"HA: not unifrom but concentrated around θ = {angle:.5f} rad")
        print("")
        print(f"Test Statistics: {C}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return C, pval


def symmetry_test(
    alpha: np.ndarray,
    median: Union[int, float, None] = None,
    verbose: bool = False,
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

    verbose: bool
        Print formatted results.

    Return
    ------
    test_statistic: float
        Test statistic
    pval: float
        p-value

    Reference
    ---------
    P631-632, Section 27.3, Example 27.6 of Zar, 2010
    """

    if median is None:
        median = circ_median(alpha=alpha)

    d = (alpha - median).round(5)
    res = wilcoxon(d, alternative="two-sided")
    test_statistic = res.statistic
    pval = res.pvalue

    if verbose:
        print("Symmetry Test")
        print("------------------------------")
        print(f"H0: symmetrical around median")
        print(f"HA: not symmetrical around median")
        print("")
        print(f"Test Statistics: {test_statistic:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return test_statistic, pval


###########################
## Two/Multi-Sample Test ##
###########################


def watson_williams_test(circs: list, verbose: bool = False) -> tuple:
    """The Watson-Williams Test for multiple samples.

    H0: All samples are from populations with the same mean angle
    H1: All samples are not from populations with the same mean angle

    Parameter
    ---------
    circs: list (k, )
        A list of Circular objects.

    verbose: bool
        Print formatted results.

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
    R = N * circ_r(
        alpha=np.hstack([circ.alpha for circ in circs]),
        w=np.hstack([circ.w for circ in circs]),
    )
    F = K * (N - k) * (np.sum(Rs) - R) / (N - np.sum(Rs)) / (k - 1)
    pval = f.sf(F, k - 1, N - k)

    if verbose:
        print("The Watson-Williams Test for multiple samples")
        print("---------------------------------------------")
        print(f"H0: all samples are from populations with the same angle.")
        print(f"HA: all samples are not from populations with the same angle.")
        print("")
        print(f"Test Statistics: {F:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return F, pval


def watson_u2_test(circs: list, verbose: bool = False) -> tuple:
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

    verbose: bool
        Print formatted results.

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

    if verbose:
        print("Watson's U2 Test for two samples")
        print("---------------------------------------------")
        print(f"H0: The two samples are from populations with the same angle.")
        print(f"HA: The two samples are not from populations with the same angle.")
        print("")
        print(f"Test Statistics: {U2:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return U2, pval


def wheeler_watson_test(circs: list, verbose: bool = False):
    """The Wheeler and Watson Two/Multi-Sample Test.

    H0: The two samples came from the same population,
        or from two populations having the same direction.
    H1: The two samples did not come from the same population,
        or from two populations having the same directions.

    Parameter
    ---------
    circs: list
        A list of Circular objects.

    verbose: bool
        Print formatted results.

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

    if verbose:
        print("The Wheeler and Watson Two/Multi-Sample Test")
        print("---------------------------------------------")
        print(f"H0: All samples are from populations with the same angle.")
        print(f"HA: All samples are not from populations with the same angle.")
        print("")
        print(f"Test Statistics: {W:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return W, pval


def wallraff_test(circs: list, angle=float, verbose: bool = False):
    """Wallraff test of angular distances / dispersion against a specified angle.

    Parameters
    ----------
    circs: list
        A list of circular object

    angle: float
        A specified angle in radian.

    verbose: bool
        Print formatted results.

    Returns
    -------
    U: float
        Test Statistics

    pval: float
        P-value.

    Reference
    ---------
    P637-638, Section 27.8, Example 27.13 of Zar, 2010
    """

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

    if verbose:
        print("Wallraff test of angular distances / dispersion")
        print("-----------------------------------------------")
        print("")
        print(f"Test Statistics: {U:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return U, pval


#####################
## Goodness-of-Fit ##
#####################


def kuiper_test(
    alpha: np.ndarray, n_simulation: int = 9999, seed: int = 2046, verbose: bool = False
) -> tuple:
    """
    Kuiper's test for Circular Uniformity.

    H0: The data in the population are distributed uniformly around the circle.
    H1: THe data in the population are not disbutrited uniformly around the circle.

    This method is for ungrouped data.

    Parameter
    ---------

    alpha: np.array
        Angles in radian.

    n_simulation: int
        Number of simulation for the p-value.
        If n_simulation=1, the p-value is asymptotically approximated.
        If n_simulation>1, the p-value is simulated.
        Default is 9999.

    seed: int
        Random seed.

    Returns
    -------
    V: float
        Test Statistics
    pval: flaot
        Asymptotic p-value

    Note
    ----
    Implementation from R package `Directional`
    https://rdrr.io/cran/Directional/src/R/kuiper.R
    """

    def compute_V(alpha):
        alpha = np.sort(alpha) / (2 * np.pi)  #
        n = len(alpha)
        i = np.arange(1, n + 1)

        D_plus = np.max(i / n - alpha)
        D_minus = np.max(alpha - (i - 1) / n)
        f = np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)
        V = f * (D_plus + D_minus)
        return V, f

    n = n = len(alpha)
    Vo, f = compute_V(alpha)

    if n_simulation == 1:
        # asymptotic p-value
        m = np.arange(1, 50) ** 2
        a1 = 4 * m * Vo**2
        a2 = np.exp(-2 * m * Vo**2)
        b1 = 2 * (a1 - 1) * a2
        b2 = 8 * Vo / (3 * f) * m * (a1 - 3) * a2
        pval = np.sum(b1 - b2)
    else:
        np.random.seed(seed)
        x = np.sort(np.random.uniform(low=0, high=2 * np.pi, size=[n, n_simulation]), 0)
        Vs = np.array(([compute_V(x[:, i])[0] for i in range(n_simulation)]))
        pval = (np.sum(Vs > Vo) + 1) / (n_simulation + 1)

    if verbose:
        print("Kuiper's Test of Circular Uniformity")
        print("------------------------------------")
        print("")
        print(f"Test Statistic: {Vo:.4f}")
        print(f"P-value = {pval} {significance_code(pval)}")

    return Vo, pval


def watson_test(
    alpha: np.ndarray, n_simulation: int = 9999, seed: int = 2046, verbose: bool = False
) -> tuple:
    """
    Watson's Goodness-of-Fit Testing, aka Watson one-sample U2 test.

    H0: The sample data come from a population distributed uniformly around the circle.
    H1: The sample data do not come from a population distributed uniformly around the circle.

    This method is for ungrouped data.

    Parameter
    ---------

    alpha: np.array
        Angles in radian.

    n_simulation: int
        Number of simulation for the p-value.
        If n_simulation=1, the p-value is asymptotically approximated.
        If n_simulation>1, the p-value is simulated.

    seed: int
        Random seed.

    Returns
    -------
    U2o: float
        Test Statistics
    pval: flaot
        Asymptotic p-value

    Note
    ----
    Implementation from R package `Directional`
    https://rdrr.io/cran/Directional/src/R/watson.R

    The code for simulated p-value in Directional (v5.7) seems to be just copied from
    kuiper(), thus yield in wrong results.

    See Also
    --------
    kuiper_test(); rao_spacing_test()
    """

    def compute_U2(alpha):
        alpha = np.sort(alpha)
        n = len(alpha)
        i = np.arange(1, n + 1)

        u = alpha / 2 / np.pi
        u2 = u**2
        iu = i * u

        U2 = np.sum(((u - (i - 0.5) / n) - (np.sum(u) / n - 0.5)) ** 2) + 1 / (12 * n)
        return U2

    n = len(alpha)
    U2o = compute_U2(alpha)

    if n_simulation == 1:
        m = np.arange(1, 51)
        pval = 2 * sum((-1) ** (m - 1) * np.exp(-2 * m**2 * np.pi**2 * U2o))
    else:
        np.random.seed(seed)
        x = np.sort(np.random.uniform(low=0, high=2 * np.pi, size=[n, n_simulation]), 0)
        U2s = np.array(([compute_U2(x[:, i]) for i in range(n_simulation)]))
        pval = (np.sum(U2s > U2o) + 1) / (n_simulation + 1)

    if verbose:
        print("Watson's One-Sample U2 Test of Circular Uniformity")
        print("--------------------------------------------------")
        print("")
        print(f"Test Statistic: {U2o:.4f}")
        print(f"P-value = {pval} {significance_code(pval)}")

    return U2o, pval


def rao_spacing_test(
    alpha: np.ndarray,
    w: Union[np.ndarray, None] = None,
    kappa: float = 1000.0,
    n_simulation: int = 9999,
    seed: int = 2046,
    verbose: bool = False,
) -> tuple:
    """Simulation based Rao's spacing test.

    H0: The sample data come from a population distributed uniformly around the circle.
    H1: The sample data do not come from a population distributed uniformly around the circle.

    This method is for both grouped and ungrouped data.

    Parameters
    ----------
    alpha: np.ndarray
        Angles in radian.

    w: np.ndarray or None
        Frequencies

    kappa: float
        Concentration parameter. Only use for grouped data.

    num_sims: int
        Number of simulations.

    seed: int
        Random seed.

    Returns
    -------
    Uo: float
        Test statistics

    pval: float
        Simulation-based p-value

    Reference
    ---------
    Landler et al. (2019)
    https://movementecologyjournal.biomedcentral.com/articles/10.1186/s40462-019-0160-x
    """

    def compute_U(alpha):
        n = len(alpha)
        f = np.sort(alpha)
        T = np.hstack([f[1:] - f[:-1], 2 * np.pi - f[-1] + f[0]])
        U = 0.5 * np.sum(np.abs(T - (2 * np.pi / n)))
        return U

    if w is not None:
        n = np.sum(w)
        m = len(alpha)
        alpha = np.repeat(alpha, w)
    else:
        n = len(alpha)

    # p-value
    np.random.seed(seed)
    Uo = compute_U(alpha)
    if w is not None:  # noncontinous / grouped data
        Us = np.array(
            [
                compute_U(
                    angrange(
                        np.floor(np.random.uniform(low=0, high=2 * np.pi, size=n))
                        * m
                        / (2 * np.pi)
                        * 2
                        * np.pi
                        / m
                        + vonmises(kappa=kappa).rvs(n)
                    )
                )
                for i in range(n_simulation)
            ]
        )
    else:  # continous / ungrouped data
        Us = np.array(
            [
                compute_U(np.random.uniform(low=0, high=2 * np.pi, size=n))
                for i in range(n_simulation)
            ]
        )

    counter = np.sum(Us > Uo)
    pval = counter / (n_simulation + 1)

    if verbose:
        print("Rao's Spacing Test of Circular Uniformity")
        print("-----------------------------------------")
        print("")
        print(f"Test Statistic: {Uo:.4f}")
        print(f"P-value = {pval}\n")

    return np.rad2deg(Uo), pval
