import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.special import comb, i0, iv
from scipy.stats import chi2, f, norm, rankdata, vonmises, wilcoxon

from .descriptive import (
    circ_dist,
    circ_kappa,
    circ_mean,
    circ_mean_and_r,
    circ_mean_ci,
    circ_median,
    circ_r,
    circ_range,
)
from .utils import A1inv, angmod, angular_distance, significance_code

###################
# One-Sample Test #
###################


@dataclass(frozen=True)
class RayleighTestResult:
    r: float  # Resultant vector length
    z: float  # Test Statistic (Rayleigh's Z)
    pval: float  # Classical P-value
    bootstrap_pval: Optional[float] = None  # Bootstrap P-value, if computed


def rayleigh_test(
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    r: Optional[float] = None,
    n: Optional[int] = None,
    B: int = 1,
    verbose: bool = False,
) -> RayleighTestResult:
    r"""
    Rayleigh's Test for Circular Uniformity.

    - H0: The data in the population are distributed uniformly around the circle.
    - H1: The data in the population are not disbutrited uniformly around the circle.

    $$ z = n \cdot r^2 $$

    and

    $$ p = \exp(\sqrt{1 + 4n + 4(n^2 - R^2)} - (1 + 2n)) $$

    This method is for ungrouped data. For testing uniformity with
    grouped data, use `chisquare_test()` or `scipy.stats.chisquare()`.

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

    B: int
        Number of bootstrap samples for p-value estimation.

    verbose: bool
        Print formatted results.

    Returns
    -------
    RayleighTestResult
        A dataclass containing:

        - r: float
            - Resultant vector length.
        - z: float
            - Test statistic (Rayleigh's Z).
        - pval: float
            - Classical p-value based on the asymptotic formula.
        - bootstrap_pval: float or None
            - Bootstrap p-value (if computed, i.e., B > 1); otherwise, None.

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
        n = np.sum(w, dtype=int)
        r = circ_r(alpha, w)

    if n is None:
        raise ValueError("Sample size `n` is missing.")

    R = n * r
    z = n * r**2  # eq(27.2)

    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))  # eq(27.4)

    if B > 1:

        tb = np.zeros(B)
        for i in range(B):
            x = np.random.normal(size=(n, 1))
            x /= np.linalg.norm(x, axis=1, keepdims=True)  # Normalize to unit sphere
            mb = np.sum(x, axis=0)
            tb[i] = np.sum(mb**2) / n

        bootstrap_pval = (np.sum(tb > z) + 1) / (B + 1)
    else:
        bootstrap_pval = None

    if verbose:
        print("Rayleigh's Test of Uniformity")
        print("-----------------------------")
        print("H0: ρ = 0")
        print("HA: ρ ≠ 0")
        print("")
        print(f"Test Statistics  (ρ | z-score): {r:.5f} | {z:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")
        if B > 1:
            print(
                f"Bootstrap P-value: {bootstrap_pval:.5f} {significance_code(bootstrap_pval)}"
            )

    return RayleighTestResult(r=r, z=z, pval=pval, bootstrap_pval=bootstrap_pval)


@dataclass(frozen=True)
class ChiSquareTestResult:
    chi2: float
    pval: float


def chisquare_test(w: np.ndarray, verbose: bool = False) -> ChiSquareTestResult:
    """Chi-Square Goodness of Fit for Circular data.

    - H0: The data in the population are distributed uniformly around the circle.
    - H1: THe data in the population are not disbutrited uniformly around the circle.

    For method is for grouped data.

    Parameters
    ----------
    w: np.ndarray
        Frequencies of angles

    verbose: bool
        Print formatted results.

    Returns
    -------
    ChiSquareTestResult
        A dataclass containing:

        - chi2: float
            - The chi-squared test statistic.
        - pval: float
            - The p-value of the test.

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
        print(f"Test Statistics (χ²): {chi2:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return ChiSquareTestResult(chi2=chi2, pval=pval)


def V_test(
    angle: Union[int, float],
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    mean: Optional[float] = None,
    r: Optional[float] = None,
    n: Optional[int] = None,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """
    Modified Rayleigh Test for Uniformity versus a Specified Angle.

    - H0: The population is uniformly distributed around the circle (i.e., H0: ρ=0)
    - H1: The population is not uniformly distributed around the circle (i.e., H1: ρ!=0),
        but has a mean of certain degree.

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
        mean, r = circ_mean_and_r(alpha, w)

    R = n * r
    V = R * np.cos(mean - angle)  # eq(27.5)
    u = V * np.sqrt(2 / n)  # eq(27.6)
    pval = 1 - norm().cdf(u)

    if verbose:
        print("Modified Rayleigh's Test of Uniformity")
        print("--------------------------------------")
        print("H0: ρ = 0")
        print("HA: ρ ≠ 0 and μ = {angle:.5f} rad")
        print("")
        print(f"Test Statistics: {V:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return V, u, pval


def one_sample_test(
    angle: Union[int, float],
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
    verbose: bool = False,
) -> bool:
    """
    To test whether the population mean angle is equal to a specified value,
    which is achieved by observing whether the angle lies within the 95% CI.

    - H0: The population has a mean of μ (μ_a = μ_0)
    - H1: The population mean is not μ (μ_a ≠ μ_0)

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

    Returns
    -------
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
) -> tuple[float, float]:
    """
    A simple alternative to the Rayleigh test, aka Hodges-Ajne test,
    which does not assume sampling from a specific distribution. This
    is called an "omnibus test" because it works well for unimodal,
    bimodal, and multimodal distributions (for ungrouped data).

    - H0: The population is uniformly distributed around the circle
    - H1: The population is not uniformly distributed.

    Parameters
    ----------
    alpha: np.array or None
        Angles in radian.

    scale: int
        Scale factor for the number of lines to be tested.

    verbose: bool
        Print formatted results.

    Returns
    -------
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

    lines_rotated = angmod((lines[:, None] - alpha)).round(5)

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
        print("H0: uniform")
        print("HA: not unifrom")
        print("")
        print(f"Test Statistics: {A:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")
    return A, pval


def batschelet_test(
    angle: Union[int, float],
    alpha: np.ndarray,
    verbose: bool = False,
) -> tuple[float, float]:
    """Modified Hodges-Ajne Test for Uniformity versus a specified Angle
    (for ungrouped data).

    - H0: The population is uniformly distributed around the circle.
    - H1: The population is not uniformly distributed around the circle, but
        is concentrated around a specified angle.

    Parameters
    ----------
    angle: np.array
        A specified angle.

    alpha: np.array or None
        Angles in radian.

    verbose: bool
        Print formatted results.

    Returns
    -------
    pval: float
        p-value

    Reference
    ---------
    P630-631, Section 27.2, Example 27.5 of Zar, 2010
    """

    from scipy.stats import binomtest

    n = len(alpha)
    angle_diff = angmod(((angle + 0.5 * np.pi) - alpha)).round(5)
    m = np.logical_and(angle_diff > 0.0, angle_diff < np.round(np.pi, 5)).sum()
    C = n - m
    pval = binomtest(C, n=n, p=0.5).pvalue

    if verbose:
        print("Batschelet Test for Uniformity")
        print("------------------------------")
        print("H0: uniform")
        print(f"HA: not unifrom but concentrated around θ = {angle:.5f} rad")
        print("")
        print(f"Test Statistics: {C}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return C, pval


def symmetry_test(
    alpha: np.ndarray,
    median: Optional[Union[int, float]] = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Non-parametric test for symmetry around the median. Works by performing a
    Wilcoxon sign rank test on the differences to the median. Also known as
    Wilcoxon paired-sample test.

    - H0: the population is symmetrical around the median
    - HA: the population is not symmetrical around the median

    Parameters
    ----------
    alpha: np.array
        Angles in radian.

    median: float or None.
        Median computed by `descriptive.median()`.

    verbose: bool
        Print formatted results.

    Returns
    -------
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


def watson_williams_test(circs: list, verbose: bool = False) -> tuple[float, float]:
    """The Watson-Williams Test for multiple samples.

    - H0: All samples are from populations with the same mean angle
    - H1: All samples are not from populations with the same mean angle

    Parameters
    ----------
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
        print("H0: all samples are from populations with the same angle.")
        print("HA: all samples are not from populations with the same angle.")
        print("")
        print(f"Test Statistics: {F:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return F, pval


def watson_u2_test(circs: list, verbose: bool = False) -> tuple[float, float]:
    """Watson's U2 Test for nonparametric two-sample testing
    (with or without ties).

    - H0: The two samples came from the same population,
        or from two populations having the same direction.
    - H1: The two samples did not come from the same population,
        or from two populations having the same directions.

    Use this instead of Watson-Williams two-sample test when at
    least one of the sampled populations is not unimodal or when
    there are other considerable departures from the assumptions
    of the latter test. It may be used on grouped data if the
    grouping interval is no greater than 5 degree.

    Parameters
    ----------
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
        print("H0: The two samples are from populations with the same angle.")
        print("HA: The two samples are not from populations with the same angle.")
        print("")
        print(f"Test Statistics: {U2:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return U2, pval


def wheeler_watson_test(circs: list, verbose: bool = False) -> tuple[float, float]:
    """The Wheeler and Watson Two/Multi-Sample Test.

    - H0: The two samples came from the same population,
        or from two populations having the same direction.
    - H1: The two samples did not come from the same population,
        or not from two populations having the same directions.

    Parameters
    ----------
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
    a, _ = np.unique(
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
        print("H0: All samples are from populations with the same angle.")
        print("HA: All samples are not from populations with the same angle.")
        print("")
        print(f"Test Statistics: {W:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return W, pval


def wallraff_test(
    circs: list, angle=float, verbose: bool = False
) -> tuple[float, float]:
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


def circ_anova_test(
    samples: list[np.ndarray], 
    method: str = "F-test", 
    kappa: Optional[float] = None, 
    f_mod: bool = True, 
    verbose: bool = False
) -> dict:
    """
    Circular Analysis of Variance (ANOVA) for multi-sample comparison of mean directions.

    - **H₀**: All groups have the same mean direction.
    - **H₁**: At least one group has a different mean direction.

    Parameters
    ----------
    samples : list of np.ndarray
        List of arrays, where each array contains circular data (angles in radians) for a group.
    method : str, optional
        The test statistic to use. Options:
        - `"F-test"` (default): High-concentration F-test (Stephens 1972).
        - `"LRT"`: Likelihood Ratio Test (Cordeiro et al. 1994).
    kappa : float, optional
        The common concentration parameter (κ). If not specified, it is estimated using MLE.
    f_mod : bool, optional
        If `True`, applies a correction factor `(1 + 3/8κ)` to the F-statistic.
    verbose : bool, optional
        If `True`, prints the test summary.

    Returns
    -------
    result : dict
        A dictionary with:
        - `'method'`: `"F-test"` or `"LRT"`
        - `'mu'`: Mean directions of each group (radians)
        - `'mu_all'`: Mean direction of all samples combined
        - `'kappa'`: Estimated concentration parameters for each group
        - `'kappa_all'`: Estimated concentration parameter for all samples combined
        - `'rho'`: Resultant vector lengths for each group
        - `'rho_all'`: Resultant vector length for all samples combined
        - `'df'`: Degrees of freedom
        - `'statistic'`: Test statistic (F-value or Chi-Square)
        - `'p_value'`: p-value
        - `'SS'`: Sum of squares (for F-test)
        - `'MS'`: Mean squares (for F-test)

    References
    ----------
    - Stephens (1972). Multi-sample tests for the von Mises distribution.
    - Cordeiro, Paula, & Botter (1994). Improved likelihood ratio tests for dispersion models.
    - Jammalamadaka & SenGupta (2001). Topics in Circular Statistics, Section 5.3.
    """

    # Number of groups
    k = len(samples)
    if k < 2:
        raise ValueError("At least two groups are required for ANOVA.")

    # Sample sizes, mean directions, and resultants
    ns = np.array([len(group) for group in samples])
    Rs = np.array([circ_r(group) * len(group) for group in samples])  # Sum of resultant vectors
    mus = np.array([circ_mean(group) for group in samples])  # Mean directions

    # Overall resultant and mean direction
    all_samples = np.hstack(samples)
    N = len(all_samples)
    R_all = circ_r(all_samples) * N
    mu_all = circ_mean(all_samples)

    # Estimate κ if not provided
    if kappa is None:
        kappa = circ_kappa(R_all / N)

    # **F-test**
    if method == "F-test":
        # Between-group and within-group sum of squares
        SS_between = np.sum(Rs) - R_all
        SS_within = N - np.sum(Rs)
        SS_total = N - R_all

        df_between = k - 1
        df_within = N - k
        df_total = N - 1

        MS_between = SS_between / df_between
        MS_within = SS_within / df_within

        # Apply correction factor (Stephens 1972)
        if f_mod:
            F_stat = (1 + 3 / (8 * kappa)) * (MS_between / MS_within)
        else:
            F_stat = MS_between / MS_within

        p_value = 1 - f.cdf(F_stat, df_between, df_within)

        result = {
            "method": "F-test",
            "mu": mus,
            "mu_all": mu_all,
            "kappa": kappa,
            "kappa_all": kappa,
            "rho": Rs,
            "rho_all": R_all,
            "df": (df_between, df_within, df_total),
            "statistic": F_stat,
            "p_value": p_value,
            "SS": (SS_between, SS_within, SS_total),
            "MS": (MS_between, MS_within)
        }

    # **Likelihood Ratio Test (LRT)**
    elif method == "LRT":
        # Compute test statistic
        term1 = 1 - (1 / (4 * kappa)) * (sum(1 / ns) - 1 / N)
        term2 = 2 * kappa * np.sum(Rs * (1 - np.cos(mus - mu_all)))
        chi_square_stat = term1 * term2

        df = k - 1
        p_value = 1 - chi2.cdf(chi_square_stat, df)

        result = {
            "method": "LRT",
            "mu": mus,
            "mu_all": mu_all,
            "kappa": kappa,
            "kappa_all": kappa,
            "rho": Rs,
            "rho_all": R_all,
            "df": df,
            "statistic": chi_square_stat,
            "p_value": p_value,
        }

    else:
        raise ValueError("Invalid method. Choose 'F-test' or 'LRT'.")

    # Print results if verbose is enabled
    if verbose:
        print("\nCircular Analysis of Variance (ANOVA)")
        print("--------------------------------------")
        print(f"Method: {result['method']}")
        print(f"Mean Directions (radians): {result['mu']}")
        print(f"Overall Mean Direction (radians): {result['mu_all']}")
        print(f"Kappa: {result['kappa']}")
        print(f"Kappa (overall): {result['kappa_all']}")
        print(f"Degrees of Freedom: {result['df']}")
        print(f"Test Statistic: {result['statistic']:.5f}")
        print(f"P-value: {result['p_value']:.5f}")
        if method == "F-test":
            print(f"Sum of Squares (Between, Within, Total): {result['SS']}")
            print(f"Mean Squares (Between, Within): {result['MS']}")
        print("--------------------------------------\n")

    return result


#####################
## Goodness-of-Fit ##
#####################


def kuiper_test(
    alpha: np.ndarray,
    n_simulation: int = 9999,
    seed: int = 2046,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Kuiper's test for Circular Uniformity.

    - H0: The data in the population are distributed uniformly around the circle.
    - H1: THe data in the population are not disbutrited uniformly around the circle.

    This method is for ungrouped data.

    Parameters
    ----------

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
    alpha: np.ndarray,
    n_simulation: int = 9999,
    seed: int = 2046,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Watson's Goodness-of-Fit Testing, aka Watson one-sample U2 test.

    - H0: The sample data come from a population distributed uniformly around the circle.
    - H1: The sample data do not come from a population distributed uniformly around the circle.

    This method is for ungrouped data.

    Parameters
    ----------

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
        # u2 = u**2
        # iu = i * u

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
) -> tuple[float, float]:
    """Simulation based Rao's spacing test.

    - H0: The sample data come from a population distributed uniformly around the circle.
    - H1: The sample data do not come from a population distributed uniformly around the circle.

    This method is for both grouped and ungrouped data.

    Parameters
    ----------
    alpha: np.ndarray
        Angles in radian.

    w: np.ndarray or None
        Frequencies

    kappa: float
        Concentration parameter. Only use for grouped data.

    n_simulation: int
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
                    angmod(
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

def circ_range_test(alpha: np.ndarray) -> tuple[float, float]:
    """
    Perform the Circular Range Test for uniformity.

    - **H0**: The data is uniformly distributed around the circle.
    - **H1**: The data is non-uniformly distributed (clustered).

    Parameters
    ----------
    alpha : np.ndarray
        Angles in radians.

    Returns
    -------
    range_stat : float
        The circular range test statistic.
    p_value : float
        The p-value indicating significance of non-uniformity.

    Reference
    ---------
    P162, Section 7.2.3 of Jammalamadaka, S. Rao and SenGupta, A. (2001)
    """
    range_stat = circ_range(alpha)  # Compute test statistic

    # Compute p-value using approximation formula from CircStats (if available)
    n = len(alpha)
    stop = int(np.floor(1 / (1 - range_stat / (2 * np.pi))))
    index = np.arange(1, stop + 1)

    # Compute p-value using series expansion
    sequence = ((-1) ** (index - 1)) * comb(n, index) * \
               (1 - index * (1 - range_stat / (2 * np.pi))) ** (n - 1)
    p_value = np.sum(sequence)

    return range_stat, p_value


def binomial_test(alpha: np.ndarray, md: float) -> float:
    """
    Perform the binomial test for the median direction of circular data.

    This test evaluates whether the population median angle is equal to a specified value.

    - **H0**: The population has median angle `md`.
    - **H1**: The population does not have median angle `md`.

    Parameters
    ----------
    alpha : np.ndarray
        Sample of angles in radians.
    md : float
        Hypothesized median angle.

    Returns
    -------
    pval : float
        p-value of the test (small values suggest rejecting H0).

    References
    ----------
    Zar, J. H. (2010). Biostatistical Analysis. Section 27.4.
    """
    from scipy.stats import binom

    alpha = np.asarray(alpha)

    if np.ndim(md) != 0:
        raise ValueError("The median (md) must be a single scalar value.")

    n = len(alpha)

    # Compute circular differences from hypothesized median
    d = circ_dist(alpha, md)

    # Count the number of angles on each side of the hypothesized median
    n1 = np.sum(d < 0)
    n2 = np.sum(d > 0)

    # Compute p-value using binomial test
    n_min = min(n1, n2)
    n_max = max(n1, n2)

    # Binomial p-value
    pval = binom.cdf(n_min, n, 0.5) + (1 - binom.cdf(n_max - 1, n, 0.5))

    return pval


def concentration_test(alpha1: np.ndarray, alpha2: np.ndarray) -> tuple[float, float]:
    """
    Parametric two-sample test for concentration equality in circular data.

    This test determines whether two von Mises-type samples have different
    concentration parameters (i.e., different dispersions).

    - **H0**: The two samples have the same concentration parameter.
    - **H1**: The two samples have different concentration parameters.

    Parameters
    ----------
    alpha1 : np.ndarray
        First sample of circular data (radians).
    alpha2 : np.ndarray
        Second sample of circular data (radians).

    Returns
    -------
    f_stat : float
        The F-statistic for the test.
    pval : float
        The p-value indicating whether the samples have significantly different concentrations.

    Notes
    -----
    - This test assumes that both samples follow von Mises distributions.
    - The **resultant vector length** of the combined samples should be greater than 0.7 for validity.
    - Based on Batschelet (1980), Section 6.9, p. 122-124.

    References
    ----------
    Batschelet, E. (1980). Circular Statistics in Biology. Academic Press.
    """
    # Ensure inputs are numpy arrays
    alpha1, alpha2 = np.asarray(alpha1), np.asarray(alpha2)
    
    # Sample sizes
    n1, n2 = len(alpha1), len(alpha2)

    # Compute resultant vector lengths
    R1 = n1 * circ_r(alpha1)
    R2 = n2 * circ_r(alpha2)

    # Compute mean resultant length of combined samples
    rbar = (R1 + R2) / (n1 + n2)

    # Warn if rbar is too low
    if rbar < 0.7:
        print("Warning: The resultant vector length should be > 0.7 for valid results.")

    # Compute F-statistic
    f_stat = ((n2 - 1) * (n1 - R1)) / ((n1 - 1) * (n2 - R2))
    
    # Compute p-value (adjusting for F-stat symmetry)
    if f_stat > 1:
        pval = 2 * (1 - f.cdf(f_stat, n1, n2))
    else:
        f_stat = 1 / f_stat
        pval = 2 * (1 - f.cdf(f_stat, n2, n1))

    return f_stat, pval


def rao_homogeneity_test(samples: list, alpha: float = 0.05) -> dict:
    """
    Perform Rao's test for homogeneity on multiple samples of angular data.

    - **Test 1**: Equality of Mean Directions (Polar Vectors)
    - **Test 2**: Equality of Dispersions

    Parameters
    ----------
    samples : list of np.ndarray
        A list where each entry is a vector of angular values (in radians).
    alpha : float, optional
        Significance level for the hypothesis test. Default is 0.05.

    Returns
    -------
    dict
        A dictionary with test statistics and p-values for both tests.

    References
    ----------
    Jammalamadaka, S. Rao and SenGupta, A. (2001). Topics in Circular Statistics, Section 7.6.1.
    Rao, J.S. (1967). Large sample tests for the homogeneity of angular data, Sankhya, Ser, B., 28.
    """
    if not isinstance(samples, list) or not all(isinstance(s, np.ndarray) for s in samples):
        raise ValueError("Input must be a list of numpy arrays.")

    k = len(samples)  # Number of samples
    n = np.array([len(s) for s in samples])  # Sample sizes

    # Compute mean cosine and sine values for each sample
    cos_means = np.array([np.mean(np.cos(s)) for s in samples])
    sin_means = np.array([np.mean(np.sin(s)) for s in samples])

    # Compute variances
    # Compute sample variances (use ddof=1 to match R)
    var_cos = np.array([np.var(np.cos(s), ddof=1) for s in samples])
    var_sin = np.array([np.var(np.sin(s), ddof=1) for s in samples])

    # Compute covariance (use ddof=1 to match R's var(x, y))
    cov_cos_sin = np.array([np.cov(np.cos(s), np.sin(s), ddof=1)[0, 1] for s in samples])

    # Compute test statistics
    s_polar = 1 / n * (var_sin / cos_means**2 + (sin_means**2 * var_cos) / cos_means**4 - (2 * sin_means * cov_cos_sin) / cos_means**3)
    tan_means = sin_means / cos_means
    H_polar = np.sum(tan_means**2 / s_polar) - (np.sum(tan_means / s_polar)**2) / np.sum(1 / s_polar)

    U = cos_means**2 + sin_means**2
    s_disp = 4 / n * (cos_means**2 * var_cos + sin_means**2 * var_sin + 2 * cos_means * sin_means * cov_cos_sin)
    H_disp = np.sum(U**2 / s_disp) - (np.sum(U / s_disp)**2) / np.sum(1 / s_disp)

    # Compute p-values
    df = k - 1  # Degrees of freedom
    pval_polar = 1 - chi2.cdf(H_polar, df)
    pval_disp = 1 - chi2.cdf(H_disp, df)

    # Determine critical values
    crit_polar = chi2.ppf(1 - alpha, df)
    crit_disp = chi2.ppf(1 - alpha, df)

    # Test decisions
    reject_polar = H_polar > crit_polar
    reject_disp = H_disp > crit_disp

    return {
        "H_polar": H_polar,
        "pval_polar": pval_polar,
        "reject_polar": reject_polar,
        "H_disp": H_disp,
        "pval_disp": pval_disp,
        "reject_disp": reject_disp,
    }


def change_point_test(alpha):
    """
    Perform a change point test for mean direction, concentration, or both.

    Parameters
    ----------
    alpha : np.ndarray
        Vector of angular measurements in radians.

    Returns
    -------
    pd.DataFrame
        DataFrame containing test statistics and estimated change point locations.

    References
    ----------
    Jammalamadaka, S. Rao and SenGupta, A. (2001). Topics in Circular Statistics, Chapter 11.

    Notes
    -----
    Ported from `change.pt()` function in the `CircStats` package for R.
    """

    def phi(x):
        """Helper function for phi computation."""
        arg = A1inv(x)
        if i0(arg) != np.inf:
            return x * A1inv(x) - np.log(i0(arg))
        else:
            return x * A1inv(x) - (arg + np.log(1 / np.sqrt(2 * np.pi * arg) * (1 + 1/(8 * arg) + 9/(128 * arg**2) + 225/(1024 * arg**3))))

    def est_rho(alpha):
        """Estimate mean resultant length (rho)."""
        return np.linalg.norm(np.sum(np.exp(1j * alpha))) / len(alpha)


    n = len(alpha)
    if n < 4:
        raise ValueError("Sample size must be at least 4 for change point test.")

    rho = est_rho(alpha)

    R1, R2, V = np.zeros(n), np.zeros(n), np.zeros(n)
    
    for k in range(1, n):  
        R1[k-1] = est_rho(alpha[:k]) * k  
        R2[k-1] = est_rho(alpha[k:]) * (n - k)

        if 2 <= k <= (n - 2): 
            V[k-1] = (k/n) * phi(R1[k-1] / k) + ((n - k) / n) * phi(R2[k-1] / (n - k))

    R1[-1] = rho * n 
    R2[-1] = 0

    R_diff = R1 + R2 - rho * n
    rmax = np.max(R_diff)
    k_r = np.argmax(R_diff)
    rave = np.mean(R_diff)

    if n > 3:
        V = V[1:n-2]
        print(f"V sliced: {V}")
        tmax = np.max(V)
        k_t = np.argmax(V) + 1
        tave = np.mean(V) 
    else:
        raise ValueError("Sample size must be at least 4.")

    return pd.DataFrame({
        "n": [n],
        "rho": [rho],
        "rmax": [rmax],
        "k.r": [k_r],
        "rave": [rave],
        "tmax": [tmax],
        "k.t": [k_t],
        "tave": [tave],
    })


def harrison_kanji_test(alpha: np.ndarray, idp: np.ndarray, idq: np.ndarray, inter: bool = True, fn: Optional[list] = None) -> tuple[tuple[float, float, float], pd.DataFrame]:
    """
    Harrison-Kanji Test (Two-Way ANOVA) for Circular Data.
    """

    if fn is None:
        fn = ['A', 'B']

    # Ensure data is in column format
    alpha = np.asarray(alpha).flatten()
    idp = np.asarray(idp).flatten()
    idq = np.asarray(idq).flatten()


    # Number of factor levels
    p = len(np.unique(idp))
    q = len(np.unique(idq))

    # Data frame for aggregation
    df = pd.DataFrame({fn[0]: idp, fn[1]: idq, 'dependent': alpha})
    n = len(df)

    # Total resultant vector length
    tr = n * circ_r(df['dependent'])
    kk = circ_kappa(tr / n)

    # Compute mean resultants per group
    gr = df.groupby(fn)
    cn = gr.count()
    cr = gr.agg(circ_r) * cn
    cn = cn.unstack(fn[1])
    cr = cr.unstack(fn[1])

    # Factor A
    gr = df.groupby(fn[0])
    pn = gr.count()['dependent']
    pr = gr.agg(circ_r)['dependent'] * pn
    pm = gr.agg(circ_mean)['dependent']

    # Factor B
    gr = df.groupby(fn[1])
    qn = gr.count()['dependent']
    qr = gr.agg(circ_r)['dependent'] * qn
    qm = gr.agg(circ_mean)['dependent']

    if kk > 2:  # Large kappa approximation
        eff_1 = sum(pr ** 2 / cn.sum(axis=1)) - tr ** 2 / n
        df_1 = p - 1
        ms_1 = eff_1 / df_1

        eff_2 = sum(qr ** 2 / cn.sum(axis=0)) - tr ** 2 / n
        df_2 = q - 1
        ms_2 = eff_2 / df_2

        eff_t = n - tr ** 2 / n
        df_t = n - 1
        m = cn.values.mean()

        if inter:
            beta = 1 / (1 - 1 / (5 * kk) - 1 / (10 * (kk ** 2)))

            eff_r = n - (cr**2./cn).values.sum()
            df_r = p * q * (m - 1)
            ms_r = eff_r / df_r

            eff_i = (cr**2./cn).values.sum() - sum(qr**2./qn) - sum(pr**2./pn) + tr**2 / n
            df_i = (p - 1) * (q - 1)
            ms_i = eff_i / df_i

            FI = ms_i / ms_r
            pI = 1 - f.cdf(FI, df_i, df_r)  # `f.cdf` is now unambiguous
        else:
            eff_r = n - sum(qr**2./qn) - sum(pr**2./pn) + tr**2 / n
            df_r = (p - 1) * (q - 1)
            ms_r = eff_r / df_r

            eff_i, df_i, ms_i, FI, pI = None, None, None, None, np.nan
            beta = 1

        F1 = beta * ms_1 / ms_r
        p1 = 1 - f.cdf(F1, df_1, df_r)

        F2 = beta * ms_2 / ms_r
        p2 = 1 - f.cdf(F2, df_2, df_r)

    else:  # Small kappa approximation
        rr = iv(1, kk) / iv(0, kk)
        kappa_factor = 2 / (1 - rr ** 2)  # Renamed `f` to `kappa_factor`

        chi1 = kappa_factor * (sum(pr**2./pn) - tr**2 / n)
        df_1 = 2 * (p - 1)
        p1 = 1 - chi2.cdf(chi1, df=df_1)

        chi2_val = kappa_factor * (sum(qr**2./qn) - tr**2 / n)
        df_2 = 2 * (q - 1)
        p2 = 1 - chi2.cdf(chi2_val, df=df_2)

        chiI = kappa_factor * ((cr**2./cn).values.sum() - sum(pr**2./pn) - sum(qr**2./qn) + tr**2 / n)
        df_i = (p - 1) * (q - 1)
        pI = chi2.sf(chiI, df=df_i)

    pval = (p1.squeeze(), p2.squeeze(), pI.squeeze())

    # Construct ANOVA Table
    if kk > 2:
        table = pd.DataFrame({
            'Source': fn + ['Interaction', 'Residual', 'Total'],
            'DoF': [df_1, df_2, df_i, df_r, df_t],
            'SS': [eff_1, eff_2, eff_i, eff_r, eff_t],
            'MS': [ms_1, ms_2, ms_i, ms_r, np.nan],
            'F': [F1.squeeze(), F2.squeeze(), FI, np.nan, np.nan],
            'p': list(pval) + [np.nan, np.nan]
        }).set_index('Source')
    else:
        table = pd.DataFrame({
            'Source': fn + ['Interaction'],
            'DoF': [df_1, df_2, df_i],
            'chi2': [chi1.squeeze(), chi2_val.squeeze(), chiI.squeeze()],
            'p': pval
        }).set_index('Source')

    return pval, table


def equal_kappa_test(samples: list[np.ndarray], verbose: bool = False) -> dict:
    """
    Test for Homogeneity of Concentration Parameters (κ) in Circular Data.

    - **H₀**: All groups have the same concentration parameter (κ).
    - **H₁**: At least one group has a different κ.

    Parameters
    ----------
    samples : list of np.ndarray
        List of circular data arrays (angles in radians) for different groups.
    verbose : bool, optional
        If `True`, prints the test summary.

    Returns
    -------
    result : dict
        A dictionary containing:
        - `'kappa'`: Estimated concentration parameters for each group.
        - `'kappa_all'`: Estimated common κ for all samples combined.
        - `'rho'`: Mean resultant lengths for each group.
        - `'rho_all'`: Mean resultant length for all samples combined.
        - `'df'`: Degrees of freedom.
        - `'statistic'`: Test statistic (Chi-Square).
        - `'p_value'`: p-value.

    Notes
    -----
    - Uses **different approximations based on mean resultant length** (`r̄`):
      - **Small `r̄` (< 0.45)**: Uses `arcsin` transformation.
      - **Moderate `r̄` (0.45 - 0.7)**: Uses `asinh` transformation.
      - **Large `r̄` (> 0.7)**: Uses Bartlett-type test (log-likelihood method).
    
    References
    ----------
    - Jammalamadaka & SenGupta (2001), Section 5.4.
    - Fisher (1993), Section 4.3.
    - `equal.kappa.test` from R's `circular` package.
    """

    # Number of groups
    k = len(samples)
    if k < 2:
        raise ValueError("At least two groups are required for the test.")

    # Sample sizes
    ns = np.array([len(group) for group in samples])

    # Mean resultant lengths
    r_bars = np.array([circ_r(group) for group in samples])
    Rs = r_bars * ns  # Unnormalized resultants

    # Overall resultant and mean resultant length
    all_samples = np.hstack(samples)
    N = len(all_samples)
    r_bar_all = circ_r(all_samples)

    # Estimate kappa values
    kappas = np.array([circ_kappa(r) for r in r_bars])
    kappa_all = circ_kappa(r_bar_all)

    # Choose test statistic based on `r̄`
    if r_bar_all < 0.45:
        # Small `r̄`: arcsin transformation
        ws = 4 * (ns - 4) / 3
        g1s = np.arcsin(np.sqrt(3/8) * 2 * r_bars)
        chi_square_stat = np.sum(ws * g1s**2) - (np.sum(ws * g1s)**2 / np.sum(ws))

    elif 0.45 <= r_bar_all <= 0.7:
        # Moderate `r̄`: asinh transformation
        ws = (ns - 3) / 0.798
        g2s = np.arcsinh((r_bars - 1.089) / 0.258)
        chi_square_stat = np.sum(ws * g2s**2) - (np.sum(ws * g2s)**2 / np.sum(ws))

    else:
        # Large `r̄`: Bartlett-type test
        vs = ns - 1
        v = N - k
        d = 1 / (3 * (k - 1)) * (np.sum(1 / vs) - 1 / v)
        chi_square_stat = (1 / (1 + d)) * (v * np.log((N - np.sum(Rs)) / v) - np.sum(vs * np.log((ns - Rs) / vs)))

    # Compute p-value
    df = k - 1
    p_value = 1 - chi2.cdf(chi_square_stat, df)

    result = {
        "kappa": kappas,
        "kappa_all": kappa_all,
        "rho": r_bars,
        "rho_all": r_bar_all,
        "df": df,
        "statistic": chi_square_stat,
        "p_value": p_value
    }

    # Print results if verbose is enabled
    if verbose:
        print("\nTest for Homogeneity of Concentration Parameters (κ)")
        print("------------------------------------------------------")
        print(f"Mean Resultant Lengths: {r_bars}")
        print(f"Overall Mean Resultant Length: {r_bar_all:.5f}")
        print(f"Estimated Kappa Values: {kappas}")
        print(f"Overall Estimated Kappa: {kappa_all:.5f}")
        print(f"Degrees of Freedom: {df}")
        print(f"Chi-Square Statistic: {chi_square_stat:.5f}")
        print(f"P-value: {p_value:.5f}")
        print("------------------------------------------------------\n")

    return result