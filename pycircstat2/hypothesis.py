import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

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
    circ_pairdist,
    circ_r,
    circ_range,
)
from .utils import (
    A1inv,
    angmod,
    angular_distance,
    is_within_circular_range,
    significance_code,
)

###################
# One-Sample Test #
###################


@dataclass(frozen=True)
class TestResult:
    """Base class for hypothesis test results."""

    def asdict(self) -> dict[str, Any]:
        """Return result data as a dictionary."""
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True)
class RayleighTestResult(TestResult):
    r: float  # Resultant vector length
    z: float  # Test Statistic (Rayleigh's Z)
    pval: float  # Classical P-value
    bootstrap_pval: Optional[float] = None  # Bootstrap P-value, if computed


@dataclass(frozen=True)
class ChiSquareTestResult(TestResult):
    chi2: float
    pval: float


@dataclass(frozen=True)
class VTestResult(TestResult):
    V: float
    u: float
    pval: float


@dataclass(frozen=True)
class OneSampleTestResult(TestResult):
    reject: bool
    angle: float
    ci: tuple[float, float]


@dataclass(frozen=True)
class OmnibusTestResult(TestResult):
    A: float
    pval: float
    m: int


@dataclass(frozen=True)
class BatscheletTestResult(TestResult):
    C: int
    pval: float


@dataclass(frozen=True)
class SymmetryTestResult(TestResult):
    statistic: float
    pval: float


@dataclass(frozen=True)
class WatsonWilliamsTestResult(TestResult):
    F: float
    pval: float
    df_between: int
    df_within: int
    k: int
    N: int


@dataclass(frozen=True)
class WatsonU2TestResult(TestResult):
    U2: float
    pval: float


@dataclass(frozen=True)
class WheelerWatsonTestResult(TestResult):
    W: float
    pval: float
    df: int


@dataclass(frozen=True)
class WallraffTestResult(TestResult):
    U: float
    pval: float


@dataclass(frozen=True)
class CircularAnovaResult(TestResult):
    method: str
    mu: np.ndarray
    mu_all: float
    kappa: Union[float, np.ndarray]
    kappa_all: float
    R: np.ndarray
    R_all: float
    df: Union[int, tuple[int, int, int]]
    statistic: float
    pval: float
    SS: Optional[tuple[float, float, float]] = None
    MS: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class AngularRandomisationTestResult(TestResult):
    statistic: float
    pval: float
    n_simulation: int


@dataclass(frozen=True)
class KuiperTestResult(TestResult):
    V: float
    pval: float
    mode: str
    n_simulation: int


@dataclass(frozen=True)
class WatsonTestResult(TestResult):
    U2: float
    pval: float
    mode: str
    n_simulation: int


@dataclass(frozen=True)
class RaoSpacingTestResult(TestResult):
    statistic: float
    pval: float
    mode: str
    n_simulation: int


@dataclass(frozen=True)
class CircularRangeTestResult(TestResult):
    range_stat: float
    pval: float


@dataclass(frozen=True)
class BinomialTestResult(TestResult):
    pval: float
    n_eff: int
    n1: int
    n2: int


@dataclass(frozen=True)
class ConcentrationTestResult(TestResult):
    f_stat: float
    pval: float
    df1: int
    df2: int


@dataclass(frozen=True)
class RaoHomogeneityTestResult(TestResult):
    H_polar: float
    pval_polar: float
    reject_polar: bool
    H_disp: float
    pval_disp: float
    reject_disp: bool


@dataclass(frozen=True)
class ChangePointTestResult(TestResult):
    n: int
    rho: float
    rmax: float
    k_r: int
    rave: float
    tmax: float
    k_t: int
    tave: float


@dataclass(frozen=True)
class HarrisonKanjiTestResult(TestResult):
    p_values: tuple[Optional[float], Optional[float], Optional[float]]
    anova_table: pd.DataFrame


@dataclass(frozen=True)
class EqualKappaTestResult(TestResult):
    kappa: np.ndarray
    kappa_all: float
    rho: np.ndarray
    rho_all: float
    df: int
    statistic: float
    pval: float
    regime: str


@dataclass(frozen=True)
class CommonMedianTestResult(TestResult):
    common_median: float
    statistic: float
    pval: float
    reject: bool


@dataclass(frozen=True)
class _CircularSample:
    alpha: np.ndarray
    w: np.ndarray
    n: int
    r: float
    R: float

    def expand(self) -> np.ndarray:
        """Return expanded sample with weights applied."""
        if self.w.size == 0:
            return np.array([], dtype=float)
        return np.repeat(self.alpha, self.w)


def _coerce_circular_samples(samples: Sequence[Any]) -> list[_CircularSample]:
    """Coerce a sequence of Circular objects or arrays into unified samples."""
    if not isinstance(samples, Sequence) or len(samples) == 0:
        raise ValueError("`samples` must be a non-empty sequence.")

    try:
        from .base import Circular
    except Exception:  # pragma: no cover - defensive import guard
        Circular = None  # type: ignore

    normalized: list[_CircularSample] = []

    for sample in samples:
        if Circular is not None and isinstance(sample, Circular):  # type: ignore[arg-type]
            alpha_arr = np.asarray(sample.alpha, dtype=float)
            weights = getattr(sample, "w", None)
            if weights is None:
                weights_arr = np.ones_like(alpha_arr, dtype=int)
            else:
                weights_arr = np.asarray(weights, dtype=float)
        else:
            alpha_arr = np.asarray(sample, dtype=float)
            if alpha_arr.ndim != 1:
                raise ValueError("Each sample must be a one-dimensional array of angles.")
            weights_arr = np.ones_like(alpha_arr, dtype=float)

        if alpha_arr.size == 0:
            raise ValueError("Each sample must contain at least one observation.")
        if weights_arr.shape != alpha_arr.shape:
            raise ValueError("Weights must match the shape of the angle data.")
        if np.any(weights_arr < 0):
            raise ValueError("Weights must be non-negative.")
        if not np.all(np.isfinite(alpha_arr)):
            raise ValueError("Angles must be finite.")
        if not np.all(np.isfinite(weights_arr)):
            raise ValueError("Weights must be finite.")

        rounded_weights = np.round(weights_arr).astype(int)
        if not np.allclose(weights_arr, rounded_weights):
            raise ValueError("All weights must be integers to support grouped data.")

        n_i = int(np.sum(rounded_weights))
        if n_i <= 0:
            raise ValueError("Each sample must have a positive total weight.")

        r_i = float(circ_r(alpha_arr, rounded_weights))
        normalized.append(
            _CircularSample(
                alpha=alpha_arr,
                w=rounded_weights,
                n=n_i,
                r=r_i,
                R=n_i * r_i,
            )
        )

    return normalized


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
    SymmetryTestResult
        Dataclass containing the Wilcoxon statistic and p-value.

    Returns
    -------
    BatscheletTestResult
        Dataclass containing the count statistic `C` and the associated p-value.

    Returns
    -------
    OneSampleTestResult
        Dataclass with the rejection decision, tested angle, and confidence interval.

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

    if B <= 0:
        raise ValueError("`B` must be a positive integer.")

    if r is None:
        if alpha is None:
            raise ValueError("If `r` is None, then `alpha` (and optionally `w`) is required.")
        alpha = np.asarray(alpha, dtype=float)
        if alpha.size == 0:
            raise ValueError("`alpha` must contain at least one angle.")
        if w is None:
            w = np.ones_like(alpha, dtype=float)
        else:
            w = np.asarray(w, dtype=float)
            if w.shape != alpha.shape:
                raise ValueError("`w` must have the same shape as `alpha`.")
        n_total = float(np.sum(w))
        if n_total <= 0:
            raise ValueError("Sample size inferred from `w` must be positive.")
        if not np.isclose(n_total, round(n_total)):
            raise ValueError("Rayleigh's test requires integer sample sizes when weights are used.")
        n = int(round(n_total))
        r = circ_r(alpha, w)
    else:
        r = float(r)

    if n is None or n <= 0:
        raise ValueError("Sample size `n` must be provided and positive when `r` is given.")

    if not (0.0 <= r <= 1.0):
        raise ValueError("`r` must lie in the interval [0, 1].")

    R = n * r
    z = n * r**2  # eq(27.2)

    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))  # eq(27.4)

    bootstrap_pval: Optional[float]
    if B > 1:
        rng = np.random.default_rng()
        uniforms = rng.uniform(0.0, 2 * np.pi, size=(B, n))
        unit_vectors = np.exp(1j * uniforms)
        resultant_lengths = np.abs(np.sum(unit_vectors, axis=1))
        bootstrap_stats = (resultant_lengths**2) / n
        bootstrap_pval = float((np.count_nonzero(bootstrap_stats >= z) + 1) / (B + 1))
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
        if B > 1 and bootstrap_pval is not None:
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

    frequencies = np.asarray(w, dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("`w` must be a one-dimensional array with at least one element.")
    if np.any(frequencies < 0):
        raise ValueError("`w` must contain non-negative frequencies.")

    res = chisquare(frequencies)
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
) -> VTestResult:
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
    VTestResult
        Dataclass containing the test statistic `V`, the normalized statistic `u`,
        and the p-value.

    Reference
    ---------
    P627, Section 27.1, Example 27.2 of Zar, 2010
    """

    angle = float(angle)

    if mean is None or r is None or n is None:
        if alpha is None:
            raise ValueError("If `mean`, `r`, or `n` is None, then `alpha` (and optionally `w`) is required.")
        alpha = np.asarray(alpha, dtype=float)
        if alpha.size == 0:
            raise ValueError("`alpha` must contain at least one angle.")
        if w is None:
            w = np.ones_like(alpha, dtype=float)
        else:
            w = np.asarray(w, dtype=float)
            if w.shape != alpha.shape:
                raise ValueError("`w` must have the same shape as `alpha`.")
        n = int(np.sum(w))
        if n <= 0:
            raise ValueError("Sample size inferred from `w` must be positive.")
        mean, r = circ_mean_and_r(alpha, w)
    else:
        mean = float(mean)
        r = float(r)
        if n <= 0:
            raise ValueError("`n` must be positive.")

    if not (0.0 <= r <= 1.0):
        raise ValueError("`r` must lie in the interval [0, 1].")

    R = n * r
    V = R * np.cos(angmod(mean - angle, bounds=[-np.pi, np.pi]))  # eq(27.5)
    u = V * np.sqrt(2.0 / n)  # eq(27.6)
    pval = float(norm.sf(u))

    if verbose:
        print("Modified Rayleigh's Test of Uniformity")
        print("--------------------------------------")
        print("H0: ρ = 0")
        print(f"HA: ρ ≠ 0 and μ = {angle:.5f} rad")
        print("")
        print(f"Test Statistics: {V:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return VTestResult(V=V, u=u, pval=pval)


def one_sample_test(
    angle: Union[int, float],
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
    verbose: bool = False,
) -> OneSampleTestResult:
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

    Reference
    ---------
    P628, Section 27.1, Example 27.3 of Zar, 2010
    """

    angle = float(angle)

    if lb is None or ub is None:
        if alpha is None:
            raise ValueError("If `lb` or `ub` is None, then `alpha` (and optionally `w`) is required.")
        alpha = np.asarray(alpha, dtype=float)
        if alpha.size == 0:
            raise ValueError("`alpha` must contain at least one angle.")
        if w is None:
            w = np.ones_like(alpha, dtype=float)
        else:
            w = np.asarray(w, dtype=float)
            if w.shape != alpha.shape:
                raise ValueError("`w` must have the same shape as `alpha`.")
        lb, ub = circ_mean_ci(alpha=alpha, w=w)

    lb = float(lb)
    ub = float(ub)

    reject = not is_within_circular_range(angle, lb, ub)

    if verbose:
        print("One-Sample Test for the Mean Angle")
        print("----------------------------------")
        print("H0: μ = μ0")
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

    return OneSampleTestResult(reject=reject, angle=angle, ci=(lb, ub))


def omnibus_test(
    alpha: np.ndarray,
    scale: int = 1,
    verbose: bool = False,
) -> OmnibusTestResult:
    """
    Hodges–Ajne omnibus test for circular uniformity.

    - H0: The population is uniformly distributed around the circle
    - H1: The population is not uniformly distributed.

    This test is distribution-free and handles uni-, bi-, and multimodal
    alternatives.  The classical p-value involves factorials and
    overflows for large *n*.  We therefore compute it in log-space
    (``math.lgamma``) and exponentiate at the very end.

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
    OmnibusTestResult
        Dataclass containing the test statistic `A`, the corresponding p-value,
        and the minimum count `m`.

    Reference
    ---------
    P629-630, Section 27.2, Example 27.4 of Zar, 2010
    """

    if scale <= 0:
        raise ValueError("`scale` must be a positive integer.")

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    lines = np.linspace(0.0, np.pi, scale * 360, endpoint=False)
    n = alpha.size

    lines_rotated = angmod(lines[:, None] - alpha)

    # # count number of points on the right half circle, excluding the boundaries
    right = n - np.logical_and(
        lines_rotated > 0.0, lines_rotated < np.pi
    ).sum(axis=1)
    m = int(np.min(right))

    # ------------------------------------------------------------------
    # 2. p-value   ———  analytical formula and its log form
    # ------------------------------------------------------------------
    #     Classical (Zar 2010, eq. 27-4):
    #
    #         p  =  (n − 2m) · n! / [ m! · (n − m)! · 2^(n−1) ]            …(1)
    #       # pval = (
    #       #    (n - 2 * m)
    #       #    * math.factorial(n)
    #       #    / (math.factorial(m) * math.factorial(n - m))
    #       #    / 2 ** (n - 1)
    #       # ) # eq(27.7)

    #     Taking natural logs and using  Γ(k+1) = k!  with  log Γ = lgamma:
    #
    #         ln p  =  ln(n − 2m)
    #                 + lgamma(n + 1)
    #                 − lgamma(m + 1)
    #                 − lgamma(n − m + 1)
    #                 − (n − 1)·ln 2                                        …(2)
    #
    #     Eq. (2) is numerically safe for very large n; we exponentiate at
    #     the end, knowing the result may under-flow to 0.0 in double precision.
    # ------------------------------------------------------------------

    denom = n - 2 * m
    if denom <= 0:
        logp = -np.inf
        pval = 0.0
        A = np.inf
    else:
        logp = (
            math.log(denom)
            + math.lgamma(n + 1)
            - math.lgamma(m + 1)
            - math.lgamma(n - m + 1)
            - (n - 1) * math.log(2.0)
        )
        pval = float(np.exp(logp))
        A = np.pi * np.sqrt(n) / (2 * denom)

    if verbose:
        print('Hodges-Ajne ("omnibus") Test for Uniformity')
        print("-------------------------------------------")
        print("H0: uniform")
        print("HA: not unifrom")
        print("")
        print(f"Test Statistics: {A:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")
    return OmnibusTestResult(A=float(A), pval=float(pval), m=int(m))


def batschelet_test(
    angle: Union[int, float],
    alpha: np.ndarray,
    verbose: bool = False,
) -> BatscheletTestResult:
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

    Reference
    ---------
    P630-631, Section 27.2, Example 27.5 of Zar, 2010
    """

    from scipy.stats import binomtest

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    angle = float(angle)

    n = alpha.size
    angle_diff = angmod((angle + 0.5 * np.pi) - alpha)
    m = np.logical_and(angle_diff > 0.0, angle_diff < np.pi).sum()
    C = int(n - m)
    pval = float(binomtest(C, n=n, p=0.5).pvalue)

    if verbose:
        print("Batschelet Test for Uniformity")
        print("------------------------------")
        print("H0: uniform")
        print(f"HA: not unifrom but concentrated around θ = {angle:.5f} rad")
        print("")
        print(f"Test Statistics: {C}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return BatscheletTestResult(C=C, pval=pval)


def symmetry_test(
    alpha: np.ndarray,
    median: Optional[float] = None,
    verbose: bool = False,
) -> SymmetryTestResult:
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

    Reference
    ---------
    P631-632, Section 27.3, Example 27.6 of Zar, 2010
    """

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    if median is None:
        median = float(circ_median(alpha=alpha))
    else:
        median = float(median)

    d = angmod(alpha - median, bounds=[-np.pi, np.pi])

    res = wilcoxon(d, alternative="two-sided")
    test_statistic = float(res.statistic)
    pval = float(res.pvalue)

    if verbose:
        print("Symmetry Test")
        print("------------------------------")
        print("H0: symmetrical around median")
        print("HA: not symmetrical around median")
        print("")
        print(f"Test Statistics: {test_statistic:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return SymmetryTestResult(statistic=test_statistic, pval=pval)


###########################
## Two/Multi-Sample Test ##
###########################


def watson_williams_test(
    samples: Sequence[Any],
    verbose: bool = False,
) -> WatsonWilliamsTestResult:
    """The Watson-Williams Test for multiple samples.

    - H0: All samples are from populations with the same mean angle
    - H1: All samples are not from populations with the same mean angle

    Parameters
    ----------
    samples: sequence
        A sequence of `Circular` objects or one-dimensional array-like radian samples.

    verbose: bool
        Print formatted results.

    Returns
    -------
    WatsonWilliamsTestResult
        Dataclass containing the F statistic, p-value, and associated degrees of freedom.

    Reference
    ---------
    P632-636, Section 27.4, Example 27.7/8 of Zar, 2010
    """

    normalized = _coerce_circular_samples(samples)
    if len(normalized) < 2:
        raise ValueError("At least two samples are required for the Watson-Williams test.")

    k = len(normalized)
    N = sum(sample.n for sample in normalized)
    if N <= k:
        raise ValueError("Combined sample size must exceed the number of groups.")

    Rs = np.array([sample.R for sample in normalized], dtype=float)
    rw = float(np.sum(Rs) / N)

    kappa_hat = float(circ_kappa(rw))
    if not np.isfinite(kappa_hat):
        kappa_hat = 0.0
    if kappa_hat <= 0.0:
        K = 1.0
        warnings.warn(
            (
                "Watson-Williams test assumes common, high concentration; "
                "estimated κ≈0. Results may be unreliable."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        K = 1.0 + 3.0 / (8.0 * kappa_hat)
        if kappa_hat < 1.0:
            warnings.warn(
                (
                    "Watson-Williams test assumes common, high concentration; "
                    f"estimated κ≈{kappa_hat:.3f}. Results may be unreliable."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

    all_alpha = np.hstack([sample.alpha for sample in normalized])
    all_weights = np.hstack([sample.w for sample in normalized])
    R = N * circ_r(alpha=all_alpha, w=all_weights)
    F = K * (N - k) * (np.sum(Rs) - R) / (N - np.sum(Rs)) / (k - 1)
    df_between = k - 1
    df_within = N - k
    pval = float(f.sf(F, df_between, df_within))

    result = WatsonWilliamsTestResult(
        F=float(F),
        pval=pval,
        df_between=df_between,
        df_within=df_within,
        k=k,
        N=N,
    )

    if verbose:
        print("The Watson-Williams Test for multiple samples")
        print("---------------------------------------------")
        print("H0: all samples are from populations with the same angle.")
        print("HA: all samples are not from populations with the same angle.")
        print("")
        print(f"Test Statistics: {result.F:.5f}")
        print(f"P-value: {result.pval:.5f} {significance_code(result.pval)}")

    return result


def watson_u2_test(
    samples: Sequence[Any],
    verbose: bool = False,
) -> WatsonU2TestResult:
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
    samples: sequence
        A sequence of `Circular` objects or one-dimensional array-like radian samples.

    verbose: bool
        Print formatted results.

    Returns
    -------
    WatsonU2TestResult
        Dataclass containing the U² statistic and the associated p-value.

    Reference
    ---------
    P637-638, Section 27.5, Example 27.9 of Zar, 2010
    P639-640, Section 27.5, Example 27.10 of Zar, 2010
    """

    from scipy.stats import rankdata

    normalized = _coerce_circular_samples(samples)
    if len(normalized) != 2:
        raise ValueError("`watson_u2_test` requires exactly two samples.")

    def cumfreq(alpha_unique: np.ndarray, sample: _CircularSample) -> np.ndarray:
        expanded = sample.expand()
        if expanded.size == 0:
            raise ValueError("Each sample must contain at least one observation.")

        idx = [np.where(np.isclose(alpha_unique, val, atol=1e-10))[0] for val in expanded]
        idx = np.concatenate(idx)
        idx = np.hstack([0, idx, alpha_unique.size])

        freq_cumsum = rankdata(expanded, method="max") / sample.n
        freq_cumsum = np.hstack([0, freq_cumsum])

        tiles = np.diff(idx)
        return np.repeat(freq_cumsum, tiles)

    expanded_samples = [sample.expand() for sample in normalized]
    a, t = np.unique(np.hstack(expanded_samples), return_counts=True)
    cfs = [cumfreq(a, sample) for sample in normalized]
    d = np.diff(cfs, axis=0)

    N = sum(sample.n for sample in normalized)
    U2 = (
        np.prod([sample.n for sample in normalized])
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

    return WatsonU2TestResult(U2=float(U2), pval=float(pval))


def wheeler_watson_test(
    samples: Sequence[Any],
    verbose: bool = False,
) -> WheelerWatsonTestResult:
    """The Wheeler and Watson Two/Multi-Sample Test.

    - H0: The two samples came from the same population,
        or from two populations having the same direction.
    - H1: The two samples did not come from the same population,
        or not from two populations having the same directions.

    Parameters
    ----------
    samples: sequence
        A sequence of `Circular` objects or one-dimensional array-like radian samples.

    verbose: bool
        Print formatted results.

    Returns
    -------
    WheelerWatsonTestResult
        Dataclass containing the W statistic, degrees of freedom, and p-value.

    Reference
    ---------
    P640-642, Section 27.5, Example 27.11 of Zar, 2010

    Note
    ----
    The current implementation doesn't consider ties in the data.
    Can be improved with P144, Pewsey et al. (2013)
    """
    from scipy.stats import chi2

    normalized = _coerce_circular_samples(samples)

    def get_circrank(alpha: np.ndarray, sample: _CircularSample, N: int) -> np.ndarray:
        expanded = sample.expand()
        rank_of_direction = (
            np.squeeze([np.where(np.isclose(alpha, value))[0] for value in expanded]) + 1
        )
        return 2 * np.pi / N * rank_of_direction

    N = sum(sample.n for sample in normalized)
    expanded_samples = [sample.expand() for sample in normalized]
    a, _ = np.unique(np.hstack(expanded_samples), return_counts=True)

    circ_ranks = [get_circrank(a, sample, N) for sample in normalized]

    k = len(circ_ranks)

    if k == 2:
        C = np.sum(np.cos(circ_ranks[0]))
        S = np.sum(np.sin(circ_ranks[0]))
        W = 2 * (N - 1) * (C**2 + S**2) / np.prod([sample.n for sample in normalized])
    elif k >= 3:
        W = 0.0
        for i in range(k):
            circ_rank = circ_ranks[i]
            C = np.sum(np.cos(circ_rank))
            S = np.sum(np.sin(circ_rank))
            W += (C**2 + S**2) / normalized[i].n
        W *= 2.0
    else:
        raise ValueError("At least two samples are required for the Wheeler-Watson test.")

    df = 2 * (k - 1)
    pval = float(chi2.sf(W, df=df))

    if verbose:
        print("The Wheeler and Watson Two/Multi-Sample Test")
        print("---------------------------------------------")
        print("H0: All samples are from populations with the same angle.")
        print("HA: All samples are not from populations with the same angle.")
        print("")
        print(f"Test Statistics: {W:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return WheelerWatsonTestResult(W=float(W), pval=pval, df=df)


def wallraff_test(
    samples: Sequence[Any],
    angle: float = 0.0,
    verbose: bool = False,
) -> WallraffTestResult:
    """Wallraff test of angular distances / dispersion against a specified angle.

    Parameters
    ----------
    samples: sequence
        A sequence of `Circular` objects or one-dimensional array-like radian samples.

    angle: float
        A specified angle in radian.

    verbose: bool
        Print formatted results.

    Returns
    -------
    WallraffTestResult
        Dataclass containing the U statistic and p-value.

    Reference
    ---------
    P637-638, Section 27.8, Example 27.13 of Zar, 2010
    """

    normalized = _coerce_circular_samples(samples)

    if len(normalized) != 2:
        raise ValueError("Current implementation only supports two-sample comparison.")

    angle_arr = np.asarray(angle, dtype=float)
    if angle_arr.ndim == 0:
        angles = np.repeat(angle_arr, len(normalized))
    else:
        if angle_arr.size != len(normalized):
            raise ValueError("`angle` must be a scalar or have the same length as `samples`.")
        angles = angle_arr

    ns = [sample.n for sample in normalized]
    distances = [angular_distance(normalized[i].alpha, angles[i]) for i in range(len(normalized))]

    rs = rankdata(np.hstack(distances))

    N = np.sum(ns)

    # mann-whitney
    R1 = np.sum(rs[: ns[0]])
    U1 = np.prod(ns) + ns[0] * (ns[0] + 1) / 2 - R1
    U2 = np.prod(ns) - U1
    U = np.min([U1, U2])

    z = (U - np.prod(ns) / 2 + 0.5) / np.sqrt(np.prod(ns) * (N + 1) / 12)
    pval = float(2 * norm.sf(abs(z)))

    if verbose:
        print("Wallraff test of angular distances / dispersion")
        print("-----------------------------------------------")
        print("")
        print(f"Test Statistics: {U:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return WallraffTestResult(U=float(U), pval=pval)


def circ_anova(
    samples: list[np.ndarray],
    method: str = "F-test",
    kappa: Optional[float] = None,
    f_mod: bool = True,
    verbose: bool = False,
) -> CircularAnovaResult:
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
    result : CircularAnovaResult
        Dataclass containing the selected statistic, p-value, and supporting metrics.

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
    Rs = np.array(
        [circ_r(group) * len(group) for group in samples]
    )  # Sum of resultant vectors
    mus = np.array([circ_mean(group) for group in samples])  # Mean directions

    # Overall resultant and mean direction
    all_samples = np.hstack(samples)
    N = len(all_samples)
    R_all = circ_r(all_samples) * N
    mu_all = circ_mean(all_samples)

    # Estimate κ if not provided
    if kappa is None:
        kappa = circ_kappa(R_all / N)
    kappa_value = float(kappa)

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

        result = CircularAnovaResult(
            method="F-test",
            mu=mus,
            mu_all=float(mu_all),
            kappa=kappa_value,
            kappa_all=kappa_value,
            R=Rs,
            R_all=float(R_all),
            df=(df_between, df_within, df_total),
            statistic=float(F_stat),
            pval=float(p_value),
            SS=(float(SS_between), float(SS_within), float(SS_total)),
            MS=(float(MS_between), float(MS_within)),
        )

    # **Likelihood Ratio Test (LRT)**
    elif method == "LRT":
        # Compute test statistic
        term1 = 1 - (1 / (4 * kappa_value)) * (sum(1 / ns) - 1 / N)
        term2 = 2 * kappa_value * np.sum(Rs * (1 - np.cos(mus - mu_all)))
        chi_square_stat = term1 * term2

        df = k - 1
        p_value = 1 - chi2.cdf(chi_square_stat, df)

        result = CircularAnovaResult(
            method="LRT",
            mu=mus,
            mu_all=float(mu_all),
            kappa=kappa_value,
            kappa_all=kappa_value,
            R=Rs,
            R_all=float(R_all),
            df=int(df),
            statistic=float(chi_square_stat),
            pval=float(p_value),
        )

    else:
        raise ValueError("Invalid method. Choose 'F-test' or 'LRT'.")

    # Print results if verbose is enabled
    if verbose:
        print("\nCircular Analysis of Variance (ANOVA)")
        print("--------------------------------------")
        print(f"Method: {result.method}")
        print(f"Mean Directions (radians): {result.mu}")
        print(f"Overall Mean Direction (radians): {result.mu_all}")
        print(f"Kappa: {result.kappa}")
        print(f"Kappa (overall): {result.kappa_all}")
        print(f"Degrees of Freedom: {result.df}")
        print(f"Test Statistic: {result.statistic:.5f}")
        print(f"P-value: {result.pval:.5f}")
        if method == "F-test":
            print(f"Sum of Squares (Between, Within, Total): {result.SS}")
            print(f"Mean Squares (Between, Within): {result.MS}")
        print("--------------------------------------\n")

    return result


def angular_randomisation_test(
    samples: Sequence[Any],
    n_simulation: int = 1000,
    verbose: bool = False,
) -> AngularRandomisationTestResult:
    """The Angular Randomization Test (ART) for homogeneity.

    - H0: The two samples come from the same population.
    - H1: The two samples do not come from the same population.

    Parameters
    ----------
    samples: sequence
        A sequence of `Circular` objects or one-dimensional array-like radian samples.
    n_simulation: int, optional
        Number of permutations for the test. Defaults to 1000.

    Returns
    -------
    AngularRandomisationTestResult
        Dataclass containing the observed statistic and permutation p-value.

    Reference
    ---------
    Jebur, A. J., & Abushilah, S. F. (2022).
    Distribution-free two-sample homogeneity test for circular data based on geodesic distance.
    International Journal of Nonlinear Analysis and Applications, 13(1), 2703-2711.
    """

    normalized = _coerce_circular_samples(samples)

    if len(normalized) != 2:
        raise ValueError("The Angular Randomization Test requires exactly two samples.")
    if n_simulation <= 0:
        raise ValueError("`n_simulation` must be a positive integer.")

    sample_arrays = [np.asarray(sample.alpha, dtype=float) for sample in normalized]
    if any(arr.size == 0 for arr in sample_arrays):
        raise ValueError("Each sample must contain at least one observation.")

    def art_statistic(S1: np.ndarray, S2: np.ndarray) -> float:
        """
        Compute the Angular Randomisation Test (ART) statistic for two groups of circular data.
        Following equations (3.1) and (4.2) from Jebur & Abushilah (2022) .

        Args:
            S1 (np.ndarray): First group of angles in radians (φ values)
            S2 (np.ndarray): Second group of angles in radians (ψ values)

        Returns:
            float: The ART test statistic
        """
        n = len(S1)
        m = len(S2)

        # Compute the scaling factor ((n+m)/(nm))^(-1/2)
        scaling_factor = np.sqrt(n * m / (n + m))

        # Compute sum of all pairwise geodesic distances
        total_distance = circ_pairdist(S1, S2, metric="geodesic", return_sum=True)

        # Scale the total distance and return
        return scaling_factor * total_distance

    # 1. Compute observed test statistic T*₀
    observed_stat = art_statistic(sample_arrays[0], sample_arrays[1])

    # Initialize counter for permutations more extreme than observed
    n_extreme = 1  # Start at 1 to count the observed statistic

    # Combine samples for permutation
    combined_data = np.concatenate(sample_arrays)
    n1 = sample_arrays[0].size

    # Perform permutation test
    rng = np.random.default_rng()

    for _ in range(n_simulation):
        # Randomly permute the combined data
        permuted_data = rng.permutation(combined_data)

        # Split into two groups of original sizes
        perm_S1 = permuted_data[:n1]
        perm_S2 = permuted_data[n1:]

        # Compute test statistic for this permutation
        perm_stat = art_statistic(perm_S1, perm_S2)

        # Count if permuted statistic is >= observed (one-sided test)
        if perm_stat >= observed_stat:
            n_extreme += 1

    # Compute p-value as in equation (4.3)
    p_value = n_extreme / (n_simulation + 1)

    if verbose:
        print("Angular Randomization Test (ART) for Homogeneity")
        print("-------------------------------------------------")
        print("H0: The two samples come from the same population.")
        print("HA: The two samples do not come from the same population.")
        print("")
        print(f"Observed Test Statistic: {observed_stat:.5f}")
        print(f"P-value: {p_value:.5f} {significance_code(p_value)}")

    return AngularRandomisationTestResult(statistic=float(observed_stat), pval=float(p_value), n_simulation=n_simulation)


#####################
## Goodness-of-Fit ##
#####################


def kuiper_test(
    alpha: np.ndarray,
    n_simulation: int = 9999,
    seed: int = 2046,
    verbose: bool = False,
) -> KuiperTestResult:
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
    KuiperTestResult
        Dataclass containing the Kuiper statistic, p-value, simulation mode, and count.

    Note
    ----
    Implementation from R package `Directional`
    https://rdrr.io/cran/Directional/src/R/kuiper.R
    """

    if n_simulation <= 0:
        raise ValueError("`n_simulation` must be a positive integer.")

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    def compute_V(sample):
        ordered = np.sort(sample) / (2 * np.pi)
        n = ordered.size
        indices = np.arange(1, n + 1, dtype=float)

        D_plus = np.max(indices / n - ordered)
        D_minus = np.max(ordered - (indices - 1) / n)
        f = np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n)
        V = f * (D_plus + D_minus)
        return float(V), float(f)

    n = alpha.size
    Vo, f = compute_V(alpha)

    if n_simulation == 1:
        # asymptotic p-value
        mode = "asymptotic"
        m = (np.arange(1, 50, dtype=float)) ** 2
        a1 = 4 * m * Vo**2
        a2 = np.exp(-2 * m * Vo**2)
        b1 = 2 * (a1 - 1) * a2
        b2 = 8 * Vo / (3 * f) * m * (a1 - 3) * a2
        pval = float(np.sum(b1 - b2))
    else:
        mode = "simulation"
        rng = np.random.default_rng(seed)
        uniforms = rng.uniform(low=0.0, high=2 * np.pi, size=(n, n_simulation))
        x = np.sort(uniforms, axis=0)
        Vs = np.array([compute_V(x[:, i])[0] for i in range(n_simulation)])
        pval = float((np.count_nonzero(Vs >= Vo) + 1) / (n_simulation + 1))

    if verbose:
        print("Kuiper's Test of Circular Uniformity")
        print("------------------------------------")
        print("")
        print(f"Test Statistic: {Vo:.4f}")
        print(f"P-value = {pval} {significance_code(pval)}")

    return KuiperTestResult(V=float(Vo), pval=float(pval), mode=mode, n_simulation=n_simulation)


def watson_test(
    alpha: np.ndarray,
    n_simulation: int = 9999,
    seed: int = 2046,
    verbose: bool = False,
) -> WatsonTestResult:
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
    WatsonTestResult
        Dataclass containing the Watson U² statistic, p-value, and simulation details.

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

    if n_simulation <= 0:
        raise ValueError("`n_simulation` must be a positive integer.")

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    def compute_U2(sample):
        ordered = np.sort(sample)
        n = ordered.size
        indices = np.arange(1, n + 1, dtype=float)

        u = ordered / (2 * np.pi)
        U2 = np.sum(((u - (indices - 0.5) / n) - (np.sum(u) / n - 0.5)) ** 2) + 1 / (12 * n)
        return float(U2)

    n = alpha.size
    U2o = compute_U2(alpha)

    if n_simulation == 1:
        mode = "asymptotic"
        m = np.arange(1, 51)
        pval = float(2 * sum((-1) ** (m - 1) * np.exp(-2 * m**2 * np.pi**2 * U2o)))
    else:
        mode = "simulation"
        rng = np.random.default_rng(seed)
        uniforms = rng.uniform(low=0.0, high=2 * np.pi, size=(n, n_simulation))
        x = np.sort(uniforms, axis=0)
        U2s = np.array([compute_U2(x[:, i]) for i in range(n_simulation)])
        pval = float((np.count_nonzero(U2s >= U2o) + 1) / (n_simulation + 1))

    if verbose:
        print("Watson's One-Sample U2 Test of Circular Uniformity")
        print("--------------------------------------------------")
        print("")
        print(f"Test Statistic: {U2o:.4f}")
        print(f"P-value = {pval} {significance_code(pval)}")

    return WatsonTestResult(U2=float(U2o), pval=float(pval), mode=mode, n_simulation=n_simulation)


def rao_spacing_test(
    alpha: np.ndarray,
    w: Union[np.ndarray, None] = None,
    kappa: float = 1000.0,
    n_simulation: int = 9999,
    seed: int = 2046,
    verbose: bool = False,
) -> RaoSpacingTestResult:
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
    RaoSpacingTestResult
        Dataclass containing the Rao spacing statistic (degrees), p-value, method, and simulation count.

    Reference
    ---------
    Landler et al. (2019)
    https://movementecologyjournal.biomedcentral.com/articles/10.1186/s40462-019-0160-x
    """

    if n_simulation <= 0:
        raise ValueError("`n_simulation` must be a positive integer.")

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    def compute_U(sample):
        ordered = np.sort(sample)
        n_local = ordered.size
        spacings = np.hstack([ordered[1:] - ordered[:-1], 2 * np.pi - ordered[-1] + ordered[0]])
        return 0.5 * np.sum(np.abs(spacings - (2 * np.pi / n_local)))

    if w is not None:
        w = np.asarray(w, dtype=float)
        if np.any(w < 0):
            raise ValueError("`w` must contain non-negative frequencies.")
        if not np.all(np.isclose(w, np.round(w))):
            raise ValueError("`w` must contain integer frequencies.")
        w = w.astype(int)
        if w.shape != alpha.shape:
            raise ValueError("`w` must have the same shape as `alpha`.")
        n = int(np.sum(w))
        if n <= 0:
            raise ValueError("Sum of weights must be positive.")
        m = alpha.size
        expanded_alpha = np.repeat(alpha, w)
        mode = "grouped"
    else:
        expanded_alpha = alpha
        n = expanded_alpha.size
        mode = "ungrouped"

    rng = np.random.default_rng(seed)

    Uo = compute_U(expanded_alpha)
    if w is not None:  # noncontinuous / grouped data
        vm_dist = vonmises(kappa=kappa)
        uniforms = rng.uniform(low=0.0, high=2 * np.pi, size=(n_simulation, n))
        snapped = np.floor(uniforms * m / (2 * np.pi)) * (2 * np.pi / m)
        noise = vm_dist.rvs(size=(n_simulation, n), random_state=rng)
        samples = angmod(snapped + noise)
        Us = np.array([compute_U(sample) for sample in samples])
    else:
        samples = rng.uniform(low=0.0, high=2 * np.pi, size=(n_simulation, n))
        Us = np.array([compute_U(sample) for sample in samples])

    counter = np.count_nonzero(Us >= Uo)
    pval = float((counter + 1) / (n_simulation + 1))

    if verbose:
        print("Rao's Spacing Test of Circular Uniformity")
        print("-----------------------------------------")
        print("")
        print(f"Test Statistic: {Uo:.4f}")
        print(f"P-value = {pval}\n")

    return RaoSpacingTestResult(
        statistic=float(np.rad2deg(Uo)),
        pval=float(pval),
        mode=mode,
        n_simulation=n_simulation,
    )


def circ_range_test(alpha: np.ndarray) -> CircularRangeTestResult:
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
    CircularRangeTestResult
        Dataclass containing the range statistic and corresponding p-value.

    Reference
    ---------
    P162, Section 7.2.3 of Jammalamadaka, S. Rao and SenGupta, A. (2001)
    """
    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    range_stat = circ_range(alpha)  # Compute test statistic

    # Compute p-value using approximation formula from CircStats (if available)
    n = alpha.size
    stop = int(np.floor(1 / (1 - range_stat / (2 * np.pi))))
    index = np.arange(1, stop + 1)

    # Compute p-value using series expansion
    sequence = (
        ((-1) ** (index - 1))
        * comb(n, index)
        * (1 - index * (1 - range_stat / (2 * np.pi))) ** (n - 1)
    )
    p_value = float(np.sum(sequence))

    return CircularRangeTestResult(range_stat=float(range_stat), pval=float(p_value))


def binomial_test(alpha: np.ndarray, md: float) -> BinomialTestResult:
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
    BinomialTestResult
        Dataclass containing the p-value and counts on each side of the hypothesized median.

    References
    ----------
    Zar, J. H. (2010). Biostatistical Analysis. Section 27.4.
    """
    from scipy.stats import binom

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    if np.ndim(md) != 0:
        raise ValueError("The median (md) must be a single scalar value.")

    # Compute circular differences from hypothesized median
    d = circ_dist(alpha, float(md))

    # Count the number of angles on each side of the hypothesized median
    n1 = int(np.sum(d < 0))
    n2 = int(np.sum(d > 0))
    n_eff = int(n1 + n2)
    if n_eff == 0:
        return BinomialTestResult(pval=1.0, n_eff=0, n1=n1, n2=n2)

    # Compute p-value using binomial test
    n_min = int(min(n1, n2))
    pval = float(2 * binom.cdf(n_min, n_eff, 0.5))
    pval = min(pval, 1.0)

    return BinomialTestResult(pval=pval, n_eff=n_eff, n1=n1, n2=n2)


def concentration_test(alpha1: np.ndarray, alpha2: np.ndarray) -> ConcentrationTestResult:
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
    ConcentrationTestResult
        Dataclass with the F statistic, p-value, and associated degrees of freedom.

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
    alpha1 = np.asarray(alpha1, dtype=float)
    alpha2 = np.asarray(alpha2, dtype=float)

    # Sample sizes
    n1, n2 = len(alpha1), len(alpha2)
    if min(n1, n2) < 2:
        raise ValueError("Both samples must contain at least two observations.")

    # Compute resultant vector lengths
    R1 = n1 * circ_r(alpha1)
    R2 = n2 * circ_r(alpha2)

    # Compute mean resultant length of combined samples
    rbar = (R1 + R2) / (n1 + n2)

    # Warn if rbar is too low
    if rbar < 0.7:
        warnings.warn(
            "The resultant vector length should exceed 0.7 for the concentration test to be reliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Compute F-statistic
    df1 = n1 - 1
    df2 = n2 - 1
    numerator = df2 * (n1 - R1)
    denominator = df1 * (n2 - R2)
    if denominator <= 0 or numerator <= 0:
        raise ValueError("Degenerate data: cannot compute concentration test statistic.")
    f_stat = numerator / denominator

    # Compute p-value (adjusting for F-stat symmetry)
    if f_stat >= 1:
        pval = 2 * f.sf(f_stat, df1, df2)
    else:
        pval = 2 * f.sf(1 / f_stat, df2, df1)

    return ConcentrationTestResult(
        f_stat=float(f_stat),
        pval=float(min(pval, 1.0)),
        df1=int(df1),
        df2=int(df2),
    )


def rao_homogeneity_test(samples: list, alpha: float = 0.05) -> RaoHomogeneityTestResult:
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
    RaoHomogeneityTestResult
        Dataclass containing test statistics, p-values, and rejection flags.

    References
    ----------
    Jammalamadaka, S. Rao and SenGupta, A. (2001). Topics in Circular Statistics, Section 7.6.1.
    Rao, J.S. (1967). Large sample tests for the homogeneity of angular data, Sankhya, Ser, B., 28.
    """
    if not isinstance(samples, list) or not all(
        isinstance(s, np.ndarray) for s in samples
    ):
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
    cov_cos_sin = np.array(
        [np.cov(np.cos(s), np.sin(s), ddof=1)[0, 1] for s in samples]
    )

    # Compute test statistics
    s_polar = (
        1
        / n
        * (
            var_sin / cos_means**2
            + (sin_means**2 * var_cos) / cos_means**4
            - (2 * sin_means * cov_cos_sin) / cos_means**3
        )
    )
    tan_means = sin_means / cos_means
    H_polar = np.sum(tan_means**2 / s_polar) - (
        np.sum(tan_means / s_polar) ** 2
    ) / np.sum(1 / s_polar)

    U = cos_means**2 + sin_means**2
    s_disp = (
        4
        / n
        * (
            cos_means**2 * var_cos
            + sin_means**2 * var_sin
            + 2 * cos_means * sin_means * cov_cos_sin
        )
    )
    H_disp = np.sum(U**2 / s_disp) - (np.sum(U / s_disp) ** 2) / np.sum(1 / s_disp)

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

    return RaoHomogeneityTestResult(
        H_polar=float(H_polar),
        pval_polar=float(pval_polar),
        reject_polar=bool(reject_polar),
        H_disp=float(H_disp),
        pval_disp=float(pval_disp),
        reject_disp=bool(reject_disp),
    )


def change_point_test(alpha) -> ChangePointTestResult:
    """
    Perform a change point test for mean direction, concentration, or both.

    Parameters
    ----------
    alpha : np.ndarray
        Vector of angular measurements in radians.

    Returns
    -------
    ChangePointTestResult
        Dataclass containing the change point statistics.

    References
    ----------
    Jammalamadaka, S. Rao and SenGupta, A. (2001). Topics in Circular Statistics, Chapter 11.

    Notes
    -----
    Ported from `change.pt()` function in the `CircStats` package for R.
    """

    def phi(x):
        """Helper function for phi computation."""
        inv = A1inv(x)
        bessel = i0(inv)
        if np.isinf(bessel):
            corr = (
                inv
                + np.log(
                    1
                    / np.sqrt(2 * np.pi * inv)
                    * (1 + 1 / (8 * inv) + 9 / (128 * inv**2) + 225 / (1024 * inv**3))
                )
            )
        else:
            corr = np.log(bessel)
        return x * inv - corr

    def est_rho(alpha):
        """Estimate mean resultant length (rho)."""
        return np.linalg.norm(np.sum(np.exp(1j * alpha))) / len(alpha)

    n = len(alpha)
    if n < 4:
        raise ValueError("Sample size must be at least 4 for change point test.")

    rho = est_rho(alpha)

    R1, R2, V = np.zeros(n), np.zeros(n), np.zeros(n)

    for k in range(1, n):
        R1[k - 1] = est_rho(alpha[:k]) * k
        R2[k - 1] = est_rho(alpha[k:]) * (n - k)

        if 2 <= k <= (n - 2):
            V[k - 1] = (k / n) * phi(R1[k - 1] / k) + ((n - k) / n) * phi(
                R2[k - 1] / (n - k)
            )

    R1[-1] = rho * n
    R2[-1] = 0

    R_diff = R1 + R2 - rho * n
    rmax = np.max(R_diff)
    k_r = np.argmax(R_diff)
    rave = np.mean(R_diff)

    if n > 3:
        V = V[1 : n - 2]
        tmax = np.max(V)
        k_t = np.argmax(V) + 1
        tave = np.mean(V)
    else:
        raise ValueError("Sample size must be at least 4.")

    return ChangePointTestResult(
        n=int(n),
        rho=float(rho),
        rmax=float(rmax),
        k_r=int(k_r),
        rave=float(rave),
        tmax=float(tmax),
        k_t=int(k_t),
        tave=float(tave),
    )


def harrison_kanji_test(
    alpha: np.ndarray,
    idp: np.ndarray,
    idq: np.ndarray,
    inter: bool = True,
    fn: Optional[list] = None,
) -> HarrisonKanjiTestResult:
    """
    Harrison-Kanji Test (Two-Way ANOVA) for Circular Data.
    """

    if fn is None:
        fn = ["A", "B"]

    # Ensure data is in column format
    alpha = np.asarray(alpha).flatten()
    idp = np.asarray(idp).flatten()
    idq = np.asarray(idq).flatten()

    # Number of factor levels
    p = len(np.unique(idp))
    q = len(np.unique(idq))

    # Data frame for aggregation
    df = pd.DataFrame({fn[0]: idp, fn[1]: idq, "dependent": alpha})
    n = len(df)

    # Total resultant vector length
    tr = n * circ_r(np.array(df["dependent"].values))
    kk = circ_kappa(tr / n)

    # Compute mean resultants per group
    gr = df.groupby(fn)
    cn = gr.count()
    cr = gr.agg(circ_r) * cn
    cn = cn.unstack(fn[1])
    cr = cr.unstack(fn[1])

    # Factor A
    gr = df.groupby(fn[0])
    pn = gr.count()["dependent"]
    pr = gr.agg(circ_r)["dependent"] * pn

    # Factor B
    gr = df.groupby(fn[1])
    qn = gr.count()["dependent"]
    qr = gr.agg(circ_r)["dependent"] * qn

    if kk > 2:  # Large kappa approximation
        eff_1 = sum(pr**2 / np.sum(cn, axis=1)) - tr**2 / n
        df_1 = p - 1
        ms_1 = eff_1 / df_1

        eff_2 = sum(qr**2 / np.sum(cn, axis=0)) - tr**2 / n
        df_2 = q - 1
        ms_2 = eff_2 / df_2

        eff_t = n - tr**2 / n
        df_t = n - 1
        m = np.asarray(cn.values).mean()

        if inter:
            beta = 1 / (1 - 1 / (5 * kk) - 1 / (10 * (kk**2)))

            eff_r = n - np.asarray((cr**2.0 / cn).values).sum()
            df_r = p * q * (m - 1)
            ms_r = eff_r / df_r

            eff_i = (
                np.asarray((cr**2.0 / cn).values).sum()
                - sum(qr**2.0 / qn)
                - sum(pr**2.0 / pn)
                + tr**2 / n
            )
            df_i = (p - 1) * (q - 1)
            ms_i = eff_i / df_i

            FI = ms_i / ms_r
            pI = 1 - f.cdf(FI, df_i, df_r)  # `f.cdf` is now unambiguous
        else:
            eff_r = n - sum(qr**2.0 / qn) - sum(pr**2.0 / pn) + tr**2 / n
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
        kappa_factor = 2 / (1 - rr**2)  # Renamed `f` to `kappa_factor`

        chi1 = kappa_factor * (sum(pr**2.0 / pn) - tr**2 / n)
        df_1 = 2 * (p - 1)
        p1 = 1 - chi2.cdf(chi1, df=df_1)

        chi2_val = kappa_factor * (sum(qr**2.0 / qn) - tr**2 / n)
        df_2 = 2 * (q - 1)
        p2 = 1 - chi2.cdf(chi2_val, df=df_2)

        chiI = kappa_factor * (
            np.asarray((cr**2.0 / cn).values).sum()
            - sum(pr**2.0 / pn)
            - sum(qr**2.0 / qn)
            + tr**2 / n
        )
        df_i = (p - 1) * (q - 1)
        pI = chi2.sf(chiI, df=df_i)

    pval = float(p1.squeeze()), float(p2.squeeze()), float(np.squeeze(pI))

    # Construct ANOVA Table
    if kk > 2:
        table = pd.DataFrame(
            {
                "Source": fn + ["Interaction", "Residual", "Total"],
                "DoF": [df_1, df_2, df_i, df_r, df_t],
                "SS": [eff_1, eff_2, eff_i, eff_r, eff_t],
                "MS": [ms_1, ms_2, ms_i, ms_r, np.nan],
                "F": [np.squeeze(F1), np.squeeze(F2), FI, np.nan, np.nan],
                "p": list(pval) + [np.nan, np.nan],
            }
        ).set_index("Source")
    else:
        table = pd.DataFrame(
            {
                "Source": fn + ["Interaction"],
                "DoF": [df_1, df_2, df_i],
                "chi2": [chi1.squeeze(), chi2_val.squeeze(), chiI.squeeze()],
                "p": pval,
            }
        ).set_index("Source")

    return HarrisonKanjiTestResult(p_values=pval, anova_table=table)


def equal_kappa_test(samples: list[np.ndarray], verbose: bool = False) -> EqualKappaTestResult:
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
    EqualKappaTestResult
        Dataclass containing the test statistic, p-value, and supporting metrics.

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

    arrays = [np.asarray(group, dtype=float) for group in samples]
    if any(arr.size == 0 for arr in arrays):
        raise ValueError("Each group must contain at least one observation.")

    # Sample sizes
    ns = np.array([arr.size for arr in arrays])
    if np.any(ns < 2):
        raise ValueError("Each group must contain at least two observations.")

    # Mean resultant lengths
    r_bars = np.array([circ_r(arr) for arr in arrays])
    Rs = r_bars * ns  # Unnormalized resultants

    # Overall resultant and mean resultant length
    all_samples = np.hstack(arrays)
    N = len(all_samples)
    r_bar_all = circ_r(all_samples)

    # Estimate kappa values
    kappas = np.array([circ_kappa(r) for r in r_bars])
    kappa_all = circ_kappa(r_bar_all)

    # Choose test statistic based on `r̄`
    if r_bar_all < 0.45:
        # Small `r̄`: arcsin transformation
        ws = 4 * (ns - 4) / 3
        g1s = np.arcsin(np.sqrt(3 / 8) * 2 * r_bars)
        chi_square_stat = np.sum(ws * g1s**2) - (np.sum(ws * g1s) ** 2 / np.sum(ws))
        regime = "small"

    elif 0.45 <= r_bar_all <= 0.7:
        # Moderate `r̄`: asinh transformation
        ws = (ns - 3) / 0.798
        g2s = np.arcsinh((r_bars - 1.089) / 0.258)
        chi_square_stat = np.sum(ws * g2s**2) - (np.sum(ws * g2s) ** 2 / np.sum(ws))
        regime = "moderate"

    else:
        # Large `r̄`: Bartlett-type test
        vs = ns - 1
        v = N - k
        d = 1 / (3 * (k - 1)) * (np.sum(1 / vs) - 1 / v)
        total_residual = N - np.sum(Rs)
        residuals = ns - Rs
        if np.any(residuals <= 0):
            raise ValueError("Degenerate data: within-group dispersion is zero.")
        if total_residual <= 0:
            raise ValueError("Degenerate data: between-group dispersion is zero.")
        chi_square_stat = (1 / (1 + d)) * (
            v * np.log(total_residual / v) - np.sum(vs * np.log(residuals / vs))
        )
        regime = "large"

    # Compute p-value
    df = k - 1
    p_value = 1 - chi2.cdf(chi_square_stat, df)

    result = EqualKappaTestResult(
        kappa=kappas,
        kappa_all=float(kappa_all),
        rho=r_bars,
        rho_all=float(r_bar_all),
        df=int(df),
        statistic=float(chi_square_stat),
        pval=float(p_value),
        regime=regime,
    )

    # Print results if verbose is enabled
    if verbose:
        print("\nTest for Homogeneity of Concentration Parameters (κ)")
        print("------------------------------------------------------")
        print(f"Mean Resultant Lengths: {result.rho}")
        print(f"Overall Mean Resultant Length: {result.rho_all:.5f}")
        print(f"Estimated Kappa Values: {result.kappa}")
        print(f"Overall Estimated Kappa: {result.kappa_all:.5f}")
        print(f"Degrees of Freedom: {result.df}")
        print(f"Chi-Square Statistic: {result.statistic:.5f}")
        print(f"P-value: {result.pval:.5f}")
        print(f"Regime: {result.regime}")
        print("------------------------------------------------------\n")

    return result


def common_median_test(
    samples: list[np.ndarray],
    alpha: float = 0.05,
    verbose: bool = False,
) -> CommonMedianTestResult:
    """
    Common Median Test (Equal Median Test) for Multiple Circular Samples.

    - **H₀**: All groups have the same circular median.
    - **H₁**: At least one group has a different circular median.

    Parameters
    ----------
    samples : list of np.ndarray
        List of circular data arrays (angles in radians) for different groups.
    alpha : float, optional
        Significance level for deciding whether to reject the null hypothesis (default 0.05).
    verbose : bool, optional
        If `True`, prints the test summary.

    Returns
    -------
    CommonMedianTestResult
        Dataclass containing the common median, test statistic, p-value, and rejection flag.

    References
    ----------
    - Fisher, N. I. (1995). Statistical Analysis of Circular Data.
    - `circ_cmtest` from MATLAB's Circular Statistics Toolbox.
    """

    # Number of groups
    if not (0 < alpha < 1):
        raise ValueError("`alpha` must be between 0 and 1.")

    k = len(samples)
    if k < 2:
        raise ValueError("At least two groups are required for the test.")

    arrays = [np.asarray(group, dtype=float) for group in samples]
    if any(arr.size == 0 for arr in arrays):
        raise ValueError("Each group must contain at least one observation.")

    # Sample sizes
    ns = np.array([arr.size for arr in arrays])
    N = int(np.sum(ns))  # Total number of observations

    # Compute the common circular median
    common_median = circ_median(np.hstack(arrays))

    # Compute deviations from the common median
    m = np.zeros(k, dtype=float)
    for i, group in enumerate(arrays):
        deviations = circ_dist(group, common_median)
        m[i] = np.sum(deviations < 0)

    # Compute test statistic
    M = np.sum(m)
    if M == 0 or M == N:
        raise ValueError("All observations fall on the same side of the median; test is undefined.")

    P = (N**2 / (M * (N - M))) * np.sum(m**2 / ns) - (N * M) / (N - M)

    # Compute p-value
    df = k - 1
    p_value = 1 - chi2.cdf(P, df)
    reject = p_value < alpha

    # If the null hypothesis is rejected, return NaN for the median
    if reject:
        common_median = np.nan

    result = CommonMedianTestResult(
        common_median=float(common_median),
        statistic=float(P),
        pval=float(p_value),
        reject=bool(reject),
    )

    # Print results if verbose is enabled
    if verbose:
        print("\nCommon Median Test (Equal Median Test)")
        print("--------------------------------------")
        median_display = result.common_median if not result.reject else "NaN"
        print(f"Estimated Common Median: {median_display}")
        print(f"Test Statistic: {result.statistic:.5f}")
        print(f"P-value: {result.pval:.5f}")
        decision = "Yes" if result.reject else "No"
        print(f"Reject H₀ (α={alpha:.2f}): {decision}")
        print("--------------------------------------\n")

    return result
