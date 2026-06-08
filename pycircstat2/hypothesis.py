import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.special import comb, i0, iv
from scipy.stats import chi2, f, norm, rankdata, wilcoxon

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
from .distributions import vonmises
from .utils import (
    A1inv,
    angmod,
    angular_distance,
    is_within_circular_range,
    significance_code,
)

__all__ = [
    "rayleigh_test",
    "chisquare_test",
    "V_test",
    "one_sample_test",
    "omnibus_test",
    "batschelet_test",
    "symmetry_test",
    "watson_williams_test",
    "watson_u2_test",
    "kuiper_two_test",
    "wheeler_watson_test",
    "wallraff_test",
    "circ_anova",
    "angular_randomisation_test",
    "kuiper_test",
    "watson_test",
    "rao_spacing_test",
    "circ_range_test",
    "binomial_test",
    "concentration_test",
    "rao_homogeneity_test",
    "change_point_test",
    "harrison_kanji_test",
    "equal_kappa_test",
    "common_median_test",
]

SeedLike = Union[
    None,
    int,
    Sequence[int],
    np.random.Generator,
    np.random.BitGenerator,
    np.random.SeedSequence,
]


def _init_rng(seed: SeedLike) -> np.random.Generator:
    """Normalize a seed-like input into a Generator instance."""

    if isinstance(seed, np.random.Generator):
        return seed

    return np.random.default_rng(seed)


def _resolve_legacy_verbose(seed: SeedLike, verbose: bool) -> tuple[SeedLike, bool]:
    """Back-compat shim for the formerly positional ``verbose`` argument.

    Before ``seed`` was introduced these tests took ``verbose`` as the trailing
    positional argument, so a legacy call such as ``test(..., True)`` now binds
    ``True`` to ``seed`` instead. Detect that exact case (``seed is True`` with
    ``verbose`` left at its default) and reinterpret it as ``verbose=True``.
    """

    if seed is True and verbose is False:
        warnings.warn(
            "Passing `verbose` as a positional argument is deprecated; use keyword arguments.",
            DeprecationWarning,
            stacklevel=3,
        )
        return 2046, True

    return seed, verbose


def _resolve_n_resamples(
    n_resamples: int,
    *,
    B: Optional[int] = None,
    n_simulation: Optional[int] = None,
    has_asymptotic: bool,
) -> int:
    """Map the deprecated ``B`` / ``n_simulation`` keywords onto ``n_resamples``.

    ``n_resamples == 0`` means "no resampling" (use the analytic p-value). For tests
    that have an analytic fallback (``has_asymptotic``) the old sentinel value ``1``
    requested exactly that, so it maps to ``0``; otherwise the count passes through.
    """

    legacy_name, legacy_value = None, None
    if B is not None:
        legacy_name, legacy_value = "B", B
    elif n_simulation is not None:
        legacy_name, legacy_value = "n_simulation", n_simulation

    if legacy_value is None:
        return n_resamples

    warnings.warn(
        f"`{legacy_name}` is deprecated; use `n_resamples` instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    if has_asymptotic and legacy_value <= 1:
        return 0
    return int(legacy_value)


def _warn_deprecated_attr(old: str, new: str) -> None:
    """Emit a deprecation warning for a renamed result attribute."""

    warnings.warn(
        f"`{old}` is deprecated; use `{new}` instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _randomization_pval(
    statistic_fn,
    pooled: np.ndarray,
    group_sizes: Sequence[int],
    observed: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> float:
    """Randomization (permutation) p-value for a multi-sample statistic.

    Each of ``n_resamples`` permutations reshuffles the pooled per-observation
    quantities into the original group sizes; ``statistic_fn`` recomputes the
    statistic from the list of group arrays. Returns
    ``(#{stat >= observed} + 1) / (n_resamples + 1)``
    (Pewsey, Neuhäuser & Ruxton 2013, §7.3.2/7.4.3/7.5.3/7.5.5).

    ``pooled`` may be 1-D (e.g. angles, deviations, sign indicators) or 2-D with one
    row per observation (e.g. cos/sin uniform-score pairs); permutation is along axis 0.
    """

    pooled = np.asarray(pooled)
    split_at = np.cumsum(group_sizes)[:-1]
    count = 0
    for _ in range(n_resamples):
        groups = np.split(rng.permutation(pooled), split_at)
        if statistic_fn(groups) >= observed:
            count += 1
    return (count + 1) / (n_resamples + 1)


def _bootstrap_pval(
    statistic_fn,
    null_sample: np.ndarray,
    n: int,
    observed: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> float:
    """Bootstrap p-value for a one-sample statistic under a null-constrained sample.

    Draws ``n_resamples`` samples of size ``n`` with replacement from ``null_sample``
    (a version of the data forced to satisfy H0 — e.g. symmetrized about the mean, or
    mean-shifted to μ₀), recomputes ``statistic_fn``, and returns
    ``(#{stat >= observed} + 1) / (n_resamples + 1)`` (Pewsey et al. 2013, §5.2.2/5.3.3).
    """

    null_sample = np.asarray(null_sample, dtype=float)
    count = 0
    for _ in range(n_resamples):
        if statistic_fn(rng.choice(null_sample, size=n, replace=True)) >= observed:
            count += 1
    return (count + 1) / (n_resamples + 1)


def _mc_uniform_pval(
    statistic_fn,
    n: int,
    observed: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> float:
    """Monte-Carlo p-value under the uniform-circle null hypothesis.

    Draws ``n_resamples`` samples of ``n`` angles ~ Uniform(0, 2π), recomputes the
    statistic, and returns ``(#{stat >= observed} + 1) / (n_resamples + 1)``. Pass a
    statistic oriented so that *larger* means *more extreme*; negate it (and ``observed``)
    for tests where small values indicate departure from uniformity.
    """

    count = 0
    for _ in range(n_resamples):
        if statistic_fn(rng.uniform(0.0, 2 * np.pi, size=n)) >= observed:
            count += 1
    return (count + 1) / (n_resamples + 1)


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

    def significance(self, attr: str = "pval") -> Optional[str]:
        """Return significance stars for the requested p-value attribute."""

        if not hasattr(self, attr):
            return None

        value = getattr(self, attr)
        if value is None:
            return None

        try:
            return significance_code(float(value))
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class RayleighTestResult(TestResult):
    r: float  # Resultant vector length
    z: float  # Test statistic (Rayleigh's Z)
    pval: float  # P-value (analytic or Monte-Carlo, per `method`)
    method: str  # "asymptotic" | "monte_carlo"
    n_resamples: int = 0

    @property
    def bootstrap_pval(self) -> Optional[float]:
        """Deprecated: the Monte-Carlo p-value, now in `pval` when `method="monte_carlo"`."""
        _warn_deprecated_attr("bootstrap_pval", "pval (with method='monte_carlo')")
        return self.pval if self.method == "monte_carlo" else None


@dataclass(frozen=True)
class ChiSquareTestResult(TestResult):
    chi2: float
    pval: float


@dataclass(frozen=True)
class VTestResult(TestResult):
    V: float
    u: float
    pval: float
    method: str = "asymptotic"  # "asymptotic" | "monte_carlo"
    n_resamples: int = 0


@dataclass(frozen=True)
class OneSampleTestResult(TestResult):
    reject: bool  # whether μ0 lies outside the 95% CI of the mean
    angle: float
    ci: tuple[float, float]
    statistic: Optional[float] = None  # eq. 5.10 z, when raw angles are supplied
    pval: Optional[float] = None  # specified-mean test p-value (eq. 5.10)
    method: Optional[str] = None  # "asymptotic" | "bootstrap" | None (CI only)
    n_resamples: int = 0


@dataclass(frozen=True)
class OmnibusTestResult(TestResult):
    A: float
    pval: float
    m: int
    method: str = "asymptotic"  # "asymptotic" (Ajne approx) | "monte_carlo"
    n_resamples: int = 0


@dataclass(frozen=True)
class BatscheletTestResult(TestResult):
    C: int
    pval: float


@dataclass(frozen=True)
class SymmetryTestResult(TestResult):
    statistic: float
    pval: float
    method: str = "wilcoxon"  # "wilcoxon" (Zar) | "pewsey" (β̄₂ test)
    n_resamples: int = 0


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
    method: str = "asymptotic"  # "asymptotic" | "randomization"
    n_resamples: int = 0


@dataclass(frozen=True)
class KuiperTwoTestResult(TestResult):
    V: float  # two-sample Kuiper statistic D+ + D-
    pval: float
    method: str = "asymptotic"  # "asymptotic" | "randomization"
    n_resamples: int = 0


@dataclass(frozen=True)
class WheelerWatsonTestResult(TestResult):
    W: float
    pval: float
    df: int
    method: str = "asymptotic"  # "asymptotic" | "randomization"
    n_resamples: int = 0


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
    n_resamples: int = 0  # >0 => `pval` is a label-randomization p-value


@dataclass(frozen=True)
class AngularRandomisationTestResult(TestResult):
    statistic: float
    pval: float
    method: str  # always "randomization"
    n_resamples: int

    @property
    def n_simulation(self) -> int:
        """Deprecated alias for `n_resamples`."""
        _warn_deprecated_attr("n_simulation", "n_resamples")
        return self.n_resamples


@dataclass(frozen=True)
class KuiperTestResult(TestResult):
    V: float
    pval: float
    method: str  # "asymptotic" | "monte_carlo"
    n_resamples: int

    @property
    def mode(self) -> str:
        """Deprecated: p-value method, now in `method` ("asymptotic"|"monte_carlo")."""
        _warn_deprecated_attr("mode", "method")
        return "asymptotic" if self.method == "asymptotic" else "simulation"

    @property
    def n_simulation(self) -> int:
        """Deprecated alias for `n_resamples`."""
        _warn_deprecated_attr("n_simulation", "n_resamples")
        return self.n_resamples


@dataclass(frozen=True)
class WatsonTestResult(TestResult):
    U2: float
    pval: float
    method: str  # "asymptotic" | "monte_carlo" | "parametric_bootstrap"
    n_resamples: int
    dist: str = "uniform"  # null tested: "uniform" | "vonmises"
    mu: Optional[float] = None  # fitted mean direction (von Mises GoF only)
    kappa: Optional[float] = None  # fitted concentration (von Mises GoF only)

    @property
    def mode(self) -> str:
        """Deprecated: p-value method, now in `method` ("asymptotic"|"monte_carlo")."""
        _warn_deprecated_attr("mode", "method")
        return "asymptotic" if self.method == "asymptotic" else "simulation"

    @property
    def n_simulation(self) -> int:
        """Deprecated alias for `n_resamples`."""
        _warn_deprecated_attr("n_simulation", "n_resamples")
        return self.n_resamples


@dataclass(frozen=True)
class RaoSpacingTestResult(TestResult):
    statistic: float
    pval: float
    method: str  # always "monte_carlo"
    data_kind: str  # "grouped" | "ungrouped"
    n_resamples: int

    @property
    def mode(self) -> str:
        """Deprecated: data descriptor, now in `data_kind` ("grouped"|"ungrouped")."""
        _warn_deprecated_attr("mode", "data_kind")
        return self.data_kind

    @property
    def n_simulation(self) -> int:
        """Deprecated alias for `n_resamples`."""
        _warn_deprecated_attr("n_simulation", "n_resamples")
        return self.n_resamples


@dataclass(frozen=True)
class CircularRangeTestResult(TestResult):
    range_stat: float
    pval: float
    method: str = "exact"  # "exact" (series) | "monte_carlo"
    n_resamples: int = 0


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
    method: str = "asymptotic"  # "asymptotic" (F-test) | "randomization"
    n_resamples: int = 0


@dataclass(frozen=True)
class RaoHomogeneityTestResult(TestResult):
    H_polar: float
    pval_polar: float
    reject_polar: bool
    H_disp: float
    pval_disp: float
    reject_disp: bool
    method: str = "asymptotic"
    n_resamples: int = 0


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
    pval_r: Optional[float] = None  # permutation p-value for rmax (mean-direction change)
    pval_t: Optional[float] = None  # permutation p-value for tmax (concentration change)
    n_resamples: int = 0


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
    method: str = "asymptotic"  # "asymptotic" | "randomization"
    n_resamples: int = 0


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


def _coerce_sample_arrays(samples: Sequence[Any]) -> list[np.ndarray]:
    """Coerce a sequence of samples into a list of 1-D float angle arrays.

    Lightweight counterpart to ``_coerce_circular_samples`` for tests that need
    only the raw angles (no weights or resultants). Each sample may be an
    array-like (``np.ndarray``, list, ...) or a ``Circular`` object; the latter
    is unwrapped to its ``alpha``. Plain arrays/lists are the canonical input —
    ``Circular`` support is a convenience.
    """
    if not isinstance(samples, Sequence) or len(samples) == 0:
        raise ValueError("`samples` must be a non-empty sequence of array-like samples.")

    try:
        from .base import Circular
    except Exception:  # pragma: no cover - defensive import guard
        Circular = None  # type: ignore

    arrays: list[np.ndarray] = []
    for sample in samples:
        if Circular is not None and isinstance(sample, Circular):  # type: ignore[arg-type]
            arr = np.asarray(sample.alpha, dtype=float)
        else:
            arr = np.asarray(sample, dtype=float)
        if arr.ndim != 1:
            raise ValueError("Each sample must be a one-dimensional array of angles.")
        if arr.size == 0:
            raise ValueError("Each sample must contain at least one observation.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Angles must be finite.")
        arrays.append(arr)

    return arrays


def rayleigh_test(
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    r: Optional[float] = None,
    n: Optional[int] = None,
    n_resamples: int = 0,
    seed: SeedLike = 2046,
    verbose: bool = False,
    *,
    B: Optional[int] = None,
) -> RayleighTestResult:
    r"""
    Rayleigh's Test for Circular Uniformity.

    - H0: The data in the population are distributed uniformly around the circle.
    - H1: The data in the population are not distributed uniformly around the circle.

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

    n_resamples: int
        If ``0`` (default), the analytic p-value (eq. 27.4) is returned. If ``>= 1``,
        that many Monte-Carlo samples drawn from the uniform null are used to estimate
        the p-value instead.

    seed: SeedLike
        Seed used to initialize the random number generator for Monte-Carlo resampling
        when ``n_resamples >= 1``. Accepts integers, sequences of integers,
        ``numpy.random.Generator``, ``numpy.random.BitGenerator``,
        ``numpy.random.SeedSequence`` or ``None``. Defaults to 2046.

    verbose: bool
        Print formatted results.

    B: int or None
        Deprecated alias for ``n_resamples`` (the old ``B=1`` meant "no resampling").

    Returns
    -------
    RayleighTestResult
        A dataclass containing:

        - r: float
            - Resultant vector length.
        - z: float
            - Test statistic (Rayleigh's Z).
        - pval: float
            - P-value, computed per ``method``.
        - method: str
            - "asymptotic" (eq. 27.4) or "monte_carlo".
        - n_resamples: int
            - Number of Monte-Carlo resamples used (0 if analytic).

    Reference
    ---------
    P625, Section 27.1, Example 27.1 of Zar, 2010
    """

    n_resamples = _resolve_n_resamples(n_resamples, B=B, has_asymptotic=True)
    if n_resamples < 0:
        raise ValueError("`n_resamples` must be a non-negative integer.")

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

    pval = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n)))  # eq(27.4)
    method = "asymptotic"

    seed, verbose = _resolve_legacy_verbose(seed, verbose)

    if n_resamples >= 1:
        rng = _init_rng(seed)
        uniforms = rng.uniform(0.0, 2 * np.pi, size=(n_resamples, n))
        resultant_lengths = np.abs(np.sum(np.exp(1j * uniforms), axis=1))
        mc_stats = (resultant_lengths**2) / n
        pval = float((np.count_nonzero(mc_stats >= z) + 1) / (n_resamples + 1))
        method = "monte_carlo"

    if verbose:
        print("Rayleigh's Test of Uniformity")
        print("-----------------------------")
        print("H0: ρ = 0")
        print("HA: ρ ≠ 0")
        print("")
        print(f"Test Statistics  (ρ | z-score): {r:.5f} | {z:.5f}")
        print(f"P-value ({method}): {pval:.5f} {significance_code(pval)}")

    return RayleighTestResult(r=r, z=z, pval=pval, method=method, n_resamples=n_resamples)


def chisquare_test(w: np.ndarray, verbose: bool = False) -> ChiSquareTestResult:
    """Chi-Square Goodness of Fit for Circular data.

    - H0: The data in the population are distributed uniformly around the circle.
    - H1: The data in the population are not distributed uniformly around the circle.

    This method is for grouped data.

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
    n_resamples: int = 0,
    seed: SeedLike = 2046,
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

    n_resamples: int
        If ``0`` (default), the p-value is the closed-form normal approximation. If
        ``>= 1``, it is estimated from that many Monte-Carlo uniform samples.

    seed: SeedLike
        Seed (or generator) for the Monte-Carlo p-value. Default 2046.

    verbose: bool
        Print formatted results.

    Returns
    -------
    VTestResult
        Dataclass containing the test statistic `V`, the normalized statistic `u`,
        the p-value, ``method`` (``"asymptotic"`` for the normal approximation, or
        ``"monte_carlo"`` when ``n_resamples >= 1``), and ``n_resamples``.

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

    if n_resamples >= 1:
        def _v_stat(sample: np.ndarray) -> float:
            return sample.size * circ_r(sample) * np.cos(circ_mean(sample) - angle)

        rng = _init_rng(seed)
        pval = _mc_uniform_pval(_v_stat, n, V, n_resamples, rng)
        method = "monte_carlo"
    else:
        pval = float(norm.sf(u))
        method = "asymptotic"

    if verbose:
        print("Modified Rayleigh's Test of Uniformity")
        print("--------------------------------------")
        print("H0: ρ = 0")
        print(f"HA: ρ ≠ 0 and μ = {angle:.5f} rad")
        print("")
        print(f"Test Statistics: {V:.5f}")
        print(f"P-value ({method}): {pval:.5f} {significance_code(pval)}")

    return VTestResult(V=V, u=u, pval=pval, method=method, n_resamples=n_resamples)


def _spec_mean_stat(alpha: np.ndarray, mu0: float, symmetric: bool) -> tuple[float, float]:
    """Statistic 5.10 (z) and bias-corrected mean μ̂BC for the specified-mean test (§5.3.3)."""
    n = alpha.size
    C1 = float(np.mean(np.cos(alpha)))
    S1 = float(np.mean(np.sin(alpha)))
    tbar = float(np.arctan2(S1, C1) % (2 * np.pi))
    Rbar = float(np.hypot(C1, S1))
    dev = alpha - tbar
    abar2 = float(np.mean(np.cos(2 * dev)))
    bbar2 = 0.0 if symmetric else float(np.mean(np.sin(2 * dev)))
    div = 2 * n * Rbar**2
    mubc = float((tbar + bbar2 / div) % (2 * np.pi))
    se = np.sqrt((1 - abar2) / div)
    dist = np.pi - abs(np.pi - abs(mubc - mu0))  # angular distance between μ̂BC and μ0
    return float(dist / se), mubc


def one_sample_test(
    angle: Union[int, float],
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    lb: Optional[float] = None,
    ub: Optional[float] = None,
    symmetric: bool = False,
    n_resamples: int = 0,
    seed: SeedLike = 2046,
    verbose: bool = False,
) -> OneSampleTestResult:
    """
    Test whether the population mean direction equals a specified value μ0.

    The decision (`reject`) is made by checking whether μ0 lies within the 95% CI
    of the mean. When the raw angles (`alpha`) are supplied, a continuous p-value
    for H0: μ = μ0 is also computed from Pewsey et al. (2013), §5.3.3 (statistic 5.10):
    the large-sample normal p-value when ``n_resamples=0``, or the bootstrap p-value
    when ``n_resamples >= 1`` (recommended for small samples).

    - H0: The population has a mean of μ0 (μ_a = μ_0)
    - H1: The population mean is not μ0 (μ_a ≠ μ_0)

    Parameters
    ----------
    angle: float or int
        Specified mean direction μ0 in radian.

    alpha: np.array or None
        Angles in radian (required for the p-value; for the CI either `alpha` or
        `lb`/`ub` is needed).

    w: np.array or None.
        Frequencies of angles.

    lb, ub: float or None
        Confidence-interval bounds from `descriptive.circ_mean_ci()`; computed from
        `alpha` when not supplied.

    symmetric: bool
        If ``True``, assume the underlying distribution is reflectively symmetric
        (zeroes the skewness bias-correction; symmetrizes the bootstrap pool about μ0).

    n_resamples: int
        ``0`` (default) → large-sample p-value; ``>= 1`` → bootstrap p-value (§5.3.3).

    seed: SeedLike
        Seed for the bootstrap RNG when ``n_resamples >= 1``.

    verbose: bool
        Print formatted results.

    Returns
    -------
    OneSampleTestResult
        Dataclass with the CI decision `reject`, the tested `angle`, the 95% CI `ci`,
        and (when `alpha` is supplied) the specified-mean `statistic` (eq. 5.10),
        `pval`, `method` ("asymptotic"|"bootstrap"), and `n_resamples`.

    Reference
    ---------
    P628, Section 27.1, Example 27.3 of Zar, 2010 (CI inclusion).
    Pewsey, Neuhäuser & Ruxton (2013), §5.3.3 (specified-mean p-value).
    """

    angle = float(angle)

    if alpha is not None:
        alpha = np.asarray(alpha, dtype=float)
        if alpha.size == 0:
            raise ValueError("`alpha` must contain at least one angle.")
        if w is None:
            w = np.ones_like(alpha, dtype=float)
        else:
            w = np.asarray(w, dtype=float)
            if w.shape != alpha.shape:
                raise ValueError("`w` must have the same shape as `alpha`.")

    if lb is None or ub is None:
        if alpha is None:
            raise ValueError("If `lb` or `ub` is None, then `alpha` (and optionally `w`) is required.")
        lb, ub = circ_mean_ci(alpha=alpha, w=w)

    lb = float(lb)
    ub = float(ub)

    reject = not is_within_circular_range(angle, lb, ub)

    # Continuous specified-mean p-value (eq. 5.10), only when raw angles are available.
    statistic: Optional[float] = None
    pval: Optional[float] = None
    method: Optional[str] = None
    used_resamples = 0
    if alpha is not None:
        sample = np.repeat(alpha, np.round(w).astype(int))
        z0, mubc = _spec_mean_stat(sample, angle, symmetric)
        statistic = z0
        if n_resamples >= 1:
            # Shift the sample to mean direction μ0 (optionally symmetrize about μ0),
            # then resample with replacement (§5.3.3 / Fisher 1993 §4.4.5).
            shifted = angmod(sample - mubc + angle)
            null_sample = (
                np.concatenate([shifted, angmod(2 * angle - shifted)]) if symmetric else shifted
            )
            rng = _init_rng(seed)
            pval = _bootstrap_pval(
                lambda b: _spec_mean_stat(b, angle, symmetric)[0],
                null_sample,
                sample.size,
                z0,
                n_resamples,
                rng,
            )
            method = "bootstrap"
            used_resamples = n_resamples
        else:
            pval = float(2 * norm.sf(z0))
            method = "asymptotic"

    if verbose:
        print("One-Sample Test for the Mean Angle")
        print("----------------------------------")
        print("H0: μ = μ0")
        print(f"HA: μ ≠ μ0 and μ0 = {angle:.5f} rad")
        print("")
        verb = "outside" if reject else "within"
        print(f"μ0 = {angle:.5f} lies {verb} the 95% CI of μ ({np.array([lb, ub]).round(5)})")
        if pval is not None:
            print(f"P-value ({method}): {pval:.5g} {significance_code(pval)}")

    return OneSampleTestResult(
        reject=reject,
        angle=angle,
        ci=(lb, ub),
        statistic=statistic,
        pval=pval,
        method=method,
        n_resamples=used_resamples,
    )


def _omnibus_m(alpha: np.ndarray, scale: int) -> int:
    """Hodges-Ajne statistic m: the minimum point count on one side of a diameter."""
    lines = np.linspace(0.0, np.pi, scale * 360, endpoint=False)
    n = alpha.size
    lines_rotated = angmod(lines[:, None] - alpha)
    right = n - np.logical_and(lines_rotated > 0.0, lines_rotated < np.pi).sum(axis=1)
    return int(np.min(right))


def omnibus_test(
    alpha: np.ndarray,
    scale: int = 1,
    n_resamples: int = 0,
    seed: SeedLike = 2046,
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

    n_resamples: int
        If ``0`` (default), the p-value is Hodges–Ajne's closed-form approximation.
        If ``>= 1``, it is estimated from that many Monte-Carlo uniform samples.

    seed: SeedLike
        Seed (or generator) for the Monte-Carlo p-value. Default 2046.

    verbose: bool
        Print formatted results.

    Returns
    -------
    OmnibusTestResult
        Dataclass containing the test statistic `A`, the corresponding p-value,
        the minimum count `m`, ``method`` (``"asymptotic"`` for the closed-form
        approximation, or ``"monte_carlo"`` when ``n_resamples >= 1``), and
        ``n_resamples``.

    Reference
    ---------
    P629-630, Section 27.2, Example 27.4 of Zar, 2010
    """

    if scale <= 0:
        raise ValueError("`scale` must be a positive integer.")

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    n = alpha.size
    m = _omnibus_m(alpha, scale)

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
        # m ≈ n/2: the data is maximally uniform and the analytic p-value
        # (valid only for m well below n/2) degenerates to 0. There is no
        # evidence against uniformity here, so do not reject.
        pval = 1.0
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

    if n_resamples >= 1:
        # Smaller m = more clustered = more extreme, so negate for the upper-tail helper.
        rng = _init_rng(seed)
        pval = _mc_uniform_pval(lambda s: -_omnibus_m(s, scale), n, -m, n_resamples, rng)
        method = "monte_carlo"
    else:
        method = "asymptotic"

    if verbose:
        print('Hodges-Ajne ("omnibus") Test for Uniformity')
        print("-------------------------------------------")
        print("H0: uniform")
        print("HA: not uniform")
        print("")
        print(f"Test Statistics: {A:.5f}")
        print(f"P-value ({method}): {pval:.5f} {significance_code(pval)}")
    return OmnibusTestResult(
        A=float(A), pval=float(pval), m=int(m), method=method, n_resamples=n_resamples
    )


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
        print(f"HA: not uniform but concentrated around θ = {angle:.5f} rad")
        print("")
        print(f"Test Statistics: {C}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return BatscheletTestResult(C=C, pval=pval)


def _rs_test_stat(alpha: np.ndarray) -> float:
    """Pewsey's (2002) studentized second sine moment |z| for reflective symmetry (eq. 5.4)."""
    n = alpha.size
    Rbar = circ_r(alpha)
    dev = alpha - circ_mean(alpha)
    abar2 = float(np.mean(np.cos(2 * dev)))
    bbar2 = float(np.mean(np.sin(2 * dev)))
    abar3 = float(np.mean(np.cos(3 * dev)))
    abar4 = float(np.mean(np.cos(4 * dev)))
    var = (
        (1 - abar4) / 2 - 2 * abar2 + (2 * abar2 / Rbar) * (abar3 + abar2 * (1 - abar2) / Rbar)
    ) / n
    return float(abs(bbar2 / np.sqrt(var)))


def symmetry_test(
    alpha: np.ndarray,
    median: Optional[float] = None,
    method: str = "wilcoxon",
    n_resamples: int = 0,
    seed: SeedLike = 2046,
    verbose: bool = False,
) -> SymmetryTestResult:
    """Test for reflective symmetry of a circular distribution.

    - H0: the population is reflectively symmetrical
    - HA: the population is not symmetrical

    Parameters
    ----------
    alpha: np.array
        Angles in radian.

    median: float or None
        Median (only used by ``method="wilcoxon"``). Computed by
        `descriptive.circ_median()` if not provided.

    method: str
        - ``"wilcoxon"`` (default): Wilcoxon signed-rank test on the angular
          deviations from the median (Zar 2010; symmetry about the median).
        - ``"pewsey"``: Pewsey's (2002) studentized second sine moment about the
          mean direction (eq. 5.4). With ``n_resamples=0`` the large-sample normal
          p-value is used (valid n >= 50); with ``n_resamples >= 1`` the §5.2.2
          bootstrap p-value (Efron symmetrization) is used — recommended for small
          samples.

    n_resamples: int
        Bootstrap resamples for ``method="pewsey"`` (default 0 = large-sample).

    seed: SeedLike
        Seed for the bootstrap RNG when ``method="pewsey"`` and ``n_resamples >= 1``.

    verbose: bool
        Print formatted results.

    Returns
    -------
    SymmetryTestResult
        Dataclass with the test statistic, p-value, ``method`` ("wilcoxon"|"pewsey"),
        and ``n_resamples``.

    Reference
    ---------
    P631-632, Section 27.3, Example 27.6 of Zar, 2010 (Wilcoxon).
    Pewsey (2002); Pewsey, Neuhäuser & Ruxton (2013), §5.2 (Pewsey β̄₂ test).
    """

    if method not in ("wilcoxon", "pewsey"):
        raise ValueError("`method` must be 'wilcoxon' or 'pewsey'.")

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    if method == "wilcoxon":
        if median is None:
            median = float(circ_median(alpha=alpha))
        else:
            median = float(median)
        d = angmod(alpha - median, bounds=[-np.pi, np.pi])
        res = wilcoxon(d, alternative="two-sided")
        statistic = float(res.statistic)
        pval = float(res.pvalue)
        used_resamples = 0
    else:  # method == "pewsey"
        statistic = _rs_test_stat(alpha)
        if n_resamples >= 1:
            theta_bar = circ_mean(alpha)
            # Efron symmetrization: reflect about the mean, pool, resample (§5.2.2).
            symmetrized = np.concatenate([alpha, 2 * theta_bar - alpha])
            rng = _init_rng(seed)
            pval = _bootstrap_pval(
                _rs_test_stat, symmetrized, alpha.size, statistic, n_resamples, rng
            )
            used_resamples = n_resamples
        else:
            pval = float(2 * norm.sf(statistic))
            used_resamples = 0

    if verbose:
        print("Symmetry Test")
        print("------------------------------")
        print(f"H0: reflectively symmetrical ({method})")
        print("HA: not symmetrical")
        print("")
        print(f"Test Statistics: {statistic:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return SymmetryTestResult(
        statistic=statistic, pval=pval, method=method, n_resamples=used_resamples
    )


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


def _watson_u2_statistic(s0: np.ndarray, s1: np.ndarray) -> float:
    """Watson's two-sample U² statistic for two arrays of angles (ties via counts)."""
    s0 = np.sort(np.asarray(s0, dtype=float))
    s1 = np.sort(np.asarray(s1, dtype=float))
    n0, n1 = s0.size, s1.size
    N = n0 + n1
    a, t = np.unique(np.concatenate([s0, s1]), return_counts=True)
    d = np.searchsorted(s0, a, side="right") / n0 - np.searchsorted(s1, a, side="right") / n1
    return float(n0 * n1 / N**2 * (np.sum(t * d**2) - np.sum(t * d) ** 2 / N))


def watson_u2_test(
    samples: Sequence[Any],
    n_resamples: int = 0,
    seed: SeedLike = 2046,
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

    n_resamples: int
        If ``0`` (default), the p-value uses Watson's (1961) approximation. If ``>= 1``,
        it is estimated from that many label randomizations (recommended for small
        samples; Pewsey et al. 2013, §7.5.5).

    seed: SeedLike
        Seed for the randomization RNG when ``n_resamples >= 1``. Defaults to 2046.

    verbose: bool
        Print formatted results.

    Returns
    -------
    WatsonU2TestResult
        Dataclass containing the U² statistic, p-value, ``method``
        ("asymptotic"|"randomization"), and ``n_resamples``.

    Reference
    ---------
    P637-638, Section 27.5, Example 27.9 of Zar, 2010
    P639-640, Section 27.5, Example 27.10 of Zar, 2010
    """

    normalized = _coerce_circular_samples(samples)
    if len(normalized) != 2:
        raise ValueError("`watson_u2_test` requires exactly two samples.")

    s0, s1 = normalized[0].expand(), normalized[1].expand()
    U2 = _watson_u2_statistic(s0, s1)

    if n_resamples >= 1:
        rng = _init_rng(seed)
        pval = _randomization_pval(
            lambda groups: _watson_u2_statistic(groups[0], groups[1]),
            np.concatenate([s0, s1]),
            [s0.size, s1.size],
            U2,
            n_resamples,
            rng,
        )
        method = "randomization"
    else:
        # Approximated P-value from Watson (1961)
        # https://github.com/pierremegevand/watsons_u2/blob/master/watsons_U2_approx_p.m
        pval = float(2 * np.exp(-19.74 * U2))
        method = "asymptotic"

    if verbose:
        print("Watson's U2 Test for two samples")
        print("---------------------------------------------")
        print("H0: The two samples are from populations with the same angle.")
        print("HA: The two samples are not from populations with the same angle.")
        print("")
        print(f"Test Statistics: {U2:.5f}")
        print(f"P-value ({method}): {pval:.5f} {significance_code(pval)}")

    return WatsonU2TestResult(U2=float(U2), pval=float(pval), method=method, n_resamples=n_resamples)


def _kuiper_pkp(lam: float) -> float:
    """Survival function of the modified two-sample Kuiper statistic.

    ``Q(λ) = 2 Σ_{j>=1} (4 j² λ² − 1) e^{−2 j² λ²}`` (Stephens 1965; Numerical Recipes
    ``probkp``), where ``λ`` is the sample-size-corrected statistic. Returns 1.0 for
    small ``λ`` (the series does not converge below the meaningful range). Reproduces
    the tabulated Kuiper critical values (λ=1.747 → 0.05, λ=2.001 → 0.01, …).
    """
    if lam <= 0:
        return 1.0
    a2 = -2.0 * lam * lam
    total = 0.0
    termbf = 0.0
    for j in range(1, 101):
        term = 2.0 * (4.0 * j * j * lam * lam - 1.0) * np.exp(a2 * j * j)
        total += term
        if abs(term) <= 1e-3 * termbf or abs(term) <= 1e-8 * total:
            return float(min(max(total, 0.0), 1.0))
        termbf = abs(term)
    return 1.0


def _kuiper_two_statistic(s0: np.ndarray, s1: np.ndarray) -> float:
    """Two-sample Kuiper statistic ``V = D⁺ + D⁻`` from the empirical CDFs.

    Exact (no resolution binning) and ties-aware via ``searchsorted`` on the pooled
    points: ``V = max(F0 − F1) − min(F0 − F1)``. Because both CDFs reach 1 the
    difference is periodic on the circle, so ``V`` is invariant to the choice of
    origin — the defining property of Kuiper's statistic vs. Kolmogorov–Smirnov.
    """
    s0 = np.sort(np.asarray(s0, dtype=float))
    s1 = np.sort(np.asarray(s1, dtype=float))
    n0, n1 = s0.size, s1.size
    a = np.unique(np.concatenate([s0, s1]))
    d = np.searchsorted(s0, a, side="right") / n0 - np.searchsorted(s1, a, side="right") / n1
    return float(d.max() - d.min())


def kuiper_two_test(
    samples: Sequence[Any],
    n_resamples: int = 0,
    seed: SeedLike = 2046,
    verbose: bool = False,
) -> KuiperTwoTestResult:
    """Two-sample Kuiper test — the circular analogue of the two-sample
    Kolmogorov–Smirnov test.

    - H0: The two samples come from the same population (F₁ = F₂).
    - H1: The two distributions differ — in mean direction, dispersion, or any
      other respect.

    Unlike Watson's U² (sensitive mainly to differences in location and dispersion),
    the Kuiper statistic ``V = D⁺ + D⁻`` responds to any difference between the two
    empirical CDFs and is invariant to the choice of origin on the circle.

    Parameters
    ----------
    samples : sequence
        Exactly two entries, each a `Circular` object or a one-dimensional
        array-like of radian angles (grouped data are expanded by frequency).
    n_resamples : int, optional
        If ``0`` (default), the p-value comes from the large-sample asymptotic
        distribution of the modified statistic (Stephens 1965). If ``>= 1``, that
        many label-randomization resamples are used instead (pool the two samples,
        permute into the original sizes, recompute ``V``); recommended for small
        samples.
    seed : int or numpy.random.Generator, optional
        Seed (or generator) for the randomization path. Default is 2046.
    verbose : bool, optional
        If ``True``, prints the test summary.

    Returns
    -------
    KuiperTwoTestResult
        Dataclass with ``V``, ``pval``, ``method`` and ``n_resamples``.

    References
    ----------
    - Kuiper, N.H. (1960). Tests concerning random points on a circle.
    - Stephens, M.A. (1965). The goodness-of-fit statistic Vₙ: distribution and
      significance points. Biometrika 52.
    - Batschelet (1981), p. 112.
    """

    normalized = _coerce_circular_samples(samples)
    if len(normalized) != 2:
        raise ValueError("`kuiper_two_test` requires exactly two samples.")

    s0, s1 = normalized[0].expand(), normalized[1].expand()
    n, m = s0.size, s1.size
    V = _kuiper_two_statistic(s0, s1)

    if n_resamples >= 1:
        rng = _init_rng(seed)
        pval = _randomization_pval(
            lambda groups: _kuiper_two_statistic(groups[0], groups[1]),
            np.concatenate([s0, s1]),
            [n, m],
            V,
            n_resamples,
            rng,
        )
        method = "randomization"
    else:
        en = np.sqrt(n * m / (n + m))
        pval = _kuiper_pkp((en + 0.155 + 0.24 / en) * V)
        method = "asymptotic"

    if verbose:
        print("Two-sample Kuiper Test")
        print("----------------------")
        print("H0: The two samples are drawn from the same distribution.")
        print("HA: The two distributions differ.")
        print("")
        print(f"Sample sizes: n1 = {n}, n2 = {m}")
        print(f"Test statistic (V = D+ + D-): {V:.5f}")
        print(f"P-value ({method}): {pval:.5f} {significance_code(pval)}")

    return KuiperTwoTestResult(V=float(V), pval=float(pval), method=method, n_resamples=n_resamples)


def wheeler_watson_test(
    samples: Sequence[Any],
    n_resamples: int = 0,
    seed: SeedLike = 2046,
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

    n_resamples: int
        If ``0`` (default), the p-value uses the χ² approximation. If ``>= 1``, it is
        estimated from that many label randomizations of the uniform scores
        (recommended when any group has fewer than ~10 observations;
        Pewsey et al. 2013, §7.5.3).

    seed: SeedLike
        Seed for the randomization RNG when ``n_resamples >= 1``. Defaults to 2046.

    verbose: bool
        Print formatted results.

    Returns
    -------
    WheelerWatsonTestResult
        Dataclass containing the W statistic, degrees of freedom, p-value, ``method``
        ("asymptotic"|"randomization"), and ``n_resamples``.

    Reference
    ---------
    P640-642, Section 27.5, Example 27.11 of Zar, 2010

    Note
    ----
    Ties are handled via midranks (Pewsey et al. 2013, P144).
    """
    normalized = _coerce_circular_samples(samples)
    k = len(normalized)
    if k < 2:
        raise ValueError("At least two samples are required for the Wheeler-Watson test.")

    expanded_samples = [sample.expand() for sample in normalized]
    ns = [e.size for e in expanded_samples]
    N = sum(ns)

    # Uniform (circular-rank) scores for the pooled sample; midranks handle ties.
    pooled = np.concatenate(expanded_samples)
    beta = 2 * np.pi * rankdata(pooled, method="average") / N
    scores = np.column_stack([np.cos(beta), np.sin(beta)])  # one [cos, sin] row per obs
    split_at = np.cumsum(ns)[:-1]
    score_groups = np.split(scores, split_at)

    def _wg(groups: list[np.ndarray]) -> float:
        # 2 * Σ_k (C_k² + S_k²) / n_k. For k=2 this is a positive multiple of the
        # special statistic `W` below, so the randomization p-value is unaffected.
        return 2.0 * sum(
            (g[:, 0].sum() ** 2 + g[:, 1].sum() ** 2) / g.shape[0] for g in groups
        )

    if k == 2:
        C = score_groups[0][:, 0].sum()
        S = score_groups[0][:, 1].sum()
        W = 2 * (N - 1) * (C**2 + S**2) / (ns[0] * ns[1])
    else:
        W = _wg(score_groups)

    df = 2 * (k - 1)
    if n_resamples >= 1:
        rng = _init_rng(seed)
        pval = _randomization_pval(_wg, scores, ns, _wg(score_groups), n_resamples, rng)
        method = "randomization"
    else:
        pval = float(chi2.sf(W, df=df))
        method = "asymptotic"

    if verbose:
        print("The Wheeler and Watson Two/Multi-Sample Test")
        print("---------------------------------------------")
        print("H0: All samples are from populations with the same angle.")
        print("HA: All samples are not from populations with the same angle.")
        print("")
        print(f"Test Statistics: {W:.5f}")
        print(f"P-value ({method}): {pval:.5f} {significance_code(pval)}")

    return WheelerWatsonTestResult(
        W=float(W), pval=pval, df=df, method=method, n_resamples=n_resamples
    )


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
    # Expand by weights so each distance vector has length ``sample.n``; this
    # keeps the Mann-Whitney rank split below correct for grouped data and is a
    # no-op for ungrouped samples.
    distances = [
        angular_distance(sample.expand(), angles[i]) for i, sample in enumerate(normalized)
    ]

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
        print("H0: The groups have equal dispersion around the specified reference angle.")
        print("HA: At least one group differs in dispersion around the specified angle.")
        print("")
        print(f"Test Statistics: {U:.5f}")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return WallraffTestResult(U=float(U), pval=pval)


def circ_anova(
    samples: Sequence[Any],
    method: str = "F-test",
    kappa: Optional[float] = None,
    f_mod: bool = True,
    n_resamples: int = 0,
    seed: SeedLike = 2046,
    verbose: bool = False,
) -> CircularAnovaResult:
    """
    Circular Analysis of Variance (ANOVA) for multi-sample comparison of mean directions.

    - **H₀**: All groups have the same mean direction.
    - **H₁**: At least one group has a different mean direction.

    Parameters
    ----------
    samples : sequence
        A sequence (one entry per group) of `Circular` objects or one-dimensional
        array-like radian samples.
    method : str, optional
        The test statistic to use. Options:
        - `"F-test"` (default): High-concentration F-test (Stephens 1972).
        - `"LRT"`: Likelihood Ratio Test (Cordeiro et al. 1994).
    kappa : float, optional
        The common concentration parameter (κ). If not specified, it is estimated using MLE.
    f_mod : bool, optional
        If `True`, applies a correction factor `(1 + 3/8κ)` to the F-statistic.
    n_resamples : int, optional
        If ``0`` (default), the p-value comes from the parametric (F or χ²) distribution.
        If ``>= 1``, it is estimated by permuting the pooled angles into the group sizes
        and recomputing the selected statistic — distribution-free, and free of the
        high-concentration assumption.
    seed : SeedLike, optional
        Seed for the randomization RNG when ``n_resamples >= 1``. Defaults to 2046.
    verbose : bool, optional
        If `True`, prints the test summary.

    Returns
    -------
    result : CircularAnovaResult
        Dataclass containing the selected statistic, p-value, supporting metrics, and
        ``n_resamples`` (>0 when the p-value is from label randomization).

    References
    ----------
    - Stephens (1972). Multi-sample tests for the von Mises distribution.
    - Cordeiro, Paula, & Botter (1994). Improved likelihood ratio tests for dispersion models.
    - Jammalamadaka & SenGupta (2001). Topics in Circular Statistics, Section 5.3.
    """

    # Number of groups
    samples = _coerce_sample_arrays(samples)
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

        if n_resamples >= 1:
            def _f_stat(groups: list[np.ndarray]) -> float:
                sumR = sum(circ_r(g) * len(g) for g in groups)
                fval = ((sumR - R_all) / df_between) / ((N - sumR) / df_within)
                return (1 + 3 / (8 * kappa_value)) * fval if f_mod else fval

            rng = _init_rng(seed)
            p_value = _randomization_pval(
                _f_stat, all_samples, ns, float(F_stat), n_resamples, rng
            )
        else:
            p_value = float(f.sf(F_stat, df_between, df_within))

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
            n_resamples=n_resamples,
        )

    # **Likelihood Ratio Test (LRT)**
    elif method == "LRT":
        # Compute test statistic
        term1 = 1 - (1 / (4 * kappa_value)) * (sum(1 / ns) - 1 / N)
        term2 = 2 * kappa_value * np.sum(Rs * (1 - np.cos(mus - mu_all)))
        chi_square_stat = term1 * term2

        df = k - 1
        if n_resamples >= 1:
            def _lrt_stat(groups: list[np.ndarray]) -> float:
                mus_p = np.array([circ_mean(g) for g in groups])
                Rs_p = np.array([circ_r(g) * len(g) for g in groups])
                return float(term1 * (2 * kappa_value * np.sum(Rs_p * (1 - np.cos(mus_p - mu_all)))))

            rng = _init_rng(seed)
            p_value = _randomization_pval(
                _lrt_stat, all_samples, ns, float(chi_square_stat), n_resamples, rng
            )
        else:
            p_value = float(chi2.sf(chi_square_stat, df))

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
            n_resamples=n_resamples,
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
    n_resamples: int = 1000,
    seed: SeedLike = 2046,
    verbose: bool = False,
    *,
    n_simulation: Optional[int] = None,
) -> AngularRandomisationTestResult:
    """The Angular Randomization Test (ART) for homogeneity.

    - H0: The two samples come from the same population.
    - H1: The two samples do not come from the same population.

    Parameters
    ----------
    samples: sequence
        A sequence of `Circular` objects or one-dimensional array-like radian samples.
    n_resamples: int, optional
        Number of random permutations for the test. Defaults to 1000.
    seed: SeedLike
        Seed used to initialize the random number generator for the permutation test.
        Accepts integers, sequences of integers, ``numpy.random.Generator``,
        ``numpy.random.BitGenerator``, ``numpy.random.SeedSequence`` or ``None``.
        Defaults to 2046.
    n_simulation: int or None
        Deprecated alias for ``n_resamples``.

    Returns
    -------
    AngularRandomisationTestResult
        Dataclass containing the observed statistic, permutation p-value,
        ``method="randomization"``, and ``n_resamples``.

    Reference
    ---------
    Jebur, A. J., & Abushilah, S. F. (2022).
    Distribution-free two-sample homogeneity test for circular data based on geodesic distance.
    International Journal of Nonlinear Analysis and Applications, 13(1), 2703-2711.
    """

    n_resamples = _resolve_n_resamples(n_resamples, n_simulation=n_simulation, has_asymptotic=False)

    normalized = _coerce_circular_samples(samples)

    if len(normalized) != 2:
        raise ValueError("The Angular Randomization Test requires exactly two samples.")
    if n_resamples <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    sample_arrays = [np.asarray(sample.alpha, dtype=float) for sample in normalized]
    if any(arr.size == 0 for arr in sample_arrays):
        raise ValueError("Each sample must contain at least one observation.")

    # ART statistic (Jebur & Abushilah 2022, eq. 3.1 & 4.2): the scaled sum of
    # all pairwise geodesic distances between the two groups,
    #     T = sqrt(n·m / (n + m)) · Σ_{i,j} d_geo(φ_i, ψ_j).
    # Under the permutation null the two group sizes (hence the scale) are fixed,
    # so precompute the full N×N geodesic distance matrix once and score every
    # permutation as a vectorized indicator quadratic form aᵀ·D·b, instead of
    # re-summing pairwise distances in a Python loop.
    n1, n2 = sample_arrays[0].size, sample_arrays[1].size
    N = n1 + n2
    scaling_factor = np.sqrt(n1 * n2 / N)

    combined = np.concatenate(sample_arrays)
    D = np.asarray(circ_pairdist(combined, combined, metric="geodesic"), dtype=float)

    # Observed statistic: the first n1 pooled angles form group 1.
    observed_stat = float(scaling_factor * D[:n1, n1:].sum())

    seed, verbose = _resolve_legacy_verbose(seed, verbose)
    rng = _init_rng(seed)

    # Each permutation draws a random partition of the N pooled angles into a
    # first group of size n1; `left`/`right` are the 0/1 group indicators.
    order = np.argsort(rng.random((n_resamples, N)), axis=1)
    left = np.zeros((n_resamples, N), dtype=float)
    np.put_along_axis(left, order[:, :n1], 1.0, axis=1)
    right = 1.0 - left

    perm_stats = scaling_factor * ((left @ D) * right).sum(axis=1)

    # +1 in numerator and denominator counts the observed statistic itself
    # (Jebur & Abushilah 2022, eq. 4.3).
    n_extreme = 1 + int(np.count_nonzero(perm_stats >= observed_stat))
    p_value = n_extreme / (n_resamples + 1)

    if verbose:
        print("Angular Randomization Test (ART) for Homogeneity")
        print("-------------------------------------------------")
        print("H0: The two samples come from the same population.")
        print("HA: The two samples do not come from the same population.")
        print("")
        print(f"Observed Test Statistic: {observed_stat:.5f}")
        print(f"P-value: {p_value:.5f} {significance_code(p_value)}")

    return AngularRandomisationTestResult(
        statistic=float(observed_stat),
        pval=float(p_value),
        method="randomization",
        n_resamples=n_resamples,
    )


#####################
## Goodness-of-Fit ##
#####################


def kuiper_test(
    alpha: np.ndarray,
    n_resamples: int = 9999,
    seed: SeedLike = 2046,
    verbose: bool = False,
    *,
    n_simulation: Optional[int] = None,
) -> KuiperTestResult:
    """
    Kuiper's test for Circular Uniformity.

    - H0: The data in the population are distributed uniformly around the circle.
    - H1: The data in the population are not distributed uniformly around the circle.

    This method is for ungrouped data.

    Parameters
    ----------

    alpha: np.array
        Angles in radian.

    n_resamples: int
        If ``0``, the p-value is the asymptotic series approximation. If ``>= 1``
        (default 9999), it is estimated from that many Monte-Carlo uniform samples.

    seed: SeedLike
        Seed used to initialize the random number generator for the Monte-Carlo
        p-value. Accepts integers, sequences of integers, ``numpy.random.Generator``,
        ``numpy.random.BitGenerator``, ``numpy.random.SeedSequence`` or ``None``.
        Defaults to 2046.

    n_simulation: int or None
        Deprecated alias for ``n_resamples`` (the old ``n_simulation=1`` meant asymptotic).

    Returns
    -------
    KuiperTestResult
        Dataclass containing the Kuiper statistic, p-value, ``method``
        ("asymptotic"|"monte_carlo"), and ``n_resamples``.

    Note
    ----
    Implementation from R package `Directional`
    https://rdrr.io/cran/Directional/src/R/kuiper.R
    """

    n_resamples = _resolve_n_resamples(n_resamples, n_simulation=n_simulation, has_asymptotic=True)
    if n_resamples < 0:
        raise ValueError("`n_resamples` must be a non-negative integer.")

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

    seed, verbose = _resolve_legacy_verbose(seed, verbose)

    if n_resamples == 0:
        # asymptotic p-value
        method = "asymptotic"
        m = (np.arange(1, 50, dtype=float)) ** 2
        a1 = 4 * m * Vo**2
        a2 = np.exp(-2 * m * Vo**2)
        b1 = 2 * (a1 - 1) * a2
        b2 = 8 * Vo / (3 * f) * m * (a1 - 3) * a2
        pval = float(np.sum(b1 - b2))
    else:
        method = "monte_carlo"
        rng = _init_rng(seed)
        uniforms = rng.uniform(low=0.0, high=2 * np.pi, size=(n, n_resamples))
        x = np.sort(uniforms, axis=0)
        Vs = np.array([compute_V(x[:, i])[0] for i in range(n_resamples)])
        pval = float((np.count_nonzero(Vs >= Vo) + 1) / (n_resamples + 1))

    if verbose:
        print("Kuiper's Test of Circular Uniformity")
        print("------------------------------------")
        print("H0: The sample is drawn from a circularly uniform distribution.")
        print("HA: The sample is not drawn from a circularly uniform distribution.")
        print("")
        print(f"Test Statistic: {Vo:.4f}")
        print(f"P-value ({method}): {pval:.5f} {significance_code(pval)}")

    return KuiperTestResult(V=float(Vo), pval=float(pval), method=method, n_resamples=n_resamples)


def _watson_u2_unit(z: np.ndarray) -> float:
    """Watson's U² for uniformity of PIT values ``z`` on [0,1) (sorted internally).

    ``U² = Σ (z_(i) − (2i−1)/(2n))² − n (z̄ − ½)² + 1/(12n)``. With ``z`` the
    probability-integral transform of the data through the hypothesised CDF
    (``α/2π`` for the uniform null, or the fitted von Mises CDF for the von Mises
    goodness-of-fit), this is the one-sample Watson statistic.
    """
    z = np.sort(np.asarray(z, dtype=float))
    n = z.size
    i = np.arange(1, n + 1, dtype=float)
    return float(np.sum((z - (2 * i - 1) / (2 * n)) ** 2) - n * (z.mean() - 0.5) ** 2 + 1 / (12 * n))


def watson_test(
    alpha: np.ndarray,
    dist: str = "uniform",
    n_resamples: int = 9999,
    seed: SeedLike = 2046,
    verbose: bool = False,
    *,
    n_simulation: Optional[int] = None,
) -> WatsonTestResult:
    """
    Watson's one-sample U² goodness-of-fit test.

    - H0: The sample is drawn from the null distribution (``dist``).
    - H1: The sample is not drawn from the null distribution.

    With ``dist="uniform"`` (default) this tests circular uniformity; with
    ``dist="vonmises"`` it tests goodness-of-fit to a von Mises distribution
    (parameters estimated from the data). This method is for ungrouped data.

    Parameters
    ----------

    alpha: np.array
        Angles in radian.

    dist: str
        Null distribution to test against: ``"uniform"`` (default) or ``"vonmises"``.

    n_resamples: int
        For ``dist="uniform"``: ``0`` gives the asymptotic series p-value, ``>= 1``
        (default 9999) a Monte-Carlo p-value from that many uniform samples. For
        ``dist="vonmises"``: the number of parametric-bootstrap resamples (refitting
        μ, κ on each); must be ``>= 1`` (there is no closed-form p-value).

    seed: SeedLike
        Seed used to initialize the random number generator for the Monte-Carlo
        p-value. Accepts integers, sequences of integers, ``numpy.random.Generator``,
        ``numpy.random.BitGenerator``, ``numpy.random.SeedSequence`` or ``None``.
        Defaults to 2046.

    n_simulation: int or None
        Deprecated alias for ``n_resamples`` (the old ``n_simulation=1`` meant asymptotic).

    Returns
    -------
    WatsonTestResult
        Dataclass containing the Watson U² statistic, p-value, ``method``
        (``"asymptotic"`` or ``"monte_carlo"`` for the uniform null;
        ``"parametric_bootstrap"`` for ``dist="vonmises"``), ``n_resamples``, the
        ``dist`` tested, and — for the von Mises GoF — the fitted ``mu``/``kappa``
        (``None`` for the uniform null).

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

    if dist not in ("uniform", "vonmises"):
        raise ValueError("`dist` must be 'uniform' or 'vonmises'.")

    n_resamples = _resolve_n_resamples(n_resamples, n_simulation=n_simulation, has_asymptotic=True)
    if n_resamples < 0:
        raise ValueError("`n_resamples` must be a non-negative integer.")

    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")
    n = alpha.size

    seed, verbose = _resolve_legacy_verbose(seed, verbose)

    if dist == "uniform":
        # PIT under the uniform null is simply α / 2π.
        U2o = _watson_u2_unit(alpha / (2 * np.pi))
        if n_resamples == 0:
            method = "asymptotic"
            m = np.arange(1, 51)
            pval = float(2 * sum((-1) ** (m - 1) * np.exp(-2 * m**2 * np.pi**2 * U2o)))
        else:
            method = "monte_carlo"
            rng = _init_rng(seed)
            uniforms = rng.uniform(low=0.0, high=2 * np.pi, size=(n, n_resamples))
            U2s = np.array(
                [_watson_u2_unit(uniforms[:, i] / (2 * np.pi)) for i in range(n_resamples)]
            )
            pval = float((np.count_nonzero(U2s >= U2o) + 1) / (n_resamples + 1))
        mu = kappa = None
    else:
        # von Mises GoF: PIT through the ML-fitted von Mises CDF, then a parametric
        # bootstrap (refit μ, κ on each simulated sample) — the null distribution of
        # U² depends on the unknown κ, so the parameters must be re-estimated each time.
        if n_resamples < 1:
            raise ValueError(
                "von Mises goodness-of-fit has no closed-form p-value; use n_resamples >= 1."
            )
        mu = float(circ_mean(alpha))
        kappa = float(circ_kappa(circ_r(alpha)))
        U2o = _watson_u2_unit(np.asarray(vonmises(mu=mu, kappa=kappa).cdf(alpha)))
        rng = _init_rng(seed)
        null = vonmises(mu=mu, kappa=kappa)
        count = 1  # the observed statistic counts itself
        for _ in range(n_resamples):
            sim = np.asarray(null.rvs(size=n, random_state=rng))
            mb = float(circ_mean(sim))
            kb = float(circ_kappa(circ_r(sim)))
            if _watson_u2_unit(np.asarray(vonmises(mu=mb, kappa=kb).cdf(sim))) >= U2o:
                count += 1
        pval = float(count / (n_resamples + 1))
        method = "parametric_bootstrap"

    if verbose:
        if dist == "uniform":
            print("Watson's One-Sample U2 Test of Circular Uniformity")
            print("--------------------------------------------------")
            print("H0: The sample is drawn from a circularly uniform distribution.")
            print("HA: The sample is not drawn from a circularly uniform distribution.")
        else:
            print("Watson's U2 Goodness-of-Fit Test for the von Mises Distribution")
            print("--------------------------------------------------------------")
            print("H0: The sample is drawn from a von Mises distribution.")
            print("HA: The sample is not drawn from a von Mises distribution.")
            print(f"Fitted parameters: mu = {mu:.4f}, kappa = {kappa:.4f}")
        print("")
        print(f"Test Statistic: {U2o:.4f}")
        print(f"P-value ({method}): {pval:.5f} {significance_code(pval)}")

    return WatsonTestResult(
        U2=float(U2o),
        pval=float(pval),
        method=method,
        n_resamples=n_resamples,
        dist=dist,
        mu=mu,
        kappa=kappa,
    )


def rao_spacing_test(
    alpha: np.ndarray,
    w: Union[np.ndarray, None] = None,
    kappa: float = 1000.0,
    n_resamples: int = 9999,
    seed: SeedLike = 2046,
    verbose: bool = False,
    *,
    n_simulation: Optional[int] = None,
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

    n_resamples: int
        Number of Monte-Carlo samples for the p-value (default 9999). Must be >= 1;
        this test has no analytic fallback.

    seed: SeedLike
        Seed used to initialize the random number generator for the Monte-Carlo
        p-value. Accepts integers, sequences of integers, ``numpy.random.Generator``,
        ``numpy.random.BitGenerator``, ``numpy.random.SeedSequence`` or ``None``.
        Defaults to 2046.

    n_simulation: int or None
        Deprecated alias for ``n_resamples``.

    Returns
    -------
    RaoSpacingTestResult
        Dataclass containing the Rao spacing statistic (degrees), p-value,
        ``method="monte_carlo"``, ``data_kind`` ("grouped"|"ungrouped"), and ``n_resamples``.

    Reference
    ---------
    Landler et al. (2019)
    https://movementecologyjournal.biomedcentral.com/articles/10.1186/s40462-019-0160-x
    """

    n_resamples = _resolve_n_resamples(n_resamples, n_simulation=n_simulation, has_asymptotic=False)
    if n_resamples <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

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
        data_kind = "grouped"
    else:
        expanded_alpha = alpha
        n = expanded_alpha.size
        data_kind = "ungrouped"

    seed, verbose = _resolve_legacy_verbose(seed, verbose)

    rng = _init_rng(seed)

    Uo = compute_U(expanded_alpha)
    if w is not None:  # noncontinuous / grouped data
        vm_dist = vonmises(mu=0.0, kappa=kappa)
        uniforms = rng.uniform(low=0.0, high=2 * np.pi, size=(n_resamples, n))
        snapped = np.floor(uniforms * m / (2 * np.pi)) * (2 * np.pi / m)
        noise = vm_dist.rvs(size=(n_resamples, n), random_state=rng)
        samples = angmod(snapped + noise)
        Us = np.array([compute_U(sample) for sample in samples])
    else:
        samples = rng.uniform(low=0.0, high=2 * np.pi, size=(n_resamples, n))
        Us = np.array([compute_U(sample) for sample in samples])

    counter = np.count_nonzero(Us >= Uo)
    pval = float((counter + 1) / (n_resamples + 1))

    if verbose:
        print("Rao's Spacing Test of Circular Uniformity")
        print("-----------------------------------------")
        print("H0: The sample is drawn from a circularly uniform distribution.")
        print("HA: The sample is not drawn from a circularly uniform distribution.")
        print("")
        print(f"Test Statistic: {np.rad2deg(Uo):.4f}°")
        print(f"P-value: {pval:.5f} {significance_code(pval)}")

    return RaoSpacingTestResult(
        statistic=float(np.rad2deg(Uo)),
        pval=float(pval),
        method="monte_carlo",
        data_kind=data_kind,
        n_resamples=n_resamples,
    )


def circ_range_test(
    alpha: np.ndarray,
    n_resamples: int = 0,
    seed: SeedLike = 2046,
    verbose: bool = False,
) -> CircularRangeTestResult:
    """
    Perform the Circular Range Test for uniformity.

    - **H0**: The data is uniformly distributed around the circle.
    - **H1**: The data is non-uniformly distributed (clustered).

    Parameters
    ----------
    alpha : np.ndarray
        Angles in radians. Values must already be wrapped into ``[-2π, 2π]``.
    n_resamples : int, optional
        If ``0`` (default), the p-value is the closed-form series. If ``>= 1``, it
        is estimated from that many Monte-Carlo uniform samples (a cross-check that
        floors at ``1/(n_resamples+1)`` in the deep tail).
    seed : SeedLike, optional
        Seed (or generator) for the Monte-Carlo p-value. Default 2046.
    verbose : bool, optional
        If ``True``, prints test details and results.

    Returns
    -------
    CircularRangeTestResult
        Dataclass containing the range statistic, the corresponding p-value,
        ``method`` (``"exact"`` for the closed-form series, or ``"monte_carlo"``
        when ``n_resamples >= 1``), and ``n_resamples``.

    Reference
    ---------
    P162, Section 7.2.3 of Jammalamadaka, S. Rao and SenGupta, A. (2001)
    """
    alpha = np.asarray(alpha, dtype=float)
    if alpha.size == 0:
        raise ValueError("`alpha` must contain at least one angle.")

    if np.any(np.abs(alpha) > 2 * np.pi + 1e-8):
        raise ValueError("`alpha` must be provided in radians within [-2π, 2π].")

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
    method = "exact"

    if n_resamples >= 1:
        # Smaller range = more clustered = more extreme, so negate for the upper-tail helper.
        rng = _init_rng(seed)
        p_value = _mc_uniform_pval(
            lambda s: -circ_range(s), n, -float(range_stat), n_resamples, rng
        )
        method = "monte_carlo"

    result = CircularRangeTestResult(
        range_stat=float(range_stat), pval=float(p_value), method=method, n_resamples=n_resamples
    )

    if verbose:
        range_deg = float(np.rad2deg(result.range_stat))
        print("Circular Range Test of Uniformity")
        print("---------------------------------")
        print("H0: The sample is uniformly distributed around the circle.")
        print("HA: The sample exhibits clustering (non-uniformity).")
        print("")
        print(f"Sample size: {n}")
        print(f"Range statistic: {result.range_stat:.5f} rad ({range_deg:.2f}°)")
        print(f"P-value: {result.pval:.5g} {significance_code(result.pval)}")

    return result


def binomial_test(
    alpha: np.ndarray,
    md: float,
    verbose: bool = False,
) -> BinomialTestResult:
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
    verbose : bool, optional
        If ``True``, prints test details and results.

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
        result = BinomialTestResult(pval=1.0, n_eff=0, n1=n1, n2=n2)
    else:
        # Compute p-value using binomial test
        n_min = int(min(n1, n2))
        pval = float(2 * binom.cdf(n_min, n_eff, 0.5))
        pval = min(pval, 1.0)
        result = BinomialTestResult(pval=pval, n_eff=n_eff, n1=n1, n2=n2)

    if verbose:
        print("Circular Binomial Test for Median Direction")
        print("--------------------------------------------")
        print(f"H0: Median direction equals {float(md):.5f} rad.")
        print("HA: Median direction differs from the hypothesized value.")
        print("")
        print(f"Effective sample size: {result.n_eff}")
        print(f"Counts below/above median: n1 = {result.n1}, n2 = {result.n2}")
        print(f"P-value: {result.pval:.5f} {significance_code(result.pval)}")

    return result


def concentration_test(
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    n_resamples: int = 0,
    seed: SeedLike = 2046,
    verbose: bool = False,
) -> ConcentrationTestResult:
    """
    Two-sample test for concentration (dispersion) equality in circular data.

    - **H0**: The two samples have the same concentration parameter.
    - **H1**: The two samples have different concentration parameters.

    Parameters
    ----------
    alpha1 : np.ndarray
        First sample of circular data (radians).
    alpha2 : np.ndarray
        Second sample of circular data (radians).
    n_resamples : int, optional
        If ``0`` (default), the p-value comes from Batschelet's parametric F-test
        (ported from MATLAB CircStat ``circ_ktest``; assumes von Mises samples with
        combined r̄ > 0.7). If ``>= 1``, a distribution-free permutation p-value is
        used instead: the deviations of each observation from its group mean are
        pooled and randomly reassigned to the two groups, and the two-sided ratio
        ``max(F, 1/F)`` is recomputed on each permutation (Pewsey et al. 2013, §7.4.3).
        Recommended when the von Mises / high-concentration assumptions fail.
    seed : SeedLike, optional
        Seed for the randomization RNG when ``n_resamples >= 1``. Defaults to 2046.
    verbose : bool, optional
        If ``True``, prints test details and results.

    Returns
    -------
    ConcentrationTestResult
        Dataclass with the F statistic, p-value, degrees of freedom, ``method``
        ("asymptotic"|"randomization"), and ``n_resamples``.

    References
    ----------
    Mardia, K. V. (1972). Statistics of Directional Data, eq. (6.3.39) & Example 6.15
        (high-concentration F-test; degrees of freedom n1-1, n2-1).
    Batschelet, E. (1980). Circular Statistics in Biology, Section 6.9, p. 122-124.
    Pewsey, Neuhäuser & Ruxton (2013), §7.4.3 (randomization version).
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

    # The parametric F-test assumes a high combined concentration; the randomization
    # version is precisely the remedy when that fails, so only warn for the F-test.
    rbar = (R1 + R2) / (n1 + n2)
    if n_resamples < 1 and rbar < 0.7:
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

    if n_resamples >= 1:
        def _kratio(groups: list[np.ndarray]) -> float:
            g0, g1 = groups
            num = (n2 - 1) * (n1 - n1 * circ_r(g0))
            den = (n1 - 1) * (n2 - n2 * circ_r(g1))
            if den <= 0 or num <= 0:
                return np.inf  # degenerate split -> treat as extreme
            ratio = num / den
            return max(ratio, 1.0 / ratio)

        # Pool the within-group deviations (location removed) and permute them.
        dev = np.concatenate([
            angmod(alpha1 - circ_mean(alpha1), bounds=[-np.pi, np.pi]),
            angmod(alpha2 - circ_mean(alpha2), bounds=[-np.pi, np.pi]),
        ])
        rng = _init_rng(seed)
        pval = _randomization_pval(
            _kratio, dev, [n1, n2], _kratio([dev[:n1], dev[n1:]]), n_resamples, rng
        )
        method = "randomization"
    else:
        # Two-sided parametric p-value (adjusting for F-stat symmetry). Statistic and
        # degrees of freedom follow Mardia (1972), eq. (6.3.39) & Example 6.15:
        #   F = [(n1-R1)/(n1-1)] / [(n2-R2)/(n2-1)] ~ F_{n1-1, n2-1}.
        # NB: MATLAB CircStat's `circ_ktest` uses df (n1, n2) here, which is a bug —
        # do not "fix" the (n-1) df below to match it.
        if f_stat >= 1:
            pval = float(min(2 * f.sf(f_stat, df1, df2), 1.0))
        else:
            pval = float(min(2 * f.sf(1 / f_stat, df2, df1), 1.0))
        method = "asymptotic"

    result = ConcentrationTestResult(
        f_stat=float(f_stat),
        pval=float(pval),
        df1=int(df1),
        df2=int(df2),
        method=method,
        n_resamples=n_resamples if method == "randomization" else 0,
    )

    if verbose:
        print("Concentration Equality Test")
        print("---------------------------")
        print("H0: Both samples share the same concentration parameter (κ).")
        print("HA: The samples have different concentration parameters.")
        print("")
        print(f"Sample sizes: n1 = {n1}, n2 = {n2}")
        print(
            f"F statistic: {result.f_stat:.5f} "
            f"(df1 = {result.df1}, df2 = {result.df2})"
        )
        print(f"P-value: {result.pval:.5f} {significance_code(result.pval)}")

    return result


def _rao_homogeneity_stats(groups: Sequence[np.ndarray]) -> tuple[float, float]:
    """Rao's two homogeneity statistics ``(H_polar, H_disp)`` for a list of groups.

    ``H_polar`` tests equality of mean directions, ``H_disp`` equality of dispersions
    (Rao 1967; Jammalamadaka & SenGupta 2001, §7.6.1). Note: both are functions of the
    per-group cos/sin means, so they are frame-dependent (not rotation invariant).
    """
    n = np.array([len(s) for s in groups])
    cos_means = np.array([np.mean(np.cos(s)) for s in groups])
    sin_means = np.array([np.mean(np.sin(s)) for s in groups])
    # Sample (co)variances with ddof=1 to match R's var()/cov().
    var_cos = np.array([np.var(np.cos(s), ddof=1) for s in groups])
    var_sin = np.array([np.var(np.sin(s), ddof=1) for s in groups])
    cov_cos_sin = np.array([np.cov(np.cos(s), np.sin(s), ddof=1)[0, 1] for s in groups])

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
    return float(H_polar), float(H_disp)


def rao_homogeneity_test(
    samples: Sequence[Any],
    alpha: float = 0.05,
    n_resamples: int = 0,
    seed: SeedLike = 2046,
    verbose: bool = False,
) -> RaoHomogeneityTestResult:
    """
    Perform Rao's test for homogeneity on multiple samples of angular data.

    - **Test 1**: Equality of Mean Directions (Polar Vectors)
    - **Test 2**: Equality of Dispersions

    Parameters
    ----------
    samples : sequence
        A sequence (one entry per group) of `Circular` objects or one-dimensional
        array-like radian samples.
    alpha : float, optional
        Significance level for the hypothesis test. Default is 0.05.
    n_resamples : int, optional
        If ``0`` (default), p-values come from Rao's large-sample χ² approximation.
        If ``>= 1``, that many randomization (permutation) resamples are used instead:
        under the homogeneity null the pooled angles are exchangeable, so they are
        permuted into the original group sizes and both statistics recomputed. This
        frees both tests from the large-sample assumption (Rao 1967 is explicitly a
        *large-sample* test); the trade-off is that the permutation reads the two
        statistics under a single joint "identically distributed" null.
    seed : int or numpy.random.Generator, optional
        Seed (or generator) for the randomization path. Default is 2046.
    verbose : bool, optional
        If ``True``, prints test details and decisions.

    Returns
    -------
    RaoHomogeneityTestResult
        Dataclass containing test statistics, p-values, and rejection flags, plus
        ``method`` (``"asymptotic"`` for Rao's large-sample χ², or
        ``"randomization"`` when ``n_resamples >= 1``) and ``n_resamples``.

    References
    ----------
    Jammalamadaka, S. Rao and SenGupta, A. (2001). Topics in Circular Statistics, Section 7.6.1.
    Rao, J.S. (1967). Large sample tests for the homogeneity of angular data, Sankhya, Ser, B., 28.
    """
    samples = _coerce_sample_arrays(samples)

    k = len(samples)  # Number of samples
    if k < 2:
        raise ValueError("At least two groups are required for the test.")
    n = np.array([len(s) for s in samples])  # Sample sizes
    if np.any(n < 2):
        raise ValueError("Each group must contain at least two observations.")

    H_polar, H_disp = _rao_homogeneity_stats(samples)

    df = k - 1  # Degrees of freedom
    if n_resamples >= 1:
        # Under the homogeneity null (groups identically distributed) the pooled angles
        # are exchangeable; permute them into the group sizes and recompute both stats.
        pooled = np.concatenate(samples)
        split_at = np.cumsum(n)[:-1]
        rng = _init_rng(seed)
        cnt_p = cnt_d = 1  # count the observed statistics themselves
        for _ in range(n_resamples):
            hp, hd = _rao_homogeneity_stats(np.split(rng.permutation(pooled), split_at))
            if hp >= H_polar:
                cnt_p += 1
            if hd >= H_disp:
                cnt_d += 1
        pval_polar = cnt_p / (n_resamples + 1)
        pval_disp = cnt_d / (n_resamples + 1)
        method = "randomization"
    else:
        pval_polar = float(chi2.sf(H_polar, df))
        pval_disp = float(chi2.sf(H_disp, df))
        method = "asymptotic"

    # Test decisions
    reject_polar = pval_polar < alpha
    reject_disp = pval_disp < alpha

    result = RaoHomogeneityTestResult(
        H_polar=float(H_polar),
        pval_polar=float(pval_polar),
        reject_polar=bool(reject_polar),
        H_disp=float(H_disp),
        pval_disp=float(pval_disp),
        reject_disp=bool(reject_disp),
        method=method,
        n_resamples=n_resamples if method == "randomization" else 0,
    )

    if verbose:
        print("Rao's Homogeneity Test")
        print("----------------------")
        print("Test 1 H0: All groups share the same mean direction.")
        print("Test 2 H0: All groups share the same dispersion.")
        print(f"P-value method: {result.method}", end="")
        print(f" ({result.n_resamples} resamples)" if result.method == "randomization" else "")
        print("")
        print(
            f"Mean directions: H = {result.H_polar:.5f}, "
            f"p = {result.pval_polar:.5f} {significance_code(result.pval_polar)}; "
            f"reject @ α={alpha}: {result.reject_polar}"
        )
        print(
            f"Dispersions:     H = {result.H_disp:.5f}, "
            f"p = {result.pval_disp:.5f} {significance_code(result.pval_disp)}; "
            f"reject @ α={alpha}: {result.reject_disp}"
        )

    return result


def change_point_test(
    alpha: np.ndarray,
    n_resamples: int = 0,
    seed: SeedLike = 2046,
    verbose: bool = False,
) -> ChangePointTestResult:
    """
    Perform a change point test for mean direction, concentration, or both.

    Parameters
    ----------
    alpha : np.ndarray
        Vector of angular measurements in radians (in sequence order).
    n_resamples : int, optional
        If ``>= 1``, permutation p-values for the rmax and tmax statistics are
        estimated from that many random reorderings of the sequence (exchangeable
        under H0 of no change point). Default ``0`` → no p-values.
    seed : SeedLike, optional
        Seed for the permutation RNG when ``n_resamples >= 1``. Defaults to 2046.
    verbose : bool, optional
        If ``True``, prints test details and summary statistics.

    Returns
    -------
    ChangePointTestResult
        Dataclass containing the change-point statistics and (when requested) the
        permutation p-values ``pval_r`` (mean direction) and ``pval_t`` (concentration).

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

    alpha = np.asarray(alpha, dtype=float)
    n = len(alpha)
    if n < 4:
        raise ValueError("Sample size must be at least 4 for change point test.")

    def _stats(a: np.ndarray) -> tuple:
        rho = est_rho(a)
        R1, R2, V = np.zeros(n), np.zeros(n), np.zeros(n)
        for k in range(1, n):
            R1[k - 1] = est_rho(a[:k]) * k
            R2[k - 1] = est_rho(a[k:]) * (n - k)
            if 2 <= k <= (n - 2):
                V[k - 1] = (k / n) * phi(R1[k - 1] / k) + ((n - k) / n) * phi(R2[k - 1] / (n - k))
        R1[-1] = rho * n
        R2[-1] = 0
        R_diff = R1 + R2 - rho * n
        # ``n >= 4`` is guaranteed by the guard above.
        Vt = V[1 : n - 2]
        return (
            float(rho),
            float(np.max(R_diff)),
            int(np.argmax(R_diff)),
            float(np.mean(R_diff)),
            float(np.max(Vt)),
            int(np.argmax(Vt)) + 1,
            float(np.mean(Vt)),
        )

    rho, rmax, k_r, rave, tmax, k_t, tave = _stats(alpha)

    pval_r = pval_t = None
    if n_resamples >= 1:
        # Under H0 (no change point) the sequence is exchangeable; permute the order
        # and count reorderings whose max statistic is at least the observed one.
        rng = _init_rng(seed)
        cnt_r = cnt_t = 1  # count the observed statistic itself
        for _ in range(n_resamples):
            perm = _stats(rng.permutation(alpha))
            if perm[1] >= rmax:
                cnt_r += 1
            if perm[4] >= tmax:
                cnt_t += 1
        pval_r = cnt_r / (n_resamples + 1)
        pval_t = cnt_t / (n_resamples + 1)

    result = ChangePointTestResult(
        n=int(n),
        rho=rho,
        rmax=rmax,
        k_r=k_r,
        rave=rave,
        tmax=tmax,
        k_t=k_t,
        tave=tave,
        pval_r=pval_r,
        pval_t=pval_t,
        n_resamples=n_resamples,
    )

    if verbose:
        print("Circular Change Point Test")
        print("--------------------------")
        print("H0: No change point in mean direction or concentration.")
        print("HA: A change point is present in the sequence.")
        print("")
        print(f"Sample size: {result.n}")
        print(f"Overall resultant length (ρ): {result.rho:.5f}")
        r_p = f" (p = {result.pval_r:.4f})" if result.pval_r is not None else ""
        t_p = f" (p = {result.pval_t:.4f})" if result.pval_t is not None else ""
        print(f"Max R statistic: {result.rmax:.5f} at k = {result.k_r}{r_p}")
        print(f"Average R statistic: {result.rave:.5f}")
        print(f"Max T statistic: {result.tmax:.5f} at k = {result.k_t}{t_p}")
        print(f"Average T statistic: {result.tave:.5f}")

    return result


def harrison_kanji_test(
    alpha: np.ndarray,
    idp: np.ndarray,
    idq: np.ndarray,
    inter: bool = True,
    fn: Optional[list] = None,
    verbose: bool = False,
) -> HarrisonKanjiTestResult:
    """
    Harrison-Kanji Test (Two-Way ANOVA) for Circular Data.

    Parameters
    ----------
    alpha : np.ndarray
        Angular measurements (radians).
    idp : np.ndarray
        Factor A identifiers for each observation.
    idq : np.ndarray
        Factor B identifiers for each observation.
    inter : bool, optional
        Whether to include the interaction term. Defaults to ``True``.
    fn : list, optional
        Names for the two factors. Defaults to ``["A", "B"]``.
    verbose : bool, optional
        If ``True``, prints test details and results.

    Returns
    -------
    HarrisonKanjiTestResult
        Dataclass containing `p_values` — the (factor A, factor B, interaction)
        p-value triple, where the interaction entry is NaN when ``inter=False`` —
        and `anova_table`, the assembled ANOVA table as a pandas DataFrame.
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
            pI = f.sf(FI, df_i, df_r)
        else:
            eff_r = n - sum(qr**2.0 / qn) - sum(pr**2.0 / pn) + tr**2 / n
            df_r = (p - 1) * (q - 1)
            ms_r = eff_r / df_r

            eff_i, df_i, ms_i, FI, pI = None, None, None, None, np.nan
            beta = 1

        F1 = beta * ms_1 / ms_r
        p1 = f.sf(F1, df_1, df_r)

        F2 = beta * ms_2 / ms_r
        p2 = f.sf(F2, df_2, df_r)

    else:  # Small kappa approximation
        rr = iv(1, kk) / iv(0, kk)
        kappa_factor = 2 / (1 - rr**2)

        chi1 = kappa_factor * (sum(pr**2.0 / pn) - tr**2 / n)
        df_1 = 2 * (p - 1)
        p1 = chi2.sf(chi1, df=df_1)

        chi2_val = kappa_factor * (sum(qr**2.0 / qn) - tr**2 / n)
        df_2 = 2 * (q - 1)
        p2 = chi2.sf(chi2_val, df=df_2)

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

    result = HarrisonKanjiTestResult(p_values=pval, anova_table=table)

    if verbose:
        p_a, p_b, p_inter = result.p_values

        def _fmt(p: Optional[float]) -> str:
            if p is None or (isinstance(p, float) and math.isnan(p)):
                return "n/a"
            return f"{p:.5f} {significance_code(p)}"

        print("Harrison-Kanji Two-Way Circular ANOVA")
        print("-------------------------------------")
        print(f"H0 ({fn[0]}): No difference in mean direction across factor {fn[0]}.")
        print(f"H0 ({fn[1]}): No difference in mean direction across factor {fn[1]}.")
        if inter:
            print("H0 (Interaction): No interaction between the two factors.")
        print("")
        print(f"{fn[0]} effect p-value: {_fmt(p_a)}")
        print(f"{fn[1]} effect p-value: {_fmt(p_b)}")
        if inter:
            print(f"Interaction p-value: {_fmt(p_inter)}")
        print("")
        print("ANOVA table (first rows):")
        print(result.anova_table.head())

    return result


def equal_kappa_test(samples: Sequence[Any], verbose: bool = False) -> EqualKappaTestResult:
    """
    Test for Homogeneity of Concentration Parameters (κ) in Circular Data.

    - **H₀**: All groups have the same concentration parameter (κ).
    - **H₁**: At least one group has a different κ.

    Parameters
    ----------
    samples : sequence
        A sequence (one entry per group) of `Circular` objects or one-dimensional
        array-like radian samples.
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
    arrays = _coerce_sample_arrays(samples)
    k = len(arrays)
    if k < 2:
        raise ValueError("At least two groups are required for the test.")

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
    p_value = chi2.sf(chi_square_stat, df)

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
    samples: Sequence[Any],
    alpha: float = 0.05,
    n_resamples: int = 0,
    seed: SeedLike = 2046,
    verbose: bool = False,
) -> CommonMedianTestResult:
    """
    Common Median Test (Equal Median Test) for Multiple Circular Samples.

    - **H₀**: All groups have the same circular median.
    - **H₁**: At least one group has a different circular median.

    Parameters
    ----------
    samples : sequence
        A sequence (one entry per group) of `Circular` objects or one-dimensional
        array-like radian samples.
    alpha : float, optional
        Significance level for deciding whether to reject the null hypothesis (default 0.05).
    n_resamples : int, optional
        If ``0`` (default), the p-value comes from the χ² approximation. If ``>= 1``, it is
        estimated from that many label randomizations (recommended for small samples;
        Pewsey et al. 2013, §7.3.2).
    seed : SeedLike, optional
        Seed for the randomization RNG when ``n_resamples >= 1``. Defaults to 2046.
    verbose : bool, optional
        If `True`, prints the test summary.

    Returns
    -------
    CommonMedianTestResult
        Dataclass containing the common median, test statistic, p-value, rejection flag,
        ``method`` ("asymptotic"|"randomization"), and ``n_resamples``.

    References
    ----------
    - Fisher, N. I. (1995). Statistical Analysis of Circular Data.
    - Pewsey, Neuhäuser & Ruxton (2013), §7.3.2 (randomization version).
    - `circ_cmtest` from MATLAB's Circular Statistics Toolbox.
    """

    # Number of groups
    if not (0 < alpha < 1):
        raise ValueError("`alpha` must be between 0 and 1.")

    arrays = _coerce_sample_arrays(samples)
    k = len(arrays)
    if k < 2:
        raise ValueError("At least two groups are required for the test.")

    # Sample sizes
    ns = np.array([arr.size for arr in arrays])
    N = int(np.sum(ns))  # Total number of observations

    # Compute the common circular median
    common_median = circ_median(np.hstack(arrays))

    # Per-observation indicator of falling below the (fixed) common median. The
    # common median and these indicators are invariant under relabelling, so the
    # randomization below only reshuffles the indicators into the group sizes.
    below = (circ_dist(np.hstack(arrays), common_median) < 0).astype(float)
    split_at = np.cumsum(ns)[:-1]
    m = np.array([g.sum() for g in np.split(below, split_at)])

    # Compute test statistic
    M = np.sum(m)
    if M == 0 or M == N:
        raise ValueError("All observations fall on the same side of the median; test is undefined.")

    def _pg(groups: list[np.ndarray]) -> float:
        mk = np.array([g.sum() for g in groups])
        return (N**2 / (M * (N - M))) * np.sum(mk**2 / ns) - (N * M) / (N - M)

    P = _pg(np.split(below, split_at))

    # Compute p-value
    df = k - 1
    if n_resamples >= 1:
        rng = _init_rng(seed)
        p_value = _randomization_pval(_pg, below, ns, P, n_resamples, rng)
        method = "randomization"
    else:
        p_value = float(chi2.sf(P, df))
        method = "asymptotic"
    reject = p_value < alpha

    # If the null hypothesis is rejected, return NaN for the median
    if reject:
        common_median = np.nan

    result = CommonMedianTestResult(
        common_median=float(common_median),
        statistic=float(P),
        pval=float(p_value),
        reject=bool(reject),
        method=method,
        n_resamples=n_resamples,
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
