from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import chi2, norm, rankdata

from .base import Circular
from .descriptive import circ_mean


@dataclass
class CorrelationResult:
    r: float
    p_value: Optional[float] = None
    reject_null: Optional[bool] = None
    test_stat: Optional[float] = None


def circ_corrcc(
    a: Union[Circular, np.ndarray, Sequence[float]],
    b: Union[Circular, np.ndarray, Sequence[float]],
    method: str = "fl",
    test: bool = False,
    strict: bool = True,
) -> Union[float, CorrelationResult]:
    r"""
    Angular-Angular / Spherical Correlation.

    Three methods are available:

    - 'fl' (Fisher & Lee, 1983): T-linear association. The correlation coefficient

    $$
    r = \frac{\sum_{i=1}^{n-1}\sum_{j=i+1}^{n} \sin(a_{ij}) \sin(b_{ij})}{\sqrt{\sum_{i=1}^{n-1}\sum_{j=i+1}^{n} \sin^2(a_{ij}) \sum_{i=1}^{n-1}\sum_{j=i+1}^{n} \sin^2(b_{ij})}}
    $$

    - 'js' (Jammalamadaka & SenGupta, 2001)

    $$
    r = \frac{\sum \sin(a_i - \bar{a}) \sin(b_i - \bar{b})}{\sqrt{\sum \sin^2(a_i - \bar{a}) \sum \sin^2(b_i - \bar{b})}}
    $$

    - 'nonparametric'

    $$
    r = \frac{\sum \cos(C \cdot \text{rankdiff})^2 + \sum \sin(C \cdot \text{rankdiff})^2}{n^2} - \frac{\sum \cos(C \cdot \text{ranksum})^2 + \sum \sin(C \cdot \text{ranksum})^2}{n^2}
    $$

    , where $C = 2\pi / n$ and $\text{rankdiff} = \text{rank}_a - \text{rank}_b$ and $\text{ranksum} = \text{rank}_a + \text{rank}_b$.


    Parameters
    ----------
    a: Circular or np.ndarray
        Angles in radian
    b: Circular or np.ndarray
        Angles in radian
    method: str
        - 'fl' (Fisher & Lee, 1983): T-linear association. The correlation coefficient
          is computed as:
        - 'js' (Jammalamadaka & SenGupta, 2001)
        - 'nonparametric'
    test: bool
        Return significant test results.
    strict: bool
        Strict mode. If True, raise an error when mean direction is
        not significant. Only for method="js" (Jammalamadaka & SenGupta, 2001).

    Returns
    -------
    r: float
        Correlation coefficient.
    reject: bool
        Return significant test if `test` is set to True.
    """

    method = method.lower()
    if method == "fl":  # Fisher & Lee (1983)
        _corr = _circ_corrcc_fl
    elif method == "js":  # Jammalamadaka & SenGupta (2001)
        _corr = _circ_corrcc_js
    elif method == "nonparametric":
        _corr = _circ_corrcc_np
    else:
        raise ValueError("Invalid method. Choose from 'fl', 'js', or 'nonparametric'.")

    result = _corr(a, b, test, strict)

    return result if test else result.r


def _coerce_angles(
    data: Union[Circular, np.ndarray, Sequence[float]],
) -> Tuple[np.ndarray, Optional[float]]:
    """Return angle array (in radians) and mean p-value if available."""

    if isinstance(data, Circular):
        return np.asarray(data.alpha, dtype=float), getattr(data, "mean_pval", None)

    array = np.asarray(data, dtype=float)
    if array.ndim == 0:
        raise ValueError("Angles must be one-dimensional; got scalar input.")
    if array.ndim != 1:
        raise ValueError("Angles must be provided as a 1-D sequence.")
    if array.size == 0:
        raise ValueError("Angles must contain at least one element.")
    return array, None


def _circ_corrcc_fl(
    a: Union[Circular, np.ndarray, Sequence[float]],
    b: Union[Circular, np.ndarray, Sequence[float]],
    test: bool,
    strict: bool,
) -> CorrelationResult:
    """Angular-Angular Correlation based on Fisher & Lee (1983)

    Also known as Circular-Circular or T-linear association (Fisher, 1993).

    Parameters
    ----------
    a: Circular or np.ndarray
        Angles in radian
    b: Circular or np.ndarray
        Angles in radian

    Returns
    -------
    CorrelationResult

    References
    ----------
    P657-658, Section 27.15(a), Example 27.20 (Zar, 2010).
    """

    a_alpha, _ = _coerce_angles(a)
    b_alpha, _ = _coerce_angles(b)

    if a_alpha.size != b_alpha.size:
        raise ValueError("`a` and `b` must have the same number of samples.")
    if a_alpha.size < 2:
        raise ValueError("At least two paired observations are required.")

    def _corr(a, b):
        diff_a = np.sin(np.subtract.outer(a, a)[np.triu_indices(len(a), k=1)])
        diff_b = np.sin(np.subtract.outer(b, b)[np.triu_indices(len(b), k=1)])
        num = np.sum(diff_a * diff_b)
        den = np.sqrt(np.sum(diff_a**2) * np.sum(diff_b**2))
        if np.isclose(den, 0.0):
            raise ValueError("Degenerate data produced zero variance in pairwise differences.")
        return num / den

    r = _corr(a_alpha, b_alpha)

    if test:
        # jackknife test (Upton & Fingleton, 1989)
        # compute raa an additional n times, each time leaving out one pair of observations
        n = len(a_alpha)
        raas = np.empty(n)
        for i in range(n):
            raas[i] = _corr(np.delete(a_alpha, i), np.delete(b_alpha, i))
        m_raas = np.mean(raas)
        s2_raas = np.var(raas, ddof=1)
        z = norm.ppf(0.975)
        lb = n * r - (n - 1) * m_raas - z * np.sqrt(s2_raas / n)
        ub = n * r - (n - 1) * m_raas + z * np.sqrt(s2_raas / n)

        reject = ~(lb <= 0 <= ub)

        return CorrelationResult(r=r, reject_null=reject)
    else:
        return CorrelationResult(r=r)


def _circ_corrcc_js(
    a: Union[Circular, np.ndarray, Sequence[float]],
    b: Union[Circular, np.ndarray, Sequence[float]],
    test: bool,
    strict: bool,
) -> CorrelationResult:
    """Implementation of Angular-Angular Correlation
    in R.Circular.

    Parameters
    ----------
    a: Circular or np.ndarray
        Angles in radian
    b: Circular or np.ndarray
        Angles in radian
    strict: bool
        if True, raise an error when mean direction is
        not significant.

    Returns
    -------
    raa: float
        correlation coefficient.

    References
    ----------
    Jammalamadaka & SenGupta (2001)
    """

    a_alpha, a_pval = _coerce_angles(a)
    b_alpha, b_pval = _coerce_angles(b)

    if a_alpha.size != b_alpha.size:
        raise ValueError("`a` and `b` must have the same number of samples.")
    if a_alpha.size < 2:
        raise ValueError("At least two paired observations are required.")

    if strict and a_pval is not None and a_pval >= 0.05:
        raise ValueError("Sample `a` appears uniform (mean_pval ≥ 0.05).")
    if strict and b_pval is not None and b_pval >= 0.05:
        raise ValueError("Sample `b` appears uniform (mean_pval ≥ 0.05).")

    a_mean = float(circ_mean(a_alpha))
    b_mean = float(circ_mean(b_alpha))

    abar = a_alpha - a_mean
    bbar = b_alpha - b_mean
    Sa = np.sin(abar)
    Sb = np.sin(bbar)
    num = np.sum(Sa * Sb)
    den = np.sqrt(np.sum(Sa**2) * np.sum(Sb**2))
    if np.isclose(den, 0.0):
        raise ValueError("Degenerate data produced zero variance around the mean direction.")
    r = num / den

    if test:
        n = len(a_alpha)
        l20 = np.mean(Sa**2)
        l02 = np.mean(Sb**2)
        l22 = np.mean(Sa**2 * Sb**2)
        if np.isclose(l22, 0.0):
            raise ValueError("Degenerate data caused division by zero in variance term.")
        test_stat = np.sqrt(n) * np.sqrt(l20 * l02 / l22) * r
        p_val = 2 * (1 - norm.cdf(np.abs(test_stat)))

        return CorrelationResult(r=r, p_value=p_val, test_stat=test_stat)
    else:
        return CorrelationResult(r=r)


def _circ_corrcc_np(
    a: Union[Circular, np.ndarray, Sequence[float]],
    b: Union[Circular, np.ndarray, Sequence[float]],
    test: bool,
    strict: bool,
) -> CorrelationResult:
    """Nonparametric angular-angular correlation."""

    a_alpha, _ = _coerce_angles(a)
    b_alpha, _ = _coerce_angles(b)

    if a_alpha.size != b_alpha.size:
        raise ValueError("`a` and `b` must have the same number of samples.")

    n = a_alpha.size
    if n < 3:
        raise ValueError("At least three paired observations are required for the nonparametric test.")
    C = 2 * np.pi / n

    rank_a = rankdata(a_alpha)
    rank_b = rankdata(b_alpha)
    rank_diff = rank_a - rank_b
    rank_sum = rank_a + rank_b

    r1 = (
        np.sum(np.cos(C * rank_diff)) ** 2 + np.sum(np.sin(C * rank_diff)) ** 2
    ) / n**2
    r2 = (np.sum(np.cos(C * rank_sum)) ** 2 + np.sum(np.sin(C * rank_sum)) ** 2) / n**2

    r = r1 - r2

    reject = (n - 1) * r > 2.99 + 2.16 / n
    return CorrelationResult(r=float(r), reject_null=bool(reject))


def circ_corrcl(
    a: Union[Circular, np.ndarray, Sequence[float]],
    x: Union[np.ndarray, Sequence[float]],
) -> CorrelationResult:
    r"""Angular-Linear / Cylindrical Correlation based on Mardia (1972).

    Also known as Linear-circular or C-linear association (Fisher, 1993).

    $$
    r = \sqrt{\frac{r_{xc}^2 + r_{xs}^2 - 2r_{xc}r_{xs}r_{cs}}{1 - r_{cs}^2}}
    $$

    where $r_{xc}$, $r_{xs}$, and $r_{cs}$ are the correlation coefficients between
    $\cos(a)$ and $x$, $x$ and $\sin(a)$, and $\sin(a)$ and $\cos(a)$, respectively.

    Parameters
    ----------
    a: Circular or np.ndarray
        Angles in radian
    x: np.ndarray
        Linear variable

    Returns
    -------
    ral: float
        correlation coefficient.
    pval: float

    Reference
    ----
    P658-659, Section 27.15(b) of Example 27.21 (Zar, 2010).
    """

    a_alpha, _ = _coerce_angles(a)
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("`x` must be a one-dimensional array.")
    if a_alpha.size != x_arr.size:
        raise ValueError("`a` and `x` must be the same length.")
    if a_alpha.size < 3:
        raise ValueError("At least three paired observations are required.")

    n = a_alpha.size

    cos_a = np.cos(a_alpha)
    sin_a = np.sin(a_alpha)

    rxc = np.corrcoef(cos_a, x_arr)[0, 1]
    rxs = np.corrcoef(x_arr, sin_a)[0, 1]
    rcs = np.corrcoef(sin_a, cos_a)[0, 1]

    num = rxc**2 + rxs**2 - 2 * rxc * rxs * rcs
    den = 1 - rcs**2
    if np.isclose(den, 0.0):
        raise ValueError("Degenerate data produced division by zero in denominator.")
    r = np.sqrt(max(num / den, 0.0))

    pval = 1 - chi2(df=2).cdf(n * r**2)

    return CorrelationResult(r=float(r), p_value=float(pval))
