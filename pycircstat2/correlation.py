from dataclasses import dataclass
from typing import Optional, Type, Union

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


def aacorr(
    a: Union[Type[Circular], np.ndarray],
    b: Union[Type[Circular], np.ndarray],
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

    if method == "fl":  # Fisher & Lee (1983)
        _corr = _aacorr_fl
    elif method == "js":  # Jammalamadaka & SenGupta (2001)
        _corr = _aacorr_js
    elif method == "nonparametric":
        _corr = _aacorr_np
    else:
        raise ValueError("Invalid method. Choose from 'fl', 'js', 'nonparametric'.")

    result = _corr(a, b, test, strict)

    if test:
        return result
    else:
        return result.r


def _aacorr_fl(
    a: Union[Type[Circular], np.ndarray],
    b: Union[Type[Circular], np.ndarray],
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

    if isinstance(a, Circular):
        a = a.alpha
    if isinstance(b, Circular):
        b = b.alpha
    assert len(a) == len(b), "`a` and `b` must be the same length."

    def _corr(a, b):
        aij = np.triu(a[:, None] - a).flatten()
        bij = np.triu(b[:, None] - b).flatten()
        num = np.sum(np.sin(aij) * np.sin(bij))
        den = np.sqrt(np.sum(np.sin(aij) ** 2) * np.sum(np.sin(bij) ** 2))
        raa = num / den
        return raa

    r = _corr(a, b)

    if test:
        # jackknife test (Fingleton, 1989)
        # compute raa an additional n times, each time leaving out one pair of observations
        n = len(a)
        raas = [_corr(np.delete(a, i), np.delete(b, i)) for i in range(n)]
        m_raas = np.mean(raas)
        s2_raas = np.var(raas, ddof=1)
        z = norm.ppf(0.975)
        lb = n * r - (n - 1) * m_raas - z * np.sqrt(s2_raas / n)
        ub = n * r - (n - 1) * m_raas + z * np.sqrt(s2_raas / n)

        reject = ~(lb <= 0 <= ub)

        return CorrelationResult(r=r, reject_null=reject)
    else:
        return CorrelationResult(r=r)


def _aacorr_js(
    a: Union[Type[Circular], np.ndarray],
    b: Union[Type[Circular], np.ndarray],
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

    if isinstance(a, Circular):
        if strict:
            assert a.mean_pval < 0.05, "Data `a` is uniformly distributed."
        a_mean = a.mean
        a = a.alpha
    else:
        a_mean = circ_mean(a)

    if isinstance(b, Circular):
        if strict:
            assert b.mean_pval < 0.05, "Data `b` is uniformly distributed."
        b_mean = b.mean
        b = b.alpha
    else:
        b_mean = circ_mean(b)

    abar = a - a_mean
    bbar = b - b_mean
    Sa = np.sin(abar)
    Sb = np.sin(bbar)
    num = np.sum(Sa * Sb)
    den = np.sqrt(np.sum(Sa**2) * np.sum(Sb**2))

    r = num / den

    if test:
        n = len(a)
        l20 = np.mean(Sa**2)
        l02 = np.mean(Sb**2)
        l22 = np.mean(Sa**2 * Sb**2)
        test_stat = np.sqrt(n) * np.sqrt(l20 * l02 / l22) * r
        p_val = 2 * (1 - norm.cdf(np.abs(test_stat)))

        return CorrelationResult(r=r, p_value=p_val, test_stat=test_stat)
    else:
        return CorrelationResult(r=r)


def _aacorr_np(
    a: Union[Type[Circular], np.ndarray],
    b: Union[Type[Circular], np.ndarray],
    test: bool,
    strict: bool,
) -> CorrelationResult:
    """Nonparametric angular-angular correlation."""

    if isinstance(a, Circular):
        a = a.alpha
    if isinstance(b, Circular):
        b = b.alpha
    assert len(a) == len(b), "`a` and `b` must be the same length."

    n = len(a)
    C = 2 * np.pi / n

    rank_a = rankdata(a)
    rank_b = rankdata(b)
    rank_diff = rank_a - rank_b
    rank_sum = rank_a + rank_b

    r1 = (
        np.sum(np.cos(C * rank_diff)) ** 2 + np.sum(np.sin(C * rank_diff)) ** 2
    ) / n**2
    r2 = (np.sum(np.cos(C * rank_sum)) ** 2 + np.sum(np.sin(C * rank_sum)) ** 2) / n**2

    r = r1 - r2

    n = len(a)
    reject = (n - 1) * r > 2.99 + 2.16 / n
    return CorrelationResult(r=r, reject_null=reject)


def alcorr(
    a: Union[Type[Circular], np.ndarray],
    x: np.ndarray,
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

    if isinstance(a, Circular):
        a = a.alpha
    assert len(a) == len(x), "`a` and `x` must be the same length."

    n = len(a)

    rxc = np.corrcoef(np.cos(a), x)[0, 1]
    rxs = np.corrcoef(x, np.sin(a))[0, 1]
    rcs = np.corrcoef(np.sin(a), np.cos(a))[0, 1]

    num = rxc**2 + rxs**2 - 2 * rxc * rxs * rcs
    den = 1 - rcs**2
    r = np.sqrt(num / den)

    pval = 1 - chi2(df=2).cdf(n * r**2)

    return CorrelationResult(r=r, p_value=pval)
