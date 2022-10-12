from typing import Type, Union

import numpy as np
from scipy.stats import norm

from pycircstat2 import Circular


def aacorr(
    a: Union[Type[Circular], np.ndarray],
    b: Union[Type[Circular], np.ndarray],
    method: str = "fl",
    test: bool = False,
) -> tuple:

    if method == "fl":  # Fisher & Lee (1983)
        _corr = _aacorr_fl

    elif method == "js":  # Jammalamadaka & SenGupta (2001)
        _corr = _aacorr_js

    r = _corr(a, b)

    if test:

        if isinstance(a, Circular):
            a = a.alpha
        if isinstance(b, Circular):
            b = b.alpha
        assert len(a) == len(b), "`a` and `b` must be the same length."

        # jackknife test (Fingleton, 1989)
        n = len(a)
        raas = [_corr(np.delete(a, i), np.delete(b, i)) for i in range(n)]
        m_raas = np.mean(raas)
        s2_raas = np.var(raas, ddof=1)
        z = norm.ppf(0.975)
        lb = n * r - (n - 1) * m_raas - z * np.sqrt(s2_raas / n)
        ub = n * r - (n - 1) * m_raas + z * np.sqrt(s2_raas / n)

        reject = ~(lb <= 0 <= ub)

        return r, reject
    else:
        return r


def _aacorr_fl(
    a: Union[Type[Circular], np.ndarray],
    b: Union[Type[Circular], np.ndarray],
) -> float:

    """Angular-Angular Correlation based on Fisher & Lee (1983)

    Parameters
    ----------
    a: Circular or np.ndarray
        Angles in radian
    b: Circular or np.ndarray
        Angles in radian

    Return
    ------
    raa: float
        correlation coefficient.

    Note
    ----
    Implementation of Example 27.20 (Zar, 2010).
    """

    if isinstance(a, Circular):
        a = a.alpha
    if isinstance(b, Circular):
        b = b.alpha
    assert len(a) == len(b), "`a` and `b` must be the same length."

    aij = np.triu(a[:, None] - a).flatten()
    bij = np.triu(b[:, None] - b).flatten()
    num = np.sum(np.sin(aij) * np.sin(bij))
    den = np.sqrt(np.sum(np.sin(aij) ** 2) * np.sum(np.sin(bij) ** 2))
    raa = num / den

    return raa


def _aacorr_js(
    a: Union[Type[Circular], np.ndarray],
    b: Union[Type[Circular], np.ndarray],
) -> float:

    """Angular-Angular Correlation based on Jammalamadaka & SenGupta (2001)

    Parameters
    ----------
    a: Circular or np.ndarray
        Angles in radian
    b: Circular or np.ndarray
        Angles in radian

    Return
    ------
    raa: float
        correlation coefficient.
    """

    if isinstance(a, Circular):
        assert a.mean_pval < 0.01, "Data `a` is uniformly distributed."
        a = a.alpha
    if isinstance(b, Circular):
        assert b.mean_pval < 0.01, "Data `b` is uniformly distributed."
        b = b.alpha

    abar = a.alpha - a.mean
    bbar = b.alpha - b.mean
    num = np.sum(np.sin(abar) * np.sin(bbar))
    den = np.sqrt(np.sum(np.sin(abar) ** 2) * np.sum(np.sin(bbar) ** 2))

    raa = num / den
    return raa


def alcorr(
    a: Union[Type[Circular], np.ndarray],
    x: np.ndarray,
) -> float:

    """Angular-Linear Correlation based on Mardia (1972)

    Parameters
    ----------
    a: Circular or np.ndarray
        Angles in radian
    x: np.ndarray
        Linear variable

    Return
    ------
    ral: float
        correlation coefficient.

    Note
    ----
    Implementation of Example 27.21 (Zar, 2010).
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
    ral = np.sqrt(num / den)

    return ral
