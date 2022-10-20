from typing import Tuple, Union

import numpy as np
from scipy.stats import chi2, norm

from .utils import angrange


def circ_mean(
    alpha: np.ndarray,
    w: Union[np.ndarray, None] = None,
) -> tuple:

    """
    Circular mean (m) and resultant vector length (r).

    Parameters
    ----------
    alpha: np.array (n, )
        Angles in radian.
    w: np.array (n,)
        Frequencies or weights

    Returns
    -------
    m: float or NaN
        Circular mean
    r: float
        Resultant vector length

    Note
    ----
    Implementation of Example 26.5 (Zar, 2010)
    """

    if w is None:
        w = np.ones_like(alpha)

    n = np.sum(w)
    Cbar = np.sum(w * np.cos(alpha)) / n
    Sbar = np.sum(w * np.sin(alpha)) / n

    # mean resultant vecotr length
    r = np.sqrt(Cbar**2 + Sbar**2)

    # angular mean
    if np.isclose(r, 0):
        m = np.nan
    else:
        if Cbar != 0 and Sbar != 0:
            m = np.arctan2(Sbar, Cbar)
        else:
            m = np.arccos(Cbar / r)

    return angrange(m), r


def circ_moment(
    alpha: np.ndarray,
    w: Union[np.ndarray, None] = None,
    p: int = 1,
    mean=None,
    centered=False,
) -> tuple:
    """
    Circular moment. When p=1, it's the same as circular mean.

    Parameters
    ----------
    alpha: np.array (n, )
        Angles in radian.
    w: None or np.array (n,)
        Frequencies or weights
    p: int
        The p-th moment to be computed.
    mean: float
        Precomputed circular mean.
    centered: bool
        If centered is True, substract mean from the alpha.

    Returns
    -------
    mp: complex
        Circular moment

    Note
    ----
    Implementation of Equation 2.24 (Fisher, 1993)
    """

    if w is None:
        w = np.ones_like(alpha)

    if centered is True and mean is None:
        mean = circ_mean(alpha, w)[0]
    elif centered is False and mean is None:
        mean = 0.0
    else:
        pass

    n = np.sum(w)
    Cbar = np.sum(w * np.cos(p * (alpha - mean))) / n
    Sbar = np.sum(w * np.sin(p * (alpha - mean))) / n

    mp = Cbar + 1j * Sbar

    return (
        mp,
        angrange(np.angle(mp)),
        mp.real,
    )


def circ_dispersion(
    alpha: np.ndarray,
    w: Union[np.ndarray, None] = None,
    mean=None,
) -> float:

    r"""
    Sample Circular Dispersion, defined by Fisher eq(2.28):

       \hat\delta = (1 - \hat\rho_{2})/(2 \hat\rho_{1}^{2})

    Parameters
    ----------

    alpha: np.array, (n, )
        Angles in radian.
    w: None or np.array, (n)
        Frequencies or weights
    mean: None or float
        Precomputed circular mean.

    Returns
    -------
    dispersion: float
        Sample Circular Dispersion

    Note
    ----
    Implementation of Equation 2.28 (Fisher, 1993)

    """

    if w is None:
        w = np.ones_like(alpha)

    r1 = circ_moment(alpha=alpha, w=w, p=1, mean=mean, centered=True)[2]  # eq(2.26)
    r2 = circ_moment(alpha=alpha, w=w, p=2, mean=mean, centered=True)[2]  # eq(2.27)

    dispersion = (1 - r2) / (2 * r1**2)  # eq(2.28)

    return dispersion


def circ_skewness(alpha: np.ndarray, w: Union[np.ndarray, None] = None) -> float:

    r"""
    Circular skewness, as defined by Fisher eq(2.29):

        \hat s = [\hat\rho_2 \sin(\hat\mu_2 - 2 \hat\mu_1)] / (1 - \hat\rho_1)^{\frac{3}{2}}

    Parameters
    ----------

    alpha: np.array, (n, )
        Angles in radian.
    w: None or np.array, (n)
        Frequencies or weights

    Returns
    -------
    skewness: float
        Circular Skewness

    Note
    ----
    Implementation of Equation 2.29 (Fisher, 1993)
    """

    if w is None:
        w = np.ones_like(alpha)

    u1, r1 = circ_mean(alpha=alpha, w=w)

    u2 = circ_moment(alpha=alpha, w=w, p=2, mean=None, centered=False)[1]  # eq(2.27)
    r2 = circ_moment(alpha=alpha, w=w, p=2, mean=u1, centered=True)[2]

    skewness = (r2 * np.sin(u2 - 2 * u1)) / (1 - r1) ** 1.5

    return skewness


def circ_kurtosis(alpha: np.ndarray, w: Union[np.ndarray, None] = None) -> float:

    r"""
    Circular kurtosis, as defined by Fisher eq(2.30):

        \hat k = [\hat\rho_2 \cos(\hat\mu_2 - 2 \hat\mu_1) - \hat\rho_1^4] / (1 - \hat\rho_1)^{2}

    Parameters
    ----------

    alpha: np.array, (n, )
        Angles in radian.
    w: None or np.array, (n)
        Frequencies or weights

    Returns
    -------
    kurtosis: float
        Circular Kurtosis

    Note
    ----
    Implementation of Equation 2.30 (Fisher, 1993)
    """

    if w is None:
        w = np.ones_like(alpha)

    u1, r1 = circ_mean(alpha=alpha, w=w)

    u2 = circ_moment(alpha=alpha, w=w, p=2, mean=None, centered=False)[1]  # eq(2.27)
    r2 = circ_moment(alpha=alpha, w=w, p=2, mean=u1, centered=True)[2]

    kurtosis = (r2 * np.cos(u2 - 2 * u1) - r1**4) / (1 - r1) ** 2

    return kurtosis


def circ_std(
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    r: Union[float, None] = None,
    bin_size: Union[float, None] = None,
) -> tuple:

    """
    Mean angular deviation (s) & Circular standard deviation (s0).

    Parameters
    ----------
    alpha: np.array (n, ) or None
        Angles in radian.
    w: np.array (n,) or None
        Frequencies or weights
    r: float or None
        Resultant vector length
    bin_size: float
        Interval size of grouped data.
        Needed for correcting biased r.

    Returns
    -------
    s: float or NaN
        Mean angular deviation.
    s0: float
        Circular standard deviation.

    Note
    ----
    Implementation of Equation 26.15-16/20-21 (Zar, 2010)
    """

    if r is None:
        assert isinstance(alpha, np.ndarray) and isinstance(
            w, np.ndarray
        ), "If `r` is None, then `alpha` and `w` are needed."
        r = circ_mean(alpha, w)[1]

    if bin_size is None:
        assert isinstance(alpha, np.ndarray) and isinstance(
            w, np.ndarray
        ), "If `bin_size` is None, then `alpha` and `w` are needed."
        if (w == w[0]).all():  #
            bin_size = 0
        else:
            bin_size = np.diff(alpha).min()

    ## corrected r if data are grouped.
    if bin_size == 0:
        rc = r
    else:
        c = bin_size / 2 / np.sin(bin_size / 2)  # eq(26.16)
        rc = r * c  # eq(26.15)

    # mean angular deviation
    s = np.sqrt(2 * (1 - rc))  # eq(26.20)
    # circular standard deviation
    s0 = np.sqrt(-2 * np.log(rc))  # eq(26.21)

    return (s, s0, rc)


def circ_median(
    alpha: np.array,
    w: Union[np.array, None] = None,
    grouped: bool = False,
    method: str = "deviation",
) -> float:

    """
    Circular median.

    Parameters
    ----------
    alpha: np.array (n, )
        Angles in radian.
    w: np.array (n,) or None
        Frequencies or weights
    grouped: bool
        Grouped data or not.

    Return
    ------
    median: float or NaN
    """

    if w is None:
        w = np.ones_like(alpha)

    # grouped data
    if grouped:
        median = _circ_median_groupped(alpha, w)
    # ungrouped data
    else:
        # find which data point that can divide the dataset into two half
        if method == "count":
            median = _circ_median_count(alpha)
        # find the angle that has the minimal mean deviation
        elif method == "deviation":
            median = _circ_median_mean_deviation(alpha, w)

    return angrange(median)


def _circ_median_groupped(
    alpha: np.array,
    w: Union[np.array, None] = None,
) -> float:

    n = np.sum(w)  # sample size
    n_bins = len(alpha)  # number of intervals
    bin_size = np.diff(alpha).min()

    # median for groupped data operated on upper bound of bins
    alpha_ub = alpha + bin_size / 2
    alpha_rotated = angrange(alpha_ub[:, None] - alpha_ub)
    right = np.logical_and(alpha_rotated >= 0.0, alpha_rotated <= np.round(np.pi, 5))
    halfcircle_right = np.array(
        [np.sum(np.roll(w, -1)[right[:, i]]) for i in range(len(alpha))]
    )
    halfcircle_left = n - halfcircle_right

    if n_bins % 2 != 0:
        offset = np.roll(w, 2) / 2  # remove half of the previous bin freq
        halfcircle_left = halfcircle_left - offset

    # find where half-freq located.
    halffreq = np.round(n / 2, 5)
    halfcircle_range = np.round(
        np.vstack([halfcircle_left, np.roll(halfcircle_left, -1)]).T, 5
    )
    idx = np.where(
        np.logical_and(
            halffreq >= halfcircle_range[:, 0], halffreq <= halfcircle_range[:, 1]
        ),
    )[0]

    # if number of potential median is the same as the number of data points,
    # meaning that the data is more or less uniformly distributed. Retrun Nan.
    if len(idx) == len(halfcircle_range):
        median = np.nan
    # get base interval, lower and upper freq
    elif len(idx) == 1:
        freq_lower, freq_upper = np.sort(halfcircle_range[idx][0])
        base = alpha_ub[idx][0]
        ratio = (halffreq - freq_lower) / (freq_upper - freq_lower)
        median = base + bin_size * ratio
    else:
        # remove empty bins.
        select = halfcircle_range[idx, 0] != halfcircle_range[idx, 1]
        # find outer bounds.
        lower = alpha_ub[idx[select][0]] + bin_size
        upper = alpha_ub[idx[select][1]]
        # circular mean of two opposite points will always be NaN.
        # in this case, we use the inner bounds instead of the outer.
        if np.isclose(upper - lower, np.pi):
            lower = lower - bin_size
            upper = upper + bin_size
        beta = np.array([lower, upper])
        median = circ_mean(alpha=beta)[0]

    return median


def _circ_median_count(alpha: np.ndarray) -> float:

    n = len(alpha)
    alpha_rotated = angrange((alpha[:, None] - alpha)).round(5)

    # count number of points on the right (0, 180), excluding the boundaries
    right = np.logical_and(alpha_rotated > 0.0, alpha_rotated < np.round(np.pi, 5)).sum(
        0
    )
    # count number of points on the boundaries
    exact = np.logical_or(
        np.isclose(alpha_rotated, 0.0), np.isclose(alpha_rotated, np.round(np.pi, 5))
    ).sum(0)
    # count number of points on the left (180, 360), excluding the boundaries
    left = n - right - 0.5 * exact
    right = right + 0.5 * exact
    # find the point(s) location where the difference of number of points
    # on right and left is/ are minimal
    diff = np.abs(right - left)
    idx_candidates = np.where(diff == diff.min())[0]
    # if number of potential median is the same as the number of data point
    # meaning that the data is more or less uniformly distributed. Retrun Nan.
    if len(idx_candidates) == len(alpha):
        median = np.nan
    # if number of potential median is 1, return it as median
    elif len(idx_candidates) == 1:
        median = alpha[idx_candidates][0]
    # if there are more than one potential median, do we need to
    # distinguish odd or even? or just calculate the circular mean?
    else:
        median = circ_mean(alpha[idx_candidates])[0]

    return median


def _circ_median_mean_deviation(
    alpha: np.array,
    w: Union[np.array, None] = None,
) -> float:

    # get pairwise circular mean deviation
    angdist = circ_mean_deviation(alpha, alpha)
    # data point(s) with minimal circular mean deviation is/are
    # potential median(s); pitfall: angdist sound be rounded!
    # (fixed in circ_mean_deviation())
    idx_candidates = np.where(angdist == angdist.min())[0]
    # if number of potential median is the same as the number of data point
    # meaning that the data is more or less uniformly distributed. Retrun Nan.
    if len(idx_candidates) == len(alpha):
        median = np.nan
    # if number of potential median is 1, return it as median
    elif len(idx_candidates) == 1:
        median = alpha[idx_candidates][0]
    # if there are more than one potential median, do we need to
    # distinguish odd or even? or just calculate the circular mean?
    else:
        median = circ_mean(alpha[idx_candidates], w[idx_candidates])[0]

    return median


def circ_mean_deviation(
    alpha: Union[np.ndarray, float, int, list],
    beta: Union[np.ndarray, float, int, list],
) -> np.ndarray:

    """
    Circular mean deviation.

    It is the mean angular distance from one data point to all others.
    The circular median of a set of data should be the point with minimal
    circular mean deviation.

    Parameters
    ---------
    alpha: np.array, int or float
        Data in radian.
    beta: np.array, int or float
        reference angle in radian.

    Return
    ------
    circular mean deviation: np.array
    """
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha])

    if not isinstance(beta, np.ndarray):
        beta = np.array([beta])

    return (np.pi - np.mean(np.abs(np.pi - np.abs(alpha - beta[:, None])), 1)).round(5)


def circ_mean_ci(
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    mean: Union[float, None] = None,
    r: Union[float, None] = None,
    n: Union[int, None] = None,
    ci: float = 0.95,
    method: str = "approximate",
    B: int = 2000,  # number of samples for bootstrap
    return_samples: bool = False,  # bootstrap option
) -> tuple:

    # TODO

    if method == "approximate":
        (lb, ub) = _circ_mean_ci_approximate(
            alpha=alpha, w=w, mean=mean, r=r, n=n, ci=ci
        )
    elif method == "dispersion":
        (lb, ub) = _circ_mean_ci_dispersion(alpha=alpha, w=w, mean=mean, ci=ci)
    elif method == "bootstrap":
        (lb, ub) = _circ_mean_ci_bootstrap(
            alpha=alpha, B=B, ci=ci, return_samples=return_samples
        )
    else:
        raise ValueError(
            f"Method `{method}` for `circ_mean_ci` is not supported.\nTry `dispersion`, `approximate` or `bootstrap`"
        )

    return angrange(lb), angrange(ub)


def _circ_mean_ci_dispersion(
    alpha: np.ndarray,
    w: Union[np.ndarray, None] = None,
    mean: Union[float, None] = None,
    ci: float = 0.95,
) -> tuple:

    """Confidence intervals based on circular dispersion.

    Note
    ----
    Implementation of Section 4.4.4b (Fisher, 1993)
    """

    if w is None:
        w = np.ones_like(alpha)
    if mean is None:
        mean, r = circ_mean(alpha, w)

    n = np.sum(w)
    if n < 25:
        raise ValueError(
            f"n={n} is too small (< 25) for computing CI with circular dispersion."
        )
    d = np.arcsin(
        np.sqrt(circ_dispersion(alpha=alpha, w=w, mean=mean) / n)
        * norm.ppf(1 - 0.5 * (1 - ci))
    )
    lb = mean - d
    ub = mean + d

    return (lb, ub)


def _circ_mean_ci_approximate(
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    mean: Union[float, None] = None,
    r: Union[float, None] = None,
    n: Union[int, None] = None,
    ci: float = 0.95,
) -> tuple:

    """
    Confidence Interval of circular mean.

    Note
    ----
    Implementation of Example 26.6 (Zar, 2010)
    """

    if r is None:
        assert isinstance(
            alpha, np.ndarray
        ), "If `r` is None, then `alpha` (and `w`) is needed."
        if w is None:
            w = np.ones_like(alpha)
        n = np.sum(w)
        mean, r = circ_mean(alpha, w)

    if n is None:
        raise ValueError("Sample size `n` is missing.")

    R = n * r

    # Zar cited (Upton, 1986) that n can be as small as 8
    # but I encountered a few cases where n<11 will result in negative
    # value within `inner`.
    if n >= 8:  #
        if r <= 0.9:  # eq(26.24)
            inner = (
                np.sqrt(
                    (2 * n * (2 * R**2 - n * chi2.isf(0.05, 1)))
                    / (4 * n - chi2.isf(1 - ci, 1))
                )
                / R
            )
        else:  # eq(26.25)
            inner = (
                np.sqrt(n**2 - (n**2 - R**2) * np.exp(chi2.isf(1 - ci, 1) / n))
                / R
            )

        d = np.arccos(inner)
        lb = mean - d
        ub = mean + d

        return (lb, ub)

    else:
        raise ValueError(
            f"n={n} is too small (<= 8) for computing CI with approximation method."
        )


def _circ_mean_ci_bootstrap(alpha, B=2000, ci=0.95, return_samples=False):
    from scipy.stats import t

    beta = np.array([_circ_mean_resample(alpha) for i in range(B)]).flatten()

    lb, ub = t.interval(
        ci,
        len(beta) - 1,
        loc=circ_mean(alpha=beta)[0],
        scale=circ_std(alpha=beta, w=np.ones_like(beta))[0],
    )

    if return_samples:
        return lb, ub, beta
    else:
        return lb, ub


def _circ_mean_resample(alpha):
    """
    Implementation of Section 8.3.5 (Fisher, 1993)
    """

    θ = np.random.choice(alpha, len(alpha), replace=True)
    X = np.cos(θ)
    Y = np.sin(θ)
    z1 = np.mean(X)
    z2 = np.mean(Y)

    u11 = np.mean((X - z1) ** 2)
    u22 = np.mean((X - z2) ** 2)
    u12 = u21 = np.mean((X - z1) * (Y - z2))

    β = (u11 - u22) / (2 * u12) - np.sqrt((u11 - u22) ** 2 / (4 * u12**2 + 1))
    t1 = np.sqrt(β**2 * u11 + 2 * β * u12 + u22) / np.sqrt(1 + β**2)
    t2 = np.sqrt(u11 - 2 * β * u12 + β**2 * u22) / np.sqrt(1 + β**2)
    v11 = (β**2 * t1 + t2) / (1 + β**2)
    v22 = (t1 + β**2 * t2) / (1 + β**2)
    v12 = v21 = β * (t1 - t2) / (1 + β**2)

    q1 = np.sqrt(1 + β**2) / np.sqrt(β**2 * u11 + 2 * β * u12 + u22)
    q2 = np.sqrt(1 + β**2) / np.sqrt(u11 - 2 * β * u12 + β**2 * u22)
    w11 = (β**2 * q1 + q2) / (1 + β**2)
    w22 = (q1 + β**2 * q2) / (1 + β**2)
    w12 = w21 = β * (t1 - q2) / (1 + β**2)

    z0 = np.array([z1, z2])
    u0 = np.array([[u11, u12], [u21, u22]])
    v0 = np.array([[v11, v12], [v21, v22]])
    w0 = np.array([[w11, w12], [w21, w22]])

    Cbar, Sbar = z0 + v0 @ w0 @ (z0 - z0)
    Cbar = np.power(Cbar**2 + Sbar**2, -0.5) * Cbar
    Sbar = np.power(Cbar**2 + Sbar**2, -0.5) * Sbar

    r = np.sqrt(Cbar**2 + Sbar**2)

    if Cbar != 0 and Sbar != 0:
        m = np.arctan2(Sbar, Cbar)
    else:
        m = np.arccos(Cbar / r)
    return angrange(m)


def circ_median_ci(
    median: float = None,
    alpha: Union[np.ndarray, None] = None,
    w: Union[np.ndarray, None] = None,
    grouped: bool = False,
    ci: float = 0.95,
) -> tuple:

    """Confidence interval for circular median

    Parameters
    ----------
    median: float or None
        Circular median.
    alpha: np.array or None
        Data in radian.
    w: np.array or None
        Frequencies or weights

    Returns
    -------
    lower, upper, ci: tuple
        confidence intervals and alpha-level
        Exception: for n = 7 ~ 13, two levels with
        difference alpha-levels will be returned.

    Note
    ----
    Implementation of section 4.4.2 (Fisher,1993)
    """

    if median is None:
        assert isinstance(
            alpha, np.ndarray
        ), "If `median` is None, then `alpha` (and `w`) is needed."
        if w is None:
            w = np.ones_like(alpha)
        median = circ_median(alpha=alpha, w=w, grouped=grouped)

    if alpha is None:
        raise ValueError(
            "`alpha` is needed for computing the confidence interval for circular median."
        )

    # alpha, counts = np.unique(alpha)
    n = len(alpha)
    alpha = np.sort(alpha)

    if n > 15:
        z = norm.ppf(1 - 0.5 * (1 - ci))

        offset = int(1 + np.floor(0.5 * np.sqrt(n) * z))  # fisher:eq(4.19)

        idx_median = np.where(alpha.round(5) < median.round(5))[0][-1]
        idx_lb = idx_median - offset + 1
        idx_ub = idx_median + offset
        if median.round(5) in alpha.round(5):  # don't count the median per se
            idx_ub += 1

        if idx_ub > n:
            idx_ub = idx_ub - n

        if idx_lb < 0:
            idx_lb = n + idx_lb

        lower, upper = alpha[int(idx_lb)], alpha[int(idx_ub)]

    else:
        lower, upper = np.nan, np.nan

    return (angrange(lower), angrange(upper), ci)


def circ_kappa(r: float, n: Union[int, None] = None) -> float:

    """Approximate kappa
    Parameters
    ----------
    r: float
        resultant vector length
    n: int or None
        sample size

    Return
    ------
    kappa: float
        concentration parameter

    Reference
    ---------
    Section 4.5.5 (P88, Fisher, 1993)
    """

    # eq 4.40
    if r < 0.53:
        kappa = 2 * r + r**3 + 5 * r**5 / 6
    elif r < 0.85:
        kappa = -0.4 + 1.39 * r + 0.43 / (1 - r)
    else:
        nom = r**3 - 4 * r**2 + 3 * r
        if nom != 0:
            kappa = 1 / nom
        else:
            # not sure how to handle this...
            kappa = 1e-16

    # eq 4.41
    if n is not None:
        if n <= 15 and r < 0.7:
            if kappa < 2:
                kappa = np.max([kappa - 2 * 1 / (n * kappa), 0])
            else:
                kappa = (n - 1) ** 3 * kappa / (n**3 + n)

    return kappa


def compute_smooth_params(r: float, n: int) -> float:

    """
    Parameters
    ----------
    r: float
        resultant vector length
    n: int
        sample size

    Return
    ------
    h: float
        smoothing parameter

    Reference
    ---------
    Section 2.2 (P26, Fisher, 1993)
    """

    kappa = circ_kappa(r, n)
    l = 1 / np.sqrt(kappa)  # eq 2.3
    h = np.sqrt(7) * l / np.power(n, 0.2)  # eq 2.4

    return h


def nonparametric_density_estimation(
    alpha: np.ndarray,  # angles in radian
    h: float,  # smoothing parameters
    radius: float = 1,  # radius of the plotted circle
) -> tuple:

    """Nonparametric density estimates with
    a quartic kernel function.

    Parameters
    ----------
    alpha: np.ndarray (n, )
        Angles in radian
    h: float
        Smoothing parameters
    radius: float
        radius of the plotted circle

    Returns
    -------
    x: np.ndarray (100, )
        grid
    f: np.ndarray (100, )
        density

    Reference
    ---------
    Section 2.2 (P26, Fisher, 1993)
    """

    # vectorized version of step 3
    a = alpha
    x = np.linspace(0, 2 * np.pi, 100)
    d = np.abs(x[:, None] - a)
    e = np.minimum(d, 2 * np.pi - d)
    e = np.minimum(e, h)
    sum = np.sum((1 - e**2 / h**2) ** 2, 1)
    f = 0.9375 * sum / len(a) / h

    f = radius * np.sqrt(1 + np.pi * f)

    return x, f


def circ_mean_of_means(
    circs: Union[list, None] = None,
    ms: Union[np.ndarray, None] = None,
    rs: Union[np.ndarray, None] = None,
) -> Tuple[float]:

    """The Mean of a set of Mean Angles

    Parameters
    ----------
    circs: list
        a list of Circular Objects

    ms: np.array (n, )
        a set of mean angles in radian

    rs: np.array (n, )
        a set of mean resultant vecotr lengths

    Return
    ------
    m: float
        mean of means in radian

    r: float
        mean of mean resultant vector lengths

    """

    if circs is None:
        assert isinstance(ms, np.ndarray) and isinstance(
            rs, np.ndarray
        ), "If `circs` is None, then `ms` and `rs` are needed."
    else:
        ms, rs = map(np.array, zip(*[(circ.mean, circ.r) for circ in circs]))

    X = np.mean(np.cos(ms) * rs)
    Y = np.mean(np.sin(ms) * rs)
    r = np.sqrt(X**2 + Y**2)
    C = X / r
    S = Y / r

    m = angrange(np.arctan2(S, C))

    return m, r
