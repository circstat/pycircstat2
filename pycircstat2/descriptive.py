from typing import Optional, Tuple, Union

import numpy as np
from scipy.stats import chi2, norm

from .utils import angmod, is_within_circular_range


def circ_r(
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    Cbar: Optional[float] = None,
    Sbar: Optional[float] = None,
) -> float:
    r"""
    Circular mean resultant vector length (r).

    $$
    r = \sqrt{\bar{C}^2 + \bar{S}^2}
    $$

    Parameters
    ----------
    alpha: np.array (n, )
        Angles in radian.
    w: np.array (n,)
        Frequencies or weights
    Cbar, Sbar: float
        Precomputed intermediate values

    Returns
    -------
    r: float
        Resultant vector length

    References
    ----------
    Implementation of Example 26.5 (Zar, 2010)
    """
    if alpha is None and (Cbar is None or Sbar is None):
        raise ValueError("`alpha` is needed for computing the resultant vector length.")

    if w is None:
        w = np.ones_like(alpha)

    if Cbar is None or Sbar is None:
        Cbar, Sbar = compute_C_and_S(alpha, w)

    # mean resultant vecotr length
    r = np.sqrt(Cbar**2 + Sbar**2)

    return r


def circ_mean(
    alpha: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Union[np.ndarray, float]:
    r"""
    Circular mean (m).

    $$\cos\bar\theta = C/R,\space \sin\bar\theta = S/R$$
    
    or 

    $$
    \bar\theta =
    \begin{cases} 
    \tan^{-1}\left(S/C\right), & \text{if } S > 0, C > 0 \\ 
    \tan^{-1}\left(S/C\right) + \pi, & \text{if } C < 0 \\ 
    \tan^{-1}\left(S/C\right) + 2\pi, & \text{S < 0, C > 0}
    \end{cases}
    $$

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

    Note
    ----
    Implementation of Example 26.5 (Zar, 2010)
    """
    if w is None:
        w = np.ones_like(alpha)

    # mean resultant vecotr length
    Cbar, Sbar = compute_C_and_S(alpha, w)
    r = circ_r(alpha, w, Cbar, Sbar)

    # angular mean
    if np.isclose(r, 0):
        m = np.nan
    else:
        m = np.arctan2(Sbar, Cbar)

    return angmod(m)


def circ_mean_and_r(
    alpha: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Tuple[Union[float, np.ndarray], float]:
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

    # mean resultant vecotr length
    Cbar, Sbar = compute_C_and_S(alpha, w)
    r = circ_r(alpha, w, Cbar, Sbar)

    # angular mean
    if np.isclose(r, 0):
        m = np.nan
        return m, r
    else:
        m = np.arctan2(Sbar, Cbar)

        return angmod(m), r


def circ_mean_and_r_of_means(
    circs: Union[list, None] = None,
    ms: Optional[np.ndarray] = None,
    rs: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """The Mean of a set of Mean Angles

    Parameters
    ----------
    circs: list
        a list of Circular Objects

    ms: np.array (n, )
        a set of mean angles in radian

    rs: np.array (n, )
        a set of mean resultant vecotr lengths

    Returns
    -------
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

    m = angmod(np.arctan2(S, C))

    return m, r


def circ_moment(
    alpha: np.ndarray,
    w: Optional[np.ndarray] = None,
    p: int = 1,
    mean: Union[float, np.ndarray, None] = None,
    centered: bool = False,
) -> complex:
    r"""
    Compute the p-th circular moment.

    $$
    m^{\prime}_{p} = \bar{C}_{p} + i\bar{S}_{p}
    $$

    Parameters
    ----------
    alpha: np.ndarray
        Angles in radian.
    w: np.ndarray, optional
        Frequencies or weights. If None, equal weights are used.
    p: int, optional
        Order of the moment to compute.
    mean: float, optional
        Precomputed circular mean. If None, mean is computed internally.
    centered: bool, optional
        If True, center alpha by subtracting the mean.

    Returns
    -------
    mp: complex
        The p-th circular moment as a complex number.

    Note
    ----
    Implementation of Equation 2.24 (Fisher, 1993).
    """
    if w is None:
        w = np.ones_like(alpha)

    if mean is None:
        mean = circ_mean(alpha, w) if centered else 0.0

    Cbar, Sbar = compute_C_and_S(alpha, w, p, mean)

    return Cbar + 1j * Sbar


def circ_dispersion(
    alpha: np.ndarray,
    w: Optional[np.ndarray] = None,
    mean=None,
) -> float:
    r"""
    Sample Circular Dispersion, defined by Equation 2.28 (Fisher, 1993):

    $$
    \hat\delta = (1 - \hat\rho_{2})/(2 \hat\rho_{1}^{2})
    $$

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
    """

    if w is None:
        w = np.ones_like(alpha)

    mp1 = circ_moment(alpha=alpha, w=w, p=1, mean=mean, centered=False)  # eq(2.26)
    mp2 = circ_moment(alpha=alpha, w=w, p=2, mean=mean, centered=False)  # eq(2.27)

    r1 = np.abs(mp1)
    r2 = np.abs(mp2)

    dispersion = (1 - r2) / (2 * r1**2)  # eq(2.28)

    return dispersion


def circ_skewness(alpha: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    r"""
    Circular skewness, as defined by Equation 2.29 (Fisher, 1993):

    $$\hat s = [\hat\rho_2 \sin(\hat\mu_2 - 2 \hat\mu_1)] / (1 - \hat\rho_1)^{\frac{3}{2}}$$

    But unlike the implementation of Fisher (1993), here we followed Pewsey et al. (2014) by NOT centering the second moment.

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
    """

    if w is None:
        w = np.ones_like(alpha)

    mp1 = circ_moment(alpha=alpha, w=w, p=1, mean=None, centered=False)
    mp2 = circ_moment(alpha=alpha, w=w, p=2, mean=None, centered=False)  # eq(2.27)

    u1, r1 = convert_moment(mp1)
    u2, r2 = convert_moment(mp2)

    skewness = (r2 * np.sin(u2 - 2 * u1)) / (1 - r1) ** 1.5

    return skewness


def circ_kurtosis(alpha: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    r"""
    Circular kurtosis, as defined by Equation 2.30 (Fisher, 1993):

    $$\hat k = [\hat\rho_2 \cos(\hat\mu_2 - 2 \hat\mu_1) - \hat\rho_1^4] / (1 - \hat\rho_1)^{2}$$

    But unlike the implementation of Fisher (1993), here we followed Pewsey et al. (2014) by **NOT** centering the second moment.

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
    """

    if w is None:
        w = np.ones_like(alpha)

    mp1 = circ_moment(alpha=alpha, w=w, p=1, mean=None, centered=False)
    mp2 = circ_moment(alpha=alpha, w=w, p=2, mean=None, centered=False)  # eq(2.27)

    u1, r1 = convert_moment(mp1)
    u2, r2 = convert_moment(mp2)

    kurtosis = (r2 * np.cos(u2 - 2 * u1) - r1**4) / (1 - r1) ** 2

    return kurtosis


def angular_var(
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    r: Optional[float] = None,
    bin_size: Optional[float] = None,
) -> float:
    r"""
    Angular variance

    Parameters
    ----------
    alpha: np.array (n, ) or None
        Angles in radian.
    w: np.array (n,) or None
        Frequencies or weights
    r: float or None
        Resultant vector length
    bin_size: float
        Interval size of grouped data. Needed for correcting biased r.

    Returns
    -------
    angular_variance: float
        Angular variance, range from 0 to 2.

    References
    ----------
    - Batschlet (1965, 1981), from Section 26.5 of Zar (2010)
    """

    variance = circ_var(alpha=alpha, w=w, r=r, bin_size=bin_size)
    angular_variance = 2 * variance
    return angular_variance


def angular_std(
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    r: Optional[float] = None,
    bin_size: Optional[float] = None,
) -> float:
    r"""
    Angular (standard) deviation

    $$
    s = \sqrt{2V} = \sqrt{2(1 - r)}
    $$

    Parameters
    ----------
    alpha: np.array (n, ) or None
        Angles in radian.
    w: np.array (n,) or None
        Frequencies or weights
    r: float or None
        Resultant vector length
    bin_size: float
        Interval size of grouped data. Needed for correcting biased r.

    Returns
    -------
    angular_std: float
        Angular (standard) deviation, range from 0 to sqrt(2).

    References
    ----------
    - Equation 26.20 of Zar (2010)
    """

    angular_variance = angular_var(alpha=alpha, w=w, r=r, bin_size=bin_size)
    angular_std = np.sqrt(angular_variance)
    return angular_std


def circ_var(
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    r: Optional[float] = None,
    bin_size: Optional[float] = None,
) -> float:
    r"""
    Circular variance

    $$ V = 1 - r $$

    Parameters
    ----------
    alpha: np.array (n, ) or None
        Angles in radian.
    w: np.array (n,) or None
        Frequencies or weights
    r: float or None
        Resultant vector length
    bin_size: float
        Interval size of grouped data. Needed for correcting biased r.

    Returns
    -------
    variance: float
        Circular variance, range from 0 to 1.

    References
    ----------
    - Equation 2.11 of Fisher (1993)
    - Equation 26.17 of Zar (2010)
    """

    if w is None:
        w = np.ones_like(alpha)

    if r is None:
        assert isinstance(alpha, np.ndarray) and isinstance(
            w, np.ndarray
        ), "If `r` is None, then `alpha` and `w` are needed."
        r = circ_r(alpha, w)

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

    variance = 1 - rc

    return variance


def circ_std(
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    r: Optional[float] = None,
    bin_size: Optional[float] = None,
) -> tuple:
    r"""
    Circular standard deviation (s).

    $$ s = \sqrt{-2 \ln(1 - V)} $$

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
    s: float
        Circular standard deviation.

    References
    ----------
    Implementation of Equation 26.15-16/20-21 (Zar, 2010)
    """
    var = circ_var(alpha=alpha, w=w, r=r, bin_size=bin_size)

    # circular standard deviation
    s = np.sqrt(-2 * np.log(1 - var))  # eq(26.21)

    return s


def circ_median(
    alpha: np.ndarray,
    w: Optional[np.ndarray] = None,
    method: str = "deviation",
    return_average: bool = True,
    average_method: str = "all",
) -> Union[float, np.ndarray]:
    r"""
    Circular median.

    Two ways to compute the circular median for ungrouped data (Fisher, 1993):

    - `deviation`: find the angle that has the minimal mean deviation.
    - `count`: find the angle that has the equally devide the number of points on the right and left of it.

    For grouped data, we use the method described in Mardia (1972).

    Parameters
    ----------
    alpha: np.array (n, )
        Angles in radian.
    w: np.array (n,) or None
        Frequencies or weights
    method: str
        - For ungrouped data, there are two ways
        - To compute the medians:
            - deviation
            - count
        - Set to `none` to return np.nan.
    return_average: bool
        Return the average of the median
    average_method: str
        - all: circular mean of all medians
        - unique: circular mean of unique medians

    Returns
    -------
    median: float or NaN

    References
    ----------
    - For ungrouped data: Section 2.3.2 of Fisher (1993)
    - For grouped data: Mardia (1972)
    """

    if w is None:
        w = np.ones_like(alpha)

    # grouped data
    if not np.all(w == 1):
        median = _circ_median_grouped(alpha, w)
    # ungrouped data
    else:
        # find which data point that can divide the dataset into two half
        if method == "count":
            median = _circ_median_count(alpha)
        # find the angle that has the minimal mean deviation
        elif method == "deviation":
            median = _circ_median_mean_deviation(alpha)
        elif method == "none" or method is None:
            median = np.nan
        else:
            raise ValueError(
                f"Method `{method}` for `circ_median` is not supported.\nTry `deviation` or `count`"
            )

    if return_average:
        if average_method == "all":
            # Circular mean of all medians
            median = circ_mean(alpha=median)
        elif average_method == "unique":
            # Circular mean of unique medians
            median = circ_mean(alpha=np.unique(median))
        else:
            raise ValueError(
                f"Average method `{average_method}` is not supported.\nTry `all` or `unique`."
            )

    return angmod(median)


def _circ_median_grouped(
    alpha: np.array,
    w: Union[np.array, None] = None,
) -> Union[float, np.array]:
    n = np.sum(w)  # sample size
    n_bins = len(alpha)  # number of intervals
    bin_size = np.diff(alpha).min()

    # median for grouped data operated on upper bound of bins
    alpha_ub = alpha + bin_size / 2
    alpha_rotated = angmod(alpha_ub[:, None] - alpha_ub)
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
        median = np.array([lower, upper])

    return median


def _circ_median_count(alpha: np.ndarray) -> float:
    n = len(alpha)
    alpha_rotated = angmod((alpha[:, None] - alpha)).round(5)

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
    # if there are more than one potential median, return them all
    else:
        median = alpha[idx_candidates]

    return median


def _circ_median_mean_deviation(alpha: np.array) -> float:
    """
    Note
    ----
    Implementation of Section 2.3.2 of Fisher (1993)
    """

    # get pairwise circular mean deviation
    if len(alpha) > 10000:
        angdist = circ_mean_deviation_chuncked(alpha, alpha)
    else:
        # get pairwise circular mean deviation
        angdist = circ_mean_deviation(alpha, alpha)
    # data point(s) with minimal circular mean deviation is/are potential median(s);
    idx_candidates = np.where(angdist == angdist.min())[0]
    # if number of potential median is the same as the number of data point
    # meaning that the data is more or less uniformly distributed. Retrun Nan.
    if len(idx_candidates) == len(alpha):
        median = np.nan
    # if number of potential median is 1, return it as median
    elif len(idx_candidates) == 1:
        median = alpha[idx_candidates][0]
    # if there are more than one potential median, return them all
    else:
        median = alpha[idx_candidates]

    return median


def circ_mean_deviation_chuncked(
    alpha: Union[np.ndarray, float, int, list],
    beta: Union[np.ndarray, float, int, list],
    chunk_size=1000,
):
    r"""
    Optimized circular mean deviation with chunking.

    $$
    \delta = \pi - \frac{1}{n} \sum^{n}_{1}\left| \pi - \left| \alpha - \beta \right| \right|
    $$

    Parameters
    ----------
    alpha : np.ndarray
        Data in radians.
    beta : np.ndarray
        Reference angles in radians.
    chunk_size : int
        Number of rows to process in chunks.

    Returns
    -------
    np.ndarray
        Circular mean deviation.
    """
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha])

    if not isinstance(beta, np.ndarray):
        beta = np.array([beta])

    n = len(beta)
    result = np.zeros(n)

    for i in range(0, n, chunk_size):
        beta_chunk = beta[i : i + chunk_size]
        angdist = np.pi - np.abs(np.pi - np.abs(alpha - beta_chunk[:, None]))
        result[i : i + chunk_size] = np.mean(angdist, axis=1).round(5)

    return result


def circ_mean_deviation(
    alpha: Union[np.ndarray, float, int, list],
    beta: Union[np.ndarray, float, int, list],
) -> np.ndarray:
    r"""
    Circular mean deviation.

    $$
    \delta = \pi - \left| \pi - \left| \alpha - \beta \right| \right| / n
    $$

    It is the mean angular distance from one data point to all others.
    The circular median of a set of data should be the point with minimal
    circular mean deviation.

    Parameters
    ---------
    alpha: np.array, int or float
        Data in radian.
    beta: np.array, int or float
        reference angle in radian.

    Returns
    -------
    circular mean deviation: np.array

    Note
    ----
    eq 2.32, Section 2.3.2, Fisher (1993)
    """
    if not isinstance(alpha, np.ndarray):
        alpha = np.array([alpha])

    if not isinstance(beta, np.ndarray):
        beta = np.array([beta])

    return (np.pi - np.mean(np.abs(np.pi - np.abs(alpha - beta[:, None])), 1)).round(5)


def circ_mean_ci(
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    mean: Optional[float] = None,
    r: Optional[float] = None,
    n: Union[int, None] = None,
    ci: float = 0.95,
    method: str = "approximate",
    B: int = 2000,  # number of samples for bootstrap
) -> tuple[float, float]:
    r"""
    Confidence interval of circular mean.

    There are three methods to compute the confidence interval of circular mean:

    - `approximate`: for n > 8
    - `bootstrap`: for 8 < n < 25
    - `dispersion`: for n >= 25

    ### Approximate Method

    For n as small as 8, and r $\le$ 0.9, r $>$ $\sqrt{\chi^{2}_{\alpha, 1}/2n}$, the confidence interval can be approximated by:

    $$
    \delta = \arccos\left(\sqrt{\frac{2n(2R^{2} - n\chi^{2}_{\alpha, 1})}{4n - \chi^{2}_{\alpha, 1}}} /R \right)
    $$

    For r $ge$ 0.9,

    $$
    \delta = \arccos \left(\sqrt{n^2 - (n^2 - R^2)e^{\chi^2_{\alpha, 1}/n} } /R \right)
    $$

    ### Bootstrap Method

    For 8 $<$ n $<$ 25, the confidence interval can be computed by bootstrapping the data.

    ### Dispersion Method

    For n $\ge$ 25, the confidence interval can be computed by the circular dispersion:

    $$ \hat\sigma = \hat\delta / n$$

    where $\hat\delta$ is the sample circular dispersion (see `circ_dispersion`). The confidence interval is then:

    $$(\hat\mu - \sin^-1(z_{\frac{1}{2}\alpha}\hat\sigma),\space \hat\mu + \sin^-1(z_{\frac{1}{2}\alpha} \hat\sigma))$$

    Parameters
    ----------
    alpha: np.array (n, )
        Angles in radian.
    w: np.array (n,) or None
        Frequencies or weights
    mean: float or None
        Precomputed circular mean.
    r: float or None
        Precomputed resultant vector length.
    n: int or None
        Sample size.
    ci: float
        Confidence interval (default is 0.95).
    method: str
        - approximate: for n > 8
        - bootstrap: for n < 25
        - dispersion: for n >= 25
    B: int
        Number of samples for bootstrap.

    Returns
    -------
    lower_bound: float
        Lower bound of the confidence interval.
    upper_bound: float
        Upper bound of the confidence

    References
    ----------
    - Section 26.7, Zar (2010)
    - Section 4.4.4a/b, Fisher (1993)
    """

    #  n > 8, according to Ch 26.7 (Zar, 2010)
    if method == "approximate":
        (lb, ub) = _circ_mean_ci_approximate(
            alpha=alpha, w=w, mean=mean, r=r, n=n, ci=ci
        )

    # n < 25, according to 4.4.4a (Fisher, 1993, P75)
    elif method == "bootstrap":
        (lb, ub) = _circ_mean_ci_bootstrap(alpha=alpha, B=B, ci=ci)

    # n >= 25, according to 4.4.4b (Fisher, 1993, P75)
    elif method == "dispersion":
        (lb, ub) = _circ_mean_ci_dispersion(alpha=alpha, w=w, mean=mean, ci=ci)

    else:
        raise ValueError(
            f"Method `{method}` for `circ_mean_ci` is not supported.\nTry `dispersion`, `approximate` or `bootstrap`"
        )

    return angmod(lb), angmod(ub)


def _circ_mean_ci_dispersion(
    alpha: np.ndarray,
    w: Optional[np.ndarray] = None,
    mean: Optional[float] = None,
    ci: float = 0.95,
) -> tuple[float, float]:
    r"""Confidence intervals based on circular dispersion.

    Parameters
    ----------
    alpha: np.array (n, )
        Angles in radian.
    w: np.array (n,) or None
        Frequencies or weights
    mean: float or None
        Precomputed circular mean.
    ci: float
        Confidence interval (default is 0.95).


    Returns
    -------
    lower_bound: float
        Lower bound of the confidence interval.
    upper_bound: float
        Upper bound of the confidence interval.


    Note
    ----
    Implementation of Section 4.4.4b (Fisher, 1993)
    """

    if w is None:
        w = np.ones_like(alpha)
    if mean is None:
        mean, r = circ_mean_and_r(alpha, w)

    n = np.sum(w)
    if n < 25:
        raise ValueError(
            f"n={n} is too small (< 25) for computing CI with circular dispersion."
        )

    # TODO: sometime return nan because x in arcsin(x) is larger than 1.
    # Should we centered the data here? <- No.
    d = np.arcsin(
        np.sqrt(circ_dispersion(alpha=alpha, w=w) / n) * norm.ppf(1 - 0.5 * (1 - ci))
    )
    lb = mean - d
    ub = mean + d

    if not is_within_circular_range(mean, lb, ub):
        lb, ub = ub, lb

    return (lb, ub)


def _circ_mean_ci_approximate(
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    mean: Optional[float] = None,
    r: Optional[float] = None,
    n: Union[int, None] = None,
    ci: float = 0.95,
) -> tuple:
    r"""
    Confidence Interval of circular mean.

    $$
    \displaylines{
    \delta = \arccos\left(\sqrt{\frac{2n(2R^{2} - n\chi^{2}_{alpha, 1})}{4n - \chi^{2}_{\alpha, 1}}}\right)/R \cr
    }
    $$

    Parameters
    ----------
    alpha: np.array (n, )
        Angles in radian.
    w: np.array (n,) or None
        Frequencies or weights
    mean: float or None
        Precomputed circular mean.
    r: float or None
        Precomputed resultant vector length.
    n: int or None
        Sample size.
    ci: float
        Confidence interval (default is 0.95).

    Returns
    -------
    lower_bound: float
        Lower bound of the confidence interval.
    upper_bound: float
        Upper bound of the confidence interval.

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
        mean, r = circ_mean_and_r(alpha, w)

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
            inner = np.sqrt(n**2 - (n**2 - R**2) * np.exp(chi2.isf(1 - ci, 1) / n)) / R

        d = np.arccos(inner)
        lb = mean - d
        ub = mean + d

        if not is_within_circular_range(mean, lb, ub):
            lb, ub = ub, lb

        return (lb, ub)

    else:
        raise ValueError(
            f"n={n} is too small (<= 8) for computing CI with approximation method."
        )


def _circ_mean_ci_bootstrap(alpha, B=2000, ci=0.95):
    """
    Implementation of Section 8.3 (Fisher, 1993, P207)
    """

    # Precompute z0 and v0 from original data
    # algo 1
    X = np.cos(alpha)
    Y = np.sin(alpha)
    z1 = np.mean(X)  # eq(8.24)
    z2 = np.mean(Y)
    z0 = np.array([z1, z2])

    # algo 2
    u11 = np.mean((X - z1) ** 2)  # eq(8.25)
    u22 = np.mean((Y - z2) ** 2)
    u12 = u21 = np.mean((X - z1) * (Y - z2))  # eq(8.26)

    β = (u11 - u22) / (2 * u12) - np.sqrt(
        (u11 - u22) ** 2 / (4 * u12**2 + 1)
    )  # eq(8.27)
    t1 = np.sqrt(β**2 * u11 + 2 * β * u12 + u22) / np.sqrt(1 + β**2)  # eq(8.28)
    t2 = np.sqrt(u11 - 2 * β * u12 + β**2 * u22) / np.sqrt(1 + β**2)  # eq(8.29)
    v11 = (β**2 * t1 + t2) / (1 + β**2)  # eq(8.30)
    v22 = (t1 + β**2 * t2) / (1 + β**2)
    v12 = v21 = β * (t1 - t2) / (1 + β**2)  # eq(8.31)
    v0 = np.array([[v11, v12], [v21, v22]])

    beta = np.array([_circ_mean_resample(alpha, z0, v0) for i in range(B)]).flatten()

    # here we use HDI instead of the percentile method
    lb, ub = compute_hdi(beta, ci=ci)

    mean = circ_mean(beta)
    if not is_within_circular_range(mean, lb, ub):
        lb, ub = ub, lb

    return lb, ub


def _circ_mean_resample(alpha, z0, v0):
    """
    Implementation of Section 8.3.5 (Fisher, 1993, P210)
    """

    θ = np.random.choice(alpha, len(alpha), replace=True)
    X = np.cos(θ)
    Y = np.sin(θ)

    # algo 1
    z1 = np.mean(X)  # eq(8.24)
    z2 = np.mean(Y)
    zB = np.array([z1, z2])

    u11 = np.mean((X - z1) ** 2)  # eq(8.25)
    u22 = np.mean((X - z2) ** 2)
    u12 = u21 = np.mean((X - z1) * (Y - z2))  # eq(8.26)

    # algo 3
    β = (u11 - u22) / (2 * u12) - np.sqrt(
        (u11 - u22) ** 2 / (4 * u12**2 + 1)
    )  # eq(8.27)
    t1 = np.sqrt(1 + β**2) / np.sqrt(β**2 * u11 + 2 * β * u12 + u22)  # eq(8.33)
    t2 = np.sqrt(1 + β**2) / np.sqrt(u11 - 2 * β * u12 + β**2 * u22)  # eq(8.34)
    w11 = (β**2 * t1 + t2) / (1 + β**2)  # eq(8.35)
    w22 = (t1 + β**2 * t2) / (1 + β**2)
    w12 = w21 = β * (t1 - t2) / (1 + β**2)  # eq(8.36)

    wB = np.array([[w11, w12], [w21, w22]])

    Cbar, Sbar = z0 + v0 @ wB @ (zB - z0)
    Cbar = np.power(Cbar**2 + Sbar**2, -0.5) * Cbar
    Sbar = np.power(Cbar**2 + Sbar**2, -0.5) * Sbar

    m = np.arctan2(Sbar, Cbar)

    return angmod(m)


def circ_median_ci(
    median: float = None,
    alpha: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    method: str = "deviation",
    ci: float = 0.95,
) -> tuple:
    r"""Confidence interval for circular median

    For n > 15, the confidence interval can be computed by:

    $$
    m = 1 + \text{integer part of} \frac{1}{2} n^{1/2} z_{\frac{1}{2}\alpha}
    $$

    For n $\le$ 15, the confidence interval can be selected from the table in Fisher (1993).

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
        median = circ_median(alpha=alpha, w=w, method=method)

    if alpha is None:
        raise ValueError(
            "`alpha` is needed for computing the confidence interval for circular median."
        )

    n = len(alpha)
    alpha = np.sort(alpha)

    if n > 15:
        z = norm.ppf(1 - 0.5 * (1 - ci))

        offset = int(1 + np.floor(0.5 * np.sqrt(n) * z))  # fisher:eq(4.19)

        # idx_median = np.where(alpha.round(5) < np.round(median, 5))[0][-1]
        arr = np.where(alpha.round(5) < np.round(median, 5))[0]
        if len(arr) == 0:
            # That means median is smaller than alpha[0] (to 5 decimals).
            # In a circular sense, the “closest index below” is alpha[-1].
            idx_median = len(alpha) - 1
        else:
            idx_median = arr[-1]

        idx_lb = idx_median - offset + 1
        idx_ub = idx_median + offset
        if np.round(median, 5) in alpha.round(5):  # don't count the median per se
            idx_ub += 1

        if idx_ub > n:
            idx_ub = idx_ub - n

        if idx_lb < 0:
            idx_lb = n + idx_lb

        lower, upper = alpha[int(idx_lb)], alpha[int(idx_ub)]

        if not is_within_circular_range(median, lower, upper):
            lower, upper = upper, lower

    # selected confidence intervals for the median direction for n < 15
    # from A6, Fisher, 1993.
    # We only return the widest CI if there are more than one in the table.

    elif n == 3:
        lower, upper = alpha[0], alpha[2]
        ci = 0.75
    elif n == 4:
        lower, upper = alpha[0], alpha[3]
        ci = 0.875
    elif n == 5:
        lower, upper = alpha[0], alpha[4]
        ci = 0.937
    elif n == 6:
        lower, upper = alpha[0], alpha[5]
        ci = 0.97
    elif n == 7:
        lower, upper = alpha[0], alpha[6]
        ci = 0.984
    elif n == 8:
        lower, upper = alpha[0], alpha[7]
        ci = 0.992
    elif n == 9:
        lower, upper = alpha[0], alpha[8]
        ci = 0.996
    elif n == 10:
        lower, upper = alpha[1], alpha[8]
        ci = 0.978
    elif n == 11:
        lower, upper = alpha[1], alpha[9]
        ci = 0.99
    elif n == 12:
        lower, upper = alpha[2], alpha[9]
        ci = 0.962
    elif n == 13:
        lower, upper = alpha[2], alpha[10]
        ci = 0.978
    elif n == 14:
        lower, upper = alpha[3], alpha[10]
        ci = 0.937
    elif n == 15:
        lower, upper = alpha[2], alpha[12]
        ci = 0.965
    else:
        lower, upper = np.nan, np.nan

    return (angmod(lower), angmod(upper), ci)


def circ_kappa(r: float, n: Union[int, None] = None) -> float:
    r"""Estimate kappa by approximation.

    $$
    \hat\kappa_{ML} =
    \begin{cases}
     2r + r^3 + 5r^5/6, , & \text{if } r < 0.53  \\
     -0.4 + 1.39 r + 0.43 / (1 - r) , & \text{if } 0.53 \le r < 0.85\\
        1 / (r^3 - 4r^2 + 3r), & \text{if } r \ge 0.85
    \end{cases}
    $$

    For $n \le 15$:

    $$
    \hat\kappa =
    \begin{cases}
        \max\left(\hat\kappa - \frac{2}{n\hat\kappa}, 0\right), & \text{if } \hat\kappa < 2 \\
        \frac{(n - 1)^3 \hat\kappa}{n^3 + n}, & \text{if } \hat\kappa \ge 2
    \end{cases}
    $$


    Parameters
    ----------
    r: float
        Resultant vector length
    n: int or None
        Sample size. If n is not None, the adjustment for small sample size will be applied.

    Returns
    -------
    kappa: float
        Concentration parameter

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

def circ_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""
    Compute the pairwise circular difference $x_i - y_i$ using complex representation.
    
    Parameters
    ----------
    x : array-like
        Sample of circular data (radians).
    y : array-like
        Sample of circular data (radians) or a single angle.

    Returns
    -------
    array
        Circular differences wrapped to [-pi, pi].
    
    References
    ----------
    - Section 27.7 (Zar, 2010, P642)

    """
    return np.angle(np.exp(1j * x) / np.exp(1j * y))


def circ_pairdist(x: np.ndarray, y: Optional[np.ndarray]=None) -> np.ndarray:
    r"""
    Compute all pairwise circular differences $x_i - y_j$ around the circle.
    
    Parameters
    ----------
    x : array-like
        Sample of circular data (radians).
    y : array-like, optional
        Sample of circular data (radians). If None, computes all pairwise 
        differences within x.
    
    Returns
    -------
    ndarray
        Matrix of pairwise circular differences wrapped to [-pi, pi].

    References
    ----------
    - Section 27.7 (Zar, 2010, P642)
    """
    x = np.asarray(x)
    
    if y is None:
        y = x  # Compute pairwise distances within x itself
    
    y = np.asarray(y)

    # Broadcasting-friendly complex exponentiation method
    return np.angle(np.exp(1j * x[:, None]) / np.exp(1j * y[None, :]))



#########################
# Convinience functions #
#########################


def convert_moment(
    mp: complex,
) -> Tuple[float, float]:
    """
    Convert complex moment to polar coordinates.

    Parameters
    ----------
    mp: complex
        Complex moment

    Returns
    -------
    u: float
        Angle in radian
    r: float
        Magnitude

    """

    u = angmod(np.angle(mp))
    r = np.abs(mp)

    return u, r


def compute_C_and_S(
    alpha: np.ndarray,
    w: np.ndarray,
    p: int = 1,
    mean: Union[float, np.ndarray] = 0.0,
) -> Tuple[float, float]:
    r"""
    Compute the intermediate values Cbar and Sbar.

    $$
    \displaylines{
    \bar{C}_{p} = \frac{\sum_{i=1}^{n} w_{i} \cos(p(\alpha_{i} - \mu))}{n} \\
    \bar{S}_{p} = \frac{\sum_{i=1}^{n} w_{i} \sin(p(\alpha_{i} - \mu))}{n}
    }
    $$

    Parameters
    ----------
    alpha: np.ndarray
        Angles in radian.
    w: np.ndarray
        Frequencies or weights.
    p: int, optional
        Order of the moment (default is 1, for the first moment).
    mean: float, optional
        Mean angle (μ) to center the computation (default is 0.0).

    Returns
    -------
    Cbar: float
        Weighted mean cosine for the given moment.
    Sbar: float
        Weighted mean sine for the given moment.
    """
    n = np.sum(w)
    Cbar = np.sum(w * np.cos(p * (alpha - mean))) / n
    Sbar = np.sum(w * np.sin(p * (alpha - mean))) / n

    return Cbar, Sbar


def compute_hdi(samples, ci=0.95):
    """
    Compute the Highest Density Interval (HDI) for circular data.

    Parameters
    ----------
    samples : np.ndarray
        Bootstrap samples of the circular mean in radians.
    ci : float, optional
        Credible interval (default is 0.95 for 95% HDI).

    Returns
    -------
    hdi : tuple
        Lower and upper bounds of the HDI in radians.
    """
    # Wrap samples to [0, 2π) for circular consistency
    wrapped_samples = angmod(samples)

    # Sort the samples
    sorted_samples = np.sort(wrapped_samples)

    # Number of samples in the HDI
    n_samples = len(sorted_samples)
    interval_idx = int(np.floor(ci * n_samples))
    if interval_idx == 0:
        raise ValueError("Insufficient data to compute HDI.")

    # Find the shortest interval
    hdi_width = np.inf
    hdi_bounds = (None, None)
    for i in range(n_samples - interval_idx):
        lower = sorted_samples[i]
        upper = sorted_samples[i + interval_idx]
        width = angmod(upper - lower)  # Handle wrapping for circularity
        if width < hdi_width:
            hdi_width = width
            hdi_bounds = (lower, upper)

    return hdi_bounds


def compute_smooth_params(r: float, n: int) -> float:
    """
    Parameters
    ----------
    r: float
        resultant vector length
    n: int
        sample size

    Returns
    -------
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
    n = len(a)
    x = np.linspace(0, 2 * np.pi, 100)
    d = np.abs(x[:, None] - a)
    e = np.minimum(d, 2 * np.pi - d)
    e = np.minimum(e, h)
    sum = np.sum((1 - e**2 / h**2) ** 2, 1)
    f = 0.9375 * sum / n / h

    f = radius * np.sqrt(1 + np.pi * f) - radius

    return x, f

def circ_range(alpha: np.ndarray) -> float:
    """
    Compute the circular range of angular data.
    
    The circular range is the difference between the maximum and minimum angles 
    in the dataset, adjusted for circular continuity.

    Parameters
    ----------
    alpha : np.ndarray
        Angles in radians.

    Returns
    -------
    float
        Circular range, a measure of clustering (higher = more clustered).

    Reference
    ---------
    P162, Section 7.2.3 of Jammalamadaka, S. Rao and SenGupta, A. (2001)
    """
    alpha = np.sort(alpha % (2 * np.pi))  # Convert to [0, 2π) and sort
    spacings = np.diff(alpha, prepend=alpha[-1] - 2 * np.pi)  # Compute spacings
    return 2 * np.pi - np.max(spacings)  # Circular range

from typing import Union

import numpy as np

from .descriptive import circ_median


def circ_quantile(
    alpha: np.ndarray,
    probs: Union[float, np.ndarray] = np.array([0, 0.25, 0.5, 0.75, 1.0]),
    type: int = 7,
) -> np.ndarray:
    """
    Compute quantiles for circular data.

    This function computes quantiles for circular data by shifting the 
    data to be centered around the circular median, applying a linear quantile function,
    and then shifting back.

    Parameters
    ----------
    alpha : np.ndarray
        Sample of circular data (radians).
    probs : float or np.ndarray, optional
        Probabilities at which to compute quantiles. Default is `[0, 0.25, 0.5, 0.75, 1.0]`.
    type : int, optional
        Quantile algorithm type (default `7`, matches R’s default quantile type).

    Returns
    -------
    np.ndarray
        Circular quantiles.

    References
    ----------
    - R's `quantile.circular` from the `circular` package.
    - Fisher (1993), Section 2.3.2.
    """

    # Convert to numpy array
    alpha = np.asarray(alpha)
    probs = np.atleast_1d(probs)

    # Compute circular median
    circular_median = circ_median(alpha)

    # If the median is NaN (e.g., uniform data), return NaNs
    if np.isnan(circular_median):
        return np.full_like(probs, np.nan)

    # Transform data relative to circular median
    shifted_alpha = (alpha - circular_median) % (2 * np.pi)
    shifted_alpha = np.where(shifted_alpha > np.pi, shifted_alpha - 2 * np.pi, shifted_alpha)

    # Compute linear quantiles on transformed data
    linear_quantiles = np.quantile(shifted_alpha, probs, method="linear" if type == 7 else "midpoint")

    # Transform back to original circular space
    circular_quantiles = (linear_quantiles + circular_median) % (2 * np.pi)

    return circular_quantiles
