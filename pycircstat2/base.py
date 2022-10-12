from typing import Union

import numpy as np
import pandas as pd

from .descriptive import (circ_mean, circ_mean_ci, circ_median, circ_median_ci,
                          circ_std)
from .hypothesis import rayleigh_test
from .plot import plot_rose, plot_scatter
from .utils import data2rad, rad2data, significance_code


class Circular:

    """
    An Object to hold circular data.
    Sufficient statistics were computed automatically when data are loaded.

    Parameters
    ----------
    data: np.array (n, )
        Raw data.

    w: np.array (n, ) or None
        Frequencies, or weights. Default is None.

    bins: int, np.array (n+1, ) or None
        Bin edges for binning raw data.

    unit: str
        Unit of the data.

    k: int
        Number of intervel. For converting raw data
        into radian `alpha = data2rad(data, k)`.

    kwargs_median: dict
        A dictionary with additional keyword argument
        for computing median.
        `median = circ_median(alpha, w, kwargs_median)`

    Reference
    ---------
    - Chapter 26 of Biostatistical Analysis, Fifth Edition, Jerrold H. Zar. (2010)
    """

    def __init__(
        self,
        data: Union[np.ndarray, list],  # angle
        w: Union[np.ndarray, list, None] = None,  # frequency
        bins: Union[int, np.array, None] = None,
        unit: str = "degree",
        k: Union[int, float] = 360,  # number of intervals in the full cycle
        kwargs_median={"method": "deviation"},
        kwargs_mean_ci=None,
    ):

        # meta
        self.k = k
        self.unit = unit
        self.kwargs_median = kwargs_median

        # data
        self.data = np.array(data) if isinstance(data, list) else data
        self.alpha = alpha = data2rad(data, k)

        # data preprocessing
        if bins is None:
            if w is None:  # ungrouped data, because no `w` is provided.
                self.w = w = np.ones_like(alpha)
                self.grouped = grouped = False
                self.bin_size = bin_size = 0.0
            else:  # grouped data
                assert len(w) == len(alpha), "`w` and `data` must be the same length."
                assert len(w) == len(
                    np.arange(0, 2 * np.pi, 2 * np.pi / len(w))
                ), "Grouped data should included empty bins."
                self.w = np.array(w) if isinstance(w, list) else w
                self.grouped = grouped = True
                self.bin_size = bin_size = np.diff(alpha).min()
                self.alpha_lb = alpha_lb = alpha - bin_size / 2
                self.alpha_ub = alpha_ub = alpha + bin_size / 2

        ## bin data usingse np.histogram
        else:
            if isinstance(bins, int) or isinstance(bins, np.ndarray):
                w, alpha = np.histogram(
                    alpha, bins=bins, range=(0, 2 * np.pi)
                )  # np.histogram return bin edges
            self.w = w
            self.alpha_lb = alpha_lb = alpha[:-1]  # bin lower bound
            self.alpha_ub = alpha_ub = alpha[1:]  # bin upper bound
            self.alpha = alpha = 0.5 * (alpha[:-1] + alpha[1:])  # get bin centers
            self.grouped = grouped = True
            self.bin_size = bin_size = np.diff(alpha).min()

        # sample size
        self.n = n = np.sum(w).astype(int)

        # angular mean and resultant vector length
        self.mean, self.r = (mean, r) = circ_mean(alpha=alpha, w=w)

        # z-score and p-value from rayleigh test for angular mean
        self.mean_z, self.mean_pval = (mean_z, mean_pval) = rayleigh_test(n=n, r=r)

        # Rayleigh's R
        self.R = n * r

        # kappa
        # self.kappa = kappa = circ_kappa(r=r, n=n)

        # confidence interval for angular mean
        # in practice, the equations for mean ci for 8 <= n <= 12 can still yield nan
        if kwargs_mean_ci is None:
            if not np.isclose(self.r, 0) and (8 < self.n < 25):
                # Approximate ci for mean of a von Mises distribution (Upton, 1986)
                self.method_mean_ci = method_mean_ci = "bootstrap"
                ci = 0.95
            elif not np.isclose(self.r, 0) and self.n >= 25:
                # Eq 4.22 (Fisher, 1995)
                self.method_mean_ci = method_mean_ci = "dispersion"
                ci = 0.95
            else:
                self.method_mean_ci = method_mean_ci = None
                ci = 0.95
        else:
            self.method_mean_ci = method_mean_ci = kwargs_mean_ci.pop(
                "method", "bootstrap"
            )

        if method_mean_ci is not None:
            self.mean_lb, self.mean_ub = mean_lb, mean_ub = circ_mean_ci(
                alpha=self.alpha,
                w=self.w,
                mean=self.mean,
                r=self.r,
                n=self.n,
                ci=ci,
                method=method_mean_ci,
            )

        # angular deviation, circular standard deviation, adjusted resultant vector length (if needed)
        self.s, self.s0, self.rc = s, s0, rc = circ_std(r=r, bin_size=bin_size)

        # angular median
        self.median = median = circ_median(
            alpha=alpha, w=w, grouped=grouped, method=kwargs_median["method"]
        )

        # confidence inerval for angular median (only for ungrouped data)
        # it's unclear how to do it for grouped data.
        if not grouped and not np.isnan(median):
            if n not in np.arange(7, 14):
                self.median_lb, self.median_ub, self.ci = (
                    median_lb,
                    median_ub,
                    ci,
                ) = circ_median_ci(median=median, alpha=alpha)
            else:
                self.median_lb, self.median_ub, self.ci = (
                    median_lb,
                    median_ub,
                    ci,
                ) = circ_median_ci(median=median, alpha=alpha)[0]

    def __repr__(self):

        unit = self.unit
        k = self.k

        docs = "Circular Data\n"
        docs += "=============\n\n"

        docs += "Summary\n"
        docs += "-------\n"
        docs += f"  Grouped?: Yes\n" if self.grouped else f"  Grouped?: No\n"
        docs += f"  Unit: {unit}\n"
        docs += f"  Sample size: {self.n}\n"

        if hasattr(self, "d"):
            docs += f"  Angular mean: {rad2data(self.mean, k=k):.02f} ± {rad2data(self.d, k=k):.02f} ( p={self.mean_pval:.04f} {significance_code(self.mean_pval)} ) \n"
        else:
            docs += f"  Angular mean: {rad2data(self.mean, k=k):.02f} ( p={self.mean_pval:.04f} {significance_code(self.mean_pval)} ) \n"

        docs += f"  Angular median: {rad2data(self.median, k=k):.02f} \n"
        if hasattr(self, "median_lb"):
            docs += f"  Angular median CI: {rad2data(self.median_lb, k=k):.02f} - {rad2data(self.median_ub, k=k):.02f}\n"

        docs += f"  Angular deviation (s): {rad2data(self.s, k=k):.02f} \n"
        docs += f"  Circular standard deviation (s0): {rad2data(self.s0, k=k):.02f} \n"
        docs += f"  Concentration (r): {self.r:0.2f}\n\n"

        docs += "Signif. codes:\n"
        docs += "--------------\n"
        docs += " 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n\n"

        docs += "Method\n"
        docs += "------\n"
        docs += f"  Angular median: {self.kwargs_median['method']}\n"
        # docs += f"  Angular mean CI: {self.method_mean_ci}"
        return docs

    def __str__(self):

        return self.__repr__()

    def plot(self, ax=None, kind=None, **kwargs):

        if kind is None:
            kind = "rose" if self.grouped else "scatter"

        if kind == "scatter":

            plot_scatter(self, ax=ax, **kwargs)

        elif kind == "rose":

            plot_rose(self, ax=ax, **kwargs)
