from typing import Union

import numpy as np
import pandas as pd

from .clustering import MoVM
from .descriptive import (
    circ_kappa,
    circ_kurtosis,
    circ_mean,
    circ_mean_ci,
    circ_median,
    circ_median_ci,
    circ_skewness,
    circ_std,
)
from .hypothesis import rayleigh_test
from .utils import data2rad, rad2data, significance_code
from .visualization import circ_plot

__names__ = ["Circular"]

# Proposal: Automatic circular data analaysis pipeline
#
# Circstat(data)
# |-> MoVM (BIC-based clustering)
# |     |-> Single Cluster
# |     |       | -> Cicular(data)
# |     |-> Multiple Clusters
# |     |       | -> Two Clusters: if means are 180 degree seperated
# |     |       |       | -> Axial(data) (TODO)
# |     |       | -> Multiple Clusters:
# |     |       |       | -> list of Circular(data_cluster)
# |     |       |               | -> MoMeans(list of Circular)
# |     |       |       | -> Circular(data) (aggregate no matter what)


class CircStat:

    """
    An automatic pipeline for circular data analysis.
    """

    pass


class Circular:

    """
    An Object to hold one set of circular data.
    Simple descriptive statistics and hypothesis testing were
    computed automatically when data are loaded.

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
        n_intervals: Union[
            int, float, None
        ] = None,  # number of intervals in the full cycle
        n_clusters_max: int = 1,  # number of clusters to be tested for mixture of von Mises
        **kwargs,
    ):
        # meta
        self.unit = unit
        if n_intervals is None:
            if unit == "degree":
                self.n_intervals = n_intervals = 360
            elif unit == "radian":
                self.n_intervals = n_intervals = 2 * np.pi
            elif unit == "hour":
                self.n_intervals = n_intervals = 24
            else:
                raise ValueError(
                    "You need to provide a value for `n_intervals` if it is not `degree`, `radian` or hour."
                )
        else:
            self.n_intervals = n_intervals

        self.n_clusters_max = n_clusters_max
        self.kwargs_median = kwargs_median = kwargs.pop(
            "kwargs_median", {"method": "deviation"}
        )
        self.kwargs_mean_ci = kwargs_mean_ci = kwargs.pop("kwargs_mean_ci", None)

        # data
        self.data = data = np.array(data) if isinstance(data, list) else data
        self.alpha = alpha = data2rad(data, n_intervals)

        # data preprocessing
        if bins is None:
            if w is None:  # ungrouped data, because no `w` is provided.
                self.w = w = np.ones_like(alpha).astype(int)
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
        self.mean, self.r = (mean, r) = circ_mean(alpha=alpha, w=w, return_r=True)

        # z-score and p-value from rayleigh test for angular mean
        self.mean_z, self.mean_pval = (mean_z, mean_pval) = rayleigh_test(n=n, r=r)

        # Rayleigh's R
        self.R = n * r

        # kappa
        self.kappa = kappa = circ_kappa(r=r, n=n)

        # confidence interval for angular mean
        # in practice, the equations for approximating mean ci for 8 <= n <= 12 in zar 2010
        # can still yield nan
        if self.kwargs_mean_ci is None:
            if mean_pval < 0.05 and (8 <= self.n < 25):
                self.method_mean_ci = method_mean_ci = "bootstrap"
                self.mean_ci_level = mean_ci_level = 0.95
            elif mean_pval < 0.05 and self.n >= 25:
                # Eq 4.22 (Fisher, 1995)
                self.method_mean_ci = method_mean_ci = "dispersion"
                self.mean_ci_level = mean_ci_level = 0.95
            else:  # mean_pval > 0.05
                self.method_mean_ci = method_mean_ci = None
                self.mean_ci_level = mean_ci_level = np.nan
        else:
            self.method_mean_ci = method_mean_ci = kwargs_mean_ci.pop(
                "method", "bootstrap"
            )
            self.mean_ci_level = mean_ci_level = 0.95

        if method_mean_ci is not None and mean_pval < 0.05:
            self.mean_lb, self.mean_ub = mean_lb, mean_ub = circ_mean_ci(
                alpha=self.alpha,
                w=self.w,
                mean=self.mean,
                r=self.r,
                n=self.n,
                ci=mean_ci_level,
                method=method_mean_ci,
            )
        else:
            self.mean_lb, self.mean_ub = mean_lb, mean_ub = np.nan, np.nan

        # angular deviation, circular standard deviation, adjusted resultant vector length (if needed)
        self.s, self.s0, self.rc = s, s0, rc = circ_std(r=r, bin_size=bin_size)

        # angular median
        self.median = median = circ_median(
            alpha=alpha, w=w, grouped=grouped, method=kwargs_median["method"]
        )

        # confidence inerval for angular median (only for ungrouped data)
        # it's unclear how to do it for grouped data.
        if not grouped and not np.isnan(median):
            self.median_lb, self.median_ub, self.median_ci_level = (
                median_lb,
                median_ub,
                median_ci_level,
            ) = circ_median_ci(median=median, alpha=alpha)

        self.skewness = circ_skewness(alpha=alpha, w=w)
        self.kurtosis = circ_kurtosis(alpha=alpha, w=w)

        # check multimodality
        self.mixtures = []
        for k in range(1, n_clusters_max + 1):
            m = MoVM(
                n_clusters=k, n_intervals=n_intervals, unit="radian", random_seed=0
            )
            m.fit(np.repeat(alpha, w))
            self.mixtures.append(m)
        self.mixtures_BIC = [m.compute_BIC() for m in self.mixtures]
        if not np.isnan(self.mixtures_BIC).all():
            self.mixture_opt = self.mixtures[np.nanargmin(self.mixtures_BIC)]
        else:
            self.mixture_opt = None

    def __repr__(self):
        unit = self.unit
        k = self.n_intervals

        docs = "Circular Data\n"
        docs += "=============\n\n"

        docs += "Summary\n"
        docs += "-------\n"
        docs += f"  Grouped?: Yes\n" if self.grouped else f"  Grouped?: No\n"
        if self.n_clusters_max > 1 and self.mixture_opt is not None:
            docs += (
                f"  Unimodal?: Yes \n"
                if len(self.mixture_opt.m) == 1
                else f"  Unimodal?: No (n_clusters={len(self.mixture_opt.m)}) \n"
            )

        docs += f"  Unit: {unit}\n"
        docs += f"  Sample size: {self.n}\n"

        if hasattr(self, "d"):
            docs += f"  Angular mean: {rad2data(self.mean, k=k):.02f} ± {rad2data(self.d, k=k):.02f} ( p={self.mean_pval:.04f} {significance_code(self.mean_pval)} ) \n"
        else:
            docs += f"  Angular mean: {rad2data(self.mean, k=k):.02f} ( p={self.mean_pval:.04f} {significance_code(self.mean_pval)} ) \n"

        if hasattr(self, "mean_lb") and not np.isnan(self.mean_lb):
            docs += f"  Angular mean CI ({self.mean_ci_level:.2f}): {rad2data(self.mean_lb, k=k):.02f} - {rad2data(self.mean_ub, k=k):.02f}\n"

        docs += f"  Angular median: {rad2data(self.median, k=k):.02f} \n"
        if hasattr(self, "median_lb") and not np.isnan(self.median_lb):
            docs += f"  Angular median CI ({self.median_ci_level:.2f}): {rad2data(self.median_lb, k=k):.02f} - {rad2data(self.median_ub, k=k):.02f}\n"

        docs += f"  Angular deviation (s): {rad2data(self.s, k=k):.02f} \n"
        docs += f"  Circular standard deviation (s0): {rad2data(self.s0, k=k):.02f} \n"
        docs += f"  Concentration (r): {self.r:0.2f}\n"
        docs += f"  Concentration (kappa): {self.kappa:0.2f}\n"
        docs += f"  Skewness: {self.skewness:0.3f}\n"
        docs += f"  Kurtosis: {self.kurtosis:0.3f}\n"

        docs += f"\n"

        docs += "Signif. codes:\n"
        docs += "--------------\n"
        docs += " 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n\n"

        docs += "Method\n"
        docs += "------\n"
        docs += f"  Angular median: {self.kwargs_median['method']}\n"
        docs += f"  Angular mean CI: {self.method_mean_ci}\n"

        return docs

    def __str__(self):
        return self.__repr__()

    def summary(self):
        """
        Summary of basis statistcs.
        """

        return self.__repr__()

    def plot(self, ax=None, kind=None, **kwargs):
        ax = circ_plot(self, ax=ax, **kwargs)


class Axial(Circular):
    def __init__(
        self,
        data: Union[np.ndarray, list],  # angle
        w: Union[np.ndarray, list, None] = None,  # frequency
        bins: Union[int, np.ndarray, None] = None,
        unit: str = "degree",
        n_intervals: Union[
            int, float, None
        ] = None,  # number of intervals in the full cycle
        n_clusters_max: int = 1,  # number of clusters to be tested for mixture of von Mises
        **kwargs,
    ):
        # doubling original data and reducing them modulo 360 degrees
        if unit == "degree":
            data_double = 2 * data % 360
        elif unit == "radian":
            data_double = 2 * data % (2 * np.pi)
        elif unit == "hour":
            data_double = 2 * data % 24
        else:
            data_double = 2 * data % n_intervals

        super().__init__(
            data=data_double,
            w=w,
            bins=bins,
            unit=unit,
            n_intervals=n_intervals,
            n_clusters_max=n_clusters_max,
            **kwargs,
        )

        self.data_double = data_double
        self.data = data
        self.alpha_double = self.alpha
        self.alpha = data2rad(data, k=self.n_intervals)
        self.mean /= 2
        self.mean_lb /= 2
        self.mean_ub /= 2
