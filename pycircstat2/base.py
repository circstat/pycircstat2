from typing import Optional, Union

import numpy as np

from .clustering import MovM
from .descriptive import (
    angular_std,
    circ_kappa,
    circ_kurtosis,
    circ_mean_and_r,
    circ_mean_ci,
    circ_median,
    circ_median_ci,
    circ_skewness,
    circ_std,
)
from .hypothesis import rayleigh_test
from .utils import data2rad, rad2data, rotate_data, significance_code
from .visualization import circ_plot

__names__ = ["Circular", "Axial"]


class Circular:
    r"""
    Circular Data Analysis Object.

    This class encapsulates circular data and provides tools for descriptive statistics,
    hypothesis testing, and visualization. It automatically computes key circular
    statistics and tests when the data are loaded.

    Parameters
    ----------
    data : array-like (n,)
        The raw circular data, typically in degrees, radians, or other angular units.

    w : array-like (n,) or None, optional
        Frequencies or weights for the data points. If None, all data points are treated equally.
        Default is None.

    bins : int, array-like (n+1,) or None, optional
        Number of bins or bin edges to group the data. If None, the data is not binned.
        Default is None.

    unit : str, optional
        Unit of the input data. Must be one of {"degree", "radian", "hour"}.
        Default is "degree".

    full_cycle : int, float, or None, optional
        The total range of values that complete one full cycle in the chosen unit.
        If None, the value is inferred based on the unit:

        - 360 for degrees,
        - $2\pi$ for radians,
        - 24 for hours.

        Custom values can be set explicitly for other units.
        Default is None.

    n_clusters_max : int, optional
        Maximum number of clusters to test for a mixture of von Mises distributions.
        Default is 1.

    kwargs : dict, optional
        Additional keyword arguments to customize the computation of statistics such as the median.

    Attributes
    ----------
    n : int
        Total sample size, including weights.

    mean : float
        Angular mean in radians.

    mean_ci : tuple of float
        Confidence interval for the angular mean, if applicable.

    median : float
        Angular median in radians.

    median_ci : tuple of float
        Confidence interval for the angular median, if computed.

    r : float
        Resultant vector length, measuring data concentration (0 to 1).

    kappa : float
        Concentration parameter, measuring data sharpness.

    s : float
        Angular deviation, measuring data dispersion.

    skewness : float
        Circular skewness of the data.

    kurtosis : float
        Circular kurtosis of the data.

    R : float
        Rayleigh's R statistic, derived from the resultant vector length.

    mixtures : list
        Mixture models of von Mises distributions fitted to the data (if `n_clusters_max > 1`).

    Methods
    -------
    summary()
        Returns a detailed summary of the computed statistics.

    plot(ax=None, kind=None, **kwargs)
        Visualizes the circular data, including histograms and other representations.

    Notes
    -----
    - Angular data is automatically converted to radians for internal computations.
    - Data can be grouped or ungrouped. Ungrouped data is handled by assigning equal weights.
    - The Rayleigh test for angular mean is computed, with p-values indicating significance.
    - Confidence intervals for the angular mean are approximated using either bootstrap
      or dispersion methods, depending on the sample size and significance.

    References
    ----------
    - Zar, J. H. (2010). Biostatistical Analysis (5th Edition). Pearson.
    - Fisher, N. I. (1995). Statistical Analysis of Circular Data. Cambridge University Press.

    Examples
    --------

    #### Basic Usage

    ```python
    data = [30, 60, 90, 120, 150]
    circ = Circular(data, unit="degree")
    print(circ.summary())
    ```

    #### Grouped Data

    ```python
    data = [0, 30, 60, 90]
    weights = [1, 2, 3, 4]
    circ = Circular(data, w=weights, unit="degree")
    print(circ.summary())
    ```
    """

    def __init__(
        self,
        data: Union[np.ndarray, list],  # angle
        w: Optional[Union[np.ndarray, list]] = None,  # frequency
        bins: Optional[Union[int, np.ndarray]] = None,
        unit: str = "degree",
        full_cycle: Optional[Union[
            int, float
        ]] = None,  # number of intervals in the full cycle
        n_clusters_max: int = 1,  # number of clusters to be tested for mixture of von Mises
        rotate: Optional[float] = None, # in rad
        **kwargs,
    ):
        # meta
        self.unit = unit
        if full_cycle is None:
            if unit == "degree":
                self.full_cycle = full_cycle = 360
            elif unit == "radian":
                self.full_cycle = full_cycle = 2 * np.pi
            elif unit == "hour":
                self.full_cycle = full_cycle = 24
            else:
                raise ValueError(
                    "You need to provide a value for `full_cycle` if it is not `degree` or `hour`."
                )
        else:
            self.full_cycle = full_cycle

        self.n_clusters_max = n_clusters_max
        self.kwargs_median = kwargs_median = {
            **{
                "method": "deviation",
                "return_average": True,
                "average_method": "all",
            },
            **kwargs.pop("kwargs_median", {}),
        }
        self.kwargs_mean_ci = kwargs_mean_ci = kwargs.pop("kwargs_mean_ci", None)

        # data
        self.data = data = np.array(data) if isinstance(data, list) else data
        self.alpha = alpha = np.array(data2rad(data, full_cycle)) if rotate is None else rotate_data(np.array(data2rad(data, full_cycle)), rotate, unit="radian")

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
                self.w = w = np.asarray(w)
                self.grouped = grouped = True
                self.bin_size = bin_size = np.diff(alpha).min()
                self.alpha_lb = alpha - bin_size / 2
                self.alpha_ub = alpha + bin_size / 2

        # bin data usingse np.histogram
        else:
            if isinstance(bins, int) or isinstance(bins, np.ndarray):
                w, alpha = np.histogram(
                    alpha, bins=bins, range=(0, 2 * np.pi)
                )  # np.histogram return bin edges
            self.w = w = np.asarray(w)
            self.alpha_lb = alpha[:-1]  # bin lower bound
            self.alpha_ub = alpha[1:]  # bin upper bound
            self.alpha = alpha = 0.5 * (alpha[:-1] + alpha[1:])  # get bin centers
            self.grouped = grouped = True
            self.bin_size = bin_size = np.diff(alpha).min()

        # sample size
        self.n = n = np.sum(w).astype(int)

        # angular mean and resultant vector length
        self.mean, self.r = (_, r) = circ_mean_and_r(alpha=alpha, w=w)

        # z-score and p-value from rayleigh test for angular mean
        self.mean_test_result = rayleigh_test_result = rayleigh_test(n=n, r=r)
        mean_pval = rayleigh_test_result.pval

        # Rayleigh's R
        self.R = n * r

        # kappa
        self.kappa = circ_kappa(r=r, n=n)

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
            self.mean_lb, self.mean_ub = np.nan, np.nan

        # angular deviation, circular standard deviation, adjusted resultant vector length (if needed)
        self.s = angular_std(r=r, bin_size=bin_size)
        self.s0 = circ_std(r=r, bin_size=bin_size)

        # angular median
        if n > 10000 and kwargs_median["method"] is not None:
            print(
                "Sample size is large (n>10000), it will take a while to find the median.\nOr set `kwargs_median={'method': None}` to skip."
            )

        self.median = median = circ_median(
            alpha=alpha,
            w=w,
            method=kwargs_median["method"],
            return_average=kwargs_median["return_average"],
            average_method=kwargs_median["average_method"],
        )

        # confidence inerval for angular median (only for ungrouped data)
        # it's unclear how to do it for grouped data.
        if not grouped and not np.isnan(median):
            self.median_lb, self.median_ub, self.median_ci_level = circ_median_ci(
                median=float(median), alpha=alpha
            )

        self.skewness = circ_skewness(alpha=alpha, w=w)
        self.kurtosis = circ_kurtosis(alpha=alpha, w=w)

        # check multimodality
        self.mixtures = []
        if n_clusters_max > 1:
            for k in range(1, n_clusters_max + 1):
                m = MovM(
                    n_clusters=k,
                    full_cycle=full_cycle,
                    unit="radian",
                    random_seed=0,
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
        k = self.full_cycle

        docs = "Circular Data\n"
        docs += "=============\n\n"

        docs += "Summary\n"
        docs += "-------\n"
        docs += "  Grouped?: Yes\n" if self.grouped else "  Grouped?: No\n"
        if self.n_clusters_max > 1 and self.mixture_opt is not None:
            docs += (
                "  Unimodal?: Yes \n"
                if len(self.mixture_opt.m_) == 1
                else f"  Unimodal?: No (n_clusters={len(self.mixture_opt.m_)}) \n"
            )

        docs += f"  Unit: {unit}\n"
        docs += f"  Sample size: {self.n}\n"

        docs += f"  Angular mean: {rad2data(self.mean, k=k):.02f} ( p={self.mean_test_result.pval:.04f} {significance_code(self.mean_test_result.pval)} ) \n"

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

        docs += "\n"

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
        r"""
        Summary of basic statistics for circular data.

        This method generates a textual summary of the key descriptive and inferential
        statistics computed for the circular data. It provides information about
        the data type, concentration, dispersion, and more.

        The summary includes the following components:

        1. **Grouping**:

            Indicates whether the data is grouped (binned) or ungrouped.

        2. **Unimodality**:

            For models with mixtures of von Mises distributions, it specifies whether
        the data is unimodal or multimodal, along with the number of clusters if applicable.

        3. **Data Characteristics**:

            - The unit of measurement (e.g., degrees, radians, hours).
            - Total sample size, including weights if provided.

        4. **Angular Mean**:

            - The angular mean, with its corresponding p-value from the Rayleigh test.
            - The confidence interval (CI) for the angular mean, if available.

        5. **Angular Median**:

            - The angular median, representing the central tendency.
            - The confidence interval (CI) for the angular median, if applicable.

        6. **Measures of Dispersion**:

            - Angular deviation ($s$): A measure of spread in circular data.
            - Circular standard deviation ($s_0$): An alternative dispersion measure.

        7. **Measures of Concentration**:

            - Resultant vector length ($r$): A measure of data concentration, ranging from 0 (uniform) to 1 (highly concentrated).
            - Concentration parameter ($\kappa$): Indicates sharpness or clustering of the data.

        8. **Higher-Order Statistics**:

            - Circular skewness: A measure of asymmetry.
            - Circular kurtosis: A measure of peakedness or flatness relative to a uniform distribution.

        9. **Significance Codes**:

            - A guide to interpret the p-values of statistical tests.

        10. **Methods**:

            - The method used for calculating the angular median.
            - The method used for estimating confidence intervals for the angular mean.
        """

        return self.__repr__()

    def plot(self, ax=None, config=None):
        """
        Visualize circular data.

        This method provides various visualization options for circular data, including scatter
        plots, density plots, and rose diagrams. It is a wrapper around the `circ_plot` function.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            The matplotlib Axes object where the plot will be drawn. If None, a new Axes object
            is created. Default is None.
        config: dict, optional
            Configuration dictionary that overrides defaults.
        
        Returns
        -------
        ax : matplotlib.axes._axes.Axes
            The matplotlib Axes object containing the plot.

        Notes
        -----
        - This method supports both grouped and ungrouped data.
        - Density estimation can be performed using either nonparametric methods or mixtures
        of von Mises distributions.
        - The rose diagram represents grouped data as a histogram over angular bins.
        - Confidence intervals for the mean and median are plotted as arcs on the circle.

        Examples
        --------
        ```
        from pycircstat2 import load_data, Circular

        data = load_data("B3", source="fisher")["θ"].values
        c = Circular(data, unit="degree")
        c.plot(config={"scatter": {"color" : "blue", "size": 15}})
        ```

        See docstring of `circ_plot` for more examples and customization options.
        """
        ax = circ_plot(self, ax=ax, config=config)


class Axial(Circular):
    def __init__(
        self,
        data: Union[np.ndarray, list],  # angle
        w: Union[np.ndarray, list, None] = None,  # frequency
        bins: Union[int, np.ndarray, None] = None,
        unit: str = "degree",
        full_cycle: Union[
            int, float, None
        ] = None,  # number of intervals in the full cycle
        n_clusters_max: int = 1,  # number of clusters to be tested for mixture of von Mises
        **kwargs,
    ):
        
        data = np.array(data) if isinstance(data, list) else data
        # doubling original data and reducing them modulo 360 degrees
        if unit == "degree":
            data_double = 2 * data % 360
        elif unit == "radian":
            data_double = 2 * data % (2 * np.pi)
        elif unit == "hour":
            data_double = 2 * data % 24
        else:
            data_double = 2 * data % full_cycle

        super().__init__(
            data=data_double,
            w=w,
            bins=bins,
            unit=unit,
            full_cycle=full_cycle,
            n_clusters_max=n_clusters_max,
            **kwargs,
        )

        self.data_double = data_double
        self.data = data
        self.alpha_double = self.alpha
        self.alpha = data2rad(data, k=self.full_cycle)
        self.mean /= 2
        self.mean_lb /= 2
        self.mean_ub /= 2
