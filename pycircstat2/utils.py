import json
from typing import Union

import numpy as np
import pandas as pd
import pkg_resources


def data2rad(
    data: Union[np.ndarray, float, int],
    k: Union[float, int] = 360,  # number of intervals in the full cycle
) -> Union[np.ndarray, float]:  # eq(26.1), zar 2010
    """Convert data measured on a circular scale to corresponding angular
    directions.
    """
    return 2 * np.pi * data / k


def rad2data(
    rad: Union[np.ndarray, float, int], k: Union[float, int] = 360
) -> Union[np.ndarray, float]:
    return k * rad / (2 * np.pi)  # eq(26.12), zar 2010


def time2float(x, sep=":"):
    """Convert string of time to float. E.g. 12:15 ->"""
    hr, min = x.split(sep)
    return float(hr) + float(min) / 60


def angrange(rad: Union[np.ndarray, float, int]) -> Union[np.ndarray, float]:
    return ((rad % (2 * np.pi)) + 2 * np.pi) % (2 * np.pi)


def significance_code(p: float) -> str:
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    elif p < 0.1:
        sig = "."
    else:
        sig = ""
    return sig


def load_data(
    name,
    source="fisher_1993",
    print_meta=False,
    return_meta=False,
):

    __source__ = ["fisher_1993", "zar_2010", "mardia_1972"]

    # check source
    if source not in __source__:
        raise ValueError(
            f"Invalid source ('{source}').\n Availble sources: {__source__}"
        )

    # load data
    csv_path = pkg_resources.resource_filename(__name__, f"data/{source}/{name}.csv")
    csv_data = pd.read_csv(csv_path, index_col=0)

    # load meta data
    json_path = pkg_resources.resource_filename(
        __name__, f"data/{source}/{name}.csv-metadata.json"
    )
    with open(json_path) as f:
        json_data = json.load(f)

    if print_meta:
        print(json.dumps(json_data, indent=4, ensure_ascii=False))

    if return_meta:
        return csv_data, json_data
    else:
        return csv_data


def compute_kappa(r: float, n: int) -> float:

    """Approximate kappa
    Parameters
    ----------
    r: float
        resultant vector length
    n: int
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
        kappa = 1 / (r**3 - 4 * r**2 + 3 * r)

    # eq 4.41
    if n <= 15 and r < 0.7:
        if kappa < 2:
            kappa = np.max(kappa - 2 * 1 / (n * kappa), 0)
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

    kappa = compute_kappa(r, n)
    l = 1 / np.sqrt(kappa)  # eq 2.3
    h = np.sqrt(7) * l / np.power(n, 0.2)  # eq 2.4

    return h


def nonparametric_density_estimation(
    alpha: np.ndarray,
    h: float,
    radius: float = 1,
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
