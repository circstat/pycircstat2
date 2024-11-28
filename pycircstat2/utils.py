import json
from importlib import resources as importlib_resources
from typing import Union

import numpy as np
import pandas as pd


def data2rad(
    data: Union[np.ndarray, float, int],
    k: Union[float, int] = 360,  # number of intervals in the full cycle
) -> Union[np.ndarray, float]:  # eq(26.1), zar 2010
    r"""Convert data measured on a circular scale to corresponding angular directions.

    $$ \alpha = \frac{2\pi \times \mathrm{data}}{k} $$

    Parameters
    ----------
    data : np.ndarray or float
        Data measured on a circular scale.
    k : float or int
        Number of intervals in the full cycle. Default is 360.

    Returns
    -------
    angle: np.ndarray or float
        Angular directions in radian.
    """
    return 2 * np.pi * data / k


def rad2data(
    rad: Union[np.ndarray, float, int], k: Union[float, int] = 360
) -> Union[np.ndarray, float]:
    return k * rad / (2 * np.pi)  # eq(26.12), zar 2010


def time2float(x: Union[np.ndarray, list, str], sep: str = ":") -> np.ndarray:
    """Convert an array of strings in time (hh:mm) to an array of floats."""

    def _t2f(x: str, sep: str):
        """Convert string of time to float. E.g. 12:45 -> 12.75"""
        hr, min = x.split(sep)
        return float(hr) + float(min) / 60

    t2f = np.vectorize(_t2f)
    return t2f(x, sep)


def angrange(rad: Union[np.ndarray, float, int]) -> Union[np.ndarray, float]:
    return ((rad % (2 * np.pi)) + 2 * np.pi) % (2 * np.pi)


def angular_distance(a: Union[np.ndarray, list, float], b: float) -> np.ndarray:
    """Angular distance between two angles.

    Parameters
    ----------
    a: np.ndarray or float
        angle(s).

    b: float
        target angle.

    Returns
    -------
    e: np.ndarray
        angular distance

    Reference
    ---------
    P642, Section 27.2, Zar, 2010
    """

    a = np.array(a) if type(a) is list else a

    c = angrange(a - b)
    d = 2 * np.pi - c
    e = np.min([c, d], axis=0)

    return e


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
    name: str,
    source: str = "fisher",
    print_meta: bool = False,
    return_meta: bool = False,
) -> Union[pd.DataFrame, tuple]:
    __source__ = ["fisher", "zar", "mardia", "pewsey"]

    # check source
    if source not in __source__:
        raise ValueError(
            f"Invalid source ('{source}').\n Availble sources: {__source__}"
        )

    # load data
    data_files = importlib_resources.files("pycircstat2")
    csv_path = data_files / f"data/{source}/{name}.csv"
    csv_data = pd.read_csv(csv_path, index_col=0)

    json_path = data_files / f"data/{source}/{name}.csv-metadata.json"
    with open(json_path) as f:
        json_data = json.load(f)

    if print_meta:
        print(json.dumps(json_data, indent=4, ensure_ascii=False))

    if return_meta:
        return csv_data, json_data
    else:
        return csv_data


def is_within_circular_range(value, lb, ub):
    """
    Check if a value lies within the circular range [lb, ub].

    Parameters
    ----------
    value : float
        The value to check.
    lb : float
        The lower bound of the range.
    ub : float
        The upper bound of the range.

    Returns
    -------
    bool
        True if the value is within the circular range, False otherwise.
    """
    value = np.mod(value, 2 * np.pi)
    lb = np.mod(lb, 2 * np.pi)
    ub = np.mod(ub, 2 * np.pi)

    if lb <= ub:
        # Standard range
        return lb <= value <= ub
    else:
        # Wrapping range
        return value >= lb or value <= ub
