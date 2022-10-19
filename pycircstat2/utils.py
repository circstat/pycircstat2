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