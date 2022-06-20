from typing import Union

import numpy as np


def data2rad(
    data: Union[np.ndarray, float, int], k: Union[float, int] = 360
) -> Union[np.ndarray, float]:  # eq(26.1)
    return 2 * np.pi * data / k


def rad2data(
    rad: Union[np.ndarray, float, int], k: Union[float, int] = 360
) -> Union[np.ndarray, float]:  # eq(26.12)
    return k * rad / (2 * np.pi)


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
    return sig
