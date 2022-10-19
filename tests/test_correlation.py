import numpy as np

from pycircstat2 import Circular, load_data
from pycircstat2.correlation import aacorr, alcorr


def test_aacorr():

    # Example 27.20 (Zar, 2010)
    d_ex20_ch27 = load_data("D20", source="zar_2010")
    a = Circular(data=d_ex20_ch27["Insect"].values, n_clusters=1)
    b = Circular(data=d_ex20_ch27["Light"].values, n_clusters=1)

    raa, reject = aacorr(a, b, test=True)

    assert reject == True
    np.testing.assert_approx_equal(raa, 0.8945, significant=4)


def test_alcorr():

    # Example 27.21 (Zar, 2010)
    d_ex21_ch27 = load_data("D21", source="zar_2010")
    a = Circular(data=d_ex21_ch27["θ"].values, n_clusters=1).alpha
    x = d_ex21_ch27["X"].values

    raa = alcorr(a, x)
    np.testing.assert_approx_equal(raa, 0.9854, significant=4)
