import numpy as np

from pycircstat2 import Circular, load_data
from pycircstat2.correlation import aacorr, alcorr


def test_aacorr():

    # Example 27.20 (Zar, 2010)
    d_ex20_ch27 = load_data("D20", source="zar")
    a = Circular(data=d_ex20_ch27["Insect"].values)
    b = Circular(data=d_ex20_ch27["Light"].values)

    raa, reject = aacorr(a, b, test=True, method="fl")

    assert reject == True
    np.testing.assert_approx_equal(raa, 0.8945, significant=4)

    # Example 27.22 (Zar, 2010)
    d_ex22_ch27 = load_data("D22", source="zar")
    a = Circular(data=d_ex22_ch27["evening"].values)
    b = Circular(data=d_ex22_ch27["morning"].values)
    raa, reject = aacorr(a, b, test=True, method="nonparametric")
    assert reject == False


def test_alcorr():

    # Example 27.21 (Zar, 2010)
    d_ex21_ch27 = load_data("D21", source="zar")
    a = Circular(data=d_ex21_ch27["θ"].values).alpha
    x = d_ex21_ch27["X"].values

    ral, pval = alcorr(a, x)
    np.testing.assert_approx_equal(ral, 0.9854, significant=4)
    assert 0.025 < pval < 0.05
