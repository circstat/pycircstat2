import numpy as np

from pycircstat2 import Circular, load_data
from pycircstat2.correlation import aacorr, alcorr


def test_aacorr():

    # Example 27.20 (Zar, 2010)
    d_ex20_ch27 = load_data("D20", source="zar")
    a = Circular(data=d_ex20_ch27["Insect"].values)
    b = Circular(data=d_ex20_ch27["Light"].values)

    # test Fisher & Lee, 1983
    res = aacorr(a, b, test=True, method="fl")

    np.testing.assert_approx_equal(res.r, 0.8945, significant=4)
    assert res.reject_null
    assert res.test_stat is None
    assert res.p_value is None

    # Example 27.22 (Zar, 2010)
    d_ex22_ch27 = load_data("D22", source="zar")
    a = Circular(data=d_ex22_ch27["evening"].values)
    b = Circular(data=d_ex22_ch27["morning"].values)

    # test nonparametric
    res = aacorr(a, b, test=True, method="nonparametric")
    assert not res.reject_null
    assert res.test_stat is None
    assert res.p_value is None

    # test Jammalamadaka & SenGupta, 2001
    d_milwaukee = load_data("milwaukee", source="jammalamadaka")
    theta = np.deg2rad(d_milwaukee["theta"].values)
    psi = np.deg2rad(d_milwaukee["psi"].values)

    res = aacorr(theta, psi, test=True, method="js")
    np.testing.assert_approx_equal(res.r, 0.2704648, significant=4)
    np.testing.assert_approx_equal(res.test_stat, 1.214025, significant=4)
    np.testing.assert_approx_equal(res.p_value, 0.2247383, significant=4)
    assert res.reject_null is None


def test_alcorr():

    # Example 27.21 (Zar, 2010)
    d_ex21_ch27 = load_data("D21", source="zar")
    a = Circular(data=d_ex21_ch27["θ"].values).alpha
    x = d_ex21_ch27["X"].values

    res = alcorr(a, x)
    np.testing.assert_approx_equal(res.r, 0.9854, significant=4)
    assert 0.025 < res.p_value < 0.05
