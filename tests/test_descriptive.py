import numpy as np

from pycircstat2 import Circular, data
from pycircstat2.descriptive import (
    circ_mean,
    circ_std,
    circ_median,
    circ_mean_deviation,
    circ_mean_ci,
)


def test_circ_mean():

    # Ch26 Example 4 (Zar, 2010)
    data_zar_ex4_ch26 = data("zar_ex2_ch26")
    circ_zar_ex4_ch26 = Circular(data=data_zar_ex4_ch26["data"])
    m, r = circ_mean(circ_zar_ex4_ch26.alpha, circ_zar_ex4_ch26.w)[:2]

    np.testing.assert_approx_equal(np.rad2deg(m), 99, significant=1)
    np.testing.assert_approx_equal(r, 0.82522, significant=5)

    # ch26 Example 5 (Zar, 2010)
    data_zar_ex5_ch26 = data("zar_ex5_ch26")
    circ_zar_ex5_ch26 = Circular(
        data=data_zar_ex5_ch26["data"], w=data_zar_ex5_ch26["w"]
    )
    m, r = circ_mean(circ_zar_ex5_ch26.alpha, circ_zar_ex5_ch26.w)[:2]

    np.testing.assert_approx_equal(np.rad2deg(m), 162, significant=1)
    np.testing.assert_approx_equal(r, 0.55064, significant=5)


def test_circ_std():

    data_zar_ex4_ch26 = data("zar_ex4_ch26")
    circ_zar_ex4_ch26 = Circular(data=data_zar_ex4_ch26["data"])

    # Angular dispersion from Ch26.5 (Zar, 2010)
    # Part of Ch26 Example 4, using data from Ch26 Example 2

    # compute directly from r
    s, s0, rc = circ_std(r=circ_zar_ex4_ch26.r, bin_size=circ_zar_ex4_ch26.bin_size)
    np.testing.assert_approx_equal(np.rad2deg(s), 34.0, significant=1.0)
    np.testing.assert_approx_equal(np.rad2deg(s0), 36.0, significant=1.0)

    # compute from alpha and w
    s, s0, rc = circ_std(alpha=circ_zar_ex4_ch26.alpha, w=circ_zar_ex4_ch26.w)
    np.testing.assert_approx_equal(np.rad2deg(s), 34.0, significant=1.0)
    np.testing.assert_approx_equal(np.rad2deg(s0), 36.0, significant=1.0)

    data_zar_ex5_ch26 = data("zar_ex5_ch26")
    circ_zar_ex5_ch26 = Circular(
        data=data_zar_ex5_ch26["data"], w=data_zar_ex5_ch26["w"]
    )

    # compute directly from r
    s, s0, rc = circ_std(r=circ_zar_ex5_ch26.r, bin_size=circ_zar_ex5_ch26.bin_size)
    np.testing.assert_approx_equal(np.rad2deg(s), 54.0, significant=1.0)
    np.testing.assert_approx_equal(np.rad2deg(s0), 63.0, significant=1.0)

    # compute from alpha and w
    s, s0, rc = circ_std(alpha=circ_zar_ex5_ch26.alpha, w=circ_zar_ex5_ch26.w)
    np.testing.assert_approx_equal(np.rad2deg(s), 54.0, significant=1.0)
    np.testing.assert_approx_equal(np.rad2deg(s0), 63.0, significant=1.0)


def test_circ_median():

    # Ch26.6 P657 (Zar, 2010)
    data_zar_ex2_ch26 = data("zar_ex2_ch26")
    circ_zar_ex2_ch26 = Circular(data=data_zar_ex2_ch26["data"])
    median = circ_median(
        circ_zar_ex2_ch26.alpha, circ_zar_ex2_ch26.w, circ_zar_ex2_ch26.grouped
    )

    np.testing.assert_approx_equal(np.rad2deg(median), 103.0, significant=1)

    # Ch26.6 P657 (Zar, 2010) droped the first point
    circ_zar_ex2_ch26_odd = Circular(data=data_zar_ex2_ch26["data"][1:])
    median = circ_median(
        circ_zar_ex2_ch26_odd.alpha,
        circ_zar_ex2_ch26_odd.w,
        circ_zar_ex2_ch26_odd.grouped,
    )

    np.testing.assert_approx_equal(np.rad2deg(median), 110.0, significant=1)

    # mallard data (mardia, 1972)
    data_mallard = data("mallard")
    circ_mallard = Circular(data=data_mallard["data"], w=data_mallard["w"])
    median = circ_median(circ_mallard.alpha_ub, circ_mallard.w, circ_mallard.grouped)

    np.testing.assert_approx_equal(np.rad2deg(median), 313.8, significant=2)


def test_circ_mean_deviation():
    pass


def test_circ_mean_ci():

    data_zar_ex4_ch26 = data("zar_ex4_ch26")
    circ_zar_ex4_ch26 = Circular(data_zar_ex4_ch26["data"])

    # computed directly from r and n
    d, lb, ub = circ_mean_ci(
        circ_zar_ex4_ch26.mean, circ_zar_ex4_ch26.r, circ_zar_ex4_ch26.n
    )

    np.testing.assert_approx_equal(np.rad2deg(d), 31, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(lb), 68, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(ub), 130, significant=1)

    # computed from alpha and w
    d, lb, ub = circ_mean_ci(alpha=circ_zar_ex4_ch26.alpha, w=circ_zar_ex4_ch26.w)
    np.testing.assert_approx_equal(np.rad2deg(d), 31, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(lb), 68, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(ub), 130, significant=1)
