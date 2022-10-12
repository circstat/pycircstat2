import numpy as np

from pycircstat2 import Circular, load_data
from pycircstat2.descriptive import (circ_mean, circ_mean_ci, circ_median,
                                     circ_median_ci, circ_std)


def test_circ_mean():

    # Example 26.4 (Zar, 2010)
    data_zar_ex4_ch26 = load_data("D1", source="zar_2010")
    circ_zar_ex4_ch26 = Circular(data=data_zar_ex4_ch26["θ"].values)
    m, r = circ_mean(circ_zar_ex4_ch26.alpha, circ_zar_ex4_ch26.w)[:2]

    np.testing.assert_approx_equal(np.rad2deg(m), 99, significant=1)
    np.testing.assert_approx_equal(r, 0.82522, significant=5)

    # ch26 Example 5 (Zar, 2010)
    data_zar_ex5_ch26 = load_data("D2", source="zar_2010")
    circ_zar_ex5_ch26 = Circular(
        data=data_zar_ex5_ch26["θ"].values, w=data_zar_ex5_ch26["w"].values
    )
    m, r = circ_mean(circ_zar_ex5_ch26.alpha, circ_zar_ex5_ch26.w)[:2]

    np.testing.assert_approx_equal(np.rad2deg(m), 162, significant=1)
    np.testing.assert_approx_equal(r, 0.55064, significant=4)


def test_circ_std():

    data_zar_ex4_ch26 = load_data("D1", source="zar_2010")
    circ_zar_ex4_ch26 = Circular(data=data_zar_ex4_ch26["θ"].values)

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

    data_zar_ex5_ch26 = load_data("D2", source="zar_2010")
    circ_zar_ex5_ch26 = Circular(
        data=data_zar_ex5_ch26["θ"].values, w=data_zar_ex5_ch26["w"].values
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
    data_zar_ex2_ch26 = load_data("D1", source="zar_2010")
    circ_zar_ex2_ch26 = Circular(data=data_zar_ex2_ch26["θ"].values)
    median = circ_median(
        circ_zar_ex2_ch26.alpha, circ_zar_ex2_ch26.w, circ_zar_ex2_ch26.grouped
    )

    np.testing.assert_approx_equal(np.rad2deg(median), 103.0, significant=1)

    # Ch26.6 P657 (Zar, 2010) droped the first point
    circ_zar_ex2_ch26_odd = Circular(data=data_zar_ex2_ch26["θ"].values[1:])
    median = circ_median(
        circ_zar_ex2_ch26_odd.alpha,
        circ_zar_ex2_ch26_odd.w,
        circ_zar_ex2_ch26_odd.grouped,
    )

    np.testing.assert_approx_equal(np.rad2deg(median), 110.0, significant=1)

    # mallard data (mardia, 1972)
    data_mallard = load_data("mallard", source="mardia_1972")
    circ_mallard = Circular(data=data_mallard["θ"].values, w=data_mallard["w"].values)
    median = circ_median(circ_mallard.alpha_ub, circ_mallard.w, circ_mallard.grouped)

    np.testing.assert_approx_equal(np.rad2deg(median), 313.8, significant=2)


def test_circ_mean_deviation():
    pass


def test_circ_mean_ci():

    # method: approximate (from P619, Zar, 2010)
    data_zar_ex4_ch26 = load_data("D1", source="zar_2010")
    circ_zar_ex4_ch26 = Circular(data=data_zar_ex4_ch26["θ"].values)

    ## computed directly from r and n
    lb, ub = circ_mean_ci(
        mean=circ_zar_ex4_ch26.mean,
        r=circ_zar_ex4_ch26.r,
        n=circ_zar_ex4_ch26.n,
        method="approximate",
    )

    np.testing.assert_approx_equal(np.rad2deg(lb), 68, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(ub), 130, significant=1)

    ## computed from alpha and w
    lb, ub = circ_mean_ci(
        alpha=circ_zar_ex4_ch26.alpha, w=circ_zar_ex4_ch26.w, method="approximate"
    )
    np.testing.assert_approx_equal(np.rad2deg(lb), 68, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(ub), 130, significant=1)

    # method: dispersion (from P78, Fisher, 1993)
    d_ex3 = load_data("B6", "fisher_1993")
    c_ex3_s2 = Circular(np.sort(d_ex3[d_ex3.set == 2]["θ"].values))
    lb, ub = circ_mean_ci(method="dispersion", alpha=c_ex3_s2.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb), 232.7, significant=4)
    np.testing.assert_approx_equal(np.rad2deg(ub), 262.5, significant=4)

    # method: bootstrap (from P78, Fisher, 1993)
    # but how to test boostrap?


def test_circ_median_ci():

    d_ex3 = load_data("B6", "fisher_1993")
    c_ex3_s0 = Circular(
        np.sort(d_ex3[d_ex3.set == 2]["θ"].values[:10]),
        kwargs_median={"method": "count"},
    )
    c_ex3_s1 = Circular(
        np.sort(d_ex3[d_ex3.set == 2]["θ"].values[:20]),
        kwargs_median={"method": "deviation"},
    )
    c_ex3_s2 = Circular(np.sort(d_ex3[d_ex3.set == 2]["θ"].values))

    lb, ub, ci = circ_median_ci(median=c_ex3_s0.median, alpha=c_ex3_s0.alpha)[0]
    np.testing.assert_approx_equal(np.rad2deg(lb.round(5)), 245.0, significant=3)
    np.testing.assert_approx_equal(np.rad2deg(ub.round(5)), 315.0, significant=3)

    lb, ub, ci = circ_median_ci(median=c_ex3_s1.median, alpha=c_ex3_s1.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb.round(5)), 229.0, significant=3)
    np.testing.assert_approx_equal(np.rad2deg(ub.round(5)), 277.0, significant=3)

    lb, ub, ci = circ_median_ci(median=c_ex3_s2.median, alpha=c_ex3_s2.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb.round(5)), 229.0, significant=3)
    np.testing.assert_approx_equal(np.rad2deg(ub.round(5)), 267.0, significant=3)
