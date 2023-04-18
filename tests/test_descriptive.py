import numpy as np

from pycircstat2 import Circular, load_data
from pycircstat2.descriptive import (
    circ_dispersion,
    circ_kurtosis,
    circ_mean,
    circ_mean_ci,
    circ_mean_of_means,
    circ_median,
    circ_median_ci,
    circ_moment,
    circ_skewness,
    circ_std,
    compute_smooth_params,
)


def test_circ_mean():
    # Example 26.4 (Zar, 2010)
    data_zar_ex4_ch26 = load_data("D1", source="zar")
    circ_zar_ex4_ch26 = Circular(data=data_zar_ex4_ch26["θ"].values)
    m, r = circ_mean(
        alpha=circ_zar_ex4_ch26.alpha, w=circ_zar_ex4_ch26.w, return_r=True
    )

    np.testing.assert_approx_equal(np.rad2deg(m), 99, significant=1)
    np.testing.assert_approx_equal(r, 0.82522, significant=5)

    # ch26 Example 5 (Zar, 2010)
    data_zar_ex5_ch26 = load_data("D2", source="zar")
    circ_zar_ex5_ch26 = Circular(
        data=data_zar_ex5_ch26["θ"].values, w=data_zar_ex5_ch26["w"].values
    )
    m, r = circ_mean(
        alpha=circ_zar_ex5_ch26.alpha, w=circ_zar_ex5_ch26.w, return_r=True
    )

    np.testing.assert_approx_equal(np.rad2deg(m), 162, significant=1)
    np.testing.assert_approx_equal(r, 0.55064, significant=4)


def test_circ_std():
    data_zar_ex4_ch26 = load_data("D1", source="zar")
    circ_zar_ex4_ch26 = Circular(data=data_zar_ex4_ch26["θ"].values)

    # Angular dispersion from Ch26.5 (Zar, 2010)
    # Part of Ch26 Example 4, using data from Ch26 Example 2

    # compute directly from r
    s, s0, rc = circ_std(r=circ_zar_ex4_ch26.r, bin_size=circ_zar_ex4_ch26.bin_size)
    np.testing.assert_approx_equal(np.rad2deg(s), 34.0, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(s0), 36.0, significant=1)

    # compute from alpha
    s, s0, rc = circ_std(alpha=circ_zar_ex4_ch26.alpha)
    np.testing.assert_approx_equal(np.rad2deg(s), 34.0, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(s0), 36.0, significant=1)

    data_zar_ex5_ch26 = load_data("D2", source="zar")
    circ_zar_ex5_ch26 = Circular(
        data=data_zar_ex5_ch26["θ"].values, w=data_zar_ex5_ch26["w"].values
    )

    # compute directly from r
    s, s0, rc = circ_std(r=circ_zar_ex5_ch26.r, bin_size=circ_zar_ex5_ch26.bin_size)
    np.testing.assert_approx_equal(np.rad2deg(s), 54.0, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(s0), 63.0, significant=1)

    # compute from alpha and w
    s, s0, rc = circ_std(alpha=circ_zar_ex5_ch26.alpha, w=circ_zar_ex5_ch26.w)
    np.testing.assert_approx_equal(np.rad2deg(s), 54.0, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(s0), 63.0, significant=1)


def test_circ_median():
    # Ch26.6 P657 (Zar, 2010)
    data_zar_ex2_ch26 = load_data("D1", source="zar")
    circ_zar_ex2_ch26 = Circular(data=data_zar_ex2_ch26["θ"].values)
    median = circ_median(
        alpha=circ_zar_ex2_ch26.alpha,
        grouped=False,
        method="deviation",
        return_average=True,
    )

    np.testing.assert_approx_equal(np.rad2deg(median), 103.0, significant=1)

    # Ch26.6 P657 (Zar, 2010) droped the first point
    circ_zar_ex2_ch26_odd = Circular(data=data_zar_ex2_ch26["θ"].values[1:])
    median = circ_median(
        alpha=circ_zar_ex2_ch26_odd.alpha,
        grouped=False,
        method="deviation",
        return_average=True,
    )

    np.testing.assert_approx_equal(np.rad2deg(median), 110.0, significant=1)

    # mallard data (mardia, 1972)
    data_mallard = load_data("mallard", source="mardia")
    circ_mallard = Circular(data=data_mallard["θ"].values, w=data_mallard["w"].values)
    median = circ_median(
        alpha=circ_mallard.alpha_ub,
        w=circ_mallard.w,
        grouped=True,
        return_average=True,
    )

    np.testing.assert_approx_equal(np.rad2deg(median), 313.8, significant=2)


def test_circ_mean_deviation():
    pass


def test_circ_mean_ci():
    # method: approximate (from P619, Zar, 2010)
    data_zar_ex4_ch26 = load_data("D1", source="zar")
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
    d_ex3 = load_data("B6", "fisher")
    c_ex3_s2 = Circular(np.sort(d_ex3[d_ex3.set == 2]["θ"].values))
    lb, ub = circ_mean_ci(method="dispersion", alpha=c_ex3_s2.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb), 232.7, significant=4)
    np.testing.assert_approx_equal(np.rad2deg(ub), 262.5, significant=4)

    # method: bootstrap (from P78, Fisher, 1993)
    # but how to test boostrap?


def test_circ_median_ci():
    d_ex3 = load_data("B6", "fisher")
    c_ex3_s0 = Circular(
        data=np.sort(d_ex3[d_ex3.set == 2]["θ"].values[:10]),
        kwargs_median={"method": "count"},
    )
    c_ex3_s1 = Circular(
        data=np.sort(d_ex3[d_ex3.set == 2]["θ"].values[:20]),
        kwargs_median={"method": "deviation"},
    )
    c_ex3_s2 = Circular(data=np.sort(d_ex3[d_ex3.set == 2]["θ"].values))

    # n is too small for proper estimation of median ci
    lb, ub, ci = circ_median_ci(median=c_ex3_s0.median, alpha=c_ex3_s0.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb.round(5)), 245.0, significant=3)
    np.testing.assert_approx_equal(np.rad2deg(ub.round(5)), 315.0, significant=3)

    lb, ub, ci = circ_median_ci(median=c_ex3_s1.median, alpha=c_ex3_s1.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb.round(5)), 229.0, significant=3)
    np.testing.assert_approx_equal(np.rad2deg(ub.round(5)), 277.0, significant=3)

    lb, ub, ci = circ_median_ci(median=c_ex3_s2.median, alpha=c_ex3_s2.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb.round(5)), 229.0, significant=3)
    np.testing.assert_approx_equal(np.rad2deg(ub.round(5)), 267.0, significant=3)


def test_circ_mean_of_means():
    data = load_data("D4", source="zar")
    ms = np.deg2rad(data.values[:, 0])
    rs = data.values[:, 1]

    m, r = circ_mean_of_means(ms=ms, rs=rs)
    np.testing.assert_approx_equal(np.rad2deg(m), 152.0, significant=3)
    np.testing.assert_approx_equal(r, 0.59634, significant=5)


def test_circ_skewness():
    b11 = load_data("B11", source="fisher")["θ"].values
    c11 = Circular(data=b11)
    skewness = circ_skewness(alpha=c11.alpha)
    np.testing.assert_approx_equal(skewness, -0.92, significant=2)


def test_circ_kurtosis():
    b11 = load_data("B11", source="fisher")["θ"].values
    c11 = Circular(data=b11)
    kurtosis = circ_kurtosis(alpha=c11.alpha)
    np.testing.assert_approx_equal(kurtosis, 6.64, significant=3)


def test_circ_dispersion():
    b11 = load_data("B11", source="fisher")["θ"].values
    c11 = Circular(data=b11)
    dispersion = circ_dispersion(alpha=c11.alpha)
    np.testing.assert_approx_equal(dispersion, 0.24, significant=2)


def test_circ_moment():
    # Section 3.2, Pewsey (2014) P24

    b11 = load_data("B11", source="fisher")["θ"].values
    c11 = Circular(data=b11)

    # first moment == mean

    u1, r1, Cbar, Sbar = circ_moment(
        alpha=c11.alpha, p=1, centered=False, return_intermediates=True
    )
    np.testing.assert_approx_equal(np.rad2deg(u1).round(2), 3.10, significant=2)
    np.testing.assert_approx_equal(r1.round(2), 0.83, significant=2)
    np.testing.assert_approx_equal(Cbar.round(2), 0.83, significant=2)
    np.testing.assert_approx_equal(Sbar.round(2), 0.04, significant=2)

    # second moment

    u2, r2, Cbar, Sbar = circ_moment(
        alpha=c11.alpha, p=2, centered=False, return_intermediates=True
    )
    np.testing.assert_approx_equal(np.rad2deg(u2).round(2), 0.64, significant=2)
    np.testing.assert_approx_equal(r2.round(2), 0.67, significant=2)
    np.testing.assert_approx_equal(Cbar.round(2), 0.67, significant=2)
    np.testing.assert_approx_equal(Sbar.round(2), 0.01, significant=2)


def test_compute_smooth_params():
    from pycircstat2.utils import time2float

    d_fisher_b1 = load_data("B1", source="fisher")["time"].values
    c_fisher_b1 = Circular(time2float(d_fisher_b1), unit="hour")
    h0 = compute_smooth_params(c_fisher_b1.r, c_fisher_b1.n)
    np.testing.assert_approx_equal(h0, 1.06, significant=2)
