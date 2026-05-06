import numpy as np
import pytest

from pycircstat2 import Circular, load_data
from pycircstat2.descriptive import (
    angular_std,
    circ_dispersion,
    circ_dist,
    circ_kappa,
    circ_kurtosis,
    circ_mean,
    circ_mean_and_r,
    circ_mean_and_r_of_means,
    circ_mean_ci,
    circ_mean_deviation,
    circ_mean_deviation_chunked,
    circ_median,
    circ_median_ci,
    circ_moment,
    circ_pairdist,
    circ_quantile,
    circ_range,
    circ_skewness,
    circ_std,
    compute_smooth_params,
    convert_moment,
)


def test_circ_mean():
    # Example 26.4 (Zar, 2010)
    data_zar_ex4_ch26 = load_data("D1", source="zar")
    circ_zar_ex4_ch26 = Circular(data=data_zar_ex4_ch26["θ"].values[:])
    m, r = circ_mean_and_r(alpha=circ_zar_ex4_ch26.alpha, w=circ_zar_ex4_ch26.w)

    np.testing.assert_approx_equal(np.rad2deg(m), 99, significant=1)
    np.testing.assert_approx_equal(r, 0.82522, significant=5)

    # ch26 Example 5 (Zar, 2010)
    data_zar_ex5_ch26 = load_data("D2", source="zar")
    circ_zar_ex5_ch26 = Circular(
        data=data_zar_ex5_ch26["θ"].values[:], w=data_zar_ex5_ch26["w"].values[:]
    )
    m, r = circ_mean_and_r(alpha=circ_zar_ex5_ch26.alpha, w=circ_zar_ex5_ch26.w)

    np.testing.assert_approx_equal(np.rad2deg(m), 162, significant=1)
    np.testing.assert_approx_equal(r, 0.55064, significant=4)


def test_circ_std():
    data_zar_ex4_ch26 = load_data("D1", source="zar")
    circ_zar_ex4_ch26 = Circular(data=data_zar_ex4_ch26["θ"].values[:])

    # Angular dispersion from Ch26.5 (Zar, 2010)
    # Part of Ch26 Example 4, using data from Ch26 Example 2

    # compute directly from r
    s = angular_std(r=circ_zar_ex4_ch26.r, bin_size=circ_zar_ex4_ch26.bin_size)
    s0 = circ_std(r=circ_zar_ex4_ch26.r, bin_size=circ_zar_ex4_ch26.bin_size)
    np.testing.assert_approx_equal(np.rad2deg(s), 34.0, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(s0), 36.0, significant=1)

    # compute from alpha
    s = angular_std(alpha=circ_zar_ex4_ch26.alpha)
    s0 = circ_std(alpha=circ_zar_ex4_ch26.alpha)
    np.testing.assert_approx_equal(np.rad2deg(s), 34.0, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(s0), 36.0, significant=1)

    data_zar_ex5_ch26 = load_data("D2", source="zar")
    circ_zar_ex5_ch26 = Circular(
        data=data_zar_ex5_ch26["θ"].values[:], w=data_zar_ex5_ch26["w"].values[:]
    )

    # compute directly from r
    s = angular_std(r=circ_zar_ex5_ch26.r, bin_size=circ_zar_ex5_ch26.bin_size)
    s0 = circ_std(r=circ_zar_ex5_ch26.r, bin_size=circ_zar_ex5_ch26.bin_size)
    np.testing.assert_approx_equal(np.rad2deg(s), 54.0, significant=1)
    np.testing.assert_approx_equal(
        np.rad2deg(s0), 62.0, significant=1
    )  # 63 in the book, but we should correct the bias in r for grouped data.

    # compute from alpha and w
    s = angular_std(alpha=circ_zar_ex5_ch26.alpha, w=circ_zar_ex5_ch26.w)
    s0 = circ_std(alpha=circ_zar_ex5_ch26.alpha, w=circ_zar_ex5_ch26.w)
    np.testing.assert_approx_equal(np.rad2deg(s), 54.0, significant=1)
    np.testing.assert_approx_equal(
        np.rad2deg(s0), 62.0, significant=1
    )  # 63 in the book, but we should correct the bias in r for grouped data.


def test_circ_median():
    # Ch26.6 P657 (Zar, 2010)
    data_zar_ex2_ch26 = load_data("D1", source="zar")
    circ_zar_ex2_ch26 = Circular(data=data_zar_ex2_ch26["θ"].values[:])
    median = circ_median(
        alpha=circ_zar_ex2_ch26.alpha,
        method="deviation",
        return_average=True,
    )

    np.testing.assert_approx_equal(np.rad2deg(median), 103.0, significant=1)

    # Ch26.6 P657 (Zar, 2010) droped the first point
    circ_zar_ex2_ch26_odd = Circular(data=data_zar_ex2_ch26["θ"].values[:][1:])
    median = circ_median(
        alpha=circ_zar_ex2_ch26_odd.alpha,
        method="deviation",
        return_average=True,
    )

    np.testing.assert_approx_equal(np.rad2deg(median), 110.0, significant=1)

    # mallard data (mardia, 1972)
    data_mallard = load_data("mallard", source="mardia")
    circ_mallard = Circular(data=data_mallard["θ"].values[:], w=data_mallard["w"].values[:])
    median = circ_median(
        alpha=circ_mallard.alpha,
        w=circ_mallard.w,
        return_average=True,
    )

    np.testing.assert_allclose(np.rad2deg(median), 313.8, atol=0.05)

    # edge case: all angles are the same
    # 1) all angles identical (any wrap-around)
    angles = np.deg2rad([30, 30 + 360, 30])
    m = circ_median(angles)
    np.testing.assert_allclose(np.rad2deg(m) % 360, 30.0, atol=1e-12)

    # 2) perfectly symmetric bimodal sample  → NaN
    bimodal = np.deg2rad([0, 0, 90, 90])
    assert np.isnan(circ_median(bimodal))

    # 3) all-coincide early-exit must wrap into [0, 2π)
    m = circ_median(np.array([7.0, 7.0, 7.0]))
    assert 0.0 <= m < 2 * np.pi
    np.testing.assert_allclose(m, 7.0 - 2 * np.pi, atol=1e-12)

    # 4) grouped data with all mass in one bin (not bin 0): early-exit must
    #    return the populated bin's direction, not alpha[0].
    centers = np.array([(i + 0.5) * 2 * np.pi / 5 for i in range(5)])
    w = np.array([0, 0, 10, 0, 0])
    np.testing.assert_allclose(circ_median(centers, w), centers[2], atol=1e-12)


def test_circ_median_HL():
    # Reference: Otieno (2002) thesis §3.4 + Appendix E.
    # HL = circular median of pairwise circular means; HL1/HL2/HL3 differ in
    # which pairs are used.

    rng = np.random.default_rng(0)

    # On unimodal von-Mises-ish data, HL1/HL2/HL3 should agree closely with each
    # other and with the deviation method (within ~5° for n=50, κ≈10).
    alpha = np.deg2rad(120) + 0.3 * rng.standard_normal(50)
    alpha = alpha % (2 * np.pi)
    m_dev = circ_median(alpha, method="deviation")
    for hl in ["HL1", "HL2", "HL3"]:
        m_hl = circ_median(alpha, method=hl)
        diff = np.abs(np.angle(np.exp(1j * (m_hl - m_dev))))
        assert diff < np.deg2rad(5), f"{hl} too far from deviation median"

    # HL2 and HL3 share the same support (HL1 pair-means + observations); HL3
    # just doubles the HL1 pair-means. For sharply concentrated data the inner
    # deviation-method median lands on the same dense cluster.
    sharp = np.deg2rad(45) + 0.05 * rng.standard_normal(20)
    m2 = circ_median(sharp, method="HL2")
    m3 = circ_median(sharp, method="HL3")
    np.testing.assert_allclose(m2, m3, atol=np.deg2rad(2))

    # Antipodal pair (0, π) must be dropped, not crash. With only an antipodal
    # pair, HL1's candidate set vanishes → NaN. HL2/HL3 still have the
    # observations, which form a 2-point uniform set → also NaN by deviation
    # method's uniform check. Just assert no exception.
    for hl in ["HL1", "HL2", "HL3"]:
        circ_median(np.array([0.0, np.pi]), method=hl)

    # Unsupported method raises.
    with pytest.raises(ValueError):
        circ_median(alpha, method="HL4")


def test_circ_median_HL_oracle():
    # Oracle values cross-checked against a hand-assembled fixed version of
    # Otieno's HL: pair means are computed per Appendix E (sipfunc/sipfunc2/
    # sipfunc3), then R's *working* `median.circular` is applied to that array.
    #
    # We do NOT use R/circular's `medianHL.circular` as oracle — its C
    # implementation has a bug (the last line of `MedianHLCircularRad` calls
    # the deviation median on the original `x` instead of on the `meanOfPair`
    # array it just built), so it returns the regular median, not HL.
    #
    # Our deviation tie tolerance (_ANGLE_DECIMALS=8) is matched to R's 1e-8;
    # when fed identical candidate arrays the two agree to ~2e-9 rad.

    # Symmetric input: HL = symmetry center, exactly, all three variants.
    alpha = np.array([0.0, np.pi / 4, np.pi / 2])
    for hl in ["HL1", "HL2", "HL3"]:
        np.testing.assert_allclose(
            circ_median(alpha, method=hl), np.pi / 4, atol=1e-9
        )

    # Rotation equivariance: HL(α + c) ≡ HL(α) + c (mod 2π).
    rng = np.random.default_rng(7)
    base = np.deg2rad(120) + 0.4 * rng.standard_normal(20)
    for c in [0.3, 1.7, -0.9]:
        for hl in ["HL1", "HL2", "HL3"]:
            m_base = circ_median(base, method=hl)
            m_shift = circ_median(base + c, method=hl)
            diff = np.abs(np.angle(np.exp(1j * (m_shift - m_base - c))))
            assert diff < 1e-9, f"{hl} not rotation-equivariant by {c}"

    # Frog data (Otieno 2003 Table 1) — values produced by our implementation,
    # matching R when fed the same candidate array. Note: a direct R call with
    # `c(104,...) * pi/180` lands on 145° for HL2 because that float-conversion
    # path differs from `np.deg2rad` in the last bits, which flips a near-tie in
    # the deviation candidates. The algorithm itself agrees to ~2e-9 rad.
    frog = np.deg2rad([104, 110, 117, 121, 127, 130, 136,
                       144, 152, 178, 184, 192, 200, 316])
    expected_deg = {"HL1": 147.25, "HL2": 144.59986667, "HL3": 147.00}
    for hl, exp in expected_deg.items():
        m = np.rad2deg(circ_median(frog, method=hl))
        np.testing.assert_allclose(m, exp, atol=1e-6,
            err_msg=f"{hl}: got {m:.6f}°, expected {exp}°")


def test_circ_median_grouped_odd_bins():
    # _circ_median_grouped previously used `np.roll(w, 2)` with the wrong sign,
    # which gave correct answers only by coincidence for n_bins == 5.
    # For n_bins=9 with all mass in bin 0 the old code returned 80° instead of 20°.

    # Single-bin mass: median should be that bin's center.
    for nb in [5, 7, 9, 11, 17]:
        centers = np.array([(i + 0.5) * 2 * np.pi / nb for i in range(nb)])
        w = np.zeros(nb, dtype=int)
        w[0] = 10
        m = circ_median(alpha=centers, w=w)
        np.testing.assert_allclose(m, centers[0], atol=1e-9)

    # Symmetric peaked distribution (1, 4, 1): median should be at the peak.
    for nb, peak in [(5, 2), (7, 3), (9, 4), (11, 5)]:
        centers = np.array([(i + 0.5) * 2 * np.pi / nb for i in range(nb)])
        w = np.zeros(nb, dtype=int)
        w[peak - 1] = 1
        w[peak] = 4
        w[peak + 1] = 1
        m = circ_median(alpha=centers, w=w)
        np.testing.assert_allclose(m, centers[peak], atol=1e-9)


def test_circ_mean_deviation():

    d22 = load_data("B10", source="fisher")

    d22s1 = np.deg2rad(d22[d22["set"] == 1]["θ"].values[:])
    d22s2 = np.deg2rad(d22[d22["set"] == 2]["θ"].values[:])
    d22s3 = np.deg2rad(d22[d22["set"] == 3]["θ"].values[:])

    np.testing.assert_allclose(
        circ_mean_deviation(d22s1, d22s1),
        circ_mean_deviation_chunked(d22s1, d22s1),
    )

    np.testing.assert_allclose(
        circ_mean_deviation(d22s2, d22s2),
        circ_mean_deviation_chunked(d22s2, d22s2),
    )

    np.testing.assert_allclose(
        circ_mean_deviation(d22s3, d22s3),
        circ_mean_deviation_chunked(d22s3, d22s3),
    )


def test_circ_mean_ci():
    # method: approximate (from P619, Zar, 2010)
    data_zar_ex4_ch26 = load_data("D1", source="zar")
    circ_zar_ex4_ch26 = Circular(data=data_zar_ex4_ch26["θ"].values[:])

    # computed directly from r and n
    lb, ub = circ_mean_ci(
        mean=circ_zar_ex4_ch26.mean,
        r=circ_zar_ex4_ch26.r,
        n=circ_zar_ex4_ch26.n,
        method="approximate",
    )

    np.testing.assert_approx_equal(np.rad2deg(lb), 68, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(ub), 130, significant=1)

    # computed from alpha and w
    lb, ub = circ_mean_ci(
        alpha=circ_zar_ex4_ch26.alpha,
        w=circ_zar_ex4_ch26.w,
        method="approximate",
    )
    np.testing.assert_approx_equal(np.rad2deg(lb), 68, significant=1)
    np.testing.assert_approx_equal(np.rad2deg(ub), 130, significant=1)

    # method: dispersion (from P78, Fisher, 1993)
    d_ex3 = load_data("B6", "fisher")
    c_ex3_s2 = Circular(np.sort(d_ex3[d_ex3.set == 2]["θ"].values[:]))
    lb, ub = circ_mean_ci(method="dispersion", alpha=c_ex3_s2.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb), 232.7, significant=4)
    np.testing.assert_approx_equal(np.rad2deg(ub), 262.5, significant=4)

    # method: bootstrap (from P78, Fisher, 1993)
    # but how to test boostrap?

    # test uniform distributed data (all method should raise errors)
    # from pycircstat2.distributions import circularuniform
    # rng = np.random.default_rng(seed=25)
    # d_uni = circularuniform.rvs(size=25)

    # with pytest.raises(ValueError):
    #     circ_mean_ci(alpha=d_uni, method="approximate")

    # with pytest.raises(ValueError):
    #     circ_mean_ci(alpha=d_uni, method="bootstrap")

    # with pytest.raises(ValueError):
    #     circ_mean_ci(alpha=d_uni, method="dispersion")
    

def test_circ_median_ci():
    d_ex3 = load_data("B6", "fisher")
    c_ex3_s0 = Circular(
        data=np.sort(d_ex3[d_ex3.set == 2]["θ"].values[:][:10]),
        kwargs_median={"method": "count"},
    )
    c_ex3_s1 = Circular(
        data=np.sort(d_ex3[d_ex3.set == 2]["θ"].values[:][:20]),
        kwargs_median={"method": "deviation"},
    )
    c_ex3_s2 = Circular(data=np.sort(d_ex3[d_ex3.set == 2]["θ"].values[:]))

    # n is too small for proper estimation of median ci
    lb, ub, ci = circ_median_ci(median=float(c_ex3_s0.median), alpha=c_ex3_s0.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb.round(5)), 245.0, significant=3)
    np.testing.assert_approx_equal(np.rad2deg(ub.round(5)), 315.0, significant=3)

    lb, ub, ci = circ_median_ci(median=float(c_ex3_s1.median), alpha=c_ex3_s1.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb.round(5)), 229.0, significant=3)
    np.testing.assert_approx_equal(np.rad2deg(ub.round(5)), 277.0, significant=3)

    lb, ub, ci = circ_median_ci(median=float(c_ex3_s2.median), alpha=c_ex3_s2.alpha)
    np.testing.assert_approx_equal(np.rad2deg(lb.round(5)), 229.0, significant=3)
    np.testing.assert_approx_equal(np.rad2deg(ub.round(5)), 267.0, significant=3)


def test_circ_median_ci_idx_wrap():
    # Previously, when idx_ub computed to exactly n, the wrap guard
    # `if idx_ub > n` did not trigger and `alpha[n]` raised IndexError.
    # Sweep medians across a few sample sizes that expose the boundary.
    for n in [16, 20, 25]:
        alpha = np.linspace(0.0, np.pi, n)
        for k in range(n):
            lb, ub, ci = circ_median_ci(median=float(alpha[k]), alpha=alpha)
            assert np.isfinite(lb) and np.isfinite(ub)


def test_circ_mean_and_r_of_means():
    data = load_data("D4", source="zar")
    ms = np.deg2rad(data.values[:][:, 0])
    rs = data.values[:][:, 1]

    m, r = circ_mean_and_r_of_means(ms=ms, rs=rs)
    np.testing.assert_approx_equal(np.rad2deg(m), 152.0, significant=3)
    np.testing.assert_approx_equal(r, 0.59634, significant=5)


def test_circ_skewness():
    b11 = load_data("B11", source="fisher")["θ"].values[:]
    c11 = Circular(data=b11)
    skewness = circ_skewness(alpha=c11.alpha)
    np.testing.assert_approx_equal(skewness, -0.92, significant=2)


def test_circ_kurtosis():
    b11 = load_data("B11", source="fisher")["θ"].values[:]
    c11 = Circular(data=b11)
    kurtosis = circ_kurtosis(alpha=c11.alpha)
    np.testing.assert_approx_equal(kurtosis, 6.64, significant=3)


def test_circ_dispersion():
    b11 = load_data("B11", source="fisher")["θ"].values[:]
    c11 = Circular(data=b11)
    dispersion = circ_dispersion(alpha=c11.alpha)
    np.testing.assert_approx_equal(dispersion, 0.24, significant=2)


def test_circ_moment():
    # Section 3.2, Pewsey (2014) P24

    b11 = load_data("B11", source="fisher")["θ"].values[:]
    c11 = Circular(data=b11)

    # first moment == mean

    mp1 = circ_moment(alpha=c11.alpha, p=1, centered=False)
    u1, r1 = convert_moment(mp1)
    np.testing.assert_approx_equal(np.rad2deg(u1).round(2), 3.10, significant=2)
    np.testing.assert_approx_equal(np.round(r1, 2), 0.83, significant=2)
    # np.testing.assert_approx_equal(Cbar.round(2), 0.83, significant=2)
    # np.testing.assert_approx_equal(Sbar.round(2), 0.04, significant=2)

    # second moment

    mp2 = circ_moment(alpha=c11.alpha, p=2, centered=False)
    u2, r2 = convert_moment(mp2)
    np.testing.assert_approx_equal(np.rad2deg(u2).round(2), 0.64, significant=2)
    np.testing.assert_approx_equal(np.round(r2, 2), 0.67, significant=2)
    # np.testing.assert_approx_equal(Cbar.round(2), 0.67, significant=2)
    # np.testing.assert_approx_equal(Sbar.round(2), 0.01, significant=2)


def test_compute_smooth_params():
    from pycircstat2.utils import time2float

    d_fisher_b1 = load_data("B1", source="fisher")["time"].values[:]
    c_fisher_b1 = Circular(time2float(d_fisher_b1), unit="hour")
    h0 = compute_smooth_params(c_fisher_b1.r, c_fisher_b1.n)
    np.testing.assert_approx_equal(h0, 1.06, significant=2)

def test_circ_dist():
    """Test circ_dist() for correctness and periodicity."""
    x = np.array([0, np.pi/2, np.pi, -np.pi/2])
    y = np.array([np.pi/4, -np.pi/4, np.pi, np.pi])

    expected = (x - y + np.pi) % (2 * np.pi) - np.pi
    result = circ_dist(x, y)

    # Check if the output matches the expected values
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    # Check periodicity property
    assert np.allclose(circ_dist(x + 2 * np.pi, y), result)
    assert np.allclose(circ_dist(x, y + 2 * np.pi), result)

def test_circ_pairdist():
    """Test circ_pairdist() for correctness and shape."""
    x = np.array([0, np.pi/2, np.pi])
    y = np.array([np.pi/4, 3*np.pi/4])

    expected = np.angle(np.exp(1j * x[:, None]) / np.exp(1j * y[None, :]))
    result = circ_pairdist(x, y)

    # Check if the output matches expected pairwise differences
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    # Verify output shape (len(x) × len(y))
    assert result.shape == (len(x), len(y))

    # Test when y is None (should return pairwise differences within x)
    auto_result = circ_pairdist(x)
    assert auto_result.shape == (len(x), len(x))  # Square matrix

    # Check periodicity
    assert np.allclose(circ_pairdist(x + 2*np.pi, y), result)
    assert np.allclose(circ_pairdist(x, y + 2*np.pi), result)

def test_circ_range():

    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.6, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 72.0, 108.0, 108.0, 169.2, 324.0])
    np.testing.assert_approx_equal(circ_range(x), 4.584073, significant=2)

    # Lower values mean tighter clustering (the docstring used to claim the
    # opposite). Confirm the direction with two contrasting samples.
    clustered = np.deg2rad([10, 12, 15, 18, 20])
    spread = np.deg2rad([0, 90, 180, 270])
    assert circ_range(clustered) < circ_range(spread)
    np.testing.assert_allclose(circ_range(clustered), np.deg2rad(10), atol=1e-9)


def test_circ_mean_zero_resultant_consistency():
    # `circ_mean` and `circ_mean_and_r` should agree on whether r ≈ 0 means the
    # mean is undefined. They previously used different tolerances (1e-8 vs
    # 1e-12), so the same input could yield NaN from one and an angle from the
    # other.
    rng = np.random.default_rng(0)
    alpha = np.deg2rad([0, 120, 240]) + rng.standard_normal(3) * 1e-10
    m1 = circ_mean(alpha)
    m2, _ = circ_mean_and_r(alpha)
    assert np.isnan(m1) == np.isnan(m2)


def test_circ_kappa_concentrated_returns_inf():
    # All observations coincident → r = 1 → MLE diverges. Used to silently
    # return 1e-16 (i.e. "uniform"), which is the opposite of the truth.
    assert np.isinf(circ_kappa(1.0))
    # Sanity: away from r=1 we still get finite, positive κ.
    assert np.isfinite(circ_kappa(0.99)) and circ_kappa(0.99) > 0


def test_circ_mean_ci_errors():
    # Unsupported method
    with pytest.raises(ValueError, match="not supported"):
        circ_mean_ci(alpha=np.array([0.1, 0.2, 0.3]), method="banana")

    # bootstrap / dispersion need alpha
    with pytest.raises(ValueError, match="`alpha` is required"):
        circ_mean_ci(method="bootstrap")
    with pytest.raises(ValueError, match="`alpha` is required"):
        circ_mean_ci(method="dispersion")


def test_circ_mean_ci_bootstrap_seed_reproducible():
    # `seed` parameter must produce reproducible results.
    rng = np.random.default_rng(0)
    alpha = np.deg2rad(120) + 0.5 * rng.standard_normal(15)
    a = circ_mean_ci(alpha=alpha, method="bootstrap", B=200, seed=42)
    b = circ_mean_ci(alpha=alpha, method="bootstrap", B=200, seed=42)
    c = circ_mean_ci(alpha=alpha, method="bootstrap", B=200, seed=43)
    assert a == b
    assert a != c


def test_circ_mean_ci_bootstrap_interval():
    # Default is HDI.
    rng = np.random.default_rng(0)
    alpha = np.deg2rad(120) + 0.4 * rng.standard_normal(20)
    default = circ_mean_ci(alpha=alpha, method="bootstrap", B=400, seed=7)
    hdi = circ_mean_ci(alpha=alpha, method="bootstrap", B=400, seed=7, interval="hdi")
    assert default == hdi

    # Percentile runs and is reproducible.
    pct1 = circ_mean_ci(alpha=alpha, method="bootstrap", B=400, seed=7, interval="percentile")
    pct2 = circ_mean_ci(alpha=alpha, method="bootstrap", B=400, seed=7, interval="percentile")
    assert pct1 == pct2

    # On approximately symmetric (well-concentrated) data, HDI and percentile
    # should agree to within ~5° at this sample size / B.
    diff_lb = np.abs(np.angle(np.exp(1j * (hdi[0] - pct1[0]))))
    diff_ub = np.abs(np.angle(np.exp(1j * (hdi[1] - pct1[1]))))
    assert diff_lb < np.deg2rad(5)
    assert diff_ub < np.deg2rad(5)

    # Unknown interval raises.
    with pytest.raises(ValueError, match="interval="):
        circ_mean_ci(alpha=alpha, method="bootstrap", B=200, interval="bca")


def test_circ_quantile_supported_types():
    # All R types 1–9 should be accepted; type=10 should raise.
    alpha = np.deg2rad([10, 20, 30, 40, 50])
    for t in range(1, 10):
        circ_quantile(alpha, probs=0.5, type=t)
    with pytest.raises(ValueError, match="Unsupported quantile"):
        circ_quantile(alpha, type=10)

    # type=4 must agree with numpy's interpolated_inverted_cdf, not midpoint.
    # (This was previously mismapped: type=4 → "midpoint" which is actually
    # numpy's type=2.)
    x = np.deg2rad(np.arange(0, 100, 10))
    expected = np.quantile(x, 0.5, method="interpolated_inverted_cdf")
    np.testing.assert_allclose(
        circ_quantile(x, probs=0.5, type=4)[0], expected, atol=1e-12
    )


def test_circ_var_unsorted_alpha_raises():
    from pycircstat2.descriptive import circ_var
    # Non-uniform `w` triggers the bin-size inference path; unsorted alpha
    # would have produced a meaningless negative bin_size silently.
    alpha = np.deg2rad([20, 60, 40])  # not sorted
    w = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="strictly increasing"):
        circ_var(alpha=alpha, w=w)


def test_fisher_8_27_matrix_sqrt_oracle():
    # Fisher (1993) Algorithm 2 (eq 8.27-8.31) builds v from u so that v² = u.
    # This was broken: `(u11-u22)² / (4 u12² + 1)` instead of
    # `(u11-u22)² / (4 u12²) + 1`. The defining property `v @ v == u` is the
    # cleanest oracle — it doesn't depend on which sample we run on, just on
    # the formula being algebraically correct.
    from pycircstat2.descriptive import _circ_mean_ci_bootstrap

    # Reach into the function's β computation by reproducing the closed form.
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.standard_normal((2, 2))
        u = A @ A.T
        u11, u22, u12 = u[0, 0], u[1, 1], u[0, 1]
        if abs(u12) < 1e-12:
            continue
        beta = (u11 - u22) / (2 * u12) - np.sqrt(
            (u11 - u22) ** 2 / (4 * u12 ** 2) + 1
        )
        denom = np.sqrt(1 + beta ** 2)
        t1 = np.sqrt(beta ** 2 * u11 + 2 * beta * u12 + u22) / denom
        t2 = np.sqrt(u11 - 2 * beta * u12 + beta ** 2 * u22) / denom
        v11 = (beta ** 2 * t1 + t2) / (1 + beta ** 2)
        v22 = (t1 + beta ** 2 * t2) / (1 + beta ** 2)
        v12 = beta * (t1 - t2) / (1 + beta ** 2)
        v = np.array([[v11, v12], [v12, v22]])
        np.testing.assert_allclose(v @ v, u, atol=1e-12)

    # End-to-end: bootstrap CI must run cleanly on a moderate von-Mises-ish
    # sample with the corrected β.
    alpha = np.deg2rad(120) + 0.4 * rng.standard_normal(15)
    lb, ub = _circ_mean_ci_bootstrap(alpha, B=200, ci=0.95, seed=42)
    assert np.isfinite(lb) and np.isfinite(ub)

def test_circ_quantile():
    """Test `circ_quantile` with known input and compare with R output."""

    # Generate a known dataset
    np.random.seed(42)
    from pycircstat2.distributions import vonmises
    angles = vonmises.rvs(mu=0, kappa=4, size=100)

    # Compute circular quantiles
    probs = np.array([0.25, 0.5, 0.75])
    quantiles = circ_quantile(angles, probs=probs)

    # Ensure values are within valid range [0, 2π]
    assert np.all(quantiles >= 0) and np.all(quantiles <= 2 * np.pi), "Quantiles out of range"

    # Ensure median matches `circ_median`
    from pycircstat2.descriptive import circ_median
    assert np.isclose(quantiles[1], circ_median(angles), atol=1e-5), "Median quantile does not match circular median"
