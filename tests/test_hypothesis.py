import warnings

import numpy as np
import pytest

from pycircstat2 import Circular, load_data
from pycircstat2.distributions import vonmises
from pycircstat2.hypothesis import (
    V_test,
    angular_randomisation_test,
    batschelet_test,
    binomial_test,
    change_point_test,
    chisquare_test,
    circ_anova,
    circ_range_test,
    common_median_test,
    concentration_test,
    equal_kappa_test,
    harrison_kanji_test,
    kuiper_test,
    omnibus_test,
    one_sample_test,
    rao_homogeneity_test,
    rao_spacing_test,
    rayleigh_test,
    symmetry_test,
    wallraff_test,
    watson_test,
    watson_u2_test,
    kuiper_two_test,
    watson_williams_test,
    wheeler_watson_test,
)


def test_rayleigh_test():
    # Ch27 Example 1 (Zar, 2010, P667)
    # Using data from Ch26 Example 2.
    data_zar_ex2_ch26 = load_data("D1", source="zar")
    circ_zar_ex1_ch27 = Circular(data=data_zar_ex2_ch26["θ"].values[:])

    # computed directly from r and n
    result = rayleigh_test(n=circ_zar_ex1_ch27.n, r=circ_zar_ex1_ch27.r)
    np.testing.assert_approx_equal(result.z, 5.448, significant=3)
    assert 0.001 < result.pval < 0.002

    # computed directly from alpha
    result = rayleigh_test(alpha=circ_zar_ex1_ch27.alpha)
    np.testing.assert_approx_equal(result.z, 5.448, significant=3)
    assert 0.001 < result.pval < 0.002


def test_V_test():
    # Ch27 Example 2 (Zar, 2010, P669)
    data_zar_ex2_ch27 = load_data("D7", source="zar")
    circ_zar_ex2_ch27 = Circular(data=data_zar_ex2_ch27["θ"].values[:])

    # computed directly from r and n
    result = V_test(
        angle=np.deg2rad(90),
        mean=circ_zar_ex2_ch27.mean,
        n=circ_zar_ex2_ch27.n,
        r=circ_zar_ex2_ch27.r,
    )

    np.testing.assert_approx_equal(result.V, 9.498, significant=3)
    np.testing.assert_approx_equal(result.u, 4.248, significant=3)
    assert result.pval < 0.0005

    # computed directly from alpha
    result = V_test(
        alpha=circ_zar_ex2_ch27.alpha,
        angle=np.deg2rad(90),
    )

    np.testing.assert_approx_equal(result.V, 9.498, significant=3)
    np.testing.assert_approx_equal(result.u, 4.248, significant=3)
    assert result.pval < 0.0005


def test_one_sample_test():
    # Ch27 Example 3 (Zar, 2010, P669)
    # Using data from Ch27 Example 2
    data_zar_ex2_ch27 = load_data("D7", source="zar")
    circ_zar_ex3_ch27 = Circular(data=data_zar_ex2_ch27["θ"].values[:], unit="degree")

    # computed directly from lb and ub
    result = one_sample_test(
        lb=circ_zar_ex3_ch27.mean_lb,
        ub=circ_zar_ex3_ch27.mean_ub,
        angle=np.deg2rad(90),
    )

    assert result.reject is False

    # computed directly from alpha
    result = one_sample_test(alpha=circ_zar_ex3_ch27.alpha, angle=np.deg2rad(90))

    assert result.reject is False


def test_omnibus_test():
    data_zar_ex4_ch27 = load_data("D8", source="zar")
    circ_zar_ex4_ch27 = Circular(data=data_zar_ex4_ch27["θ"].values[:], unit="degree")

    result = omnibus_test(alpha=circ_zar_ex4_ch27.alpha, scale=1)

    np.testing.assert_approx_equal(result.pval, 0.0043, significant=2)

    # test large sample size
    # (factorial division overflow while computing p-val)
    # fixed in PR 12
    from pycircstat2.distributions import circularuniform, vonmises

    rng = np.random.default_rng(42)
    d0 = vonmises.rvs(mu=0, kappa=1, size=10_000, random_state=rng)
    d1 = circularuniform.rvs(size=10_000, random_state=rng)

    result = omnibus_test(alpha=d0)
    assert result.pval < 0.05, "Expected significant p-value for von Mises distribution"
    result = omnibus_test(alpha=d1)
    assert result.pval > 0.05, (
        "Expected non-significant p-value for uniform distribution"
    )


def test_batschelet_test():
    data_zar_ex5_ch27 = load_data("D8", source="zar")
    circ_zar_ex5_ch27 = Circular(data=data_zar_ex5_ch27["θ"].values[:], unit="degree")

    result = batschelet_test(
        angle=np.deg2rad(45),
        alpha=circ_zar_ex5_ch27.alpha,
    )
    np.testing.assert_equal(result.C, 5)
    np.testing.assert_approx_equal(result.pval, 0.00661, significant=3)


def test_chisquare_test():
    d2 = load_data("D2", source="zar")
    c2 = Circular(data=d2["θ"].values[:], w=d2["w"].values[:])

    result = chisquare_test(c2.w)
    np.testing.assert_approx_equal(result.chi2, 66.543, significant=3)
    assert result.pval < 0.001


def test_symmetry_test():
    data_zar_ex6_ch27 = load_data("D9", source="zar")
    circ_zar_ex6_ch27 = Circular(data=data_zar_ex6_ch27["θ"].values[:], unit="degree")

    result = symmetry_test(
        median=float(circ_zar_ex6_ch27.median), alpha=circ_zar_ex6_ch27.alpha
    )
    assert result.pval > 0.5


def test_watson_williams_test():
    data = load_data("D10", source="zar")
    s1 = Circular(data=data[data["sample"] == 1]["θ"].values[:])
    s2 = Circular(data=data[data["sample"] == 2]["θ"].values[:])
    result = watson_williams_test([s1, s2])

    np.testing.assert_approx_equal(result.F, 1.61, significant=3)
    np.testing.assert_approx_equal(result.pval, 0.22, significant=2)

    # Support plain arrays
    array_result = watson_williams_test([s1.alpha, s2.alpha])
    np.testing.assert_allclose(array_result.F, result.F, rtol=1e-6)
    np.testing.assert_allclose(array_result.pval, result.pval, rtol=1e-6)

    data = load_data("D11", source="zar")
    s1 = Circular(data=data[data["sample"] == 1]["θ"].values[:])
    s2 = Circular(data=data[data["sample"] == 2]["θ"].values[:])
    s3 = Circular(data=data[data["sample"] == 3]["θ"].values[:])

    result = watson_williams_test([s1, s2, s3])

    np.testing.assert_approx_equal(result.F, 1.86, significant=3)
    np.testing.assert_approx_equal(result.pval, 0.19, significant=2)


def test_watson_u2_test():
    d = load_data("D12", source="zar")
    c0 = Circular(data=d[d["sample"] == 1]["θ"].values[:])
    c1 = Circular(data=d[d["sample"] == 2]["θ"].values[:])
    result = watson_u2_test([c0, c1])

    np.testing.assert_approx_equal(result.U2, 0.1458, significant=3)
    assert 0.1 < result.pval < 0.2

    # Array support
    array_result = watson_u2_test([c0.alpha, c1.alpha])
    np.testing.assert_allclose(array_result.U2, result.U2, rtol=1e-6)
    np.testing.assert_allclose(array_result.pval, result.pval, rtol=1e-6)

    d = load_data("D13", source="zar")
    c0 = Circular(
        data=d[d["sample"] == 1]["θ"].values[:], w=d[d["sample"] == 1]["w"].values[:]
    )
    c1 = Circular(
        data=d[d["sample"] == 2]["θ"].values[:], w=d[d["sample"] == 2]["w"].values[:]
    )
    result = watson_u2_test([c0, c1])

    np.testing.assert_approx_equal(result.U2, 0.0612, significant=3)
    assert result.pval > 0.5

    expanded0 = np.repeat(c0.alpha, c0.w)
    expanded1 = np.repeat(c1.alpha, c1.w)
    array_result = watson_u2_test([expanded0, expanded1])
    np.testing.assert_allclose(array_result.U2, result.U2, rtol=1e-6)
    np.testing.assert_allclose(array_result.pval, result.pval, rtol=1e-6)


def test_kuiper_two_test():
    """Two-sample Kuiper test: direction, dispersion sensitivity, rotation-invariance,
    grouped-data support, and asymptotic/randomization determinism."""
    rng = np.random.default_rng(0)
    a = vonmises.rvs(mu=0.0, kappa=3.0, size=40, random_state=rng)
    same = vonmises.rvs(mu=0.0, kappa=3.0, size=40, random_state=rng)
    loc_shift = vonmises.rvs(mu=1.2, kappa=3.0, size=40, random_state=rng)
    disp_change = vonmises.rvs(mu=0.0, kappa=0.4, size=40, random_state=rng)

    assert kuiper_two_test([a, same]).pval > 0.05
    assert kuiper_two_test([a, loc_shift]).pval < 0.05  # location difference
    assert kuiper_two_test([a, disp_change]).pval < 0.05  # dispersion difference

    r = kuiper_two_test([a, loc_shift])
    assert r.method == "asymptotic" and r.n_resamples == 0

    # statistic is invariant to a common rotation (Kuiper's defining property)
    shift = 2.0
    v_rot = kuiper_two_test(
        [(a + shift) % (2 * np.pi), (loc_shift + shift) % (2 * np.pi)]
    ).V
    np.testing.assert_allclose(r.V, v_rot, atol=1e-9)

    # grouped data expand consistently with raw angles
    d = load_data("D12", source="zar")
    c0 = Circular(data=d[d["sample"] == 1]["θ"].values[:])
    c1 = Circular(data=d[d["sample"] == 2]["θ"].values[:])
    np.testing.assert_allclose(
        kuiper_two_test([c0, c1]).V, kuiper_two_test([c0.alpha, c1.alpha]).V, rtol=1e-9
    )

    # randomization path: determinism (int seed == Generator)
    rr = kuiper_two_test([a, loc_shift], n_resamples=999, seed=3)
    assert rr.method == "randomization" and rr.n_resamples == 999
    p_gen = kuiper_two_test([a, loc_shift], n_resamples=999, seed=np.random.default_rng(3)).pval
    assert rr.pval == p_gen


def test_wheeler_watson_test():
    d = load_data("D12", source="zar")
    c0 = Circular(data=d[d["sample"] == 1]["θ"].values[:])
    c1 = Circular(data=d[d["sample"] == 2]["θ"].values[:])

    result = wheeler_watson_test([c0, c1])
    np.testing.assert_approx_equal(result.W, 3.678, significant=3)
    assert 0.1 < result.pval < 0.25

    array_result = wheeler_watson_test([c0.alpha, c1.alpha])
    np.testing.assert_allclose(array_result.W, result.W, rtol=1e-6)
    np.testing.assert_allclose(array_result.pval, result.pval, rtol=1e-6)


def test_wallraff_test():
    d = load_data("D14", source="zar")
    c0 = Circular(data=d[d["sex"] == "male"]["θ"].values[:])
    c1 = Circular(data=d[d["sex"] == "female"]["θ"].values[:])
    result = wallraff_test(samples=[c0, c1], angle=np.deg2rad(135))
    np.testing.assert_approx_equal(result.U, 18.5, significant=3)
    assert result.pval > 0.20

    array_result = wallraff_test(samples=[c0.alpha, c1.alpha], angle=np.deg2rad(135))
    np.testing.assert_allclose(array_result.U, result.U, rtol=1e-6)
    np.testing.assert_allclose(array_result.pval, result.pval, rtol=1e-6)

    from pycircstat2.utils import time2float

    d = load_data("D15", source="zar")
    c0 = Circular(data=time2float(d[d["sex"] == "male"]["time"].values[:]))
    c1 = Circular(data=time2float(d[d["sex"] == "female"]["time"].values[:]))
    result = wallraff_test(
        angle=np.deg2rad(time2float(["7:55", "8:15"])),
        samples=[c0, c1],
        verbose=True,
    )
    np.testing.assert_equal(result.U, 13)
    assert result.pval > 0.05


def test_kuiper_test():
    d = load_data("B5", source="fisher")["θ"].values[:]
    c = Circular(data=d, unit="degree", full_cycle=180)
    result = kuiper_test(alpha=c.alpha)
    np.testing.assert_approx_equal(result.V, 1.5864, significant=3)
    assert result.pval > 0.05


def test_watson_test():
    pigeon = np.array([20, 135, 145, 165, 170, 200, 300, 325, 335, 350, 350, 350, 355])
    c_pigeon = Circular(data=pigeon)
    result = watson_test(alpha=c_pigeon.alpha, n_resamples=9999)
    np.testing.assert_approx_equal(result.U2, 0.137, significant=3)
    assert result.pval > 0.10
    assert result.dist == "uniform" and result.mu is None


def test_watson_test_vonmises_gof():
    """watson_test(dist='vonmises') is a parametric-bootstrap GoF: it accepts von Mises
    data, rejects a non–von Mises (wrapped Cauchy) alternative, and reports fitted μ, κ."""
    from pycircstat2.distributions import wrapcauchy

    rng = np.random.default_rng(0)
    vm = np.asarray(vonmises.rvs(mu=0.7, kappa=2.0, size=60, random_state=rng))
    wc = np.asarray(wrapcauchy.rvs(mu=0.7, rho=0.7, size=60, random_state=rng))

    r_vm = watson_test(vm, dist="vonmises", n_resamples=999, seed=3)
    assert r_vm.method == "parametric_bootstrap"
    assert r_vm.dist == "vonmises" and r_vm.mu is not None and r_vm.kappa is not None
    assert r_vm.pval > 0.05  # von Mises data: do not reject

    r_wc = watson_test(wc, dist="vonmises", n_resamples=999, seed=3)
    assert r_wc.pval < 0.05  # wrapped Cauchy: reject von Mises fit

    # no closed-form p-value for the von Mises null; determinism on the bootstrap path
    with pytest.raises(ValueError):
        watson_test(vm, dist="vonmises", n_resamples=0)
    p_gen = watson_test(vm, dist="vonmises", n_resamples=300,
                        seed=np.random.default_rng(11)).pval
    assert watson_test(vm, dist="vonmises", n_resamples=300, seed=11).pval == p_gen


def test_angular_randomisation_test():
    np.random.seed(42)
    alpha1 = Circular(np.random.vonmises(mu=0, kappa=3, size=10), unit="radian")
    alpha2 = Circular(np.random.vonmises(mu=0, kappa=3, size=50), unit="radian")

    result = angular_randomisation_test([alpha1, alpha2])
    assert result.pval > 0.05, "Expected non-significant p-value"

    array_result = angular_randomisation_test([alpha1.alpha, alpha2.alpha])
    np.testing.assert_allclose(array_result.statistic, result.statistic, rtol=1e-6)


def test_rao_spacing_test():
    pigeon = np.array([20, 135, 145, 165, 170, 200, 300, 325, 335, 350, 350, 350, 355])
    c_pigeon = Circular(data=pigeon)
    result = rao_spacing_test(alpha=c_pigeon.alpha, n_resamples=9999)
    np.testing.assert_approx_equal(result.statistic, 161.92308, significant=3)
    assert 0.05 < result.pval < 0.10


def test_randomized_tests_seed_harmonization():
    alpha = np.linspace(0.0, 2 * np.pi, 12, endpoint=False)
    seed_value = 123

    def make_generator():
        return np.random.default_rng(seed_value)

    rayleigh_int = rayleigh_test(alpha=alpha, n_resamples=128, seed=seed_value)
    rayleigh_gen = rayleigh_test(alpha=alpha, n_resamples=128, seed=make_generator())
    assert rayleigh_int.pval == rayleigh_gen.pval

    samples = [alpha[:6], alpha[6:]]
    art_int = angular_randomisation_test(samples, n_resamples=128, seed=seed_value)
    art_gen = angular_randomisation_test(
        samples, n_resamples=128, seed=make_generator()
    )
    assert art_int.pval == art_gen.pval

    kuiper_int = kuiper_test(alpha=alpha, n_resamples=256, seed=seed_value)
    kuiper_gen = kuiper_test(alpha=alpha, n_resamples=256, seed=make_generator())
    assert kuiper_int.pval == kuiper_gen.pval

    watson_int = watson_test(alpha=alpha, n_resamples=256, seed=seed_value)
    watson_gen = watson_test(alpha=alpha, n_resamples=256, seed=make_generator())
    assert watson_int.pval == watson_gen.pval

    rao_int = rao_spacing_test(alpha=alpha, n_resamples=256, seed=seed_value)
    rao_gen = rao_spacing_test(alpha=alpha, n_resamples=256, seed=make_generator())
    assert rao_int.pval == rao_gen.pval


def test_circ_range_test():
    x_deg = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.6,
            36.0,
            36.0,
            36.0,
            36.0,
            36.0,
            36.0,
            72.0,
            108.0,
            108.0,
            169.2,
            324.0,
        ]
    )
    x_rad = np.deg2rad(x_deg)
    result = circ_range_test(x_rad)
    np.testing.assert_approx_equal(result.range_stat, 3.581416, significant=5)
    np.testing.assert_approx_equal(result.pval, 5.825496e-05, significant=5)


def test_circ_range_test_rejects_degree_input():
    x_deg = np.array([0.0, 10.0, 20.0])
    with pytest.raises(ValueError):
        circ_range_test(x_deg)


def test_binomial_test_uniform():
    """Test binomial_test with uniform circular data (should not reject H0)."""
    np.random.seed(42)
    alpha = np.random.uniform(0, 2 * np.pi, 100)  # Uniformly distributed angles
    md = np.pi  # Test median at π (should be non-significant)

    result = binomial_test(alpha, md)

    assert 0.05 < result.pval < 1.0, (
        f"Unexpected p-value for uniform data: {result.pval}"
    )


def test_binomial_test_skewed():
    """Test binomial_test with a skewed circular distribution (should reject H0)."""
    np.random.seed(42)
    alpha = np.random.vonmises(mu=np.pi / 4, kappa=3, size=100)  # Clustered around π/4
    md = np.pi  # Incorrect median hypothesis

    result = binomial_test(alpha, md)

    assert result.pval < 0.05, f"Expected significant p-value but got {result.pval}"


def test_binomial_test_symmetric():
    """Test binomial_test with symmetric distribution around π (should fail to reject H0)."""
    alpha = np.array([-np.pi / 4, np.pi / 4, np.pi / 2, -np.pi / 2, np.pi])
    md = np.pi  # This should be a valid median

    result = binomial_test(alpha, md)

    assert result.pval > 0.05, f"Unexpected p-value for symmetric data: {result.pval}"


def test_binomial_test_extreme_case():
    """Test binomial_test with all points clustered at π (extreme case)."""
    alpha = np.full(20, np.pi)  # All angles at π
    md = np.pi

    result = binomial_test(alpha, md)

    assert np.isclose(result.pval, 1.0), (
        f"Expected p-value of 1 for identical data but got {result.pval}"
    )


def test_concentration_identical():
    """Test concentration_test with identical von Mises distributions (should fail to reject H0)."""
    rng = np.random.default_rng(42)
    alpha1 = vonmises.rvs(mu=0, kappa=3, size=50, random_state=rng)
    alpha2 = vonmises.rvs(mu=0, kappa=3, size=50, random_state=rng)

    result = concentration_test(alpha1, alpha2)

    assert result.pval > 0.05, (
        f"Unexpectedly small p-value: {result.pval}, should not reject H0."
    )


def test_concentration_different():
    """Test concentration_test with different kappa values (should reject H0)."""
    rng = np.random.default_rng(123)
    alpha1 = vonmises.rvs(
        mu=0, kappa=3, size=50, random_state=rng
    )  # Higher concentration
    alpha2 = vonmises.rvs(
        mu=0, kappa=1, size=50, random_state=rng
    )  # Lower concentration

    result = concentration_test(alpha1, alpha2)

    assert result.pval < 0.05, f"Expected small p-value, but got {result.pval}"


def test_concentration_high_dispersion():
    """Test concentration_test with very dispersed data (should fail to reject H0)."""
    np.random.seed(42)
    alpha1 = np.random.uniform(0, 2 * np.pi, 50)  # Uniformly spread
    alpha2 = np.random.uniform(0, 2 * np.pi, 50)

    result = concentration_test(alpha1, alpha2)

    assert result.pval > 0.05, (
        f"Unexpectedly small p-value: {result.pval}, should not reject H0."
    )


def test_concentration_extreme_case():
    """Test concentration_test when both samples have extremely high concentration (should fail to reject H0)."""
    rng = np.random.default_rng(42)
    alpha1 = vonmises.rvs(mu=0, kappa=100, size=50, random_state=rng)
    alpha2 = vonmises.rvs(mu=0, kappa=100, size=50, random_state=rng)

    result = concentration_test(alpha1, alpha2)

    assert result.pval > 0.05, (
        f"Unexpectedly small p-value: {result.pval}, should not reject H0."
    )


def test_rao_homogeneity_identical():
    """Test with identical von Mises distributions (should fail to reject H0)."""
    seeds = [101, 102, 103]
    samples = [
        vonmises.rvs(mu=0, kappa=2, size=50, random_state=np.random.default_rng(seed))
        for seed in seeds
    ]

    results = rao_homogeneity_test(samples)

    assert results.pval_polar > 0.05, (
        f"Unexpectedly small p-value: {results.pval_polar}"
    )
    assert results.pval_disp > 0.05, f"Unexpectedly small p-value: {results.pval_disp}"


def test_rao_homogeneity_different_means():
    """Test with different mean directions (should reject H0 for mean equality)."""
    seeds = [201, 202, 203]
    mus = (0.0, np.pi / 4, np.pi / 2)
    samples = [
        vonmises.rvs(kappa=2, mu=mu, size=50, random_state=np.random.default_rng(seed))
        for seed, mu in zip(seeds, mus)
    ]
    results = rao_homogeneity_test(samples)

    assert results.pval_polar < 0.05, (
        f"Expected rejection but got p={results.pval_polar}"
    )


def test_rao_homogeneity_different_dispersion():
    """Test with different kappa values (should reject H0 for dispersion equality)."""
    seeds = [301, 302, 303]
    kappas = (5, 2, 1)
    samples = [
        vonmises.rvs(
            mu=0, kappa=kappa, size=50, random_state=np.random.default_rng(seed)
        )
        for seed, kappa in zip(seeds, kappas)
    ]

    results = rao_homogeneity_test(samples)

    assert results.pval_disp < 0.05, f"Expected rejection but got p={results.pval_disp}"


def test_rao_homogeneity_randomization():
    """rao_homogeneity_test(n_resamples>0) gives permutation p-values that isolate each
    effect: a mean shift flags only H_polar, a dispersion change only H_disp."""
    def mk(seeds, mus, kappas, size=50):
        return [
            vonmises.rvs(mu=m, kappa=k, size=size, random_state=np.random.default_rng(s))
            for s, m, k in zip(seeds, mus, kappas)
        ]

    identical = mk([101, 102, 103], [0, 0, 0], [2, 2, 2])
    diff_mean = mk([201, 202, 203], [0, np.pi / 4, np.pi / 2], [2, 2, 2])
    diff_disp = mk([301, 302, 303], [0, 0, 0], [5, 2, 1])

    r_id = rao_homogeneity_test(identical, n_resamples=1999, seed=7)
    assert r_id.method == "randomization" and r_id.n_resamples == 1999
    assert r_id.pval_polar > 0.05 and r_id.pval_disp > 0.05

    r_mean = rao_homogeneity_test(diff_mean, n_resamples=1999, seed=7)
    assert r_mean.pval_polar < 0.05 < r_mean.pval_disp  # only mean direction flagged

    r_disp = rao_homogeneity_test(diff_disp, n_resamples=1999, seed=7)
    assert r_disp.pval_disp < 0.05 < r_disp.pval_polar  # only dispersion flagged

    # default = asymptotic; determinism of the randomization path
    assert rao_homogeneity_test(identical).method == "asymptotic"
    a = rao_homogeneity_test(diff_mean, n_resamples=300, seed=11)
    b = rao_homogeneity_test(diff_mean, n_resamples=300, seed=np.random.default_rng(11))
    assert a.pval_polar == b.pval_polar and a.pval_disp == b.pval_disp


def test_rao_homogeneity_small_samples():
    """Test with very small sample sizes (should handle without error)."""
    seeds = [401, 402, 403]
    samples = [
        vonmises.rvs(mu=0, kappa=3, size=5, random_state=np.random.default_rng(seed))
        for seed in seeds
    ]

    results = rao_homogeneity_test(samples)

    assert isinstance(results.pval_polar, float)
    assert isinstance(results.pval_disp, float)


def test_rao_homogeneity_invalid_input():
    """Test invalid input (should raise ValueError)."""
    with pytest.raises(ValueError):
        rao_homogeneity_test([np.array([0, np.pi / 2]), "invalid_array"])


def test_change_point_basic():
    """Test change_point_test() on a simple dataset matching R."""
    alpha = np.array(
        [
            3.03,
            0.28,
            3.90,
            5.56,
            5.77,
            5.06,
            5.96,
            0.16,
            0.51,
            1.21,
            6.03,
            1.05,
            0.45,
            1.47,
            6.09,
        ]
    )

    result = change_point_test(alpha)

    # Expected values based on R output
    expected_rho = 0.52307
    expected_rmax = 2.237654
    expected_k_r = 6
    expected_rave = 1.066862
    expected_tmax = 0.602549
    expected_k_t = 6
    expected_tave = 0.460675

    assert np.isclose(result.rho, expected_rho, atol=1e-5)
    assert np.isclose(result.rmax, expected_rmax, atol=1e-5)
    assert result.k_r == expected_k_r
    assert np.isclose(result.rave, expected_rave, atol=1e-5)
    assert np.isclose(result.tmax, expected_tmax, atol=1e-5)
    assert result.k_t == expected_k_t
    assert np.isclose(result.tave, expected_tave, atol=1e-5)


def test_harrison_kanji_test():
    """Test Harrison-Kanji two-way ANOVA for circular data."""
    np.random.seed(42)
    alpha = np.random.vonmises(0, 2, 50)
    idp = np.random.choice([1, 2, 3], 50)
    idq = np.random.choice([1, 2], 50)

    result = harrison_kanji_test(alpha, idp, idq)

    assert len(result.p_values) == 3  # Should return three p-values
    assert result.anova_table.shape[0] >= 3  # At least 3 sources in ANOVA table
    assert all(0 <= p <= 1 for p in result.p_values if p is not None)  # Valid p-values


def test_harrison_kanji_vs_pycircstat():
    """Compare PyCircStat2 `harrison_kanji_test` with original PyCircStat `hktest`."""

    def hktest(alpha, idp, idq, inter=True, fn=None):
        """copied and fixed from pycircstat.hktest"""
        import pandas as pd
        from scipy import special, stats

        from pycircstat2.descriptive import circ_kappa, circ_mean, circ_r

        if fn is None:
            fn = ["A", "B"]
        p = len(np.unique(idp))
        q = len(np.unique(idq))
        df = pd.DataFrame({fn[0]: idp, fn[1]: idq, "dependent": alpha})
        n = len(df)
        tr = n * circ_r(np.asarray(df["dependent"].values))
        kk = circ_kappa(tr / n)

        # both factors
        gr = df.groupby(fn)
        cn = gr.count()
        cr = gr.agg(circ_r) * cn
        cn = cn.unstack(fn[1])
        cr = cr.unstack(fn[1])

        # factor A
        gr = df.groupby(fn[0])
        pn = gr.count()["dependent"]
        pr = gr.agg(circ_r)["dependent"] * pn
        pm = gr.agg(circ_mean)["dependent"]
        # factor B
        gr = df.groupby(fn[1])
        qn = gr.count()["dependent"]
        qr = gr.agg(circ_r)["dependent"] * qn
        qm = gr.agg(circ_mean)["dependent"]

        if kk > 2:  # large kappa
            # effect of factor 1
            eff_1 = sum(pr**2 / cn.sum(axis=1)) - tr**2 / n
            df_1 = p - 1
            ms_1 = eff_1 / df_1

            # effect of factor 2
            eff_2 = sum(qr**2.0 / cn.sum(axis=0)) - tr**2 / n
            df_2 = q - 1
            ms_2 = eff_2 / df_2

            # total effect
            eff_t = n - tr**2 / n
            df_t = n - 1
            m = cn.values[:].mean()

            if inter:
                # correction factor for improved F statistic
                beta = 1 / (1 - 1 / (5 * kk) - 1 / (10 * (kk**2)))
                # residual effects
                eff_r = n - (cr**2.0 / cn).values[:].sum()
                df_r = p * q * (m - 1)
                ms_r = eff_r / df_r

                # interaction effects
                eff_i = (
                    (cr**2.0 / cn).values[:].sum()
                    - sum(qr**2.0 / qn)
                    - sum(pr**2.0 / pn)
                    + tr**2 / n
                )
                df_i = (p - 1) * (q - 1)
                ms_i = eff_i / df_i
                # interaction test statistic
                FI = ms_i / ms_r
                pI = 1 - stats.f.cdf(FI, df_i, df_r)
            else:
                # residual effect
                eff_r = n - sum(qr**2.0 / qn) - sum(pr**2.0 / pn) + tr**2 / n
                df_r = (p - 1) * (q - 1)
                ms_r = eff_r / df_r

                # interaction effects
                eff_i = None
                df_i = None
                ms_i = None

                # interaction test statistic
                FI = None
                pI = np.nan
                beta = 1

            F1 = beta * ms_1 / ms_r
            p1 = 1 - stats.f.cdf(F1, df_1, df_r)

            F2 = beta * ms_2 / ms_r
            p2 = 1 - stats.f.cdf(F2, df_2, df_r)

        else:  # small kappa
            # correction factor
            # special.iv is Modified Bessel function of the first kind of real order
            rr = special.iv(1, kk) / special.iv(0, kk)
            f = 2 / (1 - rr**2)

            chi1 = f * (sum(pr**2.0 / pn) - tr**2 / n)
            df_1 = 2 * (p - 1)
            p1 = 1 - stats.chi2.cdf(chi1, df=df_1)

            chi2 = f * (sum(qr**2.0 / qn) - tr**2 / n)
            df_2 = 2 * (q - 1)
            p2 = 1 - stats.chi2.cdf(chi2, df=df_2)

            chiI = f * (
                (cr**2.0 / cn).values[:].sum()
                - sum(pr**2.0 / pn)
                - sum(qr**2.0 / qn)
                + tr**2 / n
            )
            df_i = (p - 1) * (q - 1)
            pI = stats.chi2.sf(chiI, df=df_i)

        pval = (p1.squeeze(), p2.squeeze(), pI.squeeze())

        if kk > 2:
            table = pd.DataFrame(
                {
                    "Source": fn + ["Interaction", "Residual", "Total"],
                    "DoF": [df_1, df_2, df_i, df_r, df_t],
                    "SS": [eff_1, eff_2, eff_i, eff_r, eff_t],
                    "MS": [ms_1, ms_2, ms_i, ms_r, np.nan],
                    "F": [F1.squeeze(), F2.squeeze(), FI, np.nan, np.nan],
                    "p": list(pval) + [np.nan, np.nan],
                }
            )
            table = table.set_index("Source")
        else:
            table = pd.DataFrame(
                {
                    "Source": fn + ["Interaction"],
                    "DoF": [df_1, df_2, df_i],
                    "chi2": [chi1.squeeze(), chi2.squeeze(), chiI.squeeze()],
                    "p": pval,
                }
            )
            table = table.set_index("Source")

        return pval, table

    alpha = np.random.vonmises(0, 2, 50)
    idp = np.random.choice([1, 2, 3], 50)
    idq = np.random.choice([1, 2], 50)

    # Run original PyCircStat test
    pval_orig, table_orig = hktest(alpha, idp, idq)

    # Run PyCircStat2 version
    result_new = harrison_kanji_test(alpha, idp, idq)
    pval_new = result_new.p_values
    table_new = result_new.anova_table

    # Compare p-values
    assert np.allclose(pval_orig, pval_new, atol=1e-6), (
        f"P-values mismatch:\nOriginal: {pval_orig}\nNew: {pval_new}"
    )

    # Compare ANOVA table values (ignoring index differences)
    table_orig_values = table_orig.to_numpy()
    table_new_values = table_new.to_numpy()

    assert np.allclose(
        table_orig_values, table_new_values, atol=1e-6, equal_nan=True
    ), f"ANOVA tables differ:\nOriginal:\n{table_orig}\nNew:\n{table_new}"


def test_circ_anova():
    """Test the Circular ANOVA (F-test & LRT) for multiple samples."""

    # Set seed for reproducibility
    np.random.seed(42)

    # Generate von Mises distributed samples with different mean directions
    group1 = np.random.vonmises(mu=0, kappa=5, size=50)
    group2 = np.random.vonmises(mu=np.pi / 4, kappa=5, size=50)
    group3 = np.random.vonmises(mu=np.pi / 2, kappa=5, size=50)

    samples = [group1, group2, group3]

    # Run F-test
    result_f = circ_anova(samples, method="F-test")
    assert result_f.method == "F-test"
    assert 0 <= result_f.pval <= 1, "F-test p-value out of range"
    assert result_f.df == (2, 147, 149), (
        f"F-test degrees of freedom mismatch: {result_f.df}"
    )
    assert result_f.SS is not None and result_f.MS is not None

    # Run Likelihood Ratio Test (LRT)
    result_lrt = circ_anova(samples, method="LRT")
    assert result_lrt.method == "LRT"
    assert 0 <= result_lrt.pval <= 1, "LRT p-value out of range"
    assert result_lrt.df == 2, f"LRT degrees of freedom mismatch: {result_lrt.df}"

    # Edge case: All groups have the same mean direction
    identical_group = np.random.vonmises(mu=0, kappa=5, size=50)
    result_identical = circ_anova([identical_group] * 3, method="F-test")
    assert result_identical.pval > 0.05, (
        "F-test should not reject H0 for identical groups"
    )

    # Edge case: Small sample sizes
    small_group1 = np.random.vonmises(mu=0, kappa=5, size=5)
    small_group2 = np.random.vonmises(mu=np.pi / 4, kappa=5, size=5)
    small_group3 = np.random.vonmises(mu=np.pi / 2, kappa=5, size=5)

    result_small = circ_anova(
        [small_group1, small_group2, small_group3], method="F-test"
    )
    assert 0 <= result_small.pval <= 1, "Small-sample p-value out of range"

    # Invalid method check
    with pytest.raises(ValueError, match="Invalid method. Choose 'F-test' or 'LRT'."):
        circ_anova(samples, method="INVALID")

    # Single group should raise error
    with pytest.raises(ValueError, match="At least two groups are required for ANOVA."):
        circ_anova([group1])


def test_equal_median_identical_samples():
    """Test if the test correctly fails to reject H₀ when all groups are identical."""
    alpha1 = np.array([0.1, 0.2, 0.3, 1.5, 1.6])
    alpha2 = np.array([0.1, 0.2, 0.3, 1.5, 1.6])
    alpha3 = np.array([0.1, 0.2, 0.3, 1.5, 1.6])

    result = common_median_test([alpha1, alpha2, alpha3])
    assert result.reject is False
    assert not np.isnan(result.common_median)


def test_equal_median_different_samples():
    """Test if the test correctly rejects H₀ when groups have different medians."""
    alpha1 = np.array([0.1, 0.2, 0.3, 1.5, 1.6])
    alpha2 = np.array([2.2, 2.3, 2.4, 3.1, 3.2])
    alpha3 = np.array([3.5, 3.6, 3.7, 4.2, 4.3])

    result = common_median_test([alpha1, alpha2, alpha3])
    assert result.reject is True
    assert np.isnan(result.common_median)


def test_equal_median_large_sample():
    """Test the function on large sample sizes with similar medians."""
    np.random.seed(42)
    alpha1 = np.random.vonmises(mu=0, kappa=2, size=500)
    alpha2 = np.random.vonmises(mu=0, kappa=2, size=500)
    alpha3 = np.random.vonmises(mu=0, kappa=2, size=500)

    result = common_median_test([alpha1, alpha2, alpha3])
    assert result.reject is False
    assert not np.isnan(result.common_median)


def test_equal_median_small_sample():
    """Test if the function handles small sample sizes correctly."""
    alpha1 = np.array([0.1, 0.2, 0.3])
    alpha2 = np.array([0.15, 0.25, 0.35])

    result = common_median_test([alpha1, alpha2])
    assert result.reject is False
    assert not np.isnan(result.common_median)


def test_omnibus_evenly_spaced_not_rejected():
    """Maximally uniform data (m ≈ n/2) drives the analytic formula's
    denominator to zero; the test must not spuriously reject uniformity."""
    alpha = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    result = omnibus_test(alpha)
    assert result.pval == 1.0


def test_multisample_input_standards():
    """circ_anova / rao_homogeneity_test / equal_kappa_test / common_median_test
    must accept np.ndarray, list-of-lists, and Circular objects interchangeably."""
    rng = np.random.default_rng(0)
    groups = [
        rng.vonmises(0.0, 3, 40),
        rng.vonmises(0.5, 3, 40),
        rng.vonmises(1.0, 3, 40),
    ]
    as_lists = [g.tolist() for g in groups]
    as_circular = [Circular(g, unit="radian") for g in groups]

    for fn in (
        lambda s: circ_anova(s).statistic,
        lambda s: rao_homogeneity_test(s).H_polar,
        lambda s: equal_kappa_test(s).statistic,
        lambda s: common_median_test(s).statistic,
    ):
        ref = fn(groups)
        np.testing.assert_allclose(fn(as_lists), ref, rtol=1e-12)
        np.testing.assert_allclose(fn(as_circular), ref, rtol=1e-12)


def test_watson_u2_test_unsorted_input():
    """watson_u2_test must not depend on the input ordering of the angles."""
    a = np.array([0.1, 2.0, 1.0, 3.0, 0.5])
    b = np.array([2.5, 0.2, 1.5, 2.8, 0.8])
    unsorted = watson_u2_test([a, b]).U2
    ordered = watson_u2_test([np.sort(a), np.sort(b)]).U2
    np.testing.assert_allclose(unsorted, ordered, rtol=1e-12)


def test_wallraff_test_grouped_matches_expanded():
    """Weighted/grouped samples must rank-split by total weight, matching the
    equivalent weight-expanded plain arrays."""
    ang = np.deg2rad(np.array([10.0, 30.0, 50.0, 70.0, 90.0]))
    w1 = np.array([2, 3, 1, 1, 1])
    w2 = np.array([1, 1, 2, 2, 1])
    grouped = wallraff_test(
        samples=[Circular(ang, w=w1, unit="radian"), Circular(ang, w=w2, unit="radian")],
        angle=0.0,
    ).U
    expanded = wallraff_test(
        samples=[np.repeat(ang, w1), np.repeat(ang, w2)], angle=0.0
    ).U
    np.testing.assert_allclose(grouped, expanded, rtol=1e-12)


def test_equal_kappa_regimes():
    """equal_kappa_test routes through and labels each r̄-regime correctly
    (small/arcsin, moderate/asinh, large/Bartlett)."""
    for kappa, expected in [(0.6, "small"), (1.5, "moderate"), (4.0, "large")]:
        rng = np.random.default_rng(7)
        groups = [rng.vonmises(0.0, kappa, 60) for _ in range(3)]
        result = equal_kappa_test(groups)
        assert result.regime == expected, (
            f"kappa={kappa}: expected regime {expected!r}, got {result.regime!r}"
        )
        assert result.df == 2
        assert 0.0 <= result.pval <= 1.0


def test_equal_kappa_detects_difference():
    """Groups with clearly different concentrations are rejected."""
    rng = np.random.default_rng(5)
    groups = [rng.vonmises(0.0, k, 60) for k in (8, 8, 1)]
    result = equal_kappa_test(groups)
    assert result.pval < 0.05, f"Expected rejection, got p={result.pval}"


def test_rao_spacing_test_grouped():
    """Grouped (weighted) mode: the observed statistic matches the equivalent
    weight-expanded sample, the mode is reported, and invalid weights raise."""
    ang = np.deg2rad(np.array([10.0, 40.0, 70.0, 100.0, 200.0, 300.0]))
    w = np.array([3, 1, 2, 1, 4, 2])

    grouped = rao_spacing_test(ang, w=w, n_resamples=999, seed=1)
    expanded = rao_spacing_test(np.repeat(ang, w), n_resamples=999, seed=1)
    assert grouped.data_kind == "grouped"
    assert grouped.method == "monte_carlo"
    np.testing.assert_allclose(grouped.statistic, expanded.statistic, rtol=1e-12)
    assert 0.0 < grouped.pval <= 1.0

    with pytest.raises(ValueError):  # negative weight
        rao_spacing_test(ang, w=np.array([1, -1, 2, 1, 1, 1]), n_resamples=99)
    with pytest.raises(ValueError):  # non-integer weight
        rao_spacing_test(ang, w=np.array([1.0, 1.5, 2.0, 1.0, 1.0, 1.0]), n_resamples=99)
    with pytest.raises(ValueError):  # shape mismatch
        rao_spacing_test(ang, w=np.array([1, 2, 3]), n_resamples=99)


def test_wheeler_watson_three_samples():
    """The k>=3 branch matches an independent uniform-scores computation and
    separates groups with different mean directions."""
    from scipy.stats import rankdata

    rng = np.random.default_rng(11)
    groups = [rng.vonmises(m, 4, 25) for m in (0.0, 0.3, 0.6)]
    result = wheeler_watson_test(groups)

    # Independent reimplementation of W = 2 * Σ_g (C_g² + S_g²) / n_g.
    pooled = np.concatenate(groups)
    N = pooled.size
    beta = 2 * np.pi * rankdata(pooled, method="ordinal") / N
    W_ref, idx = 0.0, 0
    for grp in groups:
        b = beta[idx:idx + grp.size]
        idx += grp.size
        W_ref += (np.sum(np.cos(b)) ** 2 + np.sum(np.sin(b)) ** 2) / grp.size
    W_ref *= 2.0

    np.testing.assert_allclose(result.W, W_ref, rtol=1e-12)
    assert result.df == 2 * (len(groups) - 1)

    rng = np.random.default_rng(9)
    separated = [rng.vonmises(m, 6, 30) for m in (0.0, 1.6, 3.1)]
    assert wheeler_watson_test(separated).pval < 0.05


def test_kuiper_test_asymptotic():
    """Asymptotic mode (n_resamples=0) returns a valid p-value close to the
    Monte-Carlo one."""
    d = load_data("B5", source="fisher")["θ"].values[:]
    c = Circular(data=d, unit="degree", full_cycle=180)
    asymp = kuiper_test(alpha=c.alpha, n_resamples=0)
    sim = kuiper_test(alpha=c.alpha, n_resamples=9999)
    assert asymp.method == "asymptotic"
    assert asymp.n_resamples == 0
    assert sim.method == "monte_carlo"
    assert 0.0 <= asymp.pval <= 1.0
    assert abs(asymp.pval - sim.pval) < 0.05


def test_watson_test_asymptotic():
    """Asymptotic mode (n_resamples=0) returns a valid p-value close to the
    Monte-Carlo one."""
    pigeon = np.array([20, 135, 145, 165, 170, 200, 300, 325, 335, 350, 350, 350, 355])
    c = Circular(data=pigeon)
    asymp = watson_test(alpha=c.alpha, n_resamples=0)
    sim = watson_test(alpha=c.alpha, n_resamples=9999)
    assert asymp.method == "asymptotic"
    assert asymp.n_resamples == 0
    assert sim.method == "monte_carlo"
    assert 0.0 <= asymp.pval <= 1.0
    assert abs(asymp.pval - sim.pval) < 0.05


def test_harrison_kanji_inter_false():
    """inter=False on high-concentration data exercises the large-kappa,
    no-interaction branch and suppresses the interaction term."""
    rng = np.random.default_rng(3)
    alpha = rng.vonmises(0, 6, 60)  # high kappa -> kk > 2 (large-kappa branch)
    idp = rng.choice([1, 2, 3], 60)
    idq = rng.choice([1, 2], 60)

    result = harrison_kanji_test(alpha, idp, idq, inter=False)

    p_a, p_b, p_inter = result.p_values
    assert np.isnan(p_inter)  # interaction term dropped
    assert 0.0 <= p_a <= 1.0 and 0.0 <= p_b <= 1.0

    table = result.anova_table
    assert list(table.index) == ["A", "B", "Interaction", "Residual", "Total"]
    p, q = len(np.unique(idp)), len(np.unique(idq))
    assert table.loc["Residual", "DoF"] == (p - 1) * (q - 1)


def test_one_sample_test_rejects_distant_angle():
    """An angle far from the mean direction lies outside the 95% CI -> reject."""
    rng = np.random.default_rng(0)
    alpha = rng.vonmises(0.0, 20, 30)  # tightly concentrated near 0
    assert one_sample_test(angle=0.0, alpha=alpha).reject is False
    assert one_sample_test(angle=np.pi, alpha=alpha).reject is True


def test_watson_williams_warns_low_concentration():
    """Watson-Williams warns when the common concentration is low (κ < 1)."""
    rng = np.random.default_rng(2)
    s1 = rng.uniform(0, 2 * np.pi, 40)
    s2 = rng.uniform(0, 2 * np.pi, 40)
    with pytest.warns(RuntimeWarning):
        watson_williams_test([s1, s2])


def test_circ_anova_no_correction():
    """The f_mod=False branch (no Stephens correction factor) runs."""
    rng = np.random.default_rng(1)
    groups = [rng.vonmises(m, 5, 40) for m in (0.0, 0.4, 0.8)]
    plain = circ_anova(groups, f_mod=False)
    corrected = circ_anova(groups, f_mod=True)
    assert 0.0 <= plain.pval <= 1.0
    # The correction factor (1 + 3/8κ) > 1 inflates the F statistic.
    assert corrected.statistic > plain.statistic


def test_symmetry_test_default_median():
    """When no median is supplied, symmetry_test computes it internally."""
    from pycircstat2.descriptive import circ_median

    data_zar_ex6_ch27 = load_data("D9", source="zar")
    alpha = Circular(data=data_zar_ex6_ch27["θ"].values[:], unit="degree").alpha
    auto = symmetry_test(alpha)
    explicit = symmetry_test(alpha, median=float(circ_median(alpha)))
    np.testing.assert_allclose(auto.statistic, explicit.statistic, rtol=1e-9)
    np.testing.assert_allclose(auto.pval, explicit.pval, rtol=1e-9)


def test_weighted_input_paths():
    """The alpha+w paths (n/mean/r inferred from weighted angles) match the
    weight-expanded sample for the single-sample tests that accept `w`."""
    a = np.array([0.1, 0.2, 0.3, 1.0, 1.1])
    w = np.array([2, 2, 2, 2, 2])  # total weight 10 (one_sample CI needs n >= 8)
    exp = np.repeat(a, w)

    np.testing.assert_allclose(
        rayleigh_test(alpha=a, w=w).z, rayleigh_test(alpha=exp).z, rtol=1e-12
    )
    np.testing.assert_allclose(
        V_test(angle=0.0, alpha=a, w=w).V, V_test(angle=0.0, alpha=exp).V, rtol=1e-12
    )
    weighted_ci = one_sample_test(angle=0.0, alpha=a, w=w)
    expanded_ci = one_sample_test(angle=0.0, alpha=exp)
    np.testing.assert_allclose(weighted_ci.ci, expanded_ci.ci, rtol=1e-12)
    assert weighted_ci.reject == expanded_ci.reject


def test_verbose_branches_smoke(capsys):
    """verbose=True output paths run without error across representative tests
    (including the reject / bootstrap / NaN-median display branches)."""
    rng = np.random.default_rng(0)
    alpha = rng.vonmises(0.0, 4, 30)

    rayleigh_test(alpha=alpha, n_resamples=50, verbose=True)  # monte-carlo print
    one_sample_test(angle=0.0, alpha=alpha, verbose=True)     # reject=False branch
    one_sample_test(angle=np.pi, alpha=alpha, verbose=True)   # reject=True branch
    circ_range_test(alpha, verbose=True)
    harrison_kanji_test(
        alpha, rng.choice([1, 2, 3], 30), rng.choice([1, 2], 30), verbose=True
    )
    common_median_test(
        [rng.vonmises(0, 3, 20), rng.vonmises(0.2, 3, 20)], verbose=True
    )
    common_median_test(  # rejection -> NaN-median display branch
        [rng.vonmises(0, 3, 20), rng.vonmises(3.0, 3, 20)], verbose=True
    )

    assert capsys.readouterr().out  # something was printed


def test_result_helpers():
    """TestResult.asdict() and .significance() behave correctly."""
    rng = np.random.default_rng(0)
    res = rayleigh_test(alpha=rng.vonmises(0.0, 8, 40))  # strongly non-uniform
    assert res.asdict() == {
        "r": res.r,
        "z": res.z,
        "pval": res.pval,
        "method": res.method,
        "n_resamples": res.n_resamples,
    }
    assert res.significance() == "***"
    assert res.significance("does_not_exist") is None
    # a perfectly uniform sample is not significant -> empty stars
    uniform = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    assert rayleigh_test(alpha=uniform).significance() == ""


def test_legacy_positional_verbose_warns():
    """Passing seed=True (the formerly positional `verbose`) is deprecated but
    still honored via the back-compat shim."""
    alpha = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    with pytest.warns(DeprecationWarning):
        rayleigh_test(alpha=alpha, seed=True)


def test_deprecated_resampling_aliases():
    """Old `B` / `n_simulation` kwargs and old result attributes still work,
    with a DeprecationWarning, after the n_resamples/method harmonization."""
    rng = np.random.default_rng(0)
    alpha = rng.uniform(0, 2 * np.pi, 30)

    # Param alias maps onto n_resamples (same seed/count => identical result).
    with pytest.warns(DeprecationWarning):
        old = rayleigh_test(alpha=alpha, B=200, seed=1)
    new = rayleigh_test(alpha=alpha, n_resamples=200, seed=1)
    assert old.pval == new.pval
    assert old.method == "monte_carlo"

    # Old sentinel `1` meant "no resampling" for tests with an analytic fallback.
    with pytest.warns(DeprecationWarning):
        assert rayleigh_test(alpha=alpha, B=1).method == "asymptotic"
    with pytest.warns(DeprecationWarning):
        assert kuiper_test(alpha=alpha, n_simulation=1).method == "asymptotic"
    with pytest.warns(DeprecationWarning):
        assert kuiper_test(alpha=alpha, n_simulation=500, seed=1).method == "monte_carlo"

    # Deprecated result attributes proxy the new fields.
    mc = rayleigh_test(alpha=alpha, n_resamples=200, seed=1)
    with pytest.warns(DeprecationWarning):
        assert mc.bootstrap_pval == mc.pval
    kup = kuiper_test(alpha=alpha, n_resamples=0)
    with pytest.warns(DeprecationWarning):
        assert kup.mode == "asymptotic"
    with pytest.warns(DeprecationWarning):
        assert kup.n_simulation == 0
    rao = rao_spacing_test(alpha, n_resamples=200, seed=1)
    with pytest.warns(DeprecationWarning):
        assert rao.mode == rao.data_kind  # "ungrouped"


def test_common_median_randomization():
    """common_median_test randomization reproduces the book's ant-data result and
    the χ² asymptotic value (Pewsey et al. 2013, §7.3.2; data = B10)."""
    df = load_data("B10", source="fisher")  # desert-ant directions, 3 groups
    groups = [np.deg2rad(df[df["set"] == s]["θ"].values.astype(float)) for s in (1, 2, 3)]

    asy = common_median_test(groups)
    assert asy.method == "asymptotic"
    np.testing.assert_allclose(asy.pval, 0.4293, atol=2e-3)  # book χ² p-value

    rnd = common_median_test(groups, n_resamples=9999, seed=1)
    assert rnd.method == "randomization" and rnd.n_resamples == 9999
    assert 0.40 < rnd.pval < 0.44  # book randomization p ≈ 0.4195, CI (0.410, 0.429)

    # determinism: int seed == equivalent Generator
    p_int = common_median_test(groups, n_resamples=500, seed=7).pval
    p_gen = common_median_test(groups, n_resamples=500, seed=np.random.default_rng(7)).pval
    assert p_int == p_gen


def test_watson_u2_randomization():
    """watson_u2_test randomization reproduces the book's ant-data result
    (control vs 2nd treatment; Pewsey et al. 2013, §7.5.5; data = B10)."""
    df = load_data("B10", source="fisher")
    s1 = np.deg2rad(df[df["set"] == 1]["θ"].values.astype(float))
    s3 = np.deg2rad(df[df["set"] == 3]["θ"].values.astype(float))

    rnd = watson_u2_test([s1, s3], n_resamples=9999, seed=1)
    assert rnd.method == "randomization" and rnd.n_resamples == 9999
    np.testing.assert_allclose(rnd.U2, 0.1944, atol=1e-3)  # book statistic
    assert 0.03 < rnd.pval < 0.05  # book randomization p ≈ 0.0386, CI (0.035, 0.042)

    assert watson_u2_test([s1, s3]).method == "asymptotic"  # default unchanged
    p_int = watson_u2_test([s1, s3], n_resamples=500, seed=3).pval
    p_gen = watson_u2_test([s1, s3], n_resamples=500, seed=np.random.default_rng(3)).pval
    assert p_int == p_gen


def test_wheeler_watson_randomization_with_ties():
    """wheeler_watson_test now handles tied data via midranks, and its randomization
    p-value tracks the χ² approximation (Pewsey et al. 2013, §7.5.3; data = B10)."""
    df = load_data("B10", source="fisher")  # ant data, 3 groups, contains ties
    groups = [np.deg2rad(df[df["set"] == s]["θ"].values.astype(float)) for s in (1, 2, 3)]

    asy = wheeler_watson_test(groups)  # previously crashed on ties
    rnd = wheeler_watson_test(groups, n_resamples=9999, seed=1)
    assert rnd.method == "randomization" and rnd.n_resamples == 9999
    assert 0.10 < rnd.pval < 0.17  # book ≈ 0.1407; χ² approximation ≈ 0.13
    assert abs(rnd.pval - asy.pval) < 0.03  # randomization tracks the approximation

    p_int = wheeler_watson_test(groups, n_resamples=500, seed=4).pval
    p_gen = wheeler_watson_test(groups, n_resamples=500, seed=np.random.default_rng(4)).pval
    assert p_int == p_gen

    rng = np.random.default_rng(0)
    sep = [rng.vonmises(m, 6, 25) for m in (0.0, 1.8, 3.4)]  # separated -> reject
    assert wheeler_watson_test(sep, n_resamples=2000, seed=1).pval < 0.05


def test_concentration_randomization():
    """concentration_test randomization is distribution-free: it rejects clearly
    different concentrations and not equal ones (Pewsey et al. 2013, §7.4.3)."""
    rng = np.random.default_rng(42)
    same1 = vonmises.rvs(mu=0, kappa=5, size=60, random_state=rng)
    same2 = vonmises.rvs(mu=0, kappa=5, size=60, random_state=rng)
    diff1 = vonmises.rvs(mu=0, kappa=8, size=60, random_state=rng)
    diff2 = vonmises.rvs(mu=0, kappa=1.5, size=60, random_state=rng)

    assert concentration_test(same1, same2).method == "asymptotic"  # default = F-test

    eq = concentration_test(same1, same2, n_resamples=9999, seed=1)
    ne = concentration_test(diff1, diff2, n_resamples=9999, seed=1)
    assert eq.method == "randomization" and eq.n_resamples == 9999
    assert eq.pval > 0.05
    assert ne.pval < 0.05

    p_int = concentration_test(diff1, diff2, n_resamples=500, seed=3).pval
    p_gen = concentration_test(diff1, diff2, n_resamples=500, seed=np.random.default_rng(3)).pval
    assert p_int == p_gen

    # randomization on dispersed data must not emit the rbar<0.7 warning
    u1 = rng.uniform(0, 2 * np.pi, 50)
    u2 = rng.uniform(0, 2 * np.pi, 50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        concentration_test(u1, u2, n_resamples=200, seed=1)


def test_symmetry_test_pewsey():
    """method='pewsey' = Pewsey's (2002) β̄₂ reflective-symmetry test; large-sample
    matches R's `circular` package and the bootstrap reproduces the book's cross-bed
    azimuth result (Pewsey et al. 2013, §5.2; data = B6/set1)."""
    b6 = load_data("B6", source="fisher")
    s1 = np.deg2rad(b6[b6["set"] == 1]["θ"].values.astype(float))

    large = symmetry_test(s1, method="pewsey")
    assert large.method == "pewsey"
    np.testing.assert_allclose(large.statistic, 0.594601, rtol=1e-4)  # |z|, == R RSTestStat
    np.testing.assert_allclose(large.pval, 0.552110, rtol=1e-4)

    boot = symmetry_test(s1, method="pewsey", n_resamples=9999, seed=2046)
    assert boot.method == "pewsey" and boot.n_resamples == 9999
    assert 0.52 < boot.pval < 0.56  # book 0.5391, 95% CI (0.529, 0.549)

    assert symmetry_test(s1).method == "wilcoxon"  # default unchanged
    with pytest.raises(ValueError, match="method"):
        symmetry_test(s1, method="bogus")

    p1 = symmetry_test(s1, method="pewsey", n_resamples=500, seed=3).pval
    p2 = symmetry_test(s1, method="pewsey", n_resamples=500, seed=np.random.default_rng(3)).pval
    assert p1 == p2


def test_one_sample_specified_mean():
    """one_sample_test adds the §5.3.3 specified-mean p-value (eq. 5.10) when raw
    angles are supplied; validated bit-for-bit against R's `circular` package."""
    from pycircstat2.utils import time2float

    # B1 intensive-care times with the proper hh:mm -> decimal-hour conversion (== R's
    # fisherB1c). NB: the book's published 9.126e-5 used raw fisherB1 (8.45-as-decimal).
    b1 = time2float(load_data("B1", source="fisher")["time"].values) * 2 * np.pi / 24

    r = one_sample_test(angle=3.9270, alpha=b1, symmetric=True)  # H0: mean = 15:00
    assert r.method == "asymptotic"
    np.testing.assert_allclose(r.statistic, 4.217775, rtol=1e-4)  # == R SpecMeanTestRes
    np.testing.assert_allclose(r.pval, 2.467243e-05, rtol=1e-3)
    assert r.pval < 1e-3  # 3pm emphatically rejected (book's conclusion)

    # Backward-compat: CI-only path has no p-value; `reject` is the CI decision.
    ci_only = one_sample_test(lb=0.0, ub=1.0, angle=0.5)
    assert ci_only.reject is False
    assert ci_only.pval is None and ci_only.method is None

    # bootstrap determinism
    p_int = one_sample_test(angle=3.9270, alpha=b1, symmetric=True, n_resamples=500, seed=4).pval
    p_gen = one_sample_test(
        angle=3.9270, alpha=b1, symmetric=True, n_resamples=500, seed=np.random.default_rng(4)
    ).pval
    assert p_int == p_gen


def test_circ_anova_randomization():
    """circ_anova(n_resamples>0) gives a label-randomization p-value (free of the
    high-κ assumption) that tracks the parametric one and rejects separated means."""
    rng = np.random.default_rng(42)
    same = [rng.vonmises(0.0, 5, 40) for _ in range(3)]
    diff = [rng.vonmises(m, 5, 40) for m in (0.0, 0.5, 1.0)]

    for method in ("F-test", "LRT"):
        a = circ_anova(same, method=method)
        r = circ_anova(same, method=method, n_resamples=4999, seed=1)
        assert r.n_resamples == 4999
        assert abs(r.pval - a.pval) < 0.05  # tracks the parametric p-value under H0
        assert circ_anova(diff, method=method, n_resamples=4999, seed=1).pval < 0.05

    # default unchanged (parametric); determinism of the randomization
    assert circ_anova(same).n_resamples == 0
    p_int = circ_anova(diff, n_resamples=500, seed=7).pval
    p_gen = circ_anova(diff, n_resamples=500, seed=np.random.default_rng(7)).pval
    assert p_int == p_gen


def test_mc_uniform_pvalues():
    """V_test / omnibus_test / circ_range_test gain a Monte-Carlo-under-uniform p-value
    that tracks the analytic one (the analytic Rayleigh/Ajne forms are approximations)."""
    rng = np.random.default_rng(0)
    a = rng.vonmises(np.deg2rad(80), 1.0, 30)

    v_a = V_test(angle=np.deg2rad(90), alpha=a)
    v_m = V_test(angle=np.deg2rad(90), alpha=a, n_resamples=9999, seed=1)
    assert v_a.method == "asymptotic" and v_m.method == "monte_carlo" and v_m.n_resamples == 9999
    assert abs(v_a.pval - v_m.pval) < 0.02

    d8 = Circular(data=load_data("D8", source="zar")["θ"].values[:], unit="degree")
    o_m = omnibus_test(d8.alpha, n_resamples=9999, seed=1)
    assert o_m.method == "monte_carlo" and o_m.pval < 0.05  # book/asymptotic ~0.0043
    # MC handles the degenerate (maximally uniform) case the analytic formula clamps.
    ev = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    assert omnibus_test(ev, n_resamples=2000, seed=1).pval > 0.5

    x = np.deg2rad(np.array([0.0] * 12 + [3.6, 36, 36, 36, 36, 36, 36, 72, 108, 108, 169.2, 324.0]))
    assert circ_range_test(x).method == "exact"
    assert circ_range_test(x, n_resamples=9999, seed=1).pval < 0.05  # clustered -> reject

    # determinism
    p_int = omnibus_test(d8.alpha, n_resamples=500, seed=3).pval
    p_gen = omnibus_test(d8.alpha, n_resamples=500, seed=np.random.default_rng(3)).pval
    assert p_int == p_gen


def test_change_point_permutation_pvalues():
    """change_point_test(n_resamples>0) adds permutation p-values: rmax flags a
    mean-direction change, tmax a concentration change; homogeneous data is not flagged."""
    rng = np.random.default_rng(0)
    mean_change = np.concatenate([rng.vonmises(0, 5, 15), rng.vonmises(np.pi, 5, 15)])
    conc_change = np.concatenate([rng.vonmises(0, 12, 15), rng.vonmises(0, 0.4, 15)])
    homog = rng.vonmises(0, 5, 30)

    mc = change_point_test(mean_change, n_resamples=1999, seed=1)
    assert mc.n_resamples == 1999
    assert mc.pval_r < 0.05  # mean direction changed

    cc = change_point_test(conc_change, n_resamples=1999, seed=1)
    assert cc.pval_t < 0.05  # concentration changed (likelihood statistic)

    hm = change_point_test(homog, n_resamples=1999, seed=1)
    assert hm.pval_r > 0.05 and hm.pval_t > 0.05

    # default = statistics only, no p-values; determinism
    assert change_point_test(homog).pval_r is None
    a = change_point_test(mean_change, n_resamples=300, seed=7)
    b = change_point_test(mean_change, n_resamples=300, seed=np.random.default_rng(7))
    assert a.pval_r == b.pval_r and a.pval_t == b.pval_t
