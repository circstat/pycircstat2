import numpy as np
import pytest

from pycircstat2.distributions import (
    cardioid,
    cartwright,
    circularuniform,
    triangular,
    inverse_batschelet,
    jonespewsey,
    jonespewsey_asym,
    jonespewsey_sineskewed,
    katojones,
    vonmises,
    vonmises_flattopped,
    wrapcauchy,
    wrapnorm,
)


def test_circularuniform():

    np.testing.assert_approx_equal(circularuniform.cdf(2), 0.3183, significant=5)
    np.testing.assert_approx_equal(circularuniform.ppf(1 / np.pi), 2)


def test_cardioid():

    cd = cardioid(rho=0.3, mu=np.pi / 2)
    np.testing.assert_approx_equal(cd.cdf(np.pi), 0.6909859, significant=5)
    np.testing.assert_approx_equal(cd.ppf(0.6909859), np.pi)


def test_cartwright():

    cw = cartwright(zeta=0.1, mu=np.pi / 2)
    np.testing.assert_approx_equal(
        cw.cdf(3 * np.pi / 4),
        0.9641666531258773,
        significant=5,
    )
    np.testing.assert_approx_equal(
        cw.ppf(0.9641666531258773).round(5),
        3 * np.pi / 4,
        significant=5,
    )


def test_triangular_ppf_vectorized():
    q = np.linspace(0.1, 0.9, num=5)
    out_zero = triangular.ppf(q, rho=0.0)
    np.testing.assert_allclose(out_zero, q * (2 * np.pi))


def test_triangular_pdf_periodic():
    rho = 0.3
    x_neg = -np.pi / 4
    x_mod = np.mod(x_neg, 2 * np.pi)
    np.testing.assert_allclose(
        triangular.pdf(x_neg, rho=rho),
        triangular.pdf(x_mod, rho=rho),
        atol=1e-12,
    )


def test_vonmises_periodic_evaluation():
    mu = np.pi / 3
    kappa = 1.75
    x_neg = -np.pi / 5
    x_mod = np.mod(x_neg, 2 * np.pi)

    np.testing.assert_allclose(
        vonmises.pdf(x_neg, mu=mu, kappa=kappa),
        vonmises.pdf(x_mod, mu=mu, kappa=kappa),
        atol=1e-12,
    )

    vm = vonmises(kappa=kappa, mu=mu)
    np.testing.assert_allclose(vm.pdf(x_neg), vm.pdf(x_mod), atol=1e-12)


def test_vonmises_fit_wraps_data():
    data = np.array([-0.8, 0.2, 6.6, 7.1, -3.0])

    mu_expected, kappa_expected = vonmises.fit(
        np.mod(data, 2 * np.pi), method="analytical"
    )
    mu_actual, kappa_actual = vonmises.fit(data, method="analytical")

    diff = np.mod(mu_actual - mu_expected + np.pi, 2 * np.pi) - np.pi
    np.testing.assert_allclose(diff, 0.0, atol=1e-8)
    np.testing.assert_allclose(kappa_actual, kappa_expected, atol=1e-8)


def test_wrapcauchy():

    wc = wrapcauchy(rho=0.75, mu=np.pi / 2)
    # P54, Pewsey, et al. (2014)
    np.testing.assert_approx_equal(
        wc.cdf(np.pi / 6).round(3),
        0.0320,
        significant=3,
    )

    # P54, Pewsey, et al. (2014)
    np.testing.assert_approx_equal(
        wc.ppf(0.0320),
        np.pi / 6,
        significant=3,
    )


def test_wrapnorm():

    wn = wrapnorm(rho=0.75, mu=np.pi / 2)
    np.testing.assert_approx_equal(
        wn.cdf(np.pi / 6).round(4),
        0.0645,
        significant=3,
    )
    np.testing.assert_approx_equal(
        wn.ppf(0.5),
        1.6073,
        significant=4,
    )


def test_wrapnorm_cdf_matches_numeric():
    mu, rho = 0.7, 0.45
    theta = np.linspace(0.0, 2.0 * np.pi, 7)
    analytic = wrapnorm.cdf(theta, mu, rho)
    numeric = wrapnorm._cdf_from_pdf(theta, mu, rho)
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)
    diffs = np.diff(analytic)
    assert np.all(diffs >= -1e-10)


def test_circular_loc_scale_rejected():
    rng = np.random.default_rng(1234)
    sample = vonmises.rvs(kappa=1.0, mu=0.0, size=8, random_state=rng)

    with pytest.raises(TypeError):
        vonmises.pdf(0.5, mu=0.0, kappa=1.0, loc=0.1)

    with pytest.raises(TypeError):
        vonmises.cdf(0.5, mu=0.0, kappa=1.0, scale=1.1)

    with pytest.raises(TypeError):
        vonmises.fit(sample, loc=0.2)

    with pytest.raises(TypeError):
        vonmises.fit(sample, scale=1.2)

    with pytest.raises(TypeError):
        vonmises.fit(sample, floc=0.1)

    with pytest.raises(TypeError):
        vonmises.fit(sample, fscale=0.9)


def test_vonmises():

    vm = vonmises(kappa=2.37, mu=np.pi / 2)
    np.testing.assert_approx_equal(
        vm.cdf(np.pi / 6).round(4),
        0.0543,
        significant=3,
    )
    np.testing.assert_approx_equal(
        vm.ppf(0.5),
        1.6139,
        significant=4,
    )


def test_jonespewsey():

    jp = jonespewsey(kappa=2, psi=-1.5, mu=np.pi / 2)

    np.testing.assert_approx_equal(
        jp.cdf(np.pi / 2).round(7),
        0.4401445,
        significant=7,
    )
    # take a long time to run jonespewsey.ppf()
    # might need to implement the method explicitly
    np.testing.assert_approx_equal(
        jp.ppf(q=0.4401445),
        np.pi / 2,
        significant=7,
    )


def test_vonmises_flattopped():

    vme = vonmises_flattopped(kappa=2, nu=-0.5, mu=np.pi / 2)
    np.testing.assert_approx_equal(
        vme.cdf(x=3 * np.pi / 4).round(4),
        0.7120,
        significant=4,
    )
    np.testing.assert_approx_equal(
        vme.ppf(q=0.5).round(4),
        1.7301,
        significant=4,
    )


def test_jonespewsey_sineskewed():

    jps = jonespewsey_sineskewed(kappa=2, psi=1, lmbd=0.5, xi=np.pi / 2)

    np.testing.assert_approx_equal(
        jps.cdf(x=3 * np.pi / 2).round(4),
        0.9446,
        significant=4,
    )
    np.testing.assert_approx_equal(
        jps.ppf(q=0.5).round(4),
        2.1879,
        significant=4,
    )


def test_jonespewsey_asym():

    jpa = jonespewsey_asym(kappa=2, psi=-1, nu=0.75, xi=np.pi / 2)
    np.testing.assert_approx_equal(
        jpa.cdf(x=np.pi / 2).round(4),
        0.7535,
        significant=4,
    )
    np.testing.assert_approx_equal(
        jpa.ppf(q=0.5).round(4),
        1.0499,
        significant=4,
    )


def test_inverse_batschelet():

    ib = inverse_batschelet(kappa=2, nu=-0.5, lmbd=0.7, xi=np.pi / 2)
    np.testing.assert_approx_equal(
        ib.cdf(x=np.pi / 2).round(4),
        0.1180,
        significant=4,
    )
    np.testing.assert_approx_equal(
        ib.ppf(q=0.5).round(4),
        2.5138,
        significant=4,
    )


def _angle_diff(a, b):
    return np.mod(a - b + np.pi, 2 * np.pi) - np.pi


def test_katojones_cardioid_limit():
    theta = np.linspace(0.0, 2.0 * np.pi, 9)
    mu = 1.1
    gamma = 0.3
    kj_vals = katojones.pdf(theta, mu, gamma, 0.0, 0.0)
    card_vals = cardioid.pdf(theta, mu, gamma)
    np.testing.assert_allclose(kj_vals, card_vals, rtol=1e-10, atol=1e-12)


def test_katojones_convert_alpha2_beta2():
    gamma = 0.4
    rho = 0.35
    lam = 1.25
    alpha2, beta2 = katojones.convert_rho_lambda(gamma, rho, lam)
    rho_rt, lam_rt = katojones.convert_alpha2_beta2(gamma, alpha2, beta2)
    np.testing.assert_allclose(rho_rt, rho, atol=1e-12)
    np.testing.assert_allclose(_angle_diff(lam_rt, lam), 0.0, atol=1e-12)

    with pytest.raises(ValueError):
        katojones.convert_alpha2_beta2(gamma, alpha2 + 0.5, beta2, verify=True)


def test_katojones_fit_methods_agree():
    rng = np.random.default_rng(321)
    mu, gamma, rho, lam = 0.9, 0.35, 0.25, 1.8
    data = katojones.rvs(mu, gamma, rho, lam, size=400, random_state=rng)

    mu_mom, gamma_mom, rho_mom, lam_mom = katojones.fit(data, method="moments")
    np.testing.assert_allclose(_angle_diff(mu_mom, mu), 0.0, atol=0.2)
    np.testing.assert_allclose(gamma_mom, gamma, atol=0.05)
    np.testing.assert_allclose(rho_mom, rho, atol=0.1)
    np.testing.assert_allclose(_angle_diff(lam_mom, lam), 0.0, atol=0.25)

    mu_mle, gamma_mle, rho_mle, lam_mle = katojones.fit(
        data, method="mle", options={"maxiter": 200}
    )
    np.testing.assert_allclose(_angle_diff(mu_mle, mu), 0.0, atol=0.15)
    np.testing.assert_allclose(gamma_mle, gamma, atol=0.05)
    np.testing.assert_allclose(rho_mle, rho, atol=0.08)
    np.testing.assert_allclose(_angle_diff(lam_mle, lam), 0.0, atol=0.2)
