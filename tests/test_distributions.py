import numpy as np
import pytest
from scipy import stats

from pycircstat2.distributions import (
    cardioid,
    cartwright,
    circularuniform,
    inverse_batschelet,
    jonespewsey,
    jonespewsey_asym,
    jonespewsey_sineskewed,
    katojones,
    triangular,
    vonmises,
    vonmises_flattopped,
    wrapcauchy,
    wrapnorm,
    wrapstable,
)


def _assert_monotonic_cdf_ppf(dist, theta_grid, q_grid, *, cdf_tol=1e-12, ppf_tol=1e-12):
    def _evaluate(func, grid):
        try:
            return np.asarray(func(grid), dtype=float)
        except (TypeError, ValueError):
            flat = np.asarray(grid, dtype=float).reshape(-1)
            evaluated = np.array([func(float(val)) for val in flat], dtype=float)
            return evaluated.reshape(np.shape(grid))

    cdf_vals = _evaluate(dist.cdf, theta_grid)
    ppf_vals = _evaluate(dist.ppf, q_grid)

    assert np.all(np.isfinite(cdf_vals)), "CDF produced non-finite values"
    assert np.all(np.isfinite(ppf_vals)), "PPF produced non-finite values"

    cdf_diffs = np.diff(cdf_vals)
    assert np.all(cdf_diffs >= -cdf_tol), "CDF must be non-decreasing"
    assert np.all((cdf_vals >= -cdf_tol) & (cdf_vals <= 1.0 + cdf_tol)), "CDF outside [0, 1]"

    ppf_diffs = np.diff(ppf_vals)
    assert np.all(ppf_diffs >= -ppf_tol), "PPF must be non-decreasing"
    two_pi = 2.0 * np.pi
    assert np.all((ppf_vals >= -ppf_tol) & (ppf_vals <= two_pi + ppf_tol)), "PPF outside [0, 2π]"


def test_circularuniform():
    np.testing.assert_approx_equal(circularuniform.cdf(2), 0.3183, significant=5)
    np.testing.assert_approx_equal(circularuniform.ppf(1 / np.pi), 2)


def test_circularuniform_cdf_ppf_roundtrip():
    cu = circularuniform()
    q = np.linspace(0.0, 1.0, num=9)
    theta = cu.ppf(q)
    np.testing.assert_array_less(-1e-12, theta)
    np.testing.assert_array_less(theta, 2.0 * np.pi + 1e-12)
    np.testing.assert_allclose(cu.cdf(theta), q, atol=1e-12)

    grid = np.linspace(0.0, 2.0 * np.pi, num=11)
    q_grid = cu.cdf(grid)
    theta_back = cu.ppf(q_grid)
    wrapped = np.mod(theta_back - grid + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(wrapped, 0.0, atol=1e-12)


def test_cardioid():
    cd = cardioid(rho=0.3, mu=np.pi / 2)
    np.testing.assert_approx_equal(cd.cdf(np.pi), 0.6909859, significant=5)
    np.testing.assert_approx_equal(cd.ppf(0.6909859), np.pi)


@pytest.mark.parametrize("rho", [0.0, 0.2, 0.49])
@pytest.mark.parametrize("mu", [0.0, np.pi / 3])
def test_cardioid_cdf_ppf_roundtrip(mu, rho):
    cd = cardioid(rho=rho, mu=mu)
    q = np.linspace(0.0, 1.0, num=9)
    x = cd.ppf(q)
    np.testing.assert_array_less(-1e-12, x)
    np.testing.assert_array_less(x, 2.0 * np.pi + 1e-10)

    q_back = cd.cdf(x)
    np.testing.assert_allclose(q_back, q, atol=5e-11)

    theta = np.linspace(0.0, 2.0 * np.pi, num=7)
    q_theta = cd.cdf(theta)
    theta_back = cd.ppf(q_theta)
    wrapped_diff = np.mod(theta_back - theta + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(wrapped_diff, 0.0, atol=1e-9)

@pytest.mark.parametrize("rho", [0.0, 0.3])
@pytest.mark.parametrize("mu", [0.0, np.pi / 3])
def test_cardioid_rvs_matches_ppf(mu, rho):
    rng_samples = np.random.default_rng(321)
    rng_replay = np.random.default_rng(321)
    cd = cardioid(mu=mu, rho=rho)
    samples = cd.rvs(size=512, random_state=rng_samples)
    if np.isscalar(samples):
        samples = np.array([samples])
    u = rng_replay.random(samples.size)
    expected = cd.ppf(u)
    np.testing.assert_allclose(np.sort(samples), np.sort(expected), atol=5e-3, rtol=0.0)


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


@pytest.mark.parametrize("mu", [0.0, np.pi / 3])
@pytest.mark.parametrize("zeta", [0.2, 1.5])
def test_cartwright_cdf_ppf_roundtrip(mu, zeta):
    cw = cartwright(zeta=zeta, mu=mu)
    q = np.linspace(0.0, 1.0, num=9)
    theta = cw.ppf(q)
    np.testing.assert_array_less(-1e-12, theta)
    np.testing.assert_array_less(theta, 2.0 * np.pi + 1e-12)
    np.testing.assert_allclose(cw.cdf(theta), q, atol=5e-12)

    grid = np.linspace(0.0, 2.0 * np.pi, num=9)
    q_grid = cw.cdf(grid)
    theta_back = cw.ppf(q_grid)
    wrapped = np.mod(theta_back - grid + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(wrapped, 0.0, atol=5e-8)


@pytest.mark.parametrize("mu", [0.0, np.pi / 3])
@pytest.mark.parametrize("zeta", [0.2, 1.0])
def test_cartwright_rvs_matches_ppf(mu, zeta):
    rng_samples = np.random.default_rng(456)
    rng_replay = np.random.default_rng(456)
    cw = cartwright(mu=mu, zeta=zeta)
    samples = cw.rvs(size=512, random_state=rng_samples)
    if np.isscalar(samples):
        samples = np.array([samples])
    beta_b = 1.0 / zeta + 0.5
    t = rng_replay.beta(0.5, beta_b, size=samples.size)
    sqrt_t = np.sqrt(t)
    angles = 2.0 * np.arcsin(np.clip(sqrt_t, 0.0, 1.0))
    signs = np.where(rng_replay.random(size=samples.size) < 0.5, -1.0, 1.0)
    expected = np.mod(mu + signs * angles, 2.0 * np.pi)
    np.testing.assert_allclose(np.sort(samples), np.sort(expected), atol=1e-10, rtol=0.0)


def test_triangular_ppf_vectorized():
    q = np.linspace(0.1, 0.9, num=5)
    out_zero = triangular.ppf(q, rho=0.0)
    np.testing.assert_allclose(out_zero, q * (2 * np.pi))

@pytest.mark.parametrize("rho", [0.0, 0.3])
def test_triangular_rvs_matches_ppf(rho):
    rng_samples = np.random.default_rng(123)
    rng_replay = np.random.default_rng(123)
    samples = triangular.rvs(rho=rho, size=512, random_state=rng_samples)
    if np.isscalar(samples):
        samples = np.array([samples])
    u = rng_replay.random(samples.size)
    expected = triangular.ppf(u, rho)
    expected = np.mod(expected, 2.0 * np.pi)
    np.testing.assert_allclose(np.sort(samples), np.sort(expected), atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("rho", [0.0, 0.25, 4.0 / np.pi**2])
def test_triangular_cdf_ppf_roundtrip(rho):
    tri = triangular(rho=rho)

    q = np.linspace(0.0, 1.0, num=11)
    theta = tri.ppf(q)
    np.testing.assert_array_less(-1e-12, theta)
    np.testing.assert_array_less(theta, 2.0 * np.pi + 1e-12)
    np.testing.assert_allclose(tri.cdf(theta), q, rtol=0.0, atol=2e-12)

    grid = np.linspace(0.0, 2.0 * np.pi, num=9)
    q_grid = tri.cdf(grid)
    theta_back = tri.ppf(q_grid)
    wrapped = np.mod(theta_back - grid + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(wrapped, 0.0, atol=5e-8)


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


@pytest.mark.parametrize("mu", [0.0, np.pi / 5, 1.7])
@pytest.mark.parametrize("kappa", [0.0, 0.5, 5.0, 25.0])
@pytest.mark.parametrize("nu", [-0.8, 0.0, 0.7])
def test_vonmises_flattopped_cdf_ppf_roundtrip(mu, kappa, nu):
    dist = vonmises_flattopped(mu=mu, kappa=kappa, nu=nu)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, num=129)
    q_grid = np.linspace(0.0, 1.0, num=129)
    _assert_monotonic_cdf_ppf(dist, theta_grid, q_grid, cdf_tol=5e-12, ppf_tol=5e-12)

    q = np.linspace(0.0, 1.0, num=33)
    theta = dist.ppf(q)
    q_back = dist.cdf(theta)
    np.testing.assert_allclose(q_back, q, atol=5e-12, rtol=0.0)


def test_vonmises_flattopped_uniform_limit():
    mu = 1.5
    kappa = 0.0
    nu = 0.3
    dist = vonmises_flattopped(mu=mu, kappa=kappa, nu=nu)

    theta = np.linspace(0.0, 2.0 * np.pi, num=11)
    expected = theta / (2.0 * np.pi)
    expected[np.isclose(theta, 2.0 * np.pi)] = 1.0

    np.testing.assert_allclose(dist.pdf(theta), 1.0 / (2.0 * np.pi), atol=5e-14)
    np.testing.assert_allclose(dist.cdf(theta), expected, atol=5e-12)


def test_vonmises_flattopped_rvs_pit():
    mu = 0.8
    kappa = 7.5
    nu = -0.35
    rng = np.random.default_rng(1234)

    samples = vonmises_flattopped.rvs(mu=mu, kappa=kappa, nu=nu, size=4096, random_state=rng)
    u = vonmises_flattopped.cdf(samples, mu=mu, kappa=kappa, nu=nu)
    ks_stat, _ = stats.kstest(u, stats.uniform.cdf)
    assert ks_stat < 0.035, f"PIT KS statistic too large ({ks_stat})"


def test_vonmises_flattopped_fit_recovers_parameters():
    mu_true, kappa_true, nu_true = 1.1, 4.0, -0.25
    rng = np.random.default_rng(2024)
    sample = vonmises_flattopped.rvs(mu=mu_true, kappa=kappa_true, nu=nu_true, size=6000, random_state=rng)

    estimates, info = vonmises_flattopped.fit(sample, method="mle", return_info=True)
    assert info["converged"]

    mu_hat, kappa_hat, nu_hat = estimates
    mu_diff = np.mod(mu_hat - mu_true + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(mu_diff, 0.0, atol=5e-2)
    np.testing.assert_allclose(kappa_hat, kappa_true, atol=0.6)
    np.testing.assert_allclose(nu_hat, nu_true, atol=0.08)

    moments = vonmises_flattopped.fit(sample, method="moments")
    assert moments[2] == 0.0
    np.testing.assert_allclose(np.mod(moments[0] - mu_true + np.pi, 2.0 * np.pi) - np.pi, 0.0, atol=1e-1)


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


def test_wrapcauchy_cdf_matches_numeric():
    mu, rho = 0.9, 0.65
    theta = np.linspace(0.0, 2.0 * np.pi, 9)
    analytic = wrapcauchy.cdf(theta, mu, rho)
    numeric = wrapcauchy._cdf_from_pdf(theta, mu, rho)
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)
    diffs = np.diff(analytic)
    assert np.all(diffs >= -1e-10)


@pytest.mark.parametrize("mu", [0.0, np.pi / 3])
@pytest.mark.parametrize("rho", [0.0, 0.4, 0.95])
def test_wrapcauchy_cdf_ppf_roundtrip(mu, rho):
    wc = wrapcauchy(mu=mu, rho=rho)
    q = np.linspace(0.0, 1.0, num=9)
    theta = wc.ppf(q)
    np.testing.assert_array_less(-1e-12, theta)
    np.testing.assert_array_less(theta, 2.0 * np.pi + 1e-12)
    np.testing.assert_allclose(wc.cdf(theta), q, atol=5e-12)

    grid = np.linspace(0.0, 2.0 * np.pi, num=9)
    q_grid = wc.cdf(grid)
    theta_back = wc.ppf(q_grid)
    wrapped = np.mod(theta_back - grid + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(wrapped, 0.0, atol=5e-8)


@pytest.mark.parametrize("mu", [0.0, np.pi / 3])
@pytest.mark.parametrize("rho", [0.2, 0.8])
def test_wrapcauchy_rvs_matches_constructor(mu, rho):
    rng_samples = np.random.default_rng(654)
    rng_replay = np.random.default_rng(654)

    wc = wrapcauchy(mu=mu, rho=rho)
    samples = wc.rvs(size=512, random_state=rng_samples)
    if np.isscalar(samples):
        samples = np.array([samples])

    expected = wc.dist._rvs(mu, rho, size=512, random_state=rng_replay)
    if np.isscalar(expected):
        expected = np.array([expected])

    np.testing.assert_allclose(
        np.sort(samples),
        np.sort(expected),
        atol=1e-12,
        rtol=0.0,
    )


def test_cartwright_cdf_matches_numeric():
    mu, zeta = 1.2, 0.8
    theta = np.linspace(0.0, 2.0 * np.pi, 9)
    analytic = cartwright.cdf(theta, mu, zeta)
    numeric = cartwright._cdf_from_pdf(theta, mu, zeta)
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)
    diffs = np.diff(analytic)
    assert np.all(diffs >= -1e-10)


@pytest.mark.parametrize("mu", [0.0, np.pi / 3, np.pi])
@pytest.mark.parametrize("zeta", [0.1, 1.0, 10.0])
def test_cartwright_cdf_monotonic(mu, zeta):
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    cdf_vals = cartwright.cdf(theta, mu, zeta)
    diffs = np.diff(cdf_vals)
    assert np.all(diffs >= -1e-12), "Cartwright CDF must be non-decreasing"


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


@pytest.mark.parametrize("mu", [0.0, np.pi / 4])
@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
def test_wrapnorm_cdf_ppf_roundtrip(mu, rho):
    wn = wrapnorm(rho=rho, mu=mu)
    q = np.linspace(0.0, 1.0, num=9)
    theta = wn.ppf(q)
    np.testing.assert_array_less(-1e-12, theta)
    np.testing.assert_array_less(theta, 2.0 * np.pi + 1e-10)
    np.testing.assert_allclose(wn.cdf(theta), q, atol=5e-10)

    grid = np.linspace(0.0, 2.0 * np.pi, num=9)
    q_grid = wn.cdf(grid)
    theta_back = wn.ppf(q_grid)
    wrapped = np.mod(theta_back - grid + np.pi, 2.0 * np.pi) - np.pi
    pdf_grid = wrapnorm.pdf(grid, mu=mu, rho=rho)
    high_slope = pdf_grid > 1e-4
    if np.any(high_slope):
        np.testing.assert_allclose(wrapped[high_slope], 0.0, atol=5e-6)
    if np.any(~high_slope):
        np.testing.assert_allclose(wrapped[~high_slope], 0.0, atol=1e-2)


@pytest.mark.parametrize("mu", [0.0, np.pi / 4])
@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
def test_wrapnorm_rvs_matches_constructor(mu, rho):
    rng_samples = np.random.default_rng(789)
    rng_replay = np.random.default_rng(789)

    wn = wrapnorm(mu=mu, rho=rho)
    samples = wn.rvs(size=512, random_state=rng_samples)
    if np.isscalar(samples):
        samples = np.array([samples])

    rho_clipped = np.clip(rho, np.finfo(float).tiny, 1.0 - 1e-15)
    two_pi = 2.0 * np.pi

    if rho_clipped <= 1e-12:
        expected = rng_replay.uniform(0.0, two_pi, size=samples.size)
    else:
        sigma = float(np.sqrt(-2.0 * np.log(rho_clipped)))
        mu_mod = float(np.mod(mu, two_pi))
        if sigma < 1e-12:
            expected = np.full(samples.size, mu_mod, dtype=float)
        else:
            expected = rng_replay.normal(loc=mu_mod, scale=sigma, size=samples.size)
            expected = np.mod(expected, two_pi)

    np.testing.assert_allclose(np.sort(samples), np.sort(expected), atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("mu", [0.0, np.pi / 4, np.pi])
@pytest.mark.parametrize("rho", [0.1, 0.25, 0.5, 0.9])
def test_wrapnorm_ppf_monotonic(mu, rho):
    q = np.linspace(1e-12, 1.0 - 1e-12, 512)
    theta = wrapnorm.ppf(q, mu=mu, rho=rho)
    diffs = np.diff(theta)
    assert np.all(diffs >= -1e-10), "Wrapped normal PPF must be non-decreasing"
    assert np.all(theta >= -1e-12)
    assert np.all(theta <= 2.0 * np.pi)


_MONOTONIC_CASES = [
    {
        "id": "circularuniform",
        "dist": circularuniform(),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "triangular-rho0",
        "dist": triangular(rho=0.0),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "triangular-rho0.3",
        "dist": triangular(rho=0.3),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "cardioid-rho0.2",
        "dist": cardioid(rho=0.2, mu=0.3),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "cardioid-rho0.49",
        "dist": cardioid(rho=0.49, mu=np.pi / 2),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "cartwright-zeta0.2",
        "dist": cartwright(zeta=0.2, mu=0.1),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "cartwright-zeta1",
        "dist": cartwright(zeta=1.0, mu=np.pi),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "cartwright-zeta5",
        "dist": cartwright(zeta=5.0, mu=2.0),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "wrapnorm-rho0.1",
        "dist": wrapnorm(rho=0.1, mu=0.0),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "wrapnorm-rho0.25",
        "dist": wrapnorm(rho=0.25, mu=np.pi),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "wrapnorm-rho0.9",
        "dist": wrapnorm(rho=0.9, mu=np.pi / 4),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "wrapcauchy-rho0.2",
        "dist": wrapcauchy(rho=0.2, mu=0.5),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "wrapcauchy-rho0.95",
        "dist": wrapcauchy(rho=0.95, mu=np.pi),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "vonmises-kappa0.05",
        "dist": vonmises(kappa=0.05, mu=0.0),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "vonmises-kappa5",
        "dist": vonmises(kappa=5.0, mu=np.pi / 4),
        "theta_points": 256,
        "q_points": 256,
    },
    {
        "id": "vonmises-kappa25",
        "dist": vonmises(kappa=25.0, mu=np.pi),
        "theta_points": 256,
        "q_points": 256,
        "cdf_tol": 1e-10,
    },
    {
        "id": "vonmises-flattopped",
        "dist": vonmises_flattopped(mu=0.6, kappa=2.0, nu=0.3),
        "theta_points": 192,
        "q_points": 192,
        "cdf_tol": 1e-10,
        "ppf_tol": 1e-10,
        "q_min": 1e-6,
    },
    {
        "id": "jonespewsey",
        "dist": jonespewsey(mu=0.6, kappa=1.0, psi=0.4),
        "theta_points": 192,
        "q_points": 192,
        "cdf_tol": 1e-9,
        "ppf_tol": 1e-9,
        "q_min": 1e-6,
    },
    {
        "id": "jonespewsey-sineskewed",
        "dist": jonespewsey_sineskewed(xi=1.0, kappa=1.5, psi=0.3, lmbd=0.4),
        "theta_points": 160,
        "q_points": 160,
        "cdf_tol": 5e-9,
        "ppf_tol": 5e-9,
        "q_min": 1e-5,
    },
    {
        "id": "jonespewsey-asym",
        "dist": jonespewsey_asym(xi=0.7, kappa=1.1, psi=0.2, nu=0.4),
        "theta_points": 160,
        "q_points": 160,
        "cdf_tol": 5e-9,
        "ppf_tol": 5e-9,
        "q_min": 1e-5,
    },
    {
        "id": "inverse-batschelet",
        "dist": inverse_batschelet(xi=0.8, kappa=1.3, nu=0.3, lmbd=0.2),
        "theta_points": 160,
        "q_points": 160,
        "cdf_tol": 1e-8,
        "ppf_tol": 1e-8,
        "q_min": 1e-5,
    },
    {
        "id": "katojones",
        "dist": katojones(mu=0.8, gamma=0.3, rho=0.2, lam=0.4),
        "theta_points": 96,
        "q_points": 96,
        "cdf_tol": 1e-8,
        "ppf_tol": 1e-8,
        "q_min": 1e-5,
    },
    {
        "id": "wrapstable",
        "dist": wrapstable(delta=0.9, alpha=1.5, beta=0.2, gamma=0.4),
        "theta_points": 96,
        "q_points": 96,
        "cdf_tol": 1e-8,
        "ppf_tol": 1e-8,
        "q_min": 1e-5,
    },
]


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case["id"]) for case in _MONOTONIC_CASES],
)
def test_continuous_circular_monotonic(case):
    theta_points = int(case.get("theta_points", 256))
    q_points = int(case.get("q_points", 256))
    q_min = float(case.get("q_min", 1e-12))

    theta = np.linspace(0.0, 2.0 * np.pi, theta_points)
    q = np.linspace(q_min, 1.0 - q_min, q_points)

    _assert_monotonic_cdf_ppf(
        case["dist"],
        theta,
        q,
        cdf_tol=case.get("cdf_tol", 1e-11),
        ppf_tol=case.get("ppf_tol", 1e-11),
    )


def test_vonmises_cdf_matches_numeric():
    mu, kappa = 0.6, 3.2
    theta = np.linspace(0.0, 2.0 * np.pi, 11)
    analytic = vonmises.cdf(theta, mu, kappa)
    numeric = vonmises._cdf_from_pdf(theta, mu, kappa)
    np.testing.assert_allclose(analytic, numeric, atol=5e-7)
    diffs = np.diff(analytic)
    assert np.all(diffs >= -1e-10)


@pytest.mark.parametrize("mu", [0.0, np.pi / 4])
@pytest.mark.parametrize("kappa", [0.05, 1.0, 10.0, 50.0])
def test_vonmises_cdf_ppf_roundtrip(mu, kappa):
    vm = vonmises(mu=mu, kappa=kappa)
    q = np.linspace(0.0, 1.0, num=11)
    theta = vm.ppf(q)
    np.testing.assert_array_less(-1e-12, theta)
    np.testing.assert_array_less(theta, 2.0 * np.pi + 1e-10)
    cdf_theta = vm.cdf(theta)
    np.testing.assert_allclose(cdf_theta, q, atol=5e-12)

    grid = np.linspace(0.0, 2.0 * np.pi, num=15)
    q_grid = vm.cdf(grid)
    theta_back = vm.ppf(q_grid)
    wrapped = np.mod(theta_back - grid + np.pi, 2.0 * np.pi) - np.pi
    pdf_grid = vonmises.pdf(grid, mu=mu, kappa=kappa)
    high_slope = pdf_grid > 1e-4
    if np.any(high_slope):
        np.testing.assert_allclose(wrapped[high_slope], 0.0, atol=5e-6)


@pytest.mark.parametrize("mu", [0.0, np.pi / 4])
@pytest.mark.parametrize("kappa", [0.2, 2.0, 15.0])
def test_vonmises_rvs_matches_constructor(mu, kappa):
    rng_samples = np.random.default_rng(987)
    rng_replay = np.random.default_rng(987)

    vm = vonmises(mu=mu, kappa=kappa)
    samples = vm.rvs(size=1024, random_state=rng_samples)
    if np.isscalar(samples):
        samples = np.array([samples])

    expected = vm.dist._rvs(mu, kappa, size=1024, random_state=rng_replay)
    if np.isscalar(expected):
        expected = np.array([expected])

    np.testing.assert_allclose(
        np.sort(samples),
        np.sort(expected),
        atol=1e-10,
        rtol=0.0,
    )


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


def test_jonespewsey_cdf_ppf_roundtrip():
    dist = jonespewsey(kappa=1.2, psi=0.8, mu=1.1)
    q = np.linspace(0.05, 0.95, 9)
    theta = dist.ppf(q)
    np.testing.assert_allclose(dist.cdf(theta), q, atol=5e-6)


def test_jonespewsey_rvs_reasonable():
    dist = jonespewsey(kappa=1.4, psi=-0.6, mu=1.0)
    _assert_rvs_reasonable(dist, size=256, seed=42)


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


def test_jonespewsey_sineskewed_rvs_reasonable():
    dist = jonespewsey_sineskewed(kappa=1.1, psi=0.4, lmbd=0.3, xi=1.0)
    _assert_rvs_reasonable(dist, size=256, seed=123)


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


def test_jonespewsey_asym_rvs_reasonable():
    dist = jonespewsey_asym(kappa=1.8, psi=-0.9, nu=0.4, xi=0.7)
    _assert_rvs_reasonable(dist, size=256, seed=321)


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


def test_inverse_batschelet_pdf_uniform_limit():
    theta = np.linspace(0.0, 2.0 * np.pi, 9)
    vals = inverse_batschelet.pdf(theta, xi=0.7, kappa=0.0, nu=0.3, lmbd=-0.6)
    expected = np.full_like(theta, 1.0 / (2.0 * np.pi))
    np.testing.assert_allclose(vals, expected, atol=5e-13, rtol=0.0)


def test_inverse_batschelet_pdf_scalar_consistency():
    params = dict(xi=0.5, kappa=1.8, nu=-0.2, lmbd=0.4)
    angles = np.linspace(0.0, 2.0 * np.pi, 7)
    array_vals = inverse_batschelet.pdf(angles, **params)
    scalar_vals = np.array([inverse_batschelet.pdf(float(a), **params) for a in angles])
    np.testing.assert_allclose(array_vals, scalar_vals, atol=5e-12, rtol=0.0)


def test_inverse_batschelet_cdf_matches_numeric():
    params = dict(xi=0.9, kappa=2.4, nu=-0.35, lmbd=0.6)
    theta = np.linspace(0.0, 2.0 * np.pi, 25)
    analytic = inverse_batschelet.cdf(theta, **params)
    numeric = inverse_batschelet._cdf_from_pdf(
        theta,
        params["xi"],
        params["kappa"],
        params["nu"],
        params["lmbd"],
    )
    np.testing.assert_allclose(analytic, numeric, atol=5e-5, rtol=1e-4)


def test_inverse_batschelet_cdf_monotonic():
    params = dict(xi=0.3, kappa=3.5, nu=0.4, lmbd=-0.5)
    theta = np.linspace(0.0, 2.0 * np.pi, 257)
    cdf_vals = inverse_batschelet.cdf(theta, **params)
    diffs = np.diff(cdf_vals)
    assert np.all(diffs >= -1e-11), "CDF must be non-decreasing"


def test_inverse_batschelet_ppf_roundtrip():
    params = dict(xi=0.4, kappa=2.2, nu=-0.25, lmbd=0.55)
    q = np.linspace(1e-5, 1.0 - 1e-5, 61)
    theta = inverse_batschelet.ppf(q, **params)
    q_back = inverse_batschelet.cdf(theta, **params)
    np.testing.assert_allclose(q_back, q, atol=5e-10, rtol=0.0)

    # Endpoints
    np.testing.assert_allclose(inverse_batschelet.ppf(0.0, **params), 0.0, atol=1e-12)
    np.testing.assert_allclose(inverse_batschelet.ppf(1.0, **params), 2.0 * np.pi, atol=1e-12)


def test_inverse_batschelet_rvs_reasonable():
    dist = inverse_batschelet(xi=0.6, kappa=2.8, nu=-0.3, lmbd=0.45)
    _assert_rvs_reasonable(dist, size=512, seed=987, uniform_tol=0.02)


def test_inverse_batschelet_fit_moments():
    samples = inverse_batschelet.rvs(xi=1.1, kappa=3.0, nu=0.2, lmbd=-0.3, size=600, random_state=123)
    xi_hat, kappa_hat, nu_hat, lmbd_hat = inverse_batschelet.fit(samples, method="moments")
    np.testing.assert_allclose(np.mod(xi_hat - 1.1 + np.pi, 2.0 * np.pi) - np.pi, 0.0, atol=0.3)
    assert nu_hat == 0.0
    assert lmbd_hat == 0.0
    assert kappa_hat >= 0.0


def test_inverse_batschelet_fit_mle():
    rng = np.random.default_rng(246)
    xi_true, kappa_true, nu_true, lmbd_true = 0.8, 2.5, -0.25, 0.4
    data = inverse_batschelet.rvs(xi=xi_true, kappa=kappa_true, nu=nu_true, lmbd=lmbd_true, size=800, random_state=rng)

    (xi_hat, kappa_hat, nu_hat, lmbd_hat), info = inverse_batschelet.fit(
        data,
        method="mle",
        return_info=True,
        options={"maxiter": 200},
    )

    assert info["converged"]
    np.testing.assert_allclose(np.mod(xi_hat - xi_true + np.pi, 2.0 * np.pi) - np.pi, 0.0, atol=0.2)
    np.testing.assert_allclose(kappa_hat, kappa_true, atol=0.7)
    np.testing.assert_allclose(nu_hat, nu_true, atol=0.12)
    np.testing.assert_allclose(lmbd_hat, lmbd_true, atol=0.12)


def test_wrapstable_pdf_scalar_consistency():
    params = dict(delta=0.4, alpha=1.4, beta=-0.3, gamma=0.6)
    theta = np.linspace(0.0, 2.0 * np.pi, 9)
    array_vals = wrapstable.pdf(theta, **params)
    scalar_vals = np.array([wrapstable.pdf(float(t), **params) for t in theta])
    np.testing.assert_allclose(array_vals, scalar_vals, atol=5e-13, rtol=0.0)


def test_wrapstable_pdf_matches_wrapped_normal():
    delta = 0.7
    gamma = 0.5
    theta = np.linspace(0.0, 2.0 * np.pi, 13)
    ws_vals = wrapstable.pdf(theta, delta=delta, alpha=2.0, beta=0.0, gamma=gamma)
    rho = np.exp(-(gamma ** 2))
    wn_vals = wrapnorm.pdf(theta, mu=delta, rho=rho)
    np.testing.assert_allclose(ws_vals, wn_vals, atol=1e-6, rtol=5e-6)


def test_wrapstable_pdf_matches_wrapcauchy():
    delta = 1.2
    gamma = 0.8
    theta = np.linspace(0.0, 2.0 * np.pi, 17)
    ws_vals = wrapstable.pdf(theta, delta=delta, alpha=1.0, beta=0.0, gamma=gamma)
    rho = np.exp(-gamma)
    wc_vals = wrapcauchy.pdf(theta, mu=delta, rho=rho)
    np.testing.assert_allclose(ws_vals, wc_vals, atol=1e-7, rtol=1e-6)


def test_wrapstable_series_adaptive_truncation():
    rho_vals, mu_vals, p = wrapstable._get_series_terms(delta=0.0, alpha=1.6, beta=0.1, gamma=0.02)
    assert len(p) > 150
    assert rho_vals.shape == mu_vals.shape == p.shape


def test_wrapstable_cdf_series_matches_numeric():
    params = dict(delta=0.9, alpha=1.4, beta=0.25, gamma=0.5)
    theta = np.linspace(0.0, 2.0 * np.pi, 33)
    analytic = wrapstable.cdf(theta, **params)
    numeric = wrapstable._cdf_from_pdf(
        theta,
        params["delta"],
        params["alpha"],
        params["beta"],
        params["gamma"],
    )
    np.testing.assert_allclose(analytic, numeric, atol=5e-7, rtol=1e-6)


def test_wrapstable_cdf_monotonic():
    params = dict(delta=0.2, alpha=1.8, beta=-0.2, gamma=0.7)
    theta = np.linspace(0.0, 2.0 * np.pi, 257)
    cdf_vals = wrapstable.cdf(theta, **params)
    diffs = np.diff(cdf_vals)
    assert np.all(diffs >= -1e-11)


def test_wrapstable_ppf_roundtrip():
    params = dict(delta=0.5, alpha=1.6, beta=0.3, gamma=0.4)
    q = np.linspace(1e-5, 1.0 - 1e-5, 61)
    theta = wrapstable.ppf(q, **params)
    q_back = wrapstable.cdf(theta, **params)
    np.testing.assert_allclose(q_back, q, atol=3e-5, rtol=0.0)

    np.testing.assert_allclose(wrapstable.ppf(0.0, **params), 0.0, atol=1e-12)
    np.testing.assert_allclose(wrapstable.ppf(1.0, **params), 2.0 * np.pi, atol=1e-12)


def test_wrapstable_rvs_reasonable():
    dist = wrapstable(delta=0.6, alpha=1.3, beta=-0.2, gamma=0.7)
    _assert_rvs_reasonable(dist, size=512, seed=2024, uniform_tol=0.005)


def test_wrapstable_rvs_reduces_to_wrapped_normal():
    rng = np.random.default_rng(321)
    delta = 1.0
    gamma = 0.5
    samples = wrapstable.rvs(delta=delta, alpha=2.0, beta=0.0, gamma=gamma, size=2000, random_state=rng)
    rho = np.exp(-(gamma ** 2))
    wn_samples = wrapnorm.rvs(mu=delta, rho=rho, size=2000, random_state=321)
    # Compare first trigonometric moment
    m1_ws = np.mean(np.exp(1j * samples))
    m1_wn = np.mean(np.exp(1j * wn_samples))
    np.testing.assert_allclose(m1_ws, m1_wn, atol=0.05)


def test_wrapstable_fit_moments():
    rng = np.random.default_rng(12)
    params = dict(delta=0.9, alpha=1.4, beta=-0.25, gamma=0.6)
    data = wrapstable.rvs(size=800, random_state=rng, **params)
    delta_hat, alpha_hat, beta_hat, gamma_hat = wrapstable.fit(data, method="moments")

    np.testing.assert_allclose(_angle_diff(delta_hat, params["delta"]), 0.0, atol=0.3)
    np.testing.assert_allclose(alpha_hat, params["alpha"], atol=0.35)
    np.testing.assert_allclose(beta_hat, params["beta"], atol=0.35)
    np.testing.assert_allclose(gamma_hat, params["gamma"], atol=0.3)


def test_wrapstable_fit_mle():
    rng = np.random.default_rng(34)
    params = dict(delta=0.7, alpha=1.6, beta=0.3, gamma=0.5)
    data = wrapstable.rvs(size=1200, random_state=rng, **params)

    (delta_hat, alpha_hat, beta_hat, gamma_hat), info = wrapstable.fit(
        data,
        method="mle",
        return_info=True,
        options={"maxiter": 200},
    )

    assert info["converged"]
    np.testing.assert_allclose(_angle_diff(delta_hat, params["delta"]), 0.0, atol=0.2)
    np.testing.assert_allclose(alpha_hat, params["alpha"], atol=0.2)
    np.testing.assert_allclose(beta_hat, params["beta"], atol=0.25)
    np.testing.assert_allclose(gamma_hat, params["gamma"], atol=0.2)


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


def test_katojones_cdf_matches_numeric():
    params = dict(mu=0.8, gamma=0.4, rho=0.35, lam=1.1)
    theta = np.linspace(0.0, 2.0 * np.pi, 49)
    analytic = katojones.cdf(theta, **params)
    numeric = katojones._cdf_from_pdf(
        theta,
        params["mu"],
        params["gamma"],
        params["rho"],
        params["lam"],
    )
    np.testing.assert_allclose(analytic, numeric, atol=5e-7, rtol=1e-6)


def test_katojones_ppf_roundtrip():
    params = dict(mu=0.5, gamma=0.45, rho=0.3, lam=1.4)
    q = np.linspace(1e-5, 1.0 - 1e-5, 61)
    theta = katojones.ppf(q, **params)
    q_back = katojones.cdf(theta, **params)
    np.testing.assert_allclose(q_back, q, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(katojones.ppf(0.0, **params), 0.0, atol=1e-12)
    np.testing.assert_allclose(katojones.ppf(1.0, **params), 2.0 * np.pi, atol=1e-12)


def test_katojones_rvs_reasonable():
    dist = katojones(mu=0.7, gamma=0.5, rho=0.25, lam=1.2)
    _assert_rvs_reasonable(dist, size=512, seed=2025, uniform_tol=0.01)
    
def _assert_rvs_reasonable(dist, size=256, seed=123, uniform_tol=0.05):
    rng = np.random.default_rng(seed)
    samples = dist.rvs(size=size, random_state=rng)
    samples = np.asarray(samples, dtype=float)
    assert samples.size == size

    u = dist.cdf(samples)
    u = np.mod(u, 1.0)
    stat, pvalue = stats.kstest(u, "uniform")
    assert pvalue > uniform_tol, f"kstest failed: statistic={stat}, p={pvalue}"
