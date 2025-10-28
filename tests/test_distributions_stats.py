import numpy as np
import pytest
from scipy.special import i0, i1, iv

from pycircstat2.distributions import (
    circularuniform,
    triangular,
    vonmises,
    vonmises_flattopped,
)


def test_circularuniform_stats_degenerate_mean():
    stats = circularuniform.stats()

    assert np.isnan(stats["mean"])
    assert np.isclose(stats["median"], np.pi)
    assert np.isclose(stats["r"], 0.0, atol=1e-12)
    assert np.isclose(stats["var"], 1.0, atol=1e-12)
    assert np.isinf(stats["std"])
    assert np.isinf(stats["dispersion"])
    assert np.isclose(stats["skewness"], 0.0, atol=1e-12)
    assert np.isclose(stats["kurtosis"], 0.0, atol=1e-12)


def test_vonmises_stats_match_closed_form():
    mu = 0.7
    kappa = 1.25

    stats = vonmises.stats(mu=mu, kappa=kappa)

    r1 = i1(kappa) / i0(kappa)
    r2 = iv(2, kappa) / i0(kappa)

    assert np.isclose(vonmises.mean(mu=mu, kappa=kappa), mu, atol=1e-12)
    assert np.isclose(vonmises.r(mu=mu, kappa=kappa), r1, atol=1e-12)
    assert np.isclose(vonmises.var(mu=mu, kappa=kappa), 1 - r1, atol=1e-12)
    assert np.isclose(vonmises.std(mu=mu, kappa=kappa), np.sqrt(-2 * np.log(r1)))
    assert np.isclose(
        vonmises.dispersion(mu=mu, kappa=kappa),
        (1 - r2) / (2 * r1 * r1),
        atol=1e-12,
    )

    assert np.isclose(stats["mean"], mu, atol=1e-12)
    assert np.isclose(stats["r"], r1, atol=1e-12)
    assert np.isclose(stats["var"], 1 - r1, atol=1e-12)
    assert np.isclose(stats["std"], np.sqrt(-2 * np.log(r1)))
    assert np.isclose(stats["dispersion"], (1 - r2) / (2 * r1 * r1), atol=1e-12)
    assert np.isclose(stats["skewness"], 0.0, atol=1e-12)
    expected_kurtosis = (r2 - r1**4) / (1 - r1) ** 2
    assert np.isclose(stats["kurtosis"], expected_kurtosis, atol=1e-12)


def test_triangular_trig_moment_keyword_shape():
    rho = 0.2

    moment_kw = triangular.trig_moment(1, rho=rho)
    moment_pos = triangular.trig_moment(1, rho)

    assert np.isclose(moment_kw.real, rho, atol=1e-9)
    assert np.isclose(moment_kw.imag, 0.0, atol=1e-9)
    assert np.isclose(moment_pos.real, rho, atol=1e-9)

    with pytest.raises(TypeError):
        triangular.trig_moment(1, rho, rho=rho + 0.05)


def test_triangular_std_and_dispersion_match_frozen():
    rho = 0.2
    base_std = triangular.std(rho=rho)
    frozen_std = triangular(rho=rho).std()

    base_disp = triangular.dispersion(rho=rho)
    frozen_disp = triangular(rho=rho).dispersion()

    assert np.isclose(base_std, frozen_std, atol=1e-12)
    assert np.isclose(base_disp, frozen_disp, atol=1e-12)
    assert np.isinf(triangular.std(rho=0.0))
    assert np.isinf(triangular(rho=0.0).std())


def test_frozen_vonmises_flattopped_inherits_helpers():
    frozen = vonmises_flattopped(mu=0.5, kappa=1.1, nu=0.3)
    base_disp = vonmises_flattopped.dispersion(mu=0.5, kappa=1.1, nu=0.3)

    assert np.isclose(frozen.r(), vonmises_flattopped.r(mu=0.5, kappa=1.1, nu=0.3))
    assert np.isclose(frozen.dispersion(), base_disp)
    stats = frozen.stats()
    assert set(stats.keys()) == {
        "mean",
        "median",
        "r",
        "var",
        "std",
        "dispersion",
        "skewness",
        "kurtosis",
    }
