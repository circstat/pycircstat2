from dataclasses import dataclass
from itertools import combinations_with_replacement as cwr
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pytest
from scipy import special, stats
from scipy.integrate import quad

from pycircstat2.distributions import (
    _VMFT_KAPPA_UPPER,
    ajplss,
    cardioid,
    cardlss,
    cartlss,
    cartwright,
    circularuniform,
    ibslss,
    inverse_batschelet,
    jonespewsey,
    jonespewsey_asym,
    jonespewsey_sineskewed,
    jplss,
    katojones,
    kjlss,
    pnlss,
    ssjplss,
    triangular,
    vmftlss,
    vmlss,
    vonmises,
    vonmises_flattopped,
    wclss,
    wnlss,
    wrapcauchy,
    wrapnorm,
    wrapstable,
)
from pycircstat2.regression import circ_gam


def _assert_monotonic_cdf_ppf(
    dist, theta_grid, q_grid, *, cdf_tol=1e-12, ppf_tol=1e-12
):
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
    assert np.all((cdf_vals >= -cdf_tol) & (cdf_vals <= 1.0 + cdf_tol)), (
        "CDF outside [0, 1]"
    )

    ppf_diffs = np.diff(ppf_vals)
    assert np.all(ppf_diffs >= -ppf_tol), "PPF must be non-decreasing"
    two_pi = 2.0 * np.pi
    assert np.all((ppf_vals >= -ppf_tol) & (ppf_vals <= two_pi + ppf_tol)), (
        "PPF outside [0, 2π]"
    )


@dataclass(frozen=True)
class DistributionCase:
    id: str
    factory: Callable[..., Any]
    params: Dict[str, Any]
    theta_points: int = 129
    q_points: int = 129
    q_min: float = 0.0
    cdf_tol: float = 1e-11
    ppf_tol: float = 1e-11
    ppf_slope_threshold: float = 0.0
    ppf_high_slope_tol: Optional[float] = None
    ppf_low_slope_tol: Optional[float] = None

    def dist(self):
        return self.factory(**self.params)


@dataclass(frozen=True)
class ReferenceValue:
    id: str
    factory: Callable[..., Any]
    params: Dict[str, Any]
    method: str
    arg: float
    expected: float
    atol: float = 1e-9


@dataclass(frozen=True)
class CdfFromPdfCase:
    id: str
    cdf: Callable[..., Any]
    numeric_cdf: Callable[..., Any]
    args: Tuple[Any, ...]
    theta_points: int
    atol: float


@dataclass(frozen=True)
class RvsCase:
    id: str
    factory: Callable[..., Any]
    params: Dict[str, Any]
    size: int = 1024
    seed: int = 123
    uniform_tol: float = 0.01

    def dist(self):
        return self.factory(**self.params)


def _evaluate_array(func: Callable[..., Any], grid: Any, **kwargs: Any) -> np.ndarray:
    try:
        return np.asarray(func(grid, **kwargs), dtype=float)
    except (TypeError, ValueError):
        flat = np.asarray(grid, dtype=float).reshape(-1)
        evaluated = np.array([func(float(val), **kwargs) for val in flat], dtype=float)
        return evaluated.reshape(np.shape(grid))


_ARGCHECK_CASES = [
    (
        "triangular",
        triangular,
        (np.array([-0.1, 0.1, 0.5]),),
        np.array([False, True, False]),
    ),
    (
        "cardioid",
        cardioid,
        (np.array([0.0, 2 * np.pi + 0.1]), np.array([0.2, 0.6])),
        np.array([True, False]),
    ),
    (
        "cartwright",
        cartwright,
        (np.array([0.0, -0.1]), np.array([0.5, 0.5])),
        np.array([True, False]),
    ),
    (
        "wrapnorm",
        wrapnorm,
        (np.array([0.0, 0.0]), np.array([0.5, 1.2])),
        np.array([True, False]),
    ),
    (
        "wrapcauchy",
        wrapcauchy,
        (np.array([0.0, 0.0]), np.array([0.1, -0.1])),
        np.array([True, False]),
    ),
    (
        "vonmises",
        vonmises,
        (np.array([0.0, 7.0]), np.array([0.5, 0.5])),
        np.array([True, False]),
    ),
    (
        "vonmises_flattopped",
        vonmises_flattopped,
        (
            np.array([0.0, 0.0]),
            np.array([0.5, _VMFT_KAPPA_UPPER + 1.0]),
            np.array([0.0, 0.0]),
        ),
        np.array([True, False]),
    ),
    (
        "jonespewsey",
        jonespewsey,
        (np.array([0.0, -0.1]), np.array([0.5, 0.5]), np.array([0.0, 0.0])),
        np.array([True, False]),
    ),
    (
        "jonespewsey_sineskewed",
        jonespewsey_sineskewed,
        (
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0.0, 0.0]),
            np.array([0.0, 2.0]),
        ),
        np.array([True, False]),
    ),
    (
        "jonespewsey_asym",
        jonespewsey_asym,
        (
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
            np.array([0.0, 0.0]),
            np.array([0.5, 1.2]),
        ),
        np.array([True, False]),
    ),
    (
        "inverse_batschelet",
        inverse_batschelet,
        (
            np.array([0.0, 0.0]),
            np.array([0.5, -0.5]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        ),
        np.array([True, False]),
    ),
    (
        "wrapstable",
        wrapstable,
        (
            np.array([0.0, 2 * np.pi + 0.1]),
            np.array([1.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
        ),
        np.array([True, False]),
    ),
    (
        "katojones",
        katojones,
        (
            np.array([0.0, 0.0]),
            np.array([0.5, 1.1]),
            np.array([0.2, 0.2]),
            np.array([0.1, 0.1]),
        ),
        np.array([True, False]),
    ),
]


@pytest.mark.parametrize(
    "name, dist, params, expected", _ARGCHECK_CASES, ids=[c[0] for c in _ARGCHECK_CASES]
)
def test_argcheck_vectorized_mask_all(name, dist, params, expected):
    mask = dist._argcheck(*params)
    assert isinstance(mask, np.ndarray), f"{name} should return an array mask"
    assert mask.shape == expected.shape
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask, expected)


_PDF_VECTOR_CASES = [
    ("triangular", triangular, (0.5, np.array([0.1, 0.2]))),
    ("cardioid", cardioid, (0.25, np.array([0.0, 0.5]), np.array([0.1, 0.2]))),
    ("cartwright", cartwright, (0.25, np.array([0.0, 0.2]), np.array([0.5, 0.6]))),
    ("wrapnorm", wrapnorm, (0.25, np.array([0.0, 0.2]), np.array([0.3, 0.4]))),
    ("wrapcauchy", wrapcauchy, (0.25, np.array([0.0, 0.2]), np.array([0.3, 0.4]))),
    ("vonmises", vonmises, (0.25, np.array([0.0, 0.2]), np.array([1.0, 2.0]))),
    (
        "katojones",
        katojones,
        (
            0.25,
            np.array([0.0, 0.2]),
            np.array([0.5, 0.6]),
            np.array([0.2, 0.3]),
            np.array([0.1, 0.2]),
        ),
    ),
]


@pytest.mark.parametrize(
    "name, dist, args", _PDF_VECTOR_CASES, ids=[c[0] for c in _PDF_VECTOR_CASES]
)
def test_pdf_vectorized_shape_parameters(name, dist, args):
    vals = dist.pdf(*args)
    shapes = [np.shape(arg) for arg in args[1:]]  # skip x
    expected_shape = np.broadcast_shapes(*shapes) if shapes else ()
    assert isinstance(vals, np.ndarray)
    assert vals.shape == expected_shape
    assert np.all(np.isfinite(vals)), f"{name} pdf returned non-finite values"


def test_circular_cdf_is_periodic():
    theta = np.array([0.3, 1.0, np.pi, 4.0])
    shifted = theta + 2.0 * np.pi

    uni_base = circularuniform.cdf(theta)
    uni_shift = circularuniform.cdf(shifted)
    np.testing.assert_allclose(uni_base, uni_shift, atol=1e-12)

    mu = 0.4
    rho = 0.2
    card_base = cardioid.cdf(theta, mu=mu, rho=rho)
    card_shift = cardioid.cdf(shifted, mu=mu, rho=rho)
    np.testing.assert_allclose(card_base, card_shift, atol=1e-10)

    wn_base = np.array([wrapnorm.cdf(val, mu=0.1, rho=0.5) for val in theta])
    wn_shift = np.array([wrapnorm.cdf(val, mu=0.1, rho=0.5) for val in shifted])
    np.testing.assert_allclose(wn_base, wn_shift, atol=1e-10)


_SCALAR_ONLY_CALLS = [
    (
        "vonmises_flattopped",
        lambda: vonmises_flattopped.pdf(
            0.1, mu=np.array([0.0, 0.1]), kappa=1.0, nu=0.1
        ),
    ),
    # jonespewsey + sineskewed pdf/logpdf now ACCEPT per-obs parameter arrays
    # (the Phase-1 regression contract); cdf and the other methods remain
    # scalar-only, so the guard pins those instead.
    (
        "jonespewsey_cdf",
        lambda: jonespewsey.cdf(0.1, mu=0.0, kappa=np.array([1.0, 1.1]), psi=0.1),
    ),
    (
        "jonespewsey_sineskewed_cdf",
        lambda: jonespewsey_sineskewed.cdf(
            0.1, xi=0.0, kappa=np.array([1.0, 1.1]), psi=0.1, lmbd=0.1
        ),
    ),
    (
        "jonespewsey_asym",
        lambda: jonespewsey_asym.pdf(
            0.1, xi=0.0, kappa=np.array([1.0, 1.1]), psi=0.1, nu=0.2
        ),
    ),
    (
        "vonmises_flattopped",
        lambda: vonmises_flattopped.pdf(
            0.1, mu=0.0, kappa=np.array([1.0, 1.1]), nu=0.2
        ),
    ),
    (
        "inverse_batschelet",
        lambda: inverse_batschelet.pdf(
            0.1, xi=0.0, kappa=np.array([1.0, 1.1]), nu=0.2, lmbd=0.1
        ),
    ),
    (
        "wrapstable",
        lambda: wrapstable.pdf(
            0.1, delta=np.array([0.0, 0.1]), alpha=1.0, beta=0.0, gamma=1.0
        ),
    ),
    (
        "katojones_ppf",
        lambda: katojones.ppf(
            0.5, mu=np.array([0.0, 0.1]), gamma=0.5, rho=0.2, lam=0.1
        ),
    ),
]


@pytest.mark.parametrize(
    "name, call", _SCALAR_ONLY_CALLS, ids=[c[0] for c in _SCALAR_ONLY_CALLS]
)
def test_scalar_only_distributions_reject_arrays(name, call):
    with pytest.raises(ValueError, match="scalar"):
        call()


REFERENCE_VALUES = [
    ReferenceValue(
        id="circularuniform-cdf",
        factory=circularuniform,
        params={},
        method="cdf",
        arg=2.0,
        expected=0.3183098861837907,
        atol=1e-12,
    ),
    ReferenceValue(
        id="circularuniform-ppf",
        factory=circularuniform,
        params={},
        method="ppf",
        arg=1.0 / np.pi,
        expected=2.0,
        atol=1e-12,
    ),
    ReferenceValue(
        id="cardioid-cdf",
        factory=cardioid,
        params={"rho": 0.3, "mu": np.pi / 2},
        method="cdf",
        arg=np.pi,
        expected=0.6909859317102744,
        atol=1e-9,
    ),
    ReferenceValue(
        id="cardioid-ppf",
        factory=cardioid,
        params={"rho": 0.3, "mu": np.pi / 2},
        method="ppf",
        arg=0.6909859317102744,
        expected=np.pi,
        atol=1e-9,
    ),
    ReferenceValue(
        id="cartwright-cdf",
        factory=cartwright,
        params={"zeta": 0.1, "mu": np.pi / 2},
        method="cdf",
        arg=3.0 * np.pi / 4.0,
        expected=0.9641666531258773,
        atol=1e-9,
    ),
    ReferenceValue(
        id="cartwright-ppf",
        factory=cartwright,
        params={"zeta": 0.1, "mu": np.pi / 2},
        method="ppf",
        arg=0.9641666531258773,
        expected=3.0 * np.pi / 4.0,
        atol=1e-9,
    ),
    ReferenceValue(
        id="wrapcauchy-cdf",
        factory=wrapcauchy,
        params={"rho": 0.75, "mu": np.pi / 2},
        method="cdf",
        arg=np.pi / 6.0,
        expected=0.0320432438547667,
        atol=1e-9,
    ),
    ReferenceValue(
        id="wrapcauchy-ppf",
        factory=wrapcauchy,
        params={"rho": 0.75, "mu": np.pi / 2},
        method="ppf",
        arg=0.0320432438547667,
        expected=np.pi / 6.0,
        atol=5e-6,
    ),
    ReferenceValue(
        id="wrapnorm-cdf",
        factory=wrapnorm,
        params={"rho": 0.75, "mu": np.pi / 2},
        method="cdf",
        arg=np.pi / 6.0,
        expected=0.06451975467423943,
        atol=1e-9,
    ),
    ReferenceValue(
        id="wrapnorm-ppf",
        factory=wrapnorm,
        params={"rho": 0.75, "mu": np.pi / 2},
        method="ppf",
        arg=0.5,
        expected=1.6072904842634406,
        atol=1e-9,
    ),
    ReferenceValue(
        id="vonmises-cdf",
        factory=vonmises,
        params={"kappa": 2.37, "mu": np.pi / 2},
        method="cdf",
        arg=np.pi / 6.0,
        expected=0.05432533537843656,
        atol=5e-11,
    ),
    ReferenceValue(
        id="vonmises-ppf",
        factory=vonmises,
        params={"kappa": 2.37, "mu": np.pi / 2},
        method="ppf",
        arg=0.5,
        expected=1.6138877997996237,
        atol=5e-11,
    ),
    ReferenceValue(
        id="vonmises-flattopped-cdf",
        factory=vonmises_flattopped,
        params={"kappa": 2.0, "nu": -0.5, "mu": np.pi / 2},
        method="cdf",
        arg=3.0 * np.pi / 4.0,
        expected=0.7119746660317867,
        atol=5e-9,
    ),
    ReferenceValue(
        id="vonmises-flattopped-ppf",
        factory=vonmises_flattopped,
        params={"kappa": 2.0, "nu": -0.5, "mu": np.pi / 2},
        method="ppf",
        arg=0.5,
        expected=1.7301046248783023,
        atol=5e-9,
    ),
    ReferenceValue(
        id="jonespewsey-cdf",
        factory=jonespewsey,
        params={"kappa": 2.0, "psi": -1.5, "mu": np.pi / 2},
        method="cdf",
        arg=np.pi / 2.0,
        expected=0.4401444958105559,
        atol=5e-9,
    ),
    ReferenceValue(
        id="jonespewsey-ppf",
        factory=jonespewsey,
        params={"kappa": 2.0, "psi": -1.5, "mu": np.pi / 2},
        method="ppf",
        arg=0.4401444958105559,
        expected=1.5707963291458178,
        atol=5e-9,
    ),
    ReferenceValue(
        id="jonespewsey-sineskewed-cdf",
        factory=jonespewsey_sineskewed,
        params={"kappa": 2.0, "psi": 1.0, "lmbd": 0.5, "xi": np.pi / 2},
        method="cdf",
        arg=3.0 * np.pi / 2.0,
        expected=0.9446497875304358,
        atol=5e-9,
    ),
    ReferenceValue(
        id="jonespewsey-sineskewed-ppf",
        factory=jonespewsey_sineskewed,
        params={"kappa": 2.0, "psi": 1.0, "lmbd": 0.5, "xi": np.pi / 2},
        method="ppf",
        arg=0.5,
        expected=2.1878509192906153,
        atol=5e-9,
    ),
    ReferenceValue(
        id="jonespewsey-asym-cdf",
        factory=jonespewsey_asym,
        params={"kappa": 2.0, "psi": -1.0, "nu": 0.75, "xi": np.pi / 2},
        method="cdf",
        arg=np.pi / 2.0,
        expected=0.7535176456215893,
        atol=5e-9,
    ),
    ReferenceValue(
        id="jonespewsey-asym-ppf",
        factory=jonespewsey_asym,
        params={"kappa": 2.0, "psi": -1.0, "nu": 0.75, "xi": np.pi / 2},
        method="ppf",
        arg=0.5,
        expected=1.0498801800527269,
        atol=5e-9,
    ),
    ReferenceValue(
        id="inverse-batschelet-cdf",
        factory=inverse_batschelet,
        params={"kappa": 2.0, "nu": -0.5, "lmbd": 0.7, "xi": np.pi / 2},
        method="cdf",
        arg=np.pi / 2.0,
        expected=0.11796336892075589,
        atol=5e-9,
    ),
    ReferenceValue(
        id="inverse-batschelet-ppf",
        factory=inverse_batschelet,
        params={"kappa": 2.0, "nu": -0.5, "lmbd": 0.7, "xi": np.pi / 2},
        method="ppf",
        arg=0.5,
        expected=2.5137729476810207,
        atol=5e-9,
    ),
]

_REFERENCE_LOOKUP = {case.id: case for case in REFERENCE_VALUES}


CDF_PPF_CASES = [
    DistributionCase(
        id="circularuniform",
        factory=circularuniform,
        params={},
        theta_points=256,
        q_points=256,
        cdf_tol=1e-12,
        ppf_tol=1e-12,
    ),
    DistributionCase(
        id="triangular-rho0.0",
        factory=triangular,
        params={"rho": 0.0},
        theta_points=256,
        q_points=256,
    ),
    DistributionCase(
        id="triangular-rho0.3",
        factory=triangular,
        params={"rho": 0.3},
        theta_points=256,
        q_points=256,
    ),
    DistributionCase(
        id="triangular-rho4/pi^2",
        factory=triangular,
        params={"rho": 4.0 / np.pi**2},
        theta_points=256,
        q_points=256,
    ),
    DistributionCase(
        id="cardioid-rho0.0",
        factory=cardioid,
        params={"rho": 0.0, "mu": 0.0},
        theta_points=256,
        q_points=256,
    ),
    DistributionCase(
        id="cardioid-rho0.2",
        factory=cardioid,
        params={"rho": 0.2, "mu": 0.3},
        theta_points=256,
        q_points=256,
    ),
    DistributionCase(
        id="cardioid-rho0.49",
        factory=cardioid,
        params={"rho": 0.49, "mu": np.pi / 2},
        theta_points=256,
        q_points=256,
    ),
    DistributionCase(
        id="cardioid-rho0.3-muPi/3",
        factory=cardioid,
        params={"rho": 0.3, "mu": np.pi / 3},
        theta_points=256,
        q_points=256,
    ),
    DistributionCase(
        id="cartwright-zeta0.2",
        factory=cartwright,
        params={"zeta": 0.2, "mu": 0.1},
        theta_points=256,
        q_points=256,
        ppf_slope_threshold=1e-6,
        ppf_low_slope_tol=0.1,
    ),
    DistributionCase(
        id="cartwright-zeta1.0",
        factory=cartwright,
        params={"zeta": 1.0, "mu": np.pi},
        theta_points=256,
        q_points=256,
        ppf_slope_threshold=1e-6,
        ppf_low_slope_tol=0.1,
    ),
    DistributionCase(
        id="cartwright-zeta1.5",
        factory=cartwright,
        params={"zeta": 1.5, "mu": 0.4},
        theta_points=192,
        q_points=192,
        ppf_slope_threshold=1e-6,
        ppf_low_slope_tol=0.1,
    ),
    DistributionCase(
        id="cartwright-zeta5.0",
        factory=cartwright,
        params={"zeta": 5.0, "mu": 2.0},
        theta_points=256,
        q_points=256,
        ppf_slope_threshold=1e-6,
        ppf_low_slope_tol=0.1,
    ),
    DistributionCase(
        id="wrapnorm-rho0.1",
        factory=wrapnorm,
        params={"rho": 0.1, "mu": 0.0},
        theta_points=256,
        q_points=512,
        ppf_slope_threshold=1e-4,
        ppf_high_slope_tol=5e-6,
        ppf_low_slope_tol=1e-2,
    ),
    DistributionCase(
        id="wrapnorm-rho0.5",
        factory=wrapnorm,
        params={"rho": 0.5, "mu": np.pi / 4},
        theta_points=256,
        q_points=512,
        ppf_slope_threshold=1e-4,
        ppf_high_slope_tol=5e-6,
        ppf_low_slope_tol=1e-2,
    ),
    DistributionCase(
        id="wrapnorm-rho0.9",
        factory=wrapnorm,
        params={"rho": 0.9, "mu": np.pi / 4},
        theta_points=256,
        q_points=512,
        q_min=1e-8,
        cdf_tol=5e-10,
        ppf_tol=5e-10,
        ppf_slope_threshold=1e-4,
        ppf_high_slope_tol=5e-6,
        ppf_low_slope_tol=1e-2,
    ),
    DistributionCase(
        id="wrapcauchy-rho0.0",
        factory=wrapcauchy,
        params={"rho": 0.0, "mu": 0.0},
        theta_points=256,
        q_points=256,
    ),
    DistributionCase(
        id="wrapcauchy-rho0.4",
        factory=wrapcauchy,
        params={"rho": 0.4, "mu": np.pi / 3},
        theta_points=256,
        q_points=256,
    ),
    DistributionCase(
        id="wrapcauchy-rho0.95",
        factory=wrapcauchy,
        params={"rho": 0.95, "mu": np.pi},
        theta_points=256,
        q_points=256,
        q_min=1e-6,
        cdf_tol=5e-11,
        ppf_tol=5e-11,
    ),
    DistributionCase(
        id="vonmises-kappa0.05",
        factory=vonmises,
        params={"kappa": 0.05, "mu": 0.0},
        theta_points=256,
        q_points=256,
    ),
    DistributionCase(
        id="vonmises-kappa5.0",
        factory=vonmises,
        params={"kappa": 5.0, "mu": np.pi / 4},
        theta_points=256,
        q_points=256,
        cdf_tol=5e-10,
        ppf_tol=5e-10,
    ),
    DistributionCase(
        id="vonmises-kappa25.0",
        factory=vonmises,
        params={"kappa": 25.0, "mu": np.pi},
        theta_points=256,
        q_points=256,
        cdf_tol=1e-10,
        ppf_tol=1e-10,
        ppf_slope_threshold=1e-6,
        ppf_high_slope_tol=5e-7,
        ppf_low_slope_tol=np.pi,
    ),
    DistributionCase(
        id="vonmises-flattopped",
        factory=vonmises_flattopped,
        params={"mu": 0.6, "kappa": 2.0, "nu": 0.3},
        theta_points=192,
        q_points=192,
        q_min=1e-6,
        cdf_tol=1e-9,
        ppf_tol=1e-9,
    ),
    DistributionCase(
        id="vonmises-flattopped-uniform",
        factory=vonmises_flattopped,
        params={"mu": 1.5, "kappa": 0.0, "nu": 0.3},
        theta_points=160,
        q_points=160,
        q_min=1e-6,
        cdf_tol=1e-9,
        ppf_tol=1e-9,
    ),
    DistributionCase(
        id="jonespewsey",
        factory=jonespewsey,
        params={"mu": 0.6, "kappa": 1.0, "psi": 0.4},
        theta_points=192,
        q_points=192,
        q_min=1e-6,
        cdf_tol=1e-9,
        ppf_tol=1e-9,
    ),
    DistributionCase(
        id="jonespewsey-sineskewed",
        factory=jonespewsey_sineskewed,
        params={"xi": 1.0, "kappa": 1.5, "psi": 0.3, "lmbd": 0.4},
        theta_points=160,
        q_points=160,
        q_min=1e-5,
        cdf_tol=5e-9,
        ppf_tol=5e-9,
    ),
    DistributionCase(
        id="jonespewsey-asym",
        factory=jonespewsey_asym,
        params={"xi": 0.7, "kappa": 1.1, "psi": 0.2, "nu": 0.4},
        theta_points=160,
        q_points=160,
        q_min=1e-5,
        cdf_tol=5e-9,
        ppf_tol=5e-9,
    ),
    DistributionCase(
        id="inverse-batschelet",
        factory=inverse_batschelet,
        params={"xi": 0.8, "kappa": 1.3, "nu": 0.3, "lmbd": 0.2},
        theta_points=160,
        q_points=160,
        q_min=1e-5,
        cdf_tol=1e-8,
        ppf_tol=1e-8,
    ),
    DistributionCase(
        id="katojones",
        factory=katojones,
        params={"mu": 0.8, "gamma": 0.3, "rho": 0.2, "lam": 0.4},
        theta_points=96,
        q_points=96,
        q_min=1e-5,
        cdf_tol=5e-6,
        ppf_tol=5e-6,
    ),
    DistributionCase(
        id="wrapstable",
        factory=wrapstable,
        params={"delta": 0.9, "alpha": 1.5, "beta": 0.2, "gamma": 0.4},
        theta_points=96,
        q_points=96,
        q_min=1e-5,
        cdf_tol=1e-8,
        ppf_tol=1e-8,
    ),
]


CDF_FROM_PDF_CASES = [
    CdfFromPdfCase(
        id="cartwright",
        cdf=cartwright.cdf,
        numeric_cdf=cartwright._cdf_from_pdf,
        args=(1.2, 0.8),
        theta_points=9,
        atol=1e-7,
    ),
    CdfFromPdfCase(
        id="wrapcauchy",
        cdf=wrapcauchy.cdf,
        numeric_cdf=wrapcauchy._cdf_from_pdf,
        args=(0.9, 0.65),
        theta_points=9,
        atol=1e-7,
    ),
    CdfFromPdfCase(
        id="wrapnorm",
        cdf=wrapnorm.cdf,
        numeric_cdf=wrapnorm._cdf_from_pdf,
        args=(0.7, 0.45),
        theta_points=7,
        atol=1e-7,
    ),
    CdfFromPdfCase(
        id="vonmises",
        cdf=vonmises.cdf,
        numeric_cdf=vonmises._cdf_from_pdf,
        args=(0.6, 3.2),
        theta_points=11,
        atol=5e-7,
    ),
    CdfFromPdfCase(
        id="inverse-batschelet",
        cdf=inverse_batschelet.cdf,
        numeric_cdf=inverse_batschelet._cdf_from_pdf,
        args=(0.9, 2.4, -0.35, 0.6),
        theta_points=25,
        atol=5e-5,
    ),
    CdfFromPdfCase(
        id="katojones",
        cdf=katojones.cdf,
        numeric_cdf=katojones._cdf_from_pdf,
        args=(0.8, 0.4, 0.35, 1.1),
        theta_points=49,
        atol=5e-7,
    ),
    CdfFromPdfCase(
        id="wrapstable",
        cdf=wrapstable.cdf,
        numeric_cdf=wrapstable._cdf_from_pdf,
        args=(0.9, 1.4, 0.25, 0.5),
        theta_points=33,
        atol=5e-7,
    ),
]


RVS_CASES = [
    RvsCase(
        id="circularuniform",
        factory=circularuniform,
        params={},
        size=512,
        seed=1001,
        uniform_tol=0.01,
    ),
    RvsCase(
        id="triangular-rho0.0",
        factory=triangular,
        params={"rho": 0.0},
        size=512,
        seed=123,
        uniform_tol=0.01,
    ),
    RvsCase(
        id="triangular-rho0.3",
        factory=triangular,
        params={"rho": 0.3},
        size=512,
        seed=321,
        uniform_tol=0.01,
    ),
    RvsCase(
        id="cardioid",
        factory=cardioid,
        params={"rho": 0.3, "mu": np.pi / 3},
        size=512,
        seed=321,
        uniform_tol=0.02,
    ),
    RvsCase(
        id="cartwright",
        factory=cartwright,
        params={"zeta": 0.8, "mu": np.pi / 4},
        size=512,
        seed=456,
        uniform_tol=0.02,
    ),
    RvsCase(
        id="wrapcauchy",
        factory=wrapcauchy,
        params={"rho": 0.8, "mu": np.pi / 3},
        size=512,
        seed=654,
        uniform_tol=0.015,
    ),
    RvsCase(
        id="wrapnorm",
        factory=wrapnorm,
        params={"rho": 0.5, "mu": np.pi / 4},
        size=512,
        seed=789,
        uniform_tol=0.015,
    ),
    RvsCase(
        id="vonmises",
        factory=vonmises,
        params={"kappa": 2.0, "mu": np.pi / 4},
        size=1024,
        seed=987,
        uniform_tol=0.015,
    ),
    RvsCase(
        id="vonmises-flattopped",
        factory=vonmises_flattopped,
        params={"mu": 0.8, "kappa": 7.5, "nu": -0.35},
        size=4096,
        seed=1234,
        uniform_tol=0.035,
    ),
    RvsCase(
        id="jonespewsey",
        factory=jonespewsey,
        params={"mu": 1.0, "kappa": 1.4, "psi": -0.6},
        size=256,
        seed=42,
        uniform_tol=0.02,
    ),
    RvsCase(
        id="jonespewsey-sineskewed",
        factory=jonespewsey_sineskewed,
        params={"xi": 1.0, "kappa": 1.1, "psi": 0.4, "lmbd": 0.3},
        size=256,
        seed=123,
        uniform_tol=0.02,
    ),
    RvsCase(
        id="jonespewsey-asym",
        factory=jonespewsey_asym,
        params={"xi": 0.7, "kappa": 1.8, "psi": -0.9, "nu": 0.4},
        size=256,
        seed=321,
        uniform_tol=0.02,
    ),
    RvsCase(
        id="inverse-batschelet",
        factory=inverse_batschelet,
        params={"xi": 0.6, "kappa": 2.8, "nu": -0.3, "lmbd": 0.45},
        size=512,
        seed=987,
        uniform_tol=0.02,
    ),
    RvsCase(
        id="wrapstable",
        factory=wrapstable,
        params={"delta": 0.9, "alpha": 1.5, "beta": 0.2, "gamma": 0.4},
        size=512,
        seed=2024,
        uniform_tol=0.015,
    ),
    RvsCase(
        id="katojones",
        factory=katojones,
        params={"mu": 0.7, "gamma": 0.5, "rho": 0.25, "lam": 1.2},
        size=512,
        seed=2025,
        uniform_tol=0.01,
    ),
]


@pytest.mark.parametrize("case", REFERENCE_VALUES, ids=lambda case: case.id)
def test_distribution_reference_values(case):
    dist = case.factory(**case.params)
    method = getattr(dist, case.method)
    result = float(np.asarray(method(case.arg)))
    np.testing.assert_allclose(result, case.expected, atol=case.atol, rtol=0.0)


@pytest.mark.parametrize("case", CDF_PPF_CASES, ids=lambda case: case.id)
def test_distribution_cdf_ppf_consistency(case):
    dist = case.dist()
    theta = np.linspace(0.0, 2.0 * np.pi, case.theta_points)
    q = np.linspace(case.q_min, 1.0 - case.q_min, case.q_points)
    _assert_monotonic_cdf_ppf(
        dist,
        theta,
        q,
        cdf_tol=case.cdf_tol,
        ppf_tol=case.ppf_tol,
    )

    theta_roundtrip = dist.ppf(q)
    q_back = _evaluate_array(case.factory.cdf, theta_roundtrip, **case.params)
    np.testing.assert_allclose(
        q_back,
        q,
        atol=max(case.cdf_tol * 50, 1e-12),
        rtol=0.0,
    )

    q_from_theta = _evaluate_array(case.factory.cdf, theta, **case.params)
    theta_back = dist.ppf(q_from_theta)
    wrapped = np.mod(theta_back - theta + np.pi, 2.0 * np.pi) - np.pi
    pdf_vals = _evaluate_array(case.factory.pdf, theta, **case.params)

    default_high_tol = (
        case.ppf_high_slope_tol
        if case.ppf_high_slope_tol is not None
        else max(case.ppf_tol * 50, 5e-8)
    )
    default_low_tol = (
        case.ppf_low_slope_tol
        if case.ppf_low_slope_tol is not None
        else default_high_tol
    )

    if case.ppf_slope_threshold > 0.0:
        high_slope = pdf_vals > case.ppf_slope_threshold
        if np.any(high_slope):
            np.testing.assert_allclose(
                wrapped[high_slope],
                0.0,
                atol=default_high_tol,
                rtol=0.0,
            )
        if np.any(~high_slope):
            np.testing.assert_allclose(
                wrapped[~high_slope],
                0.0,
                atol=default_low_tol,
                rtol=0.0,
            )
    else:
        np.testing.assert_allclose(
            wrapped,
            0.0,
            atol=default_high_tol,
            rtol=0.0,
        )

    for endpoint, expected in ((0.0, 0.0), (1.0, 2.0 * np.pi)):
        try:
            value = float(dist.ppf(endpoint))
        except Exception:
            continue
        if np.isfinite(value):
            np.testing.assert_allclose(
                value,
                expected,
                atol=max(case.ppf_tol * 50, 1e-8),
                rtol=0.0,
            )


def _check_textbook_reference(
    case_id: str, *, rounding: Optional[int] = None, significant: Optional[int] = None
):
    case = _REFERENCE_LOOKUP[case_id]
    dist = case.factory(**case.params)
    method = getattr(dist, case.method)
    value = float(np.asarray(method(case.arg)))
    expected = float(case.expected)
    if rounding is not None:
        value = np.round(value, rounding)
        expected = np.round(expected, rounding)
    if significant is not None:
        np.testing.assert_approx_equal(value, expected, significant=significant)
    else:
        np.testing.assert_allclose(value, expected, atol=case.atol, rtol=0.0)


# Textbook value checks retained for readability and regression safety. These reference
# published tables, so we mirror the original significant-digit comparisons.
def test_circularuniform_textbook_values():
    _check_textbook_reference("circularuniform-cdf", significant=5)
    _check_textbook_reference("circularuniform-ppf", significant=12)


def test_cardioid_textbook_values():
    _check_textbook_reference("cardioid-cdf", significant=5)
    _check_textbook_reference("cardioid-ppf", significant=5)


def test_cartwright_textbook_values():
    _check_textbook_reference("cartwright-cdf", rounding=4, significant=5)
    _check_textbook_reference("cartwright-ppf", rounding=5, significant=5)


def test_wrapcauchy_textbook_values():
    _check_textbook_reference("wrapcauchy-cdf", rounding=3, significant=3)
    _check_textbook_reference("wrapcauchy-ppf", rounding=3, significant=3)


def test_wrapnorm_textbook_values():
    _check_textbook_reference("wrapnorm-cdf", rounding=4, significant=3)
    _check_textbook_reference("wrapnorm-ppf", rounding=4, significant=4)


def test_vonmises_textbook_values():
    _check_textbook_reference("vonmises-cdf", rounding=4, significant=3)
    _check_textbook_reference("vonmises-ppf", rounding=4, significant=4)


def test_vonmises_flattopped_textbook_values():
    _check_textbook_reference("vonmises-flattopped-cdf", rounding=4, significant=4)
    _check_textbook_reference("vonmises-flattopped-ppf", rounding=4, significant=4)


def test_jonespewsey_textbook_values():
    _check_textbook_reference("jonespewsey-cdf", rounding=7, significant=7)
    _check_textbook_reference("jonespewsey-ppf", significant=7)


def test_jonespewsey_sineskewed_textbook_values():
    _check_textbook_reference("jonespewsey-sineskewed-cdf", rounding=4, significant=4)
    _check_textbook_reference("jonespewsey-sineskewed-ppf", rounding=4, significant=4)


def test_jonespewsey_asym_textbook_values():
    _check_textbook_reference("jonespewsey-asym-cdf", rounding=4, significant=4)
    _check_textbook_reference("jonespewsey-asym-ppf", rounding=4, significant=4)


def test_inverse_batschelet_textbook_values():
    _check_textbook_reference("inverse-batschelet-cdf", rounding=4, significant=4)
    _check_textbook_reference("inverse-batschelet-ppf", rounding=4, significant=4)


@pytest.mark.parametrize("case", CDF_FROM_PDF_CASES, ids=lambda case: case.id)
def test_distribution_cdf_matches_numeric(case):
    theta = np.linspace(0.0, 2.0 * np.pi, case.theta_points)
    analytic = case.cdf(theta, *case.args)
    numeric = case.numeric_cdf(theta, *case.args)
    np.testing.assert_allclose(analytic, numeric, atol=case.atol, rtol=1e-6)
    diffs = np.diff(analytic)
    assert np.all(diffs >= -1e-10)


@pytest.mark.parametrize("case", RVS_CASES, ids=lambda case: case.id)
def test_distribution_rvs_pit(case):
    def cdf_callable(values):
        return _evaluate_array(case.factory.cdf, values, **case.params)

    _assert_rvs_reasonable(
        case.dist(),
        size=case.size,
        seed=case.seed,
        uniform_tol=case.uniform_tol,
        cdf_callable=cdf_callable,
    )


def test_circularuniform_descriptive_stats():
    dist = circularuniform()
    stats_dict = dist.stats()

    assert np.isnan(dist.mean())
    assert np.isnan(stats_dict["mean"])
    np.testing.assert_allclose(dist.r(), 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(stats_dict["r"], 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(dist.var(), 1.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(stats_dict["var"], 1.0, atol=1e-12, rtol=0.0)
    assert np.isinf(dist.std())
    assert np.isinf(stats_dict["std"])
    assert np.isinf(stats_dict["dispersion"])
    np.testing.assert_allclose(stats_dict["skewness"], 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(stats_dict["kurtosis"], 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(dist.median(), np.pi, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(stats_dict["median"], np.pi, atol=1e-12, rtol=0.0)


def test_vonmises_descriptive_stats_consistency():
    mu_true, kappa_true = 1.2, 3.4
    frozen = vonmises(mu=mu_true, kappa=kappa_true)
    generator_stats = vonmises.stats(mu=mu_true, kappa=kappa_true)
    expected_r = special.i1(kappa_true) / special.i0(kappa_true)
    expected_m2 = (
        special.iv(2, kappa_true) / special.i0(kappa_true) * np.exp(2j * mu_true)
    )

    np.testing.assert_allclose(frozen.r(), expected_r, atol=5e-12, rtol=0.0)
    np.testing.assert_allclose(frozen.mean(), mu_true, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(frozen.var(), 1.0 - expected_r, atol=1e-12, rtol=0.0)
    frozen_stats = frozen.stats()
    for key, value in generator_stats.items():
        frozen_value = frozen_stats[key]
        if np.isnan(value):
            assert np.isnan(frozen_value)
        else:
            np.testing.assert_allclose(frozen_value, value, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(frozen.trig_moment(2), expected_m2, atol=5e-12, rtol=0.0)


@pytest.mark.parametrize(
    "rho_true, seed",
    [
        (0.0, 101),
        (0.15, 102),
        (4.0 / np.pi**2 - 1e-4, 103),
    ],
)
def test_triangular_fit_recovers_rho(rho_true, seed):
    rng = np.random.default_rng(seed)
    data = triangular.rvs(rho=rho_true, size=2000, random_state=rng)

    rho_mle, info = triangular.fit(data, method="mle", return_info=True)
    assert info["converged"]
    np.testing.assert_allclose(rho_mle, rho_true, atol=7e-3, rtol=0.0)

    rho_mom = triangular.fit(data, method="moments")
    np.testing.assert_allclose(rho_mom, rho_true, atol=7e-3, rtol=0.0)


def test_wrapcauchy_fit_weights_matches_replication():
    rng = np.random.default_rng(321)
    mu_true, rho_true = 0.9, 0.6
    base = wrapcauchy.rvs(mu=mu_true, rho=rho_true, size=180, random_state=rng)
    weights = np.full(base.shape, 5.0)
    replicated = np.repeat(base, 5)

    params_weighted = wrapcauchy.fit(base, method="mle", weights=weights)
    params_replicated = wrapcauchy.fit(replicated, method="mle")

    mu_weighted, rho_weighted = params_weighted
    mu_replicated, rho_replicated = params_replicated

    mu_diff = np.mod(mu_weighted - mu_replicated + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(mu_diff, 0.0, atol=5e-4, rtol=0.0)
    np.testing.assert_allclose(rho_weighted, rho_replicated, atol=5e-4, rtol=0.0)


@pytest.mark.parametrize(
    "dist, params",
    [
        (
            cardioid,
            {"mu": 0.7, "rho": 0.3},
        ),
        (
            cartwright,
            {"mu": 0.25 * np.pi, "zeta": 1.2},
        ),
        (
            wrapnorm,
            {"mu": 1.1, "rho": 0.5},
        ),
        (
            jonespewsey,
            {"mu": 0.6, "kappa": 1.3, "psi": -0.7},
        ),
        (
            inverse_batschelet,
            {"xi": 0.9, "kappa": 2.2, "nu": -0.35, "lmbd": 0.4},
        ),
    ],
)
def test_pdf_integrates_to_one(dist, params):
    theta = np.linspace(0.0, 2.0 * np.pi, 4097)
    pdf_vals = dist.pdf(theta, **params)
    area = np.trapezoid(pdf_vals, theta)
    np.testing.assert_allclose(area, 1.0, atol=5e-6, rtol=0.0)


@pytest.mark.parametrize(
    "dist, params",
    [
        (vonmises, {"mu": 0.6, "kappa": 4.0}),
        (wrapcauchy, {"mu": 1.1, "rho": 0.7}),
        (cartwright, {"mu": 0.3, "zeta": 1.5}),
    ],
)
def test_logpdf_matches_log_of_pdf(dist, params):
    theta = np.linspace(0.0, 2.0 * np.pi, 129, endpoint=False) + 1e-6
    pdf_vals = dist.pdf(theta, **params)
    logpdf_vals = dist.logpdf(theta, **params)

    assert np.all(np.isfinite(logpdf_vals))
    mask = pdf_vals > 0.0
    np.testing.assert_allclose(
        logpdf_vals[mask], np.log(pdf_vals[mask]), atol=5e-10, rtol=0.0
    )


def test_vonmises_random_state_reproducibility():
    params = {"mu": 1.05, "kappa": 2.5}

    seq_a = vonmises.rvs(size=6, random_state=1234, **params)
    seq_b = vonmises.rvs(size=6, random_state=1234, **params)
    np.testing.assert_allclose(seq_a, seq_b)

    seq_c = vonmises.rvs(size=6, random_state=np.random.default_rng(5678), **params)
    seq_d = vonmises.rvs(size=6, random_state=np.random.default_rng(5678), **params)
    np.testing.assert_allclose(seq_c, seq_d)

    seq_e = vonmises.rvs(size=6, random_state=np.random.RandomState(5678), **params)
    seq_f = vonmises.rvs(size=6, random_state=np.random.RandomState(5678), **params)
    np.testing.assert_allclose(seq_e, seq_f)


class _LongDefaultRandomState(np.random.RandomState):
    """``RandomState`` whose ``randint`` defaults to int32, as on Windows.

    ``np.dtype(int)`` is a 32-bit C long there, so any ``randint(0, 2**32)``
    call that relies on the default dtype raises "high is out of bounds".
    """

    def randint(self, low, high=None, size=None, dtype=int):
        if dtype is int:
            dtype = np.int32
        return super().randint(low, high, size, dtype=dtype)


RVS_PARAMS = [
    (circularuniform, {}),
    (triangular, {"rho": 0.2}),
    (cardioid, {"mu": 1.0, "rho": 0.3}),
    (cartwright, {"mu": 0.3, "zeta": 1.5}),
    (wrapnorm, {"mu": 0.8, "rho": 0.4}),
    (wrapcauchy, {"mu": 1.1, "rho": 0.7}),
    (vonmises, {"mu": 0.0, "kappa": 1.0}),
    (vonmises_flattopped, {"mu": 0.5, "kappa": 2.0, "nu": 0.2}),
    (jonespewsey, {"mu": 0.5, "kappa": 2.0, "psi": 0.5}),
    (jonespewsey_sineskewed, {"xi": 0.5, "kappa": 2.0, "psi": 0.5, "lmbd": 0.3}),
    (jonespewsey_asym, {"xi": 0.5, "kappa": 2.0, "psi": 0.5, "nu": 0.3}),
    (inverse_batschelet, {"xi": 0.9, "kappa": 2.2, "nu": -0.35, "lmbd": 0.4}),
    (wrapstable, {"delta": 0.0, "alpha": 1.5, "beta": 0.0, "gamma": 1.0}),
    (katojones, {"mu": 0.0, "gamma": 0.5, "rho": 0.3, "lam": 0.2}),
]


@pytest.mark.parametrize("dist, params", RVS_PARAMS)
def test_rvs_default_random_state_on_32bit_long(dist, params, monkeypatch):
    """Default ``random_state=None`` must not seed through a platform-sized int.

    Regression test for issue #22: SciPy caches NumPy's global ``RandomState``
    on every distribution, and drawing a 32-bit seed from it failed on Windows.
    """
    monkeypatch.setattr(dist, "_random_state", _LongDefaultRandomState(0), raising=False)

    samples = dist.rvs(size=5, random_state=None, **params)

    assert samples.shape == (5,)
    assert np.all(np.isfinite(samples))


def test_rvs_default_random_state_on_32bit_long_projectednormal(monkeypatch):
    from pycircstat2.distributions import projectednormal as _pn

    monkeypatch.setattr(_pn, "_random_state", _LongDefaultRandomState(0), raising=False)

    samples = _pn.rvs(mu1=1.0, mu2=1.0, size=5, random_state=None)

    assert samples.shape == (5,)
    assert np.all(np.isfinite(samples))


def test_rvs_default_random_state_follows_global_seed(monkeypatch):
    """``random_state=None`` must keep deferring to the distribution's own state.

    Caching a fresh ``Generator`` on the (module-level, shared) distribution
    would silently detach it from ``np.random.seed`` after the first call.
    """
    params = {"mu": 1.05, "kappa": 2.5}
    rs = np.random.RandomState(2046)
    monkeypatch.setattr(vonmises, "_random_state", rs)

    seq_a = vonmises.rvs(size=6, random_state=None, **params)
    rs.seed(2046)
    seq_b = vonmises.rvs(size=6, random_state=None, **params)
    np.testing.assert_allclose(seq_a, seq_b)

    seq_c = vonmises.rvs(size=6, random_state=None, **params)
    assert not np.allclose(seq_b, seq_c)

    assert vonmises._random_state is rs


@pytest.mark.parametrize(
    "dist, params",
    [
        (wrapnorm, {"mu": 0.8, "rho": 0.4}),
        (cardioid, {"mu": 1.2, "rho": 0.25}),
        (triangular, {"rho": 0.2}),
    ],
)
def test_rvs_output_shapes(dist, params):
    scalar = dist.rvs(random_state=42, **params)
    assert np.isscalar(scalar)

    array = dist.rvs(size=(3, 2), random_state=42, **params)
    assert array.shape == (3, 2)

    empty = dist.rvs(size=0, random_state=42, **params)
    assert empty.shape == (0,)


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


def test_vonmises_flattopped_fit_recovers_parameters():
    mu_true, kappa_true, nu_true = 1.1, 4.0, -0.25
    rng = np.random.default_rng(2024)
    sample = vonmises_flattopped.rvs(
        mu=mu_true, kappa=kappa_true, nu=nu_true, size=6000, random_state=rng
    )

    estimates, info = vonmises_flattopped.fit(sample, method="mle", return_info=True)
    assert info["converged"]

    mu_hat, kappa_hat, nu_hat = estimates
    mu_diff = np.mod(mu_hat - mu_true + np.pi, 2.0 * np.pi) - np.pi
    np.testing.assert_allclose(mu_diff, 0.0, atol=5e-2)
    np.testing.assert_allclose(kappa_hat, kappa_true, atol=0.6)
    np.testing.assert_allclose(nu_hat, nu_true, atol=0.08)

    moments = vonmises_flattopped.fit(sample, method="moments")
    assert moments[2] == 0.0
    np.testing.assert_allclose(
        np.mod(moments[0] - mu_true + np.pi, 2.0 * np.pi) - np.pi, 0.0, atol=1e-1
    )


def test_vonmises_fit_wraps_data():
    data = np.array([-0.8, 0.2, 6.6, 7.1, -3.0])

    mu_expected, kappa_expected = vonmises.fit(
        np.mod(data, 2 * np.pi), method="analytical"
    )
    mu_actual, kappa_actual = vonmises.fit(data, method="analytical")

    diff = np.mod(mu_actual - mu_expected + np.pi, 2 * np.pi) - np.pi
    np.testing.assert_allclose(diff, 0.0, atol=1e-8)
    np.testing.assert_allclose(kappa_actual, kappa_expected, atol=1e-8)


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


def test_inverse_batschelet_warps_match_brentq():
    """The vectorized monotone solver (`_tnu`/`_slmbdinv`) must agree with a
    per-point `brentq` inversion to ~machine precision, including the
    ν→±1 / λ→±1 near-boundary regime where the warp slope → 0. `brentq` is the
    reference the fast path is allowed to be fast against."""
    from scipy.optimize import root_scalar

    from pycircstat2.distributions import _slmbdinv, _tnu

    phi = np.linspace(-np.pi, np.pi, 257)[:-1]

    def ref_tnu(x, nu):
        out = np.empty_like(x)
        for i, p in enumerate(x):
            s = root_scalar(lambda y: y - nu * (1.0 + np.cos(y)) - p,
                            bracket=(-np.pi, np.pi), method="brentq")
            out[i] = (s.root + np.pi) % (2.0 * np.pi) - np.pi
        return out

    def ref_slmbd(x, lmbd):
        out = np.empty_like(x)
        for i, v in enumerate(x):
            s = root_scalar(lambda u: u - 0.5 * (1.0 + lmbd) * np.sin(u) - v,
                            bracket=(-np.pi, np.pi), method="brentq")
            out[i] = (s.root + np.pi) % (2.0 * np.pi) - np.pi
        return out

    for nu in (-0.97, -0.5, -1e-3, 0.3, 0.97):
        np.testing.assert_allclose(_tnu(phi, nu, 0.0), ref_tnu(phi, nu),
                                   atol=1e-10, rtol=0.0)
    for lmbd in (-0.97, -0.5, 0.0, 0.5, 0.97):
        np.testing.assert_allclose(_slmbdinv(phi, lmbd), ref_slmbd(phi, lmbd),
                                   atol=1e-10, rtol=0.0)

    # scalar inputs return floats (the descriptive contract)
    assert isinstance(_tnu(1.3, 0.4, 0.2), float)
    assert isinstance(_slmbdinv(0.7, 0.5), float)


def test_inverse_batschelet_log_c_array_matches_scalar():
    """The vectorized normalizer `_invbat_log_c_array` (the regression-path
    value + the base for the FD'd log-c gradient/Hessian) must reproduce the
    tested scalar `_c_invbatschelet` to machine precision on the full value
    grid, across the interior (κ>0, |λ|<1) the log/tanh links guarantee — and
    fall back to the scalar at the κ≈0 / |λ|≈1 edges. This pins the per-pair
    Python loop the vectorization replaced (the κ(x)/λ(x) fit hotspot)."""
    from pycircstat2.distributions import (
        _INVBAT_NUMERIC_GRID,
        _c_invbatschelet,
        _invbat_log_c_array,
    )

    kk = np.array([0.05, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 100.0, 400.0, 700.0])
    ll = np.array([-0.95, -0.7, -0.3, -0.05, 0.0, 0.05, 0.3, 0.7, 0.95])
    K, L = np.meshgrid(kk, ll)
    k, lam = K.ravel(), L.ravel()

    ref = np.array([np.log(_c_invbatschelet(float(a), float(b)))
                    for a, b in zip(k, lam)])
    vec = _invbat_log_c_array(k, lam, grid_size=_INVBAT_NUMERIC_GRID)
    # interior pairs are computed by the vectorized assembly itself
    np.testing.assert_allclose(vec, ref, atol=1e-12, rtol=0.0)
    # broadcasting preserves the input shape
    assert _invbat_log_c_array(K, L, grid_size=_INVBAT_NUMERIC_GRID).shape == K.shape

    # κ≈0 and |λ|≈1 edges defer to the scalar exact limits
    edge_k = np.array([1e-12, 1e-12])
    edge_l = np.array([0.3, -0.3])
    np.testing.assert_allclose(
        _invbat_log_c_array(edge_k, edge_l, grid_size=_INVBAT_NUMERIC_GRID),
        -np.log(2.0 * np.pi), atol=1e-12, rtol=0.0,
    )
    for lam in (1.0 - 1e-13, -1.0 + 1e-13):
        got = float(_invbat_log_c_array(np.array([3.0]), np.array([lam]),
                                        grid_size=_INVBAT_NUMERIC_GRID)[0])
        assert np.isfinite(got)
        assert got == pytest.approx(np.log(_c_invbatschelet(3.0, lam)), abs=1e-12)


@pytest.mark.parametrize(
    "params",
    [
        dict(xi=2.0, kappa=2.0, nu=0.2, lmbd=0.2),
        dict(xi=1.0, kappa=6.0, nu=-0.5, lmbd=0.5),
        dict(xi=3.5, kappa=1.0, nu=0.4, lmbd=-0.6),
        dict(xi=0.7, kappa=10.0, nu=-0.3, lmbd=-0.3),
    ],
)
def test_inverse_batschelet_dlogpdf_matches_finite_difference(params):
    """l1: the regression-overlay score (`ibslss`) vs central differences of
    `logpdf`, w.r.t. each parameter. The ξ/ν entries are fully analytic
    (implicit differentiation of the two warps); κ/λ carry the FD'd normalizer
    gradient."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, 12))
    ana = inverse_batschelet.dlogpdf(x, **params)

    def fd(name, h):
        hi, lo = dict(params), dict(params)
        hi[name] += h
        lo[name] -= h
        return (inverse_batschelet.logpdf(x, **hi)
                - inverse_batschelet.logpdf(x, **lo)) / (2.0 * h)

    for name in ("xi", "kappa", "nu", "lmbd"):
        h = 1e-5 if name == "kappa" else 1e-6
        np.testing.assert_allclose(ana[name], fd(name, h), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    "params",
    [
        dict(xi=2.0, kappa=2.0, nu=0.2, lmbd=0.2),
        dict(xi=1.0, kappa=6.0, nu=-0.5, lmbd=0.5),
        dict(xi=3.5, kappa=1.0, nu=0.4, lmbd=-0.6),
    ],
)
def test_inverse_batschelet_d2logpdf_kernel_block_matches_fd(params):
    """l2 kernel block (`ibslss`): the seven second partials that do *not*
    touch the normalizer (every pair except κκ/κλ/λλ, since c ⊥ ξ,ν) vs
    central differences of `dlogpdf`. These are FD of the analytic kernel
    gradient, so they match tightly; the κ,λ normalizer block is EFS-grade
    (direct second differences) and is gated end-to-end by intercept-only
    parity in test_regression.py, not here."""
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, 10))
    H = inverse_batschelet.d2logpdf(x, **params)

    def fd(a, b, h):
        hi, lo = dict(params), dict(params)
        hi[b] += h
        lo[b] -= h
        return (inverse_batschelet.dlogpdf(x, **hi)[a]
                - inverse_batschelet.dlogpdf(x, **lo)[a]) / (2.0 * h)

    kernel_pairs = [("xi", "xi"), ("xi", "kappa"), ("xi", "nu"),
                    ("xi", "lmbd"), ("kappa", "nu"), ("nu", "nu"),
                    ("nu", "lmbd")]
    for a, b in kernel_pairs:
        h = 1e-5 if b == "kappa" else 1e-6
        np.testing.assert_allclose(H[(a, b)], fd(a, b, h), atol=1e-4, rtol=0.0)


@pytest.mark.parametrize(
    "params",
    [
        dict(mu=2.0, kappa=2.0, nu=0.3),
        dict(mu=1.0, kappa=6.0, nu=-0.5),
        dict(mu=3.5, kappa=1.0, nu=0.7),
        dict(mu=0.7, kappa=10.0, nu=-0.8),
    ],
)
def test_vmft_dlogpdf_matches_finite_difference(params):
    """l1 of the flat-topped vM *lss (`vmftlss`): the analytic score vs central
    differences of `logpdf`. ξ is the pure forward-warp kernel term; κ/ν carry
    the grid-expectation normalizer terms (`_vmft_logZ_moments_vec`)."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, 12))
    ana = vonmises_flattopped.dlogpdf(x, **params)

    def fd(name, h):
        hi, lo = dict(params), dict(params)
        hi[name] += h
        lo[name] -= h
        return (vonmises_flattopped.logpdf(x, **hi)
                - vonmises_flattopped.logpdf(x, **lo)) / (2.0 * h)

    for name in params:
        np.testing.assert_allclose(ana[name], fd(name, 1e-6),
                                   atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "params",
    [
        dict(mu=2.0, kappa=2.0, nu=0.3),
        dict(mu=1.0, kappa=6.0, nu=-0.5),
        dict(mu=3.5, kappa=1.0, nu=0.7),
    ],
)
def test_vmft_d2logpdf_matches_finite_difference(params):
    """l2 of `vmftlss`: every unique unordered pair vs central differences of
    `dlogpdf` (symmetrized). Unlike ibslss the normalizer block is analytic
    here (grid moments + Cov identity), so all six pairs — including κκ/κν/νν —
    match tightly, not just the kernel ones."""
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, 10))
    H = vonmises_flattopped.d2logpdf(x, **params)
    names = list(params)

    def dfd(name, h):
        hi, lo = dict(params), dict(params)
        hi[name] += h
        lo[name] -= h
        d_hi = vonmises_flattopped.dlogpdf(x, **hi)
        d_lo = vonmises_flattopped.dlogpdf(x, **lo)
        return {n: (d_hi[n] - d_lo[n]) / (2.0 * h) for n in names}

    fd = {n: dfd(n, 1e-5) for n in names}
    for a, b in cwr(names, 2):
        ref = 0.5 * (fd[a][b] + fd[b][a])
        np.testing.assert_allclose(H[(a, b)], ref, atol=1e-4, rtol=0.0)


def test_vmft_reduces_to_vonmises_at_nu0():
    """At ν=0 the forward warp B=φ collapses, so `vmftlss` *is* `vmlss`: the
    μ/κ score and Hessian blocks must equal von Mises to machine precision —
    a strong, reference-backed check on the grid-expectation normalizer
    (E[cos B]→A₁(κ), Var[cos B]→A₁′(κ))."""
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, 12))
    mu, kappa = 1.3, 3.0
    dv = vonmises_flattopped.dlogpdf(x, mu=mu, kappa=kappa, nu=0.0)
    vv = vonmises.dlogpdf(x, mu=mu, kappa=kappa)
    np.testing.assert_allclose(dv["mu"], vv["mu"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(dv["kappa"], vv["kappa"], atol=1e-12, rtol=0.0)
    H = vonmises_flattopped.d2logpdf(x, mu=mu, kappa=kappa, nu=0.0)
    Hv = vonmises.d2logpdf(x, mu=mu, kappa=kappa)
    for pair in (("mu", "mu"), ("mu", "kappa"), ("kappa", "kappa")):
        np.testing.assert_allclose(H[pair], Hv[pair], atol=1e-12, rtol=0.0)


def test_vmft_log_c_vec_matches_table():
    """The vectorized normalizer `_vmft_log_c_vec` (the regression-path value +
    the base for the grid moments) must reproduce the cached scalar table's
    `log_normalizer` to machine precision at the same grid, and route κ≈0 to
    the uniform constant."""
    from pycircstat2.distributions import (
        _vmft_build_table,
        _vmft_grid_size,
        _vmft_log_c_vec,
    )

    for k in (0.3, 1.0, 3.0, 8.0, 30.0):
        for nu in (-0.8, -0.3, 0.0, 0.4, 0.9):
            table = _vmft_build_table(float(k), float(nu),
                                      _vmft_grid_size(float(k), float(nu)))
            vec = float(_vmft_log_c_vec(np.array([k]), np.array([nu]))[0])
            assert vec == pytest.approx(table["log_normalizer"], abs=1e-12)
    # κ ≈ 0 → uniform; broadcasting preserves shape
    got = _vmft_log_c_vec(np.array([1e-12, 2.0]), np.array([0.3, 0.5]))
    assert got[0] == pytest.approx(-np.log(2.0 * np.pi), abs=1e-12)
    assert _vmft_log_c_vec(np.zeros((2, 3)), np.zeros((2, 3))).shape == (2, 3)


@pytest.mark.parametrize(
    "params",
    [
        dict(xi=2.0, kappa=2.0, psi=0.5, nu=0.3),
        dict(xi=1.0, kappa=5.0, psi=-0.6, nu=-0.4),
        dict(xi=3.5, kappa=1.0, psi=1.2, nu=0.6),
        dict(xi=0.7, kappa=3.0, psi=0.0, nu=-0.7),
    ],
)
def test_ajp_dlogpdf_matches_finite_difference(params):
    """l1 of the asymmetric-extended JP *lss (`ajplss`): the analytic score vs
    central differences of `logpdf`. ξ is the JP kernel chained through the
    forward warp g=φ+ν cosφ; κ/ψ/ν carry the (κ,ψ,ν) grid-expectation
    normalizer terms (`_jp_logZ_moments_asym_vec`)."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, 12))
    ana = jonespewsey_asym.dlogpdf(x, **params)

    def fd(name, h):
        hi, lo = dict(params), dict(params)
        hi[name] += h
        lo[name] -= h
        return (jonespewsey_asym.logpdf(x, **hi)
                - jonespewsey_asym.logpdf(x, **lo)) / (2.0 * h)

    for name in params:
        np.testing.assert_allclose(ana[name], fd(name, 1e-6),
                                   atol=1e-6, rtol=0.0)


@pytest.mark.parametrize(
    "params",
    [
        dict(xi=2.0, kappa=2.0, psi=0.5, nu=0.3),
        dict(xi=3.5, kappa=1.0, psi=1.2, nu=0.6),
        dict(xi=0.7, kappa=3.0, psi=0.0, nu=-0.7),
    ],
)
def test_ajp_d2logpdf_matches_finite_difference(params):
    """l2 of `ajplss`: every unique unordered pair vs central differences of
    `dlogpdf` (symmetrized). Exercises the warp chain rule in the ξ blocks and
    the analytic (κ,ψ,ν) normalizer Hessian (grid moments + Cov identity)."""
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, 10))
    H = jonespewsey_asym.d2logpdf(x, **params)
    names = list(params)

    def dfd(name, h):
        hi, lo = dict(params), dict(params)
        hi[name] += h
        lo[name] -= h
        d_hi = jonespewsey_asym.dlogpdf(x, **hi)
        d_lo = jonespewsey_asym.dlogpdf(x, **lo)
        return {n: (d_hi[n] - d_lo[n]) / (2.0 * h) for n in names}

    fd = {n: dfd(n, 1e-5) for n in names}
    for a, b in cwr(names, 2):
        ref = 0.5 * (fd[a][b] + fd[b][a])
        np.testing.assert_allclose(H[(a, b)], ref, atol=2e-4, rtol=0.0)


def test_ajp_dlogpdf_array_params_matches_fd():
    """The regression array path: with a *distinct* (ξ, κ, ψ, ν) per datum the
    score must still match central differences. Exercises `_ajp_logpdf_vec`
    and the multi-triple `_jp_logZ_moments_asym_vec` (the path a constant-param
    fit never reaches) cheaply — the fast guard standing in for a slow
    distributional recover-the-truth fit."""
    rng = np.random.default_rng(4)
    n = 8
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
    P = dict(xi=rng.uniform(0.5, 2.5, n), kappa=rng.uniform(1.0, 5.0, n),
             psi=rng.uniform(-0.8, 1.0, n), nu=rng.uniform(-0.6, 0.6, n))
    ana = jonespewsey_asym.dlogpdf(x, **P)

    def fd(name, h):
        hi = {k: (v if k != name else v + h) for k, v in P.items()}
        lo = {k: (v if k != name else v - h) for k, v in P.items()}
        return (jonespewsey_asym.logpdf(x, **hi)
                - jonespewsey_asym.logpdf(x, **lo)) / (2.0 * h)

    for name in P:
        np.testing.assert_allclose(ana[name], fd(name, 1e-6),
                                   atol=1e-6, rtol=0.0)


def test_ajp_reduces_to_jonespewsey_at_nu0():
    """At ν=0 the forward warp g=φ collapses, so `ajplss` *is* `jplss`: the
    ξ/κ/ψ score and Hessian blocks must equal symmetric Jones–Pewsey to
    machine precision — a strong, reference-backed check on the warp chain
    rule and the (κ,ψ,ν) normalizer moments (the ν-block's E[h_φ cosφ]→0 by
    oddness)."""
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, 12))
    for mu, kappa, psi in [(1.3, 3.0, 0.5), (0.7, 2.0, -0.7)]:
        da = jonespewsey_asym.dlogpdf(x, xi=mu, kappa=kappa, psi=psi, nu=0.0)
        dj = jonespewsey.dlogpdf(x, mu=mu, kappa=kappa, psi=psi)
        np.testing.assert_allclose(da["xi"], dj["mu"], atol=1e-11, rtol=0.0)
        np.testing.assert_allclose(da["kappa"], dj["kappa"], atol=1e-11, rtol=0.0)
        np.testing.assert_allclose(da["psi"], dj["psi"], atol=1e-11, rtol=0.0)
        Ha = jonespewsey_asym.d2logpdf(x, xi=mu, kappa=kappa, psi=psi, nu=0.0)
        Hj = jonespewsey.d2logpdf(x, mu=mu, kappa=kappa, psi=psi)
        for a, b in (("xi", "xi"), ("xi", "kappa"), ("xi", "psi"),
                     ("kappa", "kappa"), ("kappa", "psi"), ("psi", "psi")):
            jb = tuple("mu" if t == "xi" else t for t in (a, b))
            np.testing.assert_allclose(Ha[(a, b)], Hj[jb], atol=1e-11, rtol=0.0)


def test_jp_log_c_asym_vec_matches_scalar():
    """The vectorized normalizer `_jp_log_c_asym_vec` (the regression-path
    value) must reproduce the lru-cached scalar `_jp_log_c_asym` exactly over
    a (κ,ψ,ν) grid, and route κ≈0 to the uniform constant."""
    from pycircstat2.distributions import _jp_log_c_asym, _jp_log_c_asym_vec

    ks, ps, ns = [], [], []
    for k in (0.5, 2.0, 6.0):
        for p in (-0.8, 0.0, 1.0):
            for nu in (-0.6, -0.2, 0.3, 0.7):
                ks.append(k)
                ps.append(p)
                ns.append(nu)
    ka, pa, na = np.array(ks), np.array(ps), np.array(ns)
    ref = np.array([_jp_log_c_asym(float(k), float(p), float(n))
                    for k, p, n in zip(ka, pa, na)])
    np.testing.assert_allclose(_jp_log_c_asym_vec(ka, pa, na), ref,
                               atol=1e-12, rtol=0.0)
    got = _jp_log_c_asym_vec(np.array([1e-12, 2.0]), np.array([0.5, 0.5]),
                             np.array([0.3, 0.3]))
    assert got[0] == pytest.approx(-np.log(2.0 * np.pi), abs=1e-12)


def test_inverse_batschelet_fit_moments():
    samples = inverse_batschelet.rvs(
        xi=1.1, kappa=3.0, nu=0.2, lmbd=-0.3, size=600, random_state=123
    )
    xi_hat, kappa_hat, nu_hat, lmbd_hat = inverse_batschelet.fit(
        samples, method="moments"
    )
    np.testing.assert_allclose(
        np.mod(xi_hat - 1.1 + np.pi, 2.0 * np.pi) - np.pi, 0.0, atol=0.3
    )
    assert nu_hat == 0.0
    assert lmbd_hat == 0.0
    assert kappa_hat >= 0.0


def test_inverse_batschelet_fit_mle():
    rng = np.random.default_rng(246)
    xi_true, kappa_true, nu_true, lmbd_true = 0.8, 2.5, -0.25, 0.4
    data = inverse_batschelet.rvs(
        xi=xi_true,
        kappa=kappa_true,
        nu=nu_true,
        lmbd=lmbd_true,
        size=800,
        random_state=rng,
    )

    (xi_hat, kappa_hat, nu_hat, lmbd_hat), info = inverse_batschelet.fit(
        data,
        method="mle",
        return_info=True,
        options={"maxiter": 200},
    )

    assert info["converged"]
    np.testing.assert_allclose(
        np.mod(xi_hat - xi_true + np.pi, 2.0 * np.pi) - np.pi, 0.0, atol=0.2
    )
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
    rho = np.exp(-(gamma**2))
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
    rho_vals, mu_vals, p = wrapstable._get_series_terms(
        delta=0.0, alpha=1.6, beta=0.1, gamma=0.02
    )
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
    samples = wrapstable.rvs(
        delta=delta, alpha=2.0, beta=0.0, gamma=gamma, size=2000, random_state=rng
    )
    rho = np.exp(-(gamma**2))
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


def _assert_rvs_reasonable(
    dist, size=256, seed=123, uniform_tol=0.05, cdf_callable=None
):
    rng = np.random.default_rng(seed)
    samples = dist.rvs(size=size, random_state=rng)
    samples = np.asarray(samples, dtype=float)
    assert samples.size == size

    if cdf_callable is None:
        u = dist.cdf(samples)
    else:
        u = cdf_callable(samples)
    u = np.mod(u, 1.0)
    stat, pvalue = stats.kstest(u, "uniform")
    assert pvalue > uniform_tol, f"kstest failed: statistic={stat}, p={pvalue}"


def _check_pdf_normalizes(dist, params=None, atol=1e-6, limit=400):
    if params is None:
        def integrand(t):
            return dist.pdf(t)
    else:
        def integrand(t):
            return dist.pdf(t, *params)

    val, err = quad(integrand, 0, 2 * np.pi, limit=limit)
    assert np.isfinite(val)
    assert abs(val - 1.0) < atol + err


def test_vonmises_flattopped_extreme_kappa():
    # Use a very large (but not pathological) kappa to stress the table building
    params = (0.7, min(150.0, _VMFT_KAPPA_UPPER - 1.0), 0.6)
    dist = vonmises_flattopped(*params)
    theta = np.linspace(0, 2 * np.pi, 257)
    _check_pdf_normalizes(dist, params=None, atol=5e-6)
    _assert_monotonic_cdf_ppf(dist, theta, np.linspace(0, 1, 257))


def test_katojones_gamma_rho_close_to_one():
    # Stay inside the feasibility disk by aligning lambda with the first moment
    params = (1.1, 0.99, 0.99, 0.0)
    dist = katojones(*params)
    theta = np.linspace(0, 2 * np.pi, 257)
    _check_pdf_normalizes(dist, params=None, atol=1e-6)
    _assert_monotonic_cdf_ppf(dist, theta, np.linspace(0, 1, 257))


@pytest.mark.parametrize("alpha", [1e-6, 1.999])  # degenerate-small and almost-Gaussian
def test_wrapstable_alpha_extremes(alpha):
    params = (0.0, alpha, 0.0, 0.7)  # delta, alpha, beta, gamma
    dist = wrapstable(*params)
    theta = np.linspace(0, 2 * np.pi, 257)
    if alpha < 0.01:
        # Degenerate small-α corner (validation plan §5): ρ_p ≈ e^{−1}
        # never decays, the series hits _WRAPSTABLE_MAX_TERMS undecayed,
        # and the "pdf" is a ~20k-harmonic Dirichlet-kernel artifact for a
        # law that tends to atom + uniform (no density exists). Its mass
        # is exactly 1 by cosine orthogonality — an algebraic identity
        # quad cannot measure through ~20k oscillations (it warned and
        # "passed" only by panel-wise cancellation) — so this cell asserts
        # graceful degradation: finite values and a monotone unit-range
        # cdf below.
        assert np.all(np.isfinite(dist.pdf(theta)))
    else:
        _check_pdf_normalizes(dist, params=None, atol=1e-6)
    _assert_monotonic_cdf_ppf(dist, theta, np.linspace(0, 1, 257), cdf_tol=1e-9, ppf_tol=1e-9)


# ---------------------------------------------------------------------------
# Phase 1 regression contract: role overlay + analytic log-density derivatives
# ---------------------------------------------------------------------------

from pycircstat2.distributions import _RegressionReady  # noqa: E402
from pycircstat2.utils import A1, A1prime  # noqa: E402


def test_vonmises_regression_overlay():
    """The role overlay derives both directions and attaches the right links."""
    assert isinstance(vonmises, _RegressionReady)
    assert vonmises.param_roles == {"mu": "location", "kappa": "concentration"}
    # role -> name(s), one-to-many
    assert vonmises.params_by_role() == {
        "location": ["mu"],
        "concentration": ["kappa"],
    }
    # name -> role -> default link
    assert vonmises.link_for("mu") == "tanhalf"
    assert vonmises.link_for("kappa") == "log"


def test_A1prime_matches_finite_difference_and_limit():
    """A1'(κ) = 1 − A1/κ − A1², with the removable κ→0 limit A1'(0)=1/2."""
    assert A1prime(0.0) == pytest.approx(0.5)
    kappa = np.array([0.05, 0.5, 1.0, 3.0, 8.0, 25.0])
    h = 1e-6
    fd = (A1(kappa + h) - A1(kappa - h)) / (2 * h)
    np.testing.assert_allclose(A1prime(kappa), fd, atol=1e-7)


def _logpdf_vm(x, mu, kappa):
    return vonmises.logpdf(x, mu, kappa)


@pytest.mark.parametrize("kappa", [0.3, 1.0, 4.0, 12.0])
@pytest.mark.parametrize("mu", [0.4, 2.5, 5.0])
def test_vonmises_dlogpdf_matches_finite_difference(mu, kappa):
    """l1: ∂logpdf/∂μ and ∂logpdf/∂κ against central differences of logpdf."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 23)
    grad = vonmises.dlogpdf(x, mu, kappa)

    hm, hk = 1e-6, 1e-6
    d_mu = (_logpdf_vm(x, mu + hm, kappa) - _logpdf_vm(x, mu - hm, kappa)) / (2 * hm)
    d_kappa = (_logpdf_vm(x, mu, kappa + hk) - _logpdf_vm(x, mu, kappa - hk)) / (2 * hk)

    np.testing.assert_allclose(grad["mu"], d_mu, atol=1e-6)
    np.testing.assert_allclose(grad["kappa"], d_kappa, atol=1e-6)


@pytest.mark.parametrize("kappa", [0.3, 1.0, 4.0, 12.0])
@pytest.mark.parametrize("mu", [0.4, 2.5, 5.0])
def test_vonmises_d2logpdf_matches_finite_difference(mu, kappa):
    """l2: the three unique second partials against finite differences of l1."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 23)
    hess = vonmises.d2logpdf(x, mu, kappa)
    hm, hk = 1e-6, 1e-6

    # ∂²/∂μ² and ∂²/∂μ∂κ from differencing the μ-score; ∂²/∂κ² from the κ-score.
    def g_mu(m, k):
        return vonmises.dlogpdf(x, m, k)["mu"]

    def g_kappa(m, k):
        return vonmises.dlogpdf(x, m, k)["kappa"]

    d_mumu = (g_mu(mu + hm, kappa) - g_mu(mu - hm, kappa)) / (2 * hm)
    d_mukappa = (g_mu(mu, kappa + hk) - g_mu(mu, kappa - hk)) / (2 * hk)
    d_kappakappa = (g_kappa(mu, kappa + hk) - g_kappa(mu, kappa - hk)) / (2 * hk)

    np.testing.assert_allclose(hess[("mu", "mu")], d_mumu, atol=1e-6)
    np.testing.assert_allclose(hess[("mu", "kappa")], d_mukappa, atol=1e-6)
    np.testing.assert_allclose(hess[("kappa", "kappa")], d_kappakappa, atol=1e-6)
    # mixed partial is symmetric: differencing the κ-score in μ agrees
    d_kappamu = (g_kappa(mu + hm, kappa) - g_kappa(mu - hm, kappa)) / (2 * hm)
    np.testing.assert_allclose(hess[("mu", "kappa")], d_kappamu, atol=1e-6)


def test_vonmises_derivatives_vectorize_over_per_obs_params():
    """logpdf and its derivatives broadcast over per-observation μ, κ arrays —
    the hard requirement for concentration smoothing."""
    rng = np.random.default_rng(0)
    n = 50
    x = rng.uniform(0, 2 * np.pi, n)
    mu = rng.uniform(0, 2 * np.pi, n)
    kappa = rng.uniform(0.2, 10.0, n)

    grad = vonmises.dlogpdf(x, mu, kappa)
    hess = vonmises.d2logpdf(x, mu, kappa)
    assert grad["mu"].shape == (n,)
    assert hess[("kappa", "kappa")].shape == (n,)
    third = vonmises.d3logpdf(x, mu, kappa)
    fourth = vonmises.d4logpdf(x, mu, kappa)
    assert all(v.shape == (n,) for v in third.values())
    assert all(v.shape == (n,) for v in fourth.values())
    # per-obs result equals the scalar call element-by-element
    for i in (0, 17, 49):
        gi = vonmises.dlogpdf(x[i], mu[i], kappa[i])
        assert gi["mu"] == pytest.approx(grad["mu"][i])
        assert gi["kappa"] == pytest.approx(grad["kappa"][i])


def test_vonmises_dlogpdf_equals_CL_inline_score():
    """Pins the §3.2 equivalence: the distribution's scores are exactly the
    expressions circ_lm(type="cl") computes inline (κ sin(θ−μ); cos(θ−μ) − A1(κ))."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 2 * np.pi, 40)
    mu, kappa = 1.3, 3.7
    grad = vonmises.dlogpdf(x, mu, kappa)
    np.testing.assert_allclose(grad["mu"], kappa * np.sin(x - mu), atol=1e-12)
    np.testing.assert_allclose(grad["kappa"], np.cos(x - mu) - A1(kappa), atol=1e-12)
    # CL's μ-weight is the Fisher information −E[∂²_{μμ}ℓ] = κ A1(κ); the
    # expectation is over the model, so integrate against the vM density.
    grid = np.linspace(0.0, 2 * np.pi, 20001)
    f = vonmises.pdf(grid, mu, kappa)
    obs_info = -vonmises.d2logpdf(grid, mu, kappa)[("mu", "mu")]
    fisher_info = np.trapezoid(obs_info * f, grid)
    assert fisher_info == pytest.approx(kappa * A1(kappa), rel=1e-3)


# ---------------------------------------------------------------------------
# Phase 1 regression contract: links (TanHalfLink + the get_link resolver)
# ---------------------------------------------------------------------------

from hea.family import Link as _HeaLink  # noqa: E402
from hea.family import LogLink as _HeaLogLink  # noqa: E402

from pycircstat2.distributions import TanHalfLink, get_link  # noqa: E402


def test_get_link_resolves_contract_names():
    """``default_links`` names resolve end-to-end to ``hea.family.Link``
    objects (the Phase-2 face); instances pass through; unknown names raise."""
    mu_link = get_link(vonmises.link_for("mu"))
    kappa_link = get_link(vonmises.link_for("kappa"))
    assert isinstance(mu_link, TanHalfLink)
    assert isinstance(mu_link, _HeaLink)
    assert isinstance(kappa_link, _HeaLogLink)
    custom = TanHalfLink()
    assert get_link(custom) is custom
    with pytest.raises(ValueError, match="unknown link"):
        get_link("no-such-link")


def test_tanhalf_roundtrip_branch_and_monotonicity():
    """``linkinv`` inverts ``link`` on the principal branch (−π, π); ``link``
    is 2π-periodic in μ (any angle parameterization hits the principal
    branch); the map is monotone (``mu_eta`` > 0) with range (−π, π)."""
    link = TanHalfLink()
    mu = np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 101)
    np.testing.assert_allclose(link.linkinv(link.link(mu)), mu, atol=1e-12)
    np.testing.assert_allclose(
        link.link(mu + 2 * np.pi), link.link(mu), rtol=1e-6, atol=1e-12
    )
    eta = np.linspace(-50.0, 50.0, 201)
    assert np.all(np.abs(link.linkinv(eta)) < np.pi)
    assert np.all(link.mu_eta(eta) > 0)


def test_tanhalf_mu_eta_matches_finite_difference():
    """``mu_eta`` = d linkinv/dη against central differences."""
    link = TanHalfLink()
    eta = np.linspace(-8.0, 8.0, 81)
    h = 1e-6
    fd = (link.linkinv(eta + h) - link.linkinv(eta - h)) / (2 * h)
    np.testing.assert_allclose(link.mu_eta(eta), fd, atol=1e-8)


def test_tanhalf_dlink_chain_matches_finite_difference():
    """``d2link``/``d3link``/``d4link`` are successive μ-derivatives of
    g(μ) = tan(μ/2): each analytic order matches a central difference of the
    order below (the FD convention of the l1/l2 contract tests). Also pins
    the mgcv identity ``mu_eta(g(μ)) = 1/g′(μ)``."""
    link = TanHalfLink()
    mu = np.linspace(-2.4, 2.4, 49)
    h = 1e-6

    def gprime(m):
        t = np.tan(0.5 * m)
        return 0.5 * (1.0 + t * t)

    fd_g1 = (link.link(mu + h) - link.link(mu - h)) / (2 * h)
    np.testing.assert_allclose(gprime(mu), fd_g1, rtol=1e-8)
    np.testing.assert_allclose(link.mu_eta(link.link(mu)), 1.0 / gprime(mu), rtol=1e-12)

    fd_g2 = (gprime(mu + h) - gprime(mu - h)) / (2 * h)
    np.testing.assert_allclose(link.d2link(mu), fd_g2, rtol=1e-6, atol=1e-8)
    fd_g3 = (link.d2link(mu + h) - link.d2link(mu - h)) / (2 * h)
    np.testing.assert_allclose(link.d3link(mu), fd_g3, rtol=1e-6, atol=1e-8)
    fd_g4 = (link.d3link(mu + h) - link.d3link(mu - h)) / (2 * h)
    np.testing.assert_allclose(link.d4link(mu), fd_g4, rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# Phase 1 regression contract: l3/l4 depth (full outer Newton for the bridge)
# ---------------------------------------------------------------------------

from pycircstat2.utils import A1prime2, A1prime3  # noqa: E402


def test_A1prime2_A1prime3_match_finite_difference_and_limits():
    """A1'' and A1''' against central differences of the order below, plus
    the κ→0 limits (0 and −3/8). The grid covers both sides of the
    series/recurrence switch at κ = 0.01 without straddling it."""
    assert A1prime2(0.0) == pytest.approx(0.0)
    assert A1prime3(0.0) == pytest.approx(-3.0 / 8.0)
    kappa = np.array([0.002, 0.02, 0.05, 0.5, 1.0, 3.0, 8.0, 25.0])
    h = 1e-6
    fd2 = (A1prime(kappa + h) - A1prime(kappa - h)) / (2 * h)
    np.testing.assert_allclose(A1prime2(kappa), fd2, atol=1e-7)
    fd3 = (A1prime2(kappa + h) - A1prime2(kappa - h)) / (2 * h)
    np.testing.assert_allclose(A1prime3(kappa), fd3, atol=1e-6)


@pytest.mark.parametrize("kappa", [0.3, 1.0, 4.0, 12.0])
@pytest.mark.parametrize("mu", [0.4, 2.5, 5.0])
def test_vonmises_d3logpdf_matches_finite_difference(mu, kappa):
    """l3: the four unique third partials against finite differences of l2."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 23)
    third = vonmises.d3logpdf(x, mu, kappa)
    h = 1e-6

    def l2(m, k):
        return vonmises.d2logpdf(x, m, k)

    d_mmm = (l2(mu + h, kappa)[("mu", "mu")] - l2(mu - h, kappa)[("mu", "mu")]) / (2 * h)
    d_mmk = (l2(mu, kappa + h)[("mu", "mu")] - l2(mu, kappa - h)[("mu", "mu")]) / (2 * h)
    d_mkk = (l2(mu, kappa + h)[("mu", "kappa")] - l2(mu, kappa - h)[("mu", "kappa")]) / (2 * h)
    d_kkk = (
        l2(mu, kappa + h)[("kappa", "kappa")] - l2(mu, kappa - h)[("kappa", "kappa")]
    ) / (2 * h)

    np.testing.assert_allclose(third[("mu", "mu", "mu")], d_mmm, atol=1e-5)
    np.testing.assert_allclose(third[("mu", "mu", "kappa")], d_mmk, atol=1e-6)
    np.testing.assert_allclose(third[("mu", "kappa", "kappa")], d_mkk, atol=1e-6)
    np.testing.assert_allclose(third[("kappa", "kappa", "kappa")], d_kkk, atol=1e-6)
    # symmetry cross-check: differencing the (μ,κ) entry in μ also gives μμκ
    d_mmk2 = (l2(mu + h, kappa)[("mu", "kappa")] - l2(mu - h, kappa)[("mu", "kappa")]) / (
        2 * h
    )
    np.testing.assert_allclose(third[("mu", "mu", "kappa")], d_mmk2, atol=1e-6)


@pytest.mark.parametrize("kappa", [0.3, 1.0, 4.0, 12.0])
@pytest.mark.parametrize("mu", [0.4, 2.5, 5.0])
def test_vonmises_d4logpdf_matches_finite_difference(mu, kappa):
    """l4: the five unique fourth partials against finite differences of l3."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 23)
    fourth = vonmises.d4logpdf(x, mu, kappa)
    h = 1e-6

    def l3(m, k):
        return vonmises.d3logpdf(x, m, k)

    pairs = [
        (("mu", "mu", "mu", "mu"), ("mu", "mu", "mu"), "mu"),
        (("mu", "mu", "mu", "kappa"), ("mu", "mu", "mu"), "kappa"),
        (("mu", "mu", "kappa", "kappa"), ("mu", "mu", "kappa"), "kappa"),
        (("mu", "kappa", "kappa", "kappa"), ("mu", "kappa", "kappa"), "kappa"),
        (("kappa", "kappa", "kappa", "kappa"), ("kappa", "kappa", "kappa"), "kappa"),
    ]
    for key4, key3, wrt in pairs:
        if wrt == "mu":
            fd = (l3(mu + h, kappa)[key3] - l3(mu - h, kappa)[key3]) / (2 * h)
        else:
            fd = (l3(mu, kappa + h)[key3] - l3(mu, kappa - h)[key3]) / (2 * h)
        np.testing.assert_allclose(fourth[key4], fd, atol=2e-5, err_msg=str(key4))


# ---------------------------------------------------------------------------
# Phase 1 regression contract: wrapped Cauchy (second Tier-1 workhorse)
# ---------------------------------------------------------------------------


def test_wrapcauchy_regression_overlay():
    """Role overlay: ρ is a (0,1)-bounded concentration, so it gets logit."""
    assert isinstance(wrapcauchy, _RegressionReady)
    assert wrapcauchy.param_roles == {"mu": "location", "rho": "concentration"}
    assert wrapcauchy.link_for("mu") == "tanhalf"
    assert wrapcauchy.link_for("rho") == "logit"
    from hea.family import LogitLink as _HeaLogitLink

    assert isinstance(get_link(wrapcauchy.link_for("rho")), _HeaLogitLink)


@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("mu", [0.4, 2.5, 5.0])
def test_wrapcauchy_dlogpdf_matches_finite_difference(mu, rho):
    """l1 against central differences of logpdf."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 23)
    grad = wrapcauchy.dlogpdf(x, mu, rho)
    h = 1e-6
    d_mu = (wrapcauchy.logpdf(x, mu + h, rho) - wrapcauchy.logpdf(x, mu - h, rho)) / (
        2 * h
    )
    d_rho = (wrapcauchy.logpdf(x, mu, rho + h) - wrapcauchy.logpdf(x, mu, rho - h)) / (
        2 * h
    )
    np.testing.assert_allclose(grad["mu"], d_mu, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad["rho"], d_rho, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("mu", [0.4, 2.5, 5.0])
def test_wrapcauchy_d2logpdf_matches_finite_difference(mu, rho):
    """l2 against central differences of l1 (both μ- and ρ-differencing for
    the mixed entry, pinning symmetry)."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 23)
    hess = wrapcauchy.d2logpdf(x, mu, rho)
    h = 1e-6

    def l1(m, r):
        return wrapcauchy.dlogpdf(x, m, r)

    d_mm = (l1(mu + h, rho)["mu"] - l1(mu - h, rho)["mu"]) / (2 * h)
    d_mr = (l1(mu, rho + h)["mu"] - l1(mu, rho - h)["mu"]) / (2 * h)
    d_rm = (l1(mu + h, rho)["rho"] - l1(mu - h, rho)["rho"]) / (2 * h)
    d_rr = (l1(mu, rho + h)["rho"] - l1(mu, rho - h)["rho"]) / (2 * h)

    np.testing.assert_allclose(hess[("mu", "mu")], d_mm, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hess[("mu", "rho")], d_mr, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hess[("mu", "rho")], d_rm, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hess[("rho", "rho")], d_rr, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("mu", [0.4, 2.5, 5.0])
def test_wrapcauchy_d3_d4logpdf_match_finite_difference(mu, rho):
    """l3 against FD of l2, l4 against FD of l3 — every unique key."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 23)
    h = 1e-5

    third = wrapcauchy.d3logpdf(x, mu, rho)
    fourth = wrapcauchy.d4logpdf(x, mu, rho)

    def l2(m, r):
        return wrapcauchy.d2logpdf(x, m, r)

    def l3(m, r):
        return wrapcauchy.d3logpdf(x, m, r)

    chain3 = [
        (("mu", "mu", "mu"), ("mu", "mu"), "mu"),
        (("mu", "mu", "rho"), ("mu", "mu"), "rho"),
        (("mu", "rho", "rho"), ("mu", "rho"), "rho"),
        (("rho", "rho", "rho"), ("rho", "rho"), "rho"),
    ]
    for key3, key2, wrt in chain3:
        if wrt == "mu":
            fd = (l2(mu + h, rho)[key2] - l2(mu - h, rho)[key2]) / (2 * h)
        else:
            fd = (l2(mu, rho + h)[key2] - l2(mu, rho - h)[key2]) / (2 * h)
        np.testing.assert_allclose(
            third[key3], fd, rtol=1e-4, atol=1e-5, err_msg=str(key3)
        )

    chain4 = [
        (("mu", "mu", "mu", "mu"), ("mu", "mu", "mu"), "mu"),
        (("mu", "mu", "mu", "rho"), ("mu", "mu", "mu"), "rho"),
        (("mu", "mu", "rho", "rho"), ("mu", "mu", "rho"), "rho"),
        (("mu", "rho", "rho", "rho"), ("mu", "rho", "rho"), "rho"),
        (("rho", "rho", "rho", "rho"), ("rho", "rho", "rho"), "rho"),
    ]
    for key4, key3, wrt in chain4:
        if wrt == "mu":
            fd = (l3(mu + h, rho)[key3] - l3(mu - h, rho)[key3]) / (2 * h)
        else:
            fd = (l3(mu, rho + h)[key3] - l3(mu, rho - h)[key3]) / (2 * h)
        np.testing.assert_allclose(
            fourth[key4], fd, rtol=1e-4, atol=1e-4, err_msg=str(key4)
        )


def test_wrapcauchy_derivatives_vectorize_over_per_obs_params():
    """logpdf and l1..l4 broadcast over per-observation μ, ρ arrays."""
    rng = np.random.default_rng(0)
    n = 50
    x = rng.uniform(0, 2 * np.pi, n)
    mu = rng.uniform(0, 2 * np.pi, n)
    rho = rng.uniform(0.05, 0.95, n)

    assert wrapcauchy.logpdf(x, mu, rho).shape == (n,)
    grad = wrapcauchy.dlogpdf(x, mu, rho)
    assert grad["mu"].shape == (n,) and grad["rho"].shape == (n,)
    assert all(v.shape == (n,) for v in wrapcauchy.d2logpdf(x, mu, rho).values())
    assert all(v.shape == (n,) for v in wrapcauchy.d3logpdf(x, mu, rho).values())
    assert all(v.shape == (n,) for v in wrapcauchy.d4logpdf(x, mu, rho).values())
    # per-obs result equals the scalar call element-by-element
    for i in (0, 17, 49):
        gi = wrapcauchy.dlogpdf(x[i], mu[i], rho[i])
        assert gi["mu"] == pytest.approx(grad["mu"][i])
        assert gi["rho"] == pytest.approx(grad["rho"][i])


@pytest.mark.parametrize("rho", [0.1, 0.3, 0.49])
@pytest.mark.parametrize("mu", [0.4, 2.5, 5.0])
def test_cardioid_d3_d4logpdf_match_finite_difference(mu, rho):
    """l3 against FD of l2, l4 against FD of l3 — every unique key (the new
    Tier-2→full-Newton lift; cardioid is the +log P twin of wrapped Cauchy)."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 23)
    h = 1e-5
    third, fourth = cardioid.d3logpdf(x, mu, rho), cardioid.d4logpdf(x, mu, rho)

    def l2(m, r):
        return cardioid.d2logpdf(x, m, r)

    def l3(m, r):
        return cardioid.d3logpdf(x, m, r)

    chain3 = [
        (("mu", "mu", "mu"), ("mu", "mu"), "mu"),
        (("mu", "mu", "rho"), ("mu", "mu"), "rho"),
        (("mu", "rho", "rho"), ("mu", "rho"), "rho"),
        (("rho", "rho", "rho"), ("rho", "rho"), "rho"),
    ]
    for key3, key2, wrt in chain3:
        fd = ((l2(mu + h, rho)[key2] - l2(mu - h, rho)[key2]) / (2 * h)
              if wrt == "mu"
              else (l2(mu, rho + h)[key2] - l2(mu, rho - h)[key2]) / (2 * h))
        np.testing.assert_allclose(third[key3], fd, rtol=1e-4, atol=1e-4,
                                   err_msg=str(key3))

    chain4 = [
        (("mu", "mu", "mu", "mu"), ("mu", "mu", "mu"), "mu"),
        (("mu", "mu", "mu", "rho"), ("mu", "mu", "mu"), "rho"),
        (("mu", "mu", "rho", "rho"), ("mu", "mu", "rho"), "rho"),
        (("mu", "rho", "rho", "rho"), ("mu", "rho", "rho"), "rho"),
        (("rho", "rho", "rho", "rho"), ("rho", "rho", "rho"), "rho"),
    ]
    for key4, key3, wrt in chain4:
        fd = ((l3(mu + h, rho)[key3] - l3(mu - h, rho)[key3]) / (2 * h)
              if wrt == "mu"
              else (l3(mu, rho + h)[key3] - l3(mu, rho - h)[key3]) / (2 * h))
        np.testing.assert_allclose(fourth[key4], fd, rtol=1e-4, atol=1e-4,
                                   err_msg=str(key4))


@pytest.mark.parametrize("zeta", [0.4, 1.0, 1.6])
@pytest.mark.parametrize("mu", [0.4, 2.5, 5.0])
def test_cartwright_d3_d4logpdf_match_finite_difference(mu, zeta):
    """l3 against FD of l2, l4 against FD of l3 — the separable
    ℓ = N(ζ) + (2/ζ)log|cos((θ−μ)/2)| lift (N‴/N⁗ via polygamma)."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 23)
    h = 1e-5
    third, fourth = cartwright.d3logpdf(x, mu, zeta), cartwright.d4logpdf(x, mu, zeta)

    def l2(m, z):
        return cartwright.d2logpdf(x, m, z)

    def l3(m, z):
        return cartwright.d3logpdf(x, m, z)

    chain3 = [
        (("mu", "mu", "mu"), ("mu", "mu"), "mu"),
        (("mu", "mu", "zeta"), ("mu", "mu"), "zeta"),
        (("mu", "zeta", "zeta"), ("mu", "zeta"), "zeta"),
        (("zeta", "zeta", "zeta"), ("zeta", "zeta"), "zeta"),
    ]
    for key3, key2, wrt in chain3:
        fd = ((l2(mu + h, zeta)[key2] - l2(mu - h, zeta)[key2]) / (2 * h)
              if wrt == "mu"
              else (l2(mu, zeta + h)[key2] - l2(mu, zeta - h)[key2]) / (2 * h))
        np.testing.assert_allclose(third[key3], fd, rtol=1e-4, atol=1e-4,
                                   err_msg=str(key3))

    chain4 = [
        (("mu", "mu", "mu", "mu"), ("mu", "mu", "mu"), "mu"),
        (("mu", "mu", "mu", "zeta"), ("mu", "mu", "mu"), "zeta"),
        (("mu", "mu", "zeta", "zeta"), ("mu", "mu", "zeta"), "zeta"),
        (("mu", "zeta", "zeta", "zeta"), ("mu", "zeta", "zeta"), "zeta"),
        (("zeta", "zeta", "zeta", "zeta"), ("zeta", "zeta", "zeta"), "zeta"),
    ]
    for key4, key3, wrt in chain4:
        fd = ((l3(mu + h, zeta)[key3] - l3(mu - h, zeta)[key3]) / (2 * h)
              if wrt == "mu"
              else (l3(mu, zeta + h)[key3] - l3(mu, zeta - h)[key3]) / (2 * h))
        np.testing.assert_allclose(fourth[key4], fd, rtol=1e-4, atol=1e-4,
                                   err_msg=str(key4))


@pytest.mark.parametrize("dist,pname,lo,hi", [
    (cardioid, "rho", 0.05, 0.49),
    (cartwright, "zeta", 0.3, 2.0),
])
def test_cardioid_cartwright_l3_l4_vectorize_over_per_obs_params(dist, pname, lo, hi):
    """l3/l4 broadcast over per-observation (μ, ·) arrays — the hard
    requirement for the smoothing path that now drives outer Newton."""
    rng = np.random.default_rng(0)
    n = 40
    x = rng.uniform(0, 2 * np.pi, n)
    kw = {"mu": rng.uniform(0, 2 * np.pi, n), pname: rng.uniform(lo, hi, n)}
    d3 = dist.d3logpdf(x, **kw)
    d4 = dist.d4logpdf(x, **kw)
    assert all(np.asarray(v).shape == (n,) for v in d3.values())
    assert all(np.asarray(v).shape == (n,) for v in d4.values())


# ---------------------------------------------------------------------------
# Family nesting / limiting identities (circlss "choosing a family" spine).
# Cheap density-grid checks that double as living documentation of how the
# circular families relate; they lock normalizer/parameterization regressions.
# ---------------------------------------------------------------------------

def test_cardioid_equals_jonespewsey_psi1():
    """JP(ψ=1) normalizes to (1 + tanh κ · cos)/2π = cardioid(ρ = ½ tanh κ)."""
    theta = np.linspace(0.0, 2 * np.pi, 401)
    for mu, kappa in [(0.7, 0.3), (2.0, 1.0), (4.5, 2.5)]:
        rho = 0.5 * np.tanh(kappa)
        np.testing.assert_allclose(
            jonespewsey.pdf(theta, mu=mu, kappa=kappa, psi=1.0),
            cardioid.pdf(theta, mu=mu, rho=rho),
            rtol=1e-9, atol=1e-12,
        )


def test_cartwright_zeta1_equals_cardioid_half():
    """Cartwright(ζ=1) = (1+cos)/2π = cardioid(ρ=½) (densities equal to ~1e-16)."""
    theta = np.linspace(0.0, 2 * np.pi, 401)
    for mu in [0.0, 1.3, 4.0]:
        np.testing.assert_allclose(
            cartwright.pdf(theta, mu=mu, zeta=1.0),
            cardioid.pdf(theta, mu=mu, rho=0.5),
            rtol=1e-10, atol=1e-13,
        )


def test_cartwright_is_jonespewsey_kappa_limit():
    """Cartwright(ζ) = lim_{κ→∞} JP(κ, ψ=ζ) — the tanh→1 boundary (Cartwright
    is *not* an interior JP member). Assert the approach: the density gap
    shrinks as κ grows and is already negligible by κ=20."""
    theta = np.linspace(0.0, 2 * np.pi, 401)
    for mu, zeta in [(0.0, 0.6), (2.0, 1.0), (4.0, 1.5)]:
        cw = cartwright.pdf(theta, mu=mu, zeta=zeta)
        g10 = np.max(np.abs(jonespewsey.pdf(theta, mu=mu, kappa=10.0, psi=zeta) - cw))
        g20 = np.max(np.abs(jonespewsey.pdf(theta, mu=mu, kappa=20.0, psi=zeta) - cw))
        assert g20 < g10                 # approaching the limit
        assert g20 < 1e-6                 # already negligible


def test_vonmises_approx_wrapnorm_matched_moment():
    """vM(κ) ≈ wrapped normal at matched first moment ρ = A₁(κ): a loose
    sanity bound (≤ ~9%, worst near κ≈1.5), not equality — distinct families."""
    from pycircstat2.utils import A1

    theta = np.linspace(0.0, 2 * np.pi, 401)
    for mu, kappa in [(1.0, 0.5), (2.0, 1.5), (3.0, 3.0), (2.0, 5.0)]:
        vm = vonmises.pdf(theta, mu=mu, kappa=kappa)
        wn = wrapnorm.pdf(theta, mu=mu, rho=A1(kappa))
        assert np.max(np.abs(vm - wn)) / np.max(vm) < 0.10


# ---------------------------------------------------------------------------
# Phase 1 regression contract: per-obs param vectorization (Tier-1/2 audit)
# ---------------------------------------------------------------------------


def test_contract_families_logpdf_vectorizes_over_per_obs_params():
    """Tier-1/2 contract families accept per-observation parameter arrays in
    ``logpdf`` — the requirement for smoothing a concentration/shape over X.
    The per-obs call must match element-wise scalar calls (jonespewsey and
    the sine-skewed extension route through the vectorized Legendre
    normalizer; the others broadcast natively)."""
    rng = np.random.default_rng(3)
    n = 11
    x = rng.uniform(0.1, 2 * np.pi - 0.1, n)
    mu = rng.uniform(0.5, 5.5, n)
    from pycircstat2.distributions import projectednormal as _pn

    cases = [
        (vonmises, dict(mu=mu, kappa=rng.uniform(0.3, 8.0, n))),
        (wrapcauchy, dict(mu=mu, rho=rng.uniform(0.05, 0.9, n))),
        (_pn, dict(mu1=rng.uniform(-2.0, 2.0, n), mu2=rng.uniform(-2.0, 2.0, n))),
        (cartwright, dict(mu=mu, zeta=rng.uniform(0.1, 2.0, n))),
        (
            jonespewsey,
            dict(mu=mu, kappa=rng.uniform(0.3, 5.0, n), psi=rng.uniform(-1.5, 1.5, n)),
        ),
        (
            jonespewsey_sineskewed,
            dict(
                xi=mu,
                kappa=rng.uniform(0.3, 5.0, n),
                psi=rng.uniform(-1.5, 1.5, n),
                lmbd=rng.uniform(-0.8, 0.8, n),
            ),
        ),
        (
            katojones,
            dict(
                mu=mu,
                gamma=rng.uniform(0.05, 0.3, n),
                rho=rng.uniform(0.05, 0.3, n),
                lam=rng.uniform(0.0, 1.0, n),
            ),
        ),
    ]
    for dist, params in cases:
        vec = dist.logpdf(x, **params)
        assert np.shape(vec) == (n,), f"{dist.name}: shape {np.shape(vec)}"
        assert np.all(np.isfinite(vec)), f"{dist.name}: non-finite logpdf"
        for i in (0, n // 2, n - 1):
            scal = dist.logpdf(x[i], **{k: v[i] for k, v in params.items()})
            np.testing.assert_allclose(
                vec[i], scal, rtol=1e-9, atol=1e-12, err_msg=dist.name
            )


# ---------------------------------------------------------------------------
# Projected normal (Tier-1 workhorse, added by the Phase 1 contract)
# ---------------------------------------------------------------------------

from pycircstat2.distributions import projectednormal  # noqa: E402


def test_projectednormal_pdf_normalizes_and_uniform_limit():
    """∫f = 1 across concentration regimes; ‖μ‖ = 0 is the circular uniform;
    logpdf is the exact log of pdf."""
    grid = np.linspace(0.0, 2.0 * np.pi, 20001)
    for m in [(0.0, 0.0), (1.5, 0.8), (4.0, -3.0), (0.05, 0.0)]:
        f = projectednormal.pdf(grid, *m)
        assert np.trapezoid(f, grid) == pytest.approx(1.0, abs=1e-8)
    assert projectednormal.pdf(1.0, 0.0, 0.0) == pytest.approx(1.0 / (2.0 * np.pi))
    x = np.linspace(0.1, 2 * np.pi - 0.1, 17)
    np.testing.assert_allclose(
        projectednormal.logpdf(x, 1.5, 0.8),
        np.log(projectednormal.pdf(x, 1.5, 0.8)),
        atol=1e-12,
    )


def test_projectednormal_regression_overlay():
    """Both Cartesian components share the 'location' role (the documented
    one-to-many case) and resolve to hea's identity link."""
    from hea.family import IdentityLink as _HeaIdentityLink

    assert isinstance(projectednormal, _RegressionReady)
    assert projectednormal.param_roles == {"mu1": "location", "mu2": "location"}
    assert projectednormal.params_by_role() == {"location": ["mu1", "mu2"]}
    assert projectednormal.link_for("mu1") == "identity"
    assert isinstance(
        get_link(projectednormal.link_for("mu2")), _HeaIdentityLink
    )


@pytest.mark.parametrize("mu1,mu2", [(0.3, 0.2), (1.5, 0.8), (-1.2, 0.7), (4.0, -3.0)])
def test_projectednormal_derivatives_match_finite_difference(mu1, mu2):
    """The full l1..l4 chain against central differences of the order below
    (the contract convention), including a high-concentration case where t
    is strongly negative at the antimode (stresses the Mills-ratio path)."""
    x = np.linspace(0.1, 2 * np.pi - 0.1, 23)
    h = 1e-6

    def fd(f, key, wrt):
        if wrt == "mu1":
            hi, lo = f(x, mu1 + h, mu2), f(x, mu1 - h, mu2)
        else:
            hi, lo = f(x, mu1, mu2 + h), f(x, mu1, mu2 - h)
        hi = hi if key is None else hi[key]
        lo = lo if key is None else lo[key]
        return (hi - lo) / (2 * h)

    grad = projectednormal.dlogpdf(x, mu1, mu2)
    np.testing.assert_allclose(grad["mu1"], fd(projectednormal.logpdf, None, "mu1"), atol=1e-6)
    np.testing.assert_allclose(grad["mu2"], fd(projectednormal.logpdf, None, "mu2"), atol=1e-6)

    hess = projectednormal.d2logpdf(x, mu1, mu2)
    np.testing.assert_allclose(hess[("mu1", "mu1")], fd(projectednormal.dlogpdf, "mu1", "mu1"), atol=1e-5)
    np.testing.assert_allclose(hess[("mu1", "mu2")], fd(projectednormal.dlogpdf, "mu1", "mu2"), atol=1e-5)
    np.testing.assert_allclose(hess[("mu1", "mu2")], fd(projectednormal.dlogpdf, "mu2", "mu1"), atol=1e-5)
    np.testing.assert_allclose(hess[("mu2", "mu2")], fd(projectednormal.dlogpdf, "mu2", "mu2"), atol=1e-5)

    third = projectednormal.d3logpdf(x, mu1, mu2)
    chain3 = [
        (("mu1", "mu1", "mu1"), ("mu1", "mu1"), "mu1"),
        (("mu1", "mu1", "mu2"), ("mu1", "mu1"), "mu2"),
        (("mu1", "mu2", "mu2"), ("mu1", "mu2"), "mu2"),
        (("mu2", "mu2", "mu2"), ("mu2", "mu2"), "mu2"),
    ]
    for key3, key2, wrt in chain3:
        np.testing.assert_allclose(
            third[key3], fd(projectednormal.d2logpdf, key2, wrt),
            atol=1e-5, err_msg=str(key3),
        )

    fourth = projectednormal.d4logpdf(x, mu1, mu2)
    chain4 = [
        (("mu1", "mu1", "mu1", "mu1"), ("mu1", "mu1", "mu1"), "mu1"),
        (("mu1", "mu1", "mu1", "mu2"), ("mu1", "mu1", "mu1"), "mu2"),
        (("mu1", "mu1", "mu2", "mu2"), ("mu1", "mu1", "mu2"), "mu2"),
        (("mu1", "mu2", "mu2", "mu2"), ("mu1", "mu2", "mu2"), "mu2"),
        (("mu2", "mu2", "mu2", "mu2"), ("mu2", "mu2", "mu2"), "mu2"),
    ]
    for key4, key3, wrt in chain4:
        np.testing.assert_allclose(
            fourth[key4], fd(projectednormal.d3logpdf, key3, wrt),
            atol=2e-5, err_msg=str(key4),
        )


def test_projectednormal_derivatives_vectorize_over_per_obs_params():
    """logpdf and l1..l4 broadcast over per-observation (μ₁, μ₂) arrays —
    the two-linear-predictor regression requirement."""
    rng = np.random.default_rng(0)
    n = 50
    x = rng.uniform(0, 2 * np.pi, n)
    mu1 = rng.uniform(-2.5, 2.5, n)
    mu2 = rng.uniform(-2.5, 2.5, n)

    assert projectednormal.logpdf(x, mu1, mu2).shape == (n,)
    grad = projectednormal.dlogpdf(x, mu1, mu2)
    assert grad["mu1"].shape == (n,) and grad["mu2"].shape == (n,)
    assert all(v.shape == (n,) for v in projectednormal.d2logpdf(x, mu1, mu2).values())
    assert all(v.shape == (n,) for v in projectednormal.d3logpdf(x, mu1, mu2).values())
    assert all(v.shape == (n,) for v in projectednormal.d4logpdf(x, mu1, mu2).values())
    for i in (0, 17, 49):
        gi = projectednormal.dlogpdf(x[i], mu1[i], mu2[i])
        assert gi["mu1"] == pytest.approx(grad["mu1"][i])
        assert gi["mu2"] == pytest.approx(grad["mu2"][i])


def test_projectednormal_rvs_matches_density():
    """Sampling by direct projection agrees with the closed-form density:
    the empirical CDF tracks the numeric CDF and the sample mean direction
    recovers atan2(μ₂, μ₁)."""
    mu1, mu2 = 1.5, 0.8
    s = projectednormal.rvs(mu1, mu2, size=20000, random_state=42)
    assert np.all((s >= 0.0) & (s < 2.0 * np.pi))
    dir_hat = float(np.angle(np.mean(np.exp(1j * s))))
    assert dir_hat == pytest.approx(np.arctan2(mu2, mu1), abs=0.03)
    for q in (1.0, 2.5, 4.5):
        emp = float(np.mean(s <= q))
        assert emp == pytest.approx(
            float(projectednormal.cdf(q, mu1, mu2)), abs=0.02
        )


def test_projectednormal_cdf_monotone_and_bounded():
    grid = np.linspace(0.0, 2.0 * np.pi, 25)
    cdf = np.array([float(projectednormal.cdf(v, 1.2, -0.6)) for v in grid])
    assert cdf[0] == pytest.approx(0.0, abs=1e-9)
    assert cdf[-1] == pytest.approx(1.0, abs=1e-7)
    assert np.all(np.diff(cdf) >= -1e-10)


def test_projectednormal_fit_recovers_truth():
    mu1, mu2 = 1.2, -0.8
    s = projectednormal.rvs(mu1, mu2, size=4000, random_state=7)
    (m1, m2), info = projectednormal.fit(s, return_info=True)
    assert info["converged"] is True
    assert m1 == pytest.approx(mu1, abs=0.12)
    assert m2 == pytest.approx(mu2, abs=0.12)
    with pytest.raises(ValueError, match="method"):
        projectednormal.fit(s, method="moments")


# ===========================================================================
# CircularLL residual contract: pearson standardization + saturated-reference
# deviance (the family-layer quantities circ_resid / circ_check are built on)
# ===========================================================================
def test_circularll_pearson_residual_closed_forms():
    """The circular Pearson residual is sin(y-mu)/sqrt(Var sin), with
    Var(sin(y-mu)) = (1-alpha2)/2 read from the family's centered 2nd cosine
    moment. The closed-form families pin it exactly: von Mises -> A1(k)/k,
    wrapped Cauchy -> (1-rho^2)/2, cardioid -> 1/2 (first-harmonic only)."""
    from pycircstat2.distributions import cardlss, vmlss, wclss
    from pycircstat2.utils import A1

    rng = np.random.default_rng(0)
    y = np.mod(rng.uniform(0, 2 * np.pi, 256), 2 * np.pi)
    col = lambda v: np.full(y.size, v)  # noqa: E731

    mu, k = 1.1, 2.7
    d = np.angle(np.exp(1j * (y - mu)))
    np.testing.assert_allclose(
        vmlss.residuals(y, np.column_stack([col(mu), col(k)]), type="pearson"),
        np.sin(d) / np.sqrt(float(A1(k)) / k), atol=1e-12,
    )

    mu, rho = 0.6, 0.55
    d = np.angle(np.exp(1j * (y - mu)))
    np.testing.assert_allclose(
        wclss.residuals(y, np.column_stack([col(mu), col(rho)]), type="pearson"),
        np.sin(d) / np.sqrt((1 - rho**2) / 2), atol=1e-9,
    )

    mu, rho = 0.3, 0.2
    d = np.angle(np.exp(1j * (y - mu)))
    np.testing.assert_allclose(
        cardlss.residuals(y, np.column_stack([col(mu), col(rho)]), type="pearson"),
        np.sin(d) / np.sqrt(0.5), atol=1e-9,
    )


def test_circularll_pearson_aliases_deviance_for_cartesian_location():
    """A family with no single circular location (the projected normal's
    Cartesian mu1/mu2 pair) has no sin-residual standardization, so the Pearson
    residual aliases the deviance residual -- circlss's pnlss convention."""
    from pycircstat2.distributions import pnlss

    rng = np.random.default_rng(1)
    y = np.mod(rng.uniform(0, 2 * np.pi, 200), 2 * np.pi)
    fit = np.column_stack([np.full(y.size, 1.5), np.full(y.size, 0.8)])
    np.testing.assert_allclose(
        pnlss.residuals(y, fit, type="pearson"),
        pnlss.residuals(y, fit, type="deviance"), atol=1e-12,
    )


def test_katojones_pearson_uses_gamma_scale():
    """KatoJonesLL regresses in chart coordinates that are not the
    distribution's book parameters, so the generic centered-moment path cannot
    apply; its Pearson variance is the wrapped-Cauchy first-moment scale
    (1-gamma^2)/2 (gamma the concentration LP), matching circlss."""
    from pycircstat2.distributions import kjlss

    rng = np.random.default_rng(2)
    y = np.mod(rng.uniform(0, 2 * np.pi, 200), 2 * np.pi)
    mu, g = 1.0, 0.4
    fit = np.column_stack(
        [np.full(y.size, mu), np.full(y.size, g),
         np.full(y.size, 0.2), np.full(y.size, -0.1)]
    )
    d = np.angle(np.exp(1j * (y - mu)))
    np.testing.assert_allclose(
        kjlss.residuals(y, fit, type="pearson"),
        np.sin(d) / np.sqrt((1 - g * g) / 2), atol=1e-9,
    )


def test_circularll_deviance_saturated_reference_is_density_peak():
    """The deviance residual's saturated reference is the true density peak
    (max_theta logpdf), not the value at the location anchor. A symmetric
    family (mode == location) is unchanged; a skewed family (sine-skewed JP)
    takes the off-anchor peak, matching a fine-grid maximum."""
    from pycircstat2.distributions import (
        jonespewsey_sineskewed,
        ssjplss,
        vmlss,
    )

    rng = np.random.default_rng(3)
    y = np.mod(rng.uniform(0, 2 * np.pi, 200), 2 * np.pi)

    # symmetric von Mises: deviance equals the anchor-reference value, since the
    # mode IS the location.
    mu, k = 1.2, 3.0
    params = {"mu": np.full(y.size, mu), "kappa": np.full(y.size, k)}
    l_obs = vmlss._loglik_values(y, params)
    l_loc = vmlss._loglik_values(np.full(y.size, mu), params)
    anchor = np.sign(np.angle(np.exp(1j * (y - mu)))) * np.sqrt(
        2 * np.clip(l_loc - l_obs, 0, None)
    )
    fit = np.column_stack([params["mu"], params["kappa"]])
    np.testing.assert_allclose(
        vmlss.residuals(y, fit, type="deviance"), anchor, atol=1e-7
    )

    # skewed sine-skewed JP: the saturated reference exceeds the anchor value
    # and matches a dense-grid maximum of the log-density.
    xi, kap, psi, lmbd = 1.0, 2.0, 0.5, 0.6
    ps = {
        "xi": np.full(3, xi), "kappa": np.full(3, kap),
        "psi": np.full(3, psi), "lmbd": np.full(3, lmbd),
    }
    l_peak = float(ssjplss._peak_loglik(ps, "xi")[0])
    l_anchor = float(
        jonespewsey_sineskewed.logpdf(
            np.array([xi]), xi=xi, kappa=kap, psi=psi, lmbd=lmbd
        )[0]
    )
    grid_truth = float(
        np.max(
            jonespewsey_sineskewed.logpdf(
                np.linspace(0, 2 * np.pi, 40001),
                xi=xi, kappa=kap, psi=psi, lmbd=lmbd,
            )
        )
    )
    assert l_peak > l_anchor + 1e-3  # the mode sits off the anchor
    # the parabola-refined 1024-pt peak lands on the true vertex to ~1e-9; the
    # tol is tight enough to catch a regression to a plain grid max / the wrong
    # (4*denom) refinement constant, both of which err ~8e-7 here.
    assert l_peak == pytest.approx(grid_truth, abs=1e-7)


# ---------------------------------------------------------------------------
# Prior-weights ("weighted likelihood") contract for the *lss general families.
#
# Ported from circlss tests/testthat/helper-weights.R + test-weights.R +
# test-vmlss-weights.R. The contract: weighting a row by w is identical to
# duplicating that row w times, which is what lets a weighted fit
# (circ_gam(weights=), e.g. a finite-mixture EM M-step) reach the weighted MLE.
# pycircstat2 has a single shared CircularLL.ll, so this one weighting path
# covers every family (KatoJonesLL inherits it). Checked at the ll level (no
# fitting) across all 12 families, plus one end-to-end fit-level identity.
# ---------------------------------------------------------------------------

_LSS_FAMILIES = {
    "cardlss": cardlss, "cartlss": cartlss, "wnlss": wnlss, "wclss": wclss,
    "vmlss": vmlss, "pnlss": pnlss, "vmftlss": vmftlss, "jplss": jplss,
    "ssjplss": ssjplss, "ajplss": ajplss, "ibslss": ibslss, "kjlss": kjlss,
}


def _lss_weight_design(fam, n=40, seed=0):
    """A stacked design (intercept + one covariate per LP), small random
    coefficients (so the inverse-linked parameters stay well inside each
    family's domain) and moderate circular responses. The exact identity below
    is coefficient/data-agnostic, so any well-posed design exercises it."""
    rng = np.random.default_rng(seed)
    nlp = fam.n_lp
    x = rng.standard_normal(n)
    X = np.hstack([np.column_stack([np.ones(n), x]) for _ in range(nlp)])
    lpi = [np.arange(2 * j, 2 * j + 2) for j in range(nlp)]
    coef = rng.standard_normal(2 * nlp) * 0.1
    y = rng.vonmises(0.0, 2.0, n)
    return X, lpi, coef, y, rng


@pytest.mark.parametrize("name", list(_LSS_FAMILIES), ids=list(_LSS_FAMILIES))
def test_lss_weighted_ll_equals_row_duplication(name):
    """EXACT gate: for positive integer weights w, ll(wt=w) on n rows equals
    ll(wt=1) on the design with row i repeated w_i times -- objective l,
    gradient lb and Hessian lbb, to floating point. An identity, not an
    approximation, so it is robust even where the log-density is stiff."""
    fam = _LSS_FAMILIES[name]
    X, lpi, coef, y, rng = _lss_weight_design(fam)
    w = rng.integers(1, 5, len(y)).astype(float)
    idx = np.repeat(np.arange(len(y)), w.astype(int))

    rw = fam.ll(y, X, coef, w, lpi=lpi, deriv=1)
    rd = fam.ll(y[idx], X[idx], coef, np.ones(len(idx)), lpi=lpi, deriv=1)
    assert rw["l"] == pytest.approx(rd["l"], abs=1e-8)
    np.testing.assert_allclose(rw["lb"], rd["lb"], atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(rw["lbb"]), np.asarray(rd["lbb"]), atol=1e-8
    )


@pytest.mark.parametrize("name", list(_LSS_FAMILIES), ids=list(_LSS_FAMILIES))
def test_lss_weighted_ll_l0_unweighted_and_none_noop(name):
    """l0 is the per-observation log-density: it must NEVER be scaled by wt
    (only the scalar objective l is), and a None wt must behave like unit
    weights at deriv=1 (objective, gradient and Hessian)."""
    fam = _LSS_FAMILIES[name]
    X, lpi, coef, y, rng = _lss_weight_design(fam)
    n = len(y)
    wt = rng.uniform(0.2, 3.0, n)

    r_w = fam.ll(y, X, coef, wt, lpi=lpi, deriv=0)
    r_1 = fam.ll(y, X, coef, np.ones(n), lpi=lpi, deriv=0)
    np.testing.assert_array_equal(r_w["l0"], r_1["l0"])          # l0 never scaled
    assert r_w["l"] == pytest.approx(float(np.sum(wt * r_w["l0"])))  # l IS weighted
    assert r_1["l"] == pytest.approx(float(np.sum(r_1["l0"])))

    r_unit = fam.ll(y, X, coef, np.ones(n), lpi=lpi, deriv=1)
    r_none = fam.ll(y, X, coef, None, lpi=lpi, deriv=1)
    assert r_none["l"] == pytest.approx(r_unit["l"])
    np.testing.assert_array_equal(r_none["lb"], r_unit["lb"])
    np.testing.assert_array_equal(
        np.asarray(r_none["lbb"]), np.asarray(r_unit["lbb"])
    )


def test_vmlss_weighted_gradient_finite_differences():
    """Independent cross-check on a well-conditioned design: ll(deriv=1) lb/lbb
    match central differences of the weighted objective ll(deriv=0).l."""
    fam = vmlss
    X, lpi, coef, y, rng = _lss_weight_design(fam, n=50, seed=3)
    wt = rng.uniform(0.2, 3.0, len(y))
    h = 1e-5
    p = len(coef)

    def L(b):
        return fam.ll(y, X, b, wt, lpi=lpi, deriv=0)["l"]

    ret = fam.ll(y, X, coef, wt, lpi=lpi, deriv=1)
    eye = np.eye(p)
    g_fd = np.array([(L(coef + h * eye[j]) - L(coef - h * eye[j])) / (2 * h)
                     for j in range(p)])
    np.testing.assert_allclose(ret["lb"], g_fd, atol=1e-5)

    H_fd = np.zeros((p, p))
    for j in range(p):
        gp = fam.ll(y, X, coef + h * eye[j], wt, lpi=lpi, deriv=1)["lb"]
        gm = fam.ll(y, X, coef - h * eye[j], wt, lpi=lpi, deriv=1)["lb"]
        H_fd[:, j] = (gp - gm) / (2 * h)
    np.testing.assert_allclose(
        np.asarray(ret["lbb"]), (H_fd + H_fd.T) / 2, atol=1e-5
    )


def test_circ_gam_weighted_equals_duplicated_fit():
    """End-to-end: an integer-weighted circ_gam fit equals the unweighted fit on
    the row-duplicated frame. The comprehensive gate -- it drives gam.fit5's full
    deriv<=4 path, so it validates the higher-order (l3/l4) weight scaling the
    ll-level deriv=1 tests cannot reach on their own."""
    import polars as pl

    rng = np.random.default_rng(7)
    n = 200
    x = rng.uniform(-1.0, 1.0, n)
    mu = 2.0 * np.arctan(0.9 + 2.0 * x)
    y = np.mod(mu + rng.vonmises(0.0, 6.0, n), 2.0 * np.pi)
    w = rng.integers(1, 4, n)
    idx = np.repeat(np.arange(n), w)

    fW = circ_gam("y ~ x", pl.DataFrame({"y": y, "x": x}),
                  family="vmlss", weights=w.astype(float), method="ML")
    fD = circ_gam("y ~ x", pl.DataFrame({"y": y[idx], "x": x[idx]}),
                  family="vmlss", method="ML")
    np.testing.assert_allclose(
        np.asarray(fW.coef), np.asarray(fD.coef), atol=1e-6
    )
