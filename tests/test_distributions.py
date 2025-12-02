from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pytest
from scipy import stats, special

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
from pycircstat2.distributions import _VMFT_KAPPA_UPPER


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
    ("triangular", triangular, (np.array([-0.1, 0.1, 0.5]),), np.array([False, True, False])),
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
        (np.array([0.0, 0.0]), np.array([0.5, _VMFT_KAPPA_UPPER + 1.0]), np.array([0.0, 0.0])),
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
        (np.array([0.0, 0.0]), np.array([0.5, 0.5]), np.array([0.0, 0.0]), np.array([0.0, 2.0])),
        np.array([True, False]),
    ),
    (
        "jonespewsey_asym",
        jonespewsey_asym,
        (np.array([0.0, 0.0]), np.array([0.5, 0.5]), np.array([0.0, 0.0]), np.array([0.5, 1.2])),
        np.array([True, False]),
    ),
    (
        "inverse_batschelet",
        inverse_batschelet,
        (np.array([0.0, 0.0]), np.array([0.5, -0.5]), np.array([0.0, 0.0]), np.array([0.0, 0.0])),
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


@pytest.mark.parametrize("name, dist, params, expected", _ARGCHECK_CASES, ids=[c[0] for c in _ARGCHECK_CASES])
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
        (0.25, np.array([0.0, 0.2]), np.array([0.5, 0.6]), np.array([0.2, 0.3]), np.array([0.1, 0.2])),
    ),
]


@pytest.mark.parametrize("name, dist, args", _PDF_VECTOR_CASES, ids=[c[0] for c in _PDF_VECTOR_CASES])
def test_pdf_vectorized_shape_parameters(name, dist, args):
    vals = dist.pdf(*args)
    shapes = [np.shape(arg) for arg in args[1:]]  # skip x
    expected_shape = np.broadcast_shapes(*shapes) if shapes else ()
    assert isinstance(vals, np.ndarray)
    assert vals.shape == expected_shape
    assert np.all(np.isfinite(vals)), f"{name} pdf returned non-finite values"


_SCALAR_ONLY_CALLS = [
    ("vonmises_flattopped", lambda: vonmises_flattopped.pdf(0.1, mu=np.array([0.0, 0.1]), kappa=1.0, nu=0.1)),
    ("jonespewsey", lambda: jonespewsey.pdf(0.1, mu=0.0, kappa=np.array([1.0, 1.1]), psi=0.1)),
    (
        "jonespewsey_sineskewed",
        lambda: jonespewsey_sineskewed.pdf(0.1, xi=0.0, kappa=np.array([1.0, 1.1]), psi=0.1, lmbd=0.1),
    ),
    (
        "jonespewsey_asym",
        lambda: jonespewsey_asym.pdf(0.1, xi=0.0, kappa=np.array([1.0, 1.1]), psi=0.1, nu=0.2),
    ),
    (
        "inverse_batschelet",
        lambda: inverse_batschelet.pdf(0.1, xi=0.0, kappa=np.array([1.0, 1.1]), nu=0.2, lmbd=0.1),
    ),
    ("wrapstable", lambda: wrapstable.pdf(0.1, delta=np.array([0.0, 0.1]), alpha=1.0, beta=0.0, gamma=1.0)),
    ("katojones_ppf", lambda: katojones.ppf(0.5, mu=np.array([0.0, 0.1]), gamma=0.5, rho=0.2, lam=0.1)),
]


@pytest.mark.parametrize("name, call", _SCALAR_ONLY_CALLS, ids=[c[0] for c in _SCALAR_ONLY_CALLS])
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
        params={"rho": 4.0 / np.pi ** 2},
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

    default_high_tol = case.ppf_high_slope_tol if case.ppf_high_slope_tol is not None else max(case.ppf_tol * 50, 5e-8)
    default_low_tol = case.ppf_low_slope_tol if case.ppf_low_slope_tol is not None else default_high_tol

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


def _check_textbook_reference(case_id: str, *, rounding: Optional[int] = None, significant: Optional[int] = None):
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
    cdf_callable = lambda values: _evaluate_array(case.factory.cdf, values, **case.params)
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
    expected_m2 = special.iv(2, kappa_true) / special.i0(kappa_true) * np.exp(2j * mu_true)

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
    np.testing.assert_allclose(logpdf_vals[mask], np.log(pdf_vals[mask]), atol=5e-10, rtol=0.0)


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
    
def _assert_rvs_reasonable(dist, size=256, seed=123, uniform_tol=0.05, cdf_callable=None):
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
