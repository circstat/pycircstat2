"""Size-aware MAP-penalty degeneracy guard (port of circlss ``internal-degen.R``).

The guard adds a diffuse-prior penalty to a *lss family's degeneracy-prone
natural parameter, active ONLY in a reweighted circ_mix M-step (``map_lambda>0``);
for a standalone ``circ_gam`` it is inert and the fit is byte-for-byte unchanged.
pycircstat2 has no circ_mix yet, so these tests are the only exercise the guard
gets — they lock the kernel math, the assembler placement, the activation gate,
and the "standalone unchanged" invariant against the R implementation.
"""

import numpy as np
import polars as pl
import pytest

from pycircstat2 import distributions as D
from pycircstat2.distributions import (
    _degen_active,
    _degen_boundary_sym,
    _degen_boundary_upper,
    _degen_linear,
    _degen_ridge,
    _lss_map_penalty,
)
from pycircstat2.regression import circ_gam

# Expected (parameter, kernel-type) spec per family — the circlss roster.
# cartlss is deliberately guardless (bounded power-of-cosine, no wall).
_EXPECTED = {
    "vmlss": [("kappa", "linear")],
    "wclss": [("rho", "linear")],
    "wnlss": [("rho", "bupper")],
    "cardlss": [("rho", "bupper")],
    "cartlss": [],
    "pnlss": [("mu1", "ridge"), ("mu2", "ridge")],
    "vmftlss": [("kappa", "linear"), ("nu", "bsym")],
    "jplss": [("kappa", "linear"), ("psi", "ridge")],
    "ssjplss": [("kappa", "linear"), ("psi", "ridge"), ("lmbd", "bsym")],
    "ajplss": [("kappa", "linear"), ("psi", "ridge"), ("nu", "bsym")],
    "ibslss": [("kappa", "linear"), ("nu", "bsym"), ("lmbd", "bsym")],
    "kjlss": [("u1", "ridge"), ("u2", "ridge")],
}


@pytest.mark.parametrize("alias,expected", list(_EXPECTED.items()))
def test_degen_spec_resolves_to_lp_params(alias, expected):
    """Each family declares the circlss spec, and every penalized parameter is a
    real LP coordinate of that family (name, not index — no off-by-one)."""
    fam = getattr(D, alias)
    assert [k.param for k in fam.degen] == [p for p, _ in expected]
    assert all(k.param in fam.params for k in fam.degen)
    assert fam.map_lambda is None  # the module alias never carries a live lambda


def test_kernels_match_closed_form():
    v = np.array([0.1, 0.5, 0.9])
    k = _degen_linear("x", 3.0)  # rho = s*v
    assert np.allclose(k.rho0(v), 3 * v)
    assert np.allclose(k.rho1(v), 3.0)
    assert np.allclose(k.rho2(v), 0.0)
    k = _degen_ridge("x", 2.0)  # rho = s*v^2
    assert np.allclose(k.rho0(v), 2 * v * v)
    assert np.allclose(k.rho1(v), 4 * v)
    assert np.allclose(k.rho2(v), 4.0)
    k = _degen_boundary_upper("x", 1.0, 1.0)  # rho = -s*log(1 - v/vmax)
    z = 1 - v
    assert np.allclose(k.rho0(v), -np.log(z))
    assert np.allclose(k.rho1(v), 1 / z)
    assert np.allclose(k.rho2(v), 1 / z**2)
    k = _degen_boundary_sym("x", 1.0, 1.0)  # rho = -s*log(1 - (v/vmax)^2)
    z = 1 - v * v
    assert np.allclose(k.rho0(v), -np.log(z))
    assert np.allclose(k.rho1(v), 2 * v / z)
    assert np.allclose(k.rho2(v), (2 / z) + (4 * v * v) / z**2)


def test_boundary_kernel_finite_on_wall():
    """eps-floored denominator keeps a parameter sitting exactly on the wall
    large-but-finite (not inf/nan)."""
    ku = _degen_boundary_upper("x", 0.5, 1.0)
    ks = _degen_boundary_sym("x", 1.0, 1.0)
    for k, wall in ((ku, 0.5), (ks, 1.0)):
        w = np.array([wall])
        assert np.all(np.isfinite([k.rho0(w)[0], k.rho1(w)[0], k.rho2(w)[0]]))


def test_gate_inert_unless_positive_map_lambda():
    fam = D.vmlss(name="vmlss")
    assert _degen_active(fam) is False  # map_lambda None
    fam.map_lambda = 0.0
    assert _degen_active(fam) is False
    fam.map_lambda = np.inf
    assert _degen_active(fam) is False
    fam.map_lambda = 2.0
    assert _degen_active(fam) is True


def test_assembler_placement_and_values():
    """l1 gradient entries and the l2 DIAGONAL only, at the family's own trind
    columns; values match circlss's ``.lss_map_penalty`` byte-for-byte."""
    # vmlss: linear on kappa (lam=0.5, kappa=4) -> l0=-2, l1[kappa]=-0.5, l2[kk]=0
    fam = D.vmlss(name="vmlss")
    fam.map_lambda = 0.5
    p = _lss_map_penalty(fam, {"mu": np.zeros(3), "kappa": np.full(3, 4.0)}, 0.5)
    kk = int(fam.tri["i2"][1, 1])
    assert np.allclose(p["l0"], -2.0)
    assert np.allclose(p["l1"][:, 1], -0.5) and np.allclose(p["l1"][:, 0], 0.0)
    assert np.allclose(p["l2"], 0.0)  # linear has zero curvature
    assert np.allclose(p["l2"][:, kk], 0.0)

    # kjlss: ridge on chart coords u1,u2 (lam=1.5) -> l1=-1.2/-0.9, l2 diag=-3
    famk = D.kjlss(name="kjlss")
    famk.map_lambda = 1.5
    pk = _lss_map_penalty(
        famk,
        {"mu": np.zeros(2), "gamma": np.full(2, 0.5),
         "u1": np.full(2, 0.4), "u2": np.full(2, 0.3)},
        1.5,
    )
    u1u1, u2u2 = int(famk.tri["i2"][2, 2]), int(famk.tri["i2"][3, 3])
    assert np.allclose(pk["l1"][:, 2], -1.2) and np.allclose(pk["l1"][:, 3], -0.9)
    assert np.allclose(pk["l2"][:, u1u1], -3.0)
    assert np.allclose(pk["l2"][:, u2u2], -3.0)
    # off-diagonal l2 columns stay zero (penalty is separable)
    off = [c for c in range(pk["l2"].shape[1]) if c not in (u1u1, u2u2)]
    assert np.allclose(pk["l2"][:, off], 0.0)


def _intercept_ll(fam, targets, lam, deriv=1):
    """ll() on an intercept-only design at given natural params, with/without the
    penalty active. Returns (inert_result, active_result)."""
    n_lp = fam.n_lp
    n = 60
    X = np.column_stack([np.ones(n)] * n_lp)
    lpi = [np.array([j]) for j in range(n_lp)]
    coef = np.array([fam.links[j].link(float(targets[p]))
                     for j, p in enumerate(fam.params)])
    rng = np.random.default_rng(0)
    y = np.mod(rng.uniform(0, 2 * np.pi, n), 2 * np.pi)
    fam.map_lambda = None
    r0 = fam.ll(y, X, coef, lpi=lpi, deriv=deriv)
    fam.map_lambda = lam
    r1 = fam.ll(y, X, coef, lpi=lpi, deriv=deriv)
    return r0, r1


def test_ll_scalar_and_gradient_shift_vonmises():
    """The penalty shifts the objective by -lam*sum(rho0) and pulls the kappa
    gradient down by exactly the link-chained penalty; l0 (E-step density) is
    untouched. This is the A1(kappa_hat) = Rbar_w - c/N_k identity at the ll seam."""
    fam = D.vmlss(name="vmlss")
    lam, kappa, n = 0.5, 4.0, 60
    r0, r1 = _intercept_ll(fam, {"mu": 0.3, "kappa": kappa}, lam)
    assert np.allclose(r0["l0"], r1["l0"])  # density never penalized
    assert np.isclose(r1["l"] - r0["l"], -lam * n * kappa)  # rho0(linear)=kappa
    # log link: d kappa/d eta = kappa; rho1=1 -> Δgrad = -lam*kappa*sum(X)
    assert np.isclose(r1["lb"][1] - r0["lb"][1], -lam * kappa * n)
    assert np.isclose(r1["lb"][0], r0["lb"][0])  # mu-intercept unaffected


@pytest.mark.parametrize(
    "alias,targets",
    [
        ("wclss", {"mu": 0.3, "rho": 0.6}),
        ("pnlss", {"mu1": 0.5, "mu2": 0.5}),
        ("cardlss", {"mu": 0.3, "rho": 0.4}),
        ("wnlss", {"mu": 0.3, "rho": 0.8}),
        ("vmftlss", {"mu": 0.3, "kappa": 4.0, "nu": 0.5}),
        ("jplss", {"mu": 0.3, "kappa": 3.0, "psi": 0.5}),
        ("ssjplss", {"xi": 0.3, "kappa": 3.0, "psi": 0.5, "lmbd": 0.4}),
        ("ajplss", {"xi": 0.3, "kappa": 3.0, "psi": 0.5, "nu": 0.4}),
        ("ibslss", {"xi": 0.3, "kappa": 3.0, "nu": 0.4, "lmbd": 0.3}),
        ("kjlss", {"mu": 0.3, "gamma": 0.5, "u1": 0.4, "u2": 0.3}),
    ],
)
def test_active_path_finite_and_moves(alias, targets):
    """Every guarded family: active penalty keeps l/lb/lbb/l0 finite, leaves l0
    untouched, and actually moves the gradient (the penalty bites)."""
    fam = getattr(D, alias)(name=alias)
    r0, r1 = _intercept_ll(fam, targets, lam=0.5)
    for key in ("l", "lb", "lbb", "l0"):
        assert np.all(np.isfinite(r1[key]))
    assert np.allclose(r0["l0"], r1["l0"])
    assert not np.allclose(r0["lb"], r1["lb"])


def test_cartlss_has_no_guard():
    """cartlss declares no degen -> the gradient never moves, whatever map_lambda."""
    fam = D.cartlss(name="cartlss")
    r0, r1 = _intercept_ll(fam, {"mu": 0.3, "zeta": 2.0}, lam=0.5)
    assert not fam.degen
    assert np.allclose(r0["lb"], r1["lb"])
    assert np.allclose(r0["l"], r1["l"])


def test_standalone_circ_gam_unchanged_and_alias_unmutated():
    """A standalone circ_gam never activates the guard: it fits normally and the
    module-level *lss alias is not mutated (map_lambda stays None)."""
    rng = np.random.default_rng(0)
    n = 250
    x = rng.uniform(0, 1, n)
    mu = 2 * np.arctan(1.2 * np.sin(2 * np.pi * x))
    y = np.mod(rng.vonmises(mu, 3.0), 2 * np.pi)
    df = pl.DataFrame({"y": y, "x": x})
    fit = circ_gam(["y ~ s(x)", "~ 1"], df, family="vmlss")
    assert np.isfinite(float(fit.logLik))
    assert D.vmlss.map_lambda is None
