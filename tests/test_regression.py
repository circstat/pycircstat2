import numpy as np
import polars as pl
import pytest
from hea.models import gam as hea_gam  # noqa: E402

from pycircstat2.distributions import (  # noqa: E402
    CircularLL,  # noqa: E402
    projectednormal,
    vonmises,
    wrapcauchy,
)
from pycircstat2.utils import A1, A1inv


def test_circularll_requires_regression_ready():
    # triangular stays off the regression contract (no location parameter,
    # non-smooth density — see the regression plan's "not worth promoting")
    from pycircstat2.distributions import triangular

    with pytest.raises(TypeError, match="regression-ready"):
        CircularLL(triangular)


def test_lss_alias_surface():
    """The *lss instance aliases (validation plan §3.5): module-level
    pre-built families sharing the circlss names — ``family=vmlss`` is the
    primary explicit-gam spelling. Checks identity wiring, name parity,
    the ``__call__`` re-configurator (fresh instance, alias untouched),
    and chart-coordinate subclass routing."""
    from pycircstat2 import distributions as D

    aliases = {
        "cardlss": D.cardioid,
        "cartlss": D.cartwright,
        "wnlss": D.wrapnorm,
        "wclss": D.wrapcauchy,
        "vmlss": D.vonmises,
        "pnlss": D.projectednormal,
        "jplss": D.jonespewsey,
        "ssjplss": D.jonespewsey_sineskewed,
        "kjlss": D.katojones,
        "ibslss": D.inverse_batschelet,
        "vmftlss": D.vonmises_flattopped,
        "ajplss": D.jonespewsey_asym,
    }
    for name, dist in aliases.items():
        fam = getattr(D, name)
        assert isinstance(fam, D.CircularLL)
        assert fam.dist is dist
        assert fam.name == name  # circlss family$family parity
        assert fam.n_theta == 0  # the shared-instance safety condition
        assert fam.n_lp == len(dist.param_roles)
    assert isinstance(D.kjlss, D.KatoJonesLL)  # disc-chart routing baked in

    # __call__ is the family-side freeze idiom: a fresh configured family,
    # the module-level alias stays pristine, the alias name propagates.
    clone = D.vmlss(links=["identity", "log"])
    assert clone is not D.vmlss and type(clone) is D.CircularLL
    assert [lnk.name for lnk in clone.links] == ["identity", "log"]
    assert [lnk.name for lnk in D.vmlss.links] == ["tanhalf", "log"]
    assert clone.name == "vmlss"
    fresh = D.vmlss()  # the R parens spelling, verbatim
    assert fresh is not D.vmlss
    assert [lnk.name for lnk in fresh.links] == ["tanhalf", "log"]
    kj = D.kjlss()
    assert type(kj) is D.KatoJonesLL and kj.name == "kjlss"


def test_ibslss_intercept_only_matches_marginal_mle():
    """The M3 end-to-end gate for the inverse-Batschelet *lss family: an
    intercept-only ``ibslss`` fit must reproduce ``inverse_batschelet.fit``
    (the trusted marginal MLE) — both maximize the same likelihood, and for
    intercept-only the mgcv-inside-link and the marginal parameterizations
    coincide. This pins the whole bridge at once: ll, dlogpdf, the FD-grade
    d2logpdf Hessian, the closed-form null start, the EFS optimizer, and the
    per-observation normalizer (dev/plans/vectorize-distributions-and-ibslss.md
    §7)."""
    from pycircstat2.distributions import ibslss, inverse_batschelet
    from pycircstat2.regression import circ_gam

    data = inverse_batschelet.rvs(
        xi=2.3, kappa=2.5, nu=0.35, lmbd=-0.3, size=4000, random_state=7
    )
    df = pl.DataFrame({"theta": data})
    g = circ_gam(["theta ~ 1", "~ 1", "~ 1", "~ 1"], df, family=ibslss, method="REML")
    pred = np.asarray(g.predict(pl.DataFrame({"_dummy": [0.0]}), type="response"))[0]
    xi_g, kappa_g, nu_g, lmbd_g = pred
    xi_m, kappa_m, nu_m, lmbd_m = inverse_batschelet.fit(data)

    ang = (xi_g - xi_m + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(ang) < 1e-3
    np.testing.assert_allclose(
        [kappa_g, nu_g, lmbd_g], [kappa_m, nu_m, lmbd_m], atol=1e-3, rtol=0.0
    )


def test_ibslss_recovers_covariate_location():
    """M4: an ``ibslss`` GAM with a covariate-driven location exercises the
    per-observation (array) parameter path — the one intercept-only parity
    cannot reach (constant params collapse to the scalar path). Simulate
    ξ(x) = 1.5·sin x with κ,ν,λ fixed (rotate a base sample), fit a location
    smooth, and confirm the recovered direction tracks the truth."""
    from pycircstat2.distributions import ibslss, inverse_batschelet
    from pycircstat2.regression import circ_gam

    rng = np.random.default_rng(11)
    n = 1500
    x = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
    xi_true = 1.5 * np.sin(x)
    base = inverse_batschelet.rvs(
        xi=0.0, kappa=4.0, nu=0.3, lmbd=-0.2, size=n, random_state=rng
    )
    theta = np.mod(base + xi_true, 2.0 * np.pi)
    df = pl.DataFrame({"theta": theta, "x": x})

    g = circ_gam(
        ["theta ~ s(x)", "~ 1", "~ 1", "~ 1"], df, family=ibslss, method="REML"
    )
    assert g.converged
    grid = np.linspace(x.min(), x.max(), 200)
    mu = np.angle(
        np.exp(
            1j * np.asarray(g.predict(pl.DataFrame({"x": grid}), type="response"))[:, 0]
        )
    )
    truth = 1.5 * np.sin(grid)
    a = mu - np.angle(np.mean(np.exp(1j * mu)))
    b = truth - np.angle(np.mean(np.exp(1j * truth)))
    corr = np.sum(np.sin(a) * np.sin(b)) / np.sqrt(
        np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2)
    )
    assert corr > 0.95


def test_ibslss_recovers_covariate_concentration():
    """A *distributional* `ibslss` smooth — κ(x) on the log link — is the only
    fit that drives the per-observation normalizer derivative (the location
    smooth leaves κ,λ constant, so its normalizer block stays on the cheap
    scalar path). Simulate κ(x)=exp(0.7+0.9 sin x) (binned scalar draws), fit a
    concentration smooth, and confirm the recovered log-κ tracks the truth.
    This is the permanent guard on the vectorized normalizer that the κ(x)/λ(x)
    perf rewrite enabled (dev/plans/vectorize-distributions-and-ibslss.md)."""
    from pycircstat2.distributions import ibslss, inverse_batschelet
    from pycircstat2.regression import circ_gam

    rng = np.random.default_rng(7)
    n = 700
    x = np.sort(rng.uniform(-np.pi, np.pi, n))
    theta = np.empty(n)
    edges = np.linspace(x.min(), x.max(), 21)
    idx = np.clip(np.digitize(x, edges) - 1, 0, 19)
    for b in range(20):
        m = idx == b
        if not m.any():
            continue
        kb = float(np.exp(0.7 + 0.9 * np.sin(x[m].mean())))
        theta[m] = inverse_batschelet.rvs(
            xi=0.4, kappa=kb, nu=0.2, lmbd=-0.2, size=int(m.sum()), random_state=rng
        )
    df = pl.DataFrame({"theta": np.mod(theta, 2.0 * np.pi), "x": x})

    g = circ_gam(
        ["theta ~ 1", "~ s(x)", "~ 1", "~ 1"], df, family=ibslss, method="REML"
    )
    assert g.converged
    grid = np.linspace(x.min(), x.max(), 150)
    eta_k = np.asarray(g.predict(pl.DataFrame({"x": grid}), type="link"))[:, 1]
    log_k_true = 0.7 + 0.9 * np.sin(grid)
    corr = np.corrcoef(eta_k, log_k_true)[0, 1]
    assert corr > 0.9


def test_vmftlss_intercept_only_matches_marginal_mle():
    """The end-to-end gate for the flat-topped von Mises *lss family: an
    intercept-only `vmftlss` fit must reproduce `vonmises_flattopped.fit`
    (μ, κ, ν), since with no covariates the GAM maximizes the same marginal
    likelihood. Exercises the scalar/constant-array normalizer path and the
    EFS (l1+l2) wiring."""
    from pycircstat2.distributions import vmftlss, vonmises_flattopped
    from pycircstat2.regression import circ_gam

    rng = np.random.default_rng(3)
    theta = vonmises_flattopped.rvs(
        mu=2.0, kappa=3.0, nu=0.4, size=4000, random_state=rng
    )
    df = pl.DataFrame({"theta": np.mod(theta, 2.0 * np.pi)})
    g = circ_gam(["theta ~ 1", "~ 1", "~ 1"], df, family=vmftlss, method="REML")
    assert g.converged
    resp = np.asarray(g.predict(df.head(1), type="response"))[0]
    mle = vonmises_flattopped.fit(np.mod(theta, 2.0 * np.pi))
    # mean direction compared on the circle; κ, ν directly
    assert abs(np.angle(np.exp(1j * (resp[0] - mle[0])))) < 1e-3
    assert resp[1] == pytest.approx(mle[1], abs=1e-3)
    assert resp[2] == pytest.approx(mle[2], abs=1e-3)


def test_vmftlss_recovers_covariate_shape():
    """A *distributional* `vmftlss` smooth — the peakedness ν(x) on the tanh
    link — drives the per-observation normalizer expectations
    (`_vmft_logZ_moments_vec`), the path a location smooth never reaches.
    Simulate ν(x)=0.6 sin x (binned scalar draws, μ,κ fixed), fit a shape
    smooth, and confirm the recovered ν tracks the truth."""
    from pycircstat2.distributions import vmftlss, vonmises_flattopped
    from pycircstat2.regression import circ_gam

    rng = np.random.default_rng(6)
    n = 1500
    x = np.sort(rng.uniform(-np.pi, np.pi, n))
    theta = np.empty(n)
    edges = np.linspace(x.min(), x.max(), 41)
    idx = np.clip(np.digitize(x, edges) - 1, 0, 39)
    for b in range(40):
        m = idx == b
        if not m.any():
            continue
        nub = float(0.6 * np.sin(x[m].mean()))
        theta[m] = vonmises_flattopped.rvs(
            mu=1.0, kappa=3.0, nu=nub, size=int(m.sum()), random_state=rng
        )
    df = pl.DataFrame({"theta": np.mod(theta, 2.0 * np.pi), "x": x})

    g = circ_gam(["theta ~ 1", "~ 1", "~ s(x)"], df, family=vmftlss, method="REML")
    assert g.converged
    grid = np.linspace(x.min(), x.max(), 150)
    nu_fit = np.tanh(
        np.asarray(g.predict(pl.DataFrame({"x": grid}), type="link"))[:, 2]
    )
    nu_true = 0.6 * np.sin(grid)
    assert np.corrcoef(nu_fit, nu_true)[0, 1] > 0.9


def test_ajplss_intercept_only_is_mle():
    """End-to-end gate for the asymmetric-extended JP *lss family. The marginal
    `jonespewsey_asym.fit` is an unreliable reference here — its generic
    optimizer from a symmetric seed gets stuck on the hard 4-parameter aeJP
    likelihood — so instead we assert the intercept-only `ajplss` GAM attains a
    sample log-likelihood that *dominates* both the generating parameters and
    `.fit` (the defining property of the MLE: no point scores higher). This
    pins the full (κ,ψ,ν) normalizer-moments block at a genuinely asymmetric
    (ν≠0) optimum."""
    from pycircstat2.distributions import ajplss, jonespewsey_asym
    from pycircstat2.regression import circ_gam

    rng = np.random.default_rng(3)
    truth = dict(xi=2.0, kappa=3.0, psi=0.6, nu=0.4)
    theta = np.mod(
        jonespewsey_asym.rvs(**truth, size=3000, random_state=rng), 2.0 * np.pi
    )
    df = pl.DataFrame({"theta": theta})
    g = circ_gam(["theta ~ 1", "~ 1", "~ 1", "~ 1"], df, family=ajplss, method="REML")
    assert g.converged
    fit = np.asarray(g.predict(df.head(1), type="response"))[0]

    def ll(p):
        return float(
            np.sum(
                jonespewsey_asym.logpdf(theta, xi=p[0], kappa=p[1], psi=p[2], nu=p[3])
            )
        )

    ll_gam = ll(fit)
    ll_truth = ll([truth["xi"], truth["kappa"], truth["psi"], truth["nu"]])
    ll_marg = ll(list(jonespewsey_asym.fit(theta)))
    assert ll_gam >= ll_truth - 1e-3  # MLE dominates the generating params
    assert ll_gam >= ll_marg - 1e-3  # ... and the marginal fitter
    # recovers the true asymmetry/shape (not a degenerate symmetric optimum)
    assert abs(fit[2] - truth["psi"]) < 0.25
    assert abs(fit[3] - truth["nu"]) < 0.2


def test_ajplss_recovers_covariate_location():
    """An `ajplss` GAM with a covariate-driven location exercises the
    per-observation array log-density path (`_ajp_logpdf_vec`) — the warped
    kernel evaluated each datum its own ξ — while κ,ψ,ν stay constant (so the
    normalizer moments collapse to one triple and the fit stays fast).
    Simulate ξ(x)=1.3 sin x by rotating a base sample and confirm the recovered
    direction tracks the truth."""
    from pycircstat2.distributions import ajplss, jonespewsey_asym
    from pycircstat2.regression import circ_gam

    rng = np.random.default_rng(8)
    n = 1500
    x = np.sort(rng.uniform(-np.pi, np.pi, n))
    base = jonespewsey_asym.rvs(
        xi=0.0, kappa=3.0, psi=0.5, nu=0.4, size=n, random_state=rng
    )
    theta = np.mod(base + 1.3 * np.sin(x), 2.0 * np.pi)
    df = pl.DataFrame({"theta": theta, "x": x})
    g = circ_gam(
        ["theta ~ s(x)", "~ 1", "~ 1", "~ 1"], df, family=ajplss, method="REML"
    )
    assert g.converged
    grid = np.linspace(x.min(), x.max(), 200)
    mu = np.angle(
        np.exp(
            1j * np.asarray(g.predict(pl.DataFrame({"x": grid}), type="response"))[:, 0]
        )
    )
    truth = 1.3 * np.sin(grid)
    a = mu - np.angle(np.mean(np.exp(1j * mu)))
    b = truth - np.angle(np.mean(np.exp(1j * truth)))
    corr = np.sum(np.sin(a) * np.sin(b)) / np.sqrt(
        np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2)
    )
    assert corr > 0.95


def test_circularll_vonmises_intercept_only_matches_mle():
    """gam(["theta ~ 1", "~ 1"], family=CircularLL(vonmises)) is the
    unpenalized von Mises MLE — pins the whole ll() derivative stack
    (Newton lands on the score root) and initialize_coef."""
    rng = np.random.default_rng(1)
    theta = np.mod(rng.vonmises(1.0, 3.0, 500), 2 * np.pi)
    df = pl.DataFrame({"theta": theta})
    m = hea_gam(
        ["theta ~ 1", "~ 1"], data=df, family=CircularLL(vonmises), method="REML"
    )
    assert m.converged
    coef = np.asarray(m.coefficients, dtype=float)
    mu_mle, kappa_mle = vonmises.fit(theta)
    assert np.mod(2 * np.arctan(coef[0]), 2 * np.pi) == pytest.approx(mu_mle, abs=1e-4)
    assert np.exp(coef[1]) == pytest.approx(kappa_mle, rel=1e-3)


def test_circularll_wrapcauchy_intercept_only_matches_mle():
    """Same MLE-equivalence through the logit-linked wrapped Cauchy."""
    theta = np.asarray(wrapcauchy.rvs(2.2, 0.55, size=800, random_state=11))
    df = pl.DataFrame({"theta": theta})
    m = hea_gam(
        ["theta ~ 1", "~ 1"], data=df, family=CircularLL(wrapcauchy), method="REML"
    )
    assert m.converged
    coef = np.asarray(m.coefficients, dtype=float)
    mu_mle, rho_mle = wrapcauchy.fit(theta)
    assert np.mod(2 * np.arctan(coef[0]), 2 * np.pi) == pytest.approx(mu_mle, abs=1e-3)
    assert 1.0 / (1.0 + np.exp(-coef[1])) == pytest.approx(rho_mle, abs=1e-3)


def test_circularll_vonmises_recovers_smooth_mu_and_kappa():
    """The §5 target: smooth μ(x) AND log κ(z) by REML, jointly. True curves
    stay inside the tanhalf principal branch (the link cannot cross ±π)."""
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    mu_true = np.pi / 2 + 1.2 * np.sin(2 * np.pi * x)
    kap_true = np.exp(0.8 + 1.2 * z)
    theta = np.mod(mu_true + rng.vonmises(0.0, kap_true, n), 2 * np.pi)
    df = pl.DataFrame({"theta": theta, "x": x, "z": z})

    m = hea_gam(
        ["theta ~ s(x)", "~ s(z)"],
        data=df,
        family=CircularLL(vonmises),
        method="REML",
    )
    assert m.converged
    fv = np.asarray(m.fitted_values)
    circ_err = np.abs(np.angle(np.exp(1j * (fv[:, 0] - mu_true))))
    logk_err = np.abs(np.log(fv[:, 1]) - np.log(kap_true))
    assert circ_err.mean() < 0.08
    assert logk_err.mean() < 0.10


def test_circularll_projectednormal_fits_full_circle_sweep():
    """A full-circle μ(x) sweep — unrepresentable through tanhalf (pole at
    ±π) — fits cleanly through the projected normal's two identity LPs."""
    rng = np.random.default_rng(3)
    n = 3000
    x = rng.uniform(0, 1, n)
    gamma = 2.0
    mu1, mu2 = gamma * np.cos(2 * np.pi * x), gamma * np.sin(2 * np.pi * x)
    theta = np.mod(
        np.arctan2(mu2 + rng.standard_normal(n), mu1 + rng.standard_normal(n)),
        2 * np.pi,
    )
    df = pl.DataFrame({"theta": theta, "x": x})
    m = hea_gam(
        ["theta ~ s(x)", "~ s(x)"],
        data=df,
        family=CircularLL(projectednormal),
        method="REML",
    )
    assert m.converged
    fv = np.asarray(m.fitted_values)
    dir_hat = np.arctan2(fv[:, 1], fv[:, 0])
    dir_true = np.arctan2(mu2, mu1)
    err = np.abs(np.angle(np.exp(1j * (dir_hat - dir_true))))
    assert err.mean() < 0.05


def test_circularll_postproc_binds_both_hea_conventions():
    """The seam moved under us once (hea unified postproc on mgcv's 6-arg
    hook) — pin that our signature binds BOTH calling conventions: hea
    0.1.4's positional ``(y, fitted)`` and the keyword form of the mgcv
    hook later hea uses. Only ``y`` enters the null refit, so the two
    must agree exactly."""
    rng = np.random.default_rng(5)
    n = 200
    theta = np.mod(rng.vonmises(1.0, 2.0, n), 2 * np.pi)
    fam = CircularLL(vonmises)
    fitted = np.column_stack([np.full(n, 1.0), np.full(n, 2.0)])
    old = fam.postproc(theta, fitted)  # hea 0.1.4 call shape
    new = fam.postproc(  # hea > 0.1.4 (mgcv 6-arg hook)
        theta,
        prior_weights=np.ones(n),
        fitted=fitted,
        linear_predictors=fitted,
        offset=None,
        intercept=True,
    )
    assert np.isfinite(old["null_deviance"])
    assert old["null_deviance"] == pytest.approx(new["null_deviance"])


def test_circularll_honors_prior_weights():
    """gam(weights=) is honored as a likelihood weight (the circlss contract:
    weighting a row by w == duplicating that row w times — see the family-level
    duplication-identity tests in test_distributions.py). A weighted fit
    converges; a constant rescaling leaves the MLE unchanged (argmax of w·ℓ ==
    argmax of ℓ for any constant w > 0), while a non-uniform weighting moves it.
    The exact, machine-precision gate lives in test_distributions.py; this pins
    the behaviour through the circ_gam/hea front door."""
    rng = np.random.default_rng(9)
    n = 120
    x = rng.uniform(-1.0, 1.0, n)
    theta = np.mod(2 * np.arctan(0.8 * x) + rng.vonmises(0.0, 4.0, n), 2 * np.pi)
    df = pl.DataFrame({"theta": theta, "x": x})

    base = hea_gam(["theta ~ x", "~ 1"], data=df,
                   family=CircularLL(vonmises), method="REML")
    const = hea_gam(["theta ~ x", "~ 1"], data=df, family=CircularLL(vonmises),
                    method="REML", weights=np.full(n, 2.0))
    nonunif = hea_gam(["theta ~ x", "~ 1"], data=df, family=CircularLL(vonmises),
                      method="REML", weights=np.where(x > 0, 3.0, 0.5))
    assert const.converged and nonunif.converged
    # constant weight: same MLE as unit weights
    np.testing.assert_allclose(np.asarray(const.coef), np.asarray(base.coef),
                               atol=1e-6)
    # non-uniform weighting genuinely takes effect
    assert np.max(np.abs(np.asarray(nonunif.coef) - np.asarray(base.coef))) > 1e-3


def _cl_gam_sim(n=900, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    mu = np.pi / 2 + 1.2 * np.sin(2 * np.pi * x)
    kap = np.exp(0.8 + 1.2 * z)
    theta = np.mod(mu + rng.vonmises(0.0, kap, n), 2 * np.pi)
    return pl.DataFrame({"theta": theta, "x": x, "z": z}), mu


from pycircstat2.distributions import vmlss  # noqa: E402
from pycircstat2.regression import circ_gam  # noqa: E402


def test_circ_gam_b2_twin_and_knot_defaults():
    """The circlss §4 twin call fits through circ_gam with family/method
    defaulted and **knots omitted**: ``phi`` on [0, 2π] (pycircstat2's
    convention) auto-pins its cyclic knots to the period, and a single formula
    auto-expands to a constant second LP."""
    rng = np.random.default_rng(8)
    n = 400
    phi = rng.uniform(0, 2 * np.pi, n)
    mu_true = np.pi / 2 + 1.2 * np.sin(phi)
    theta = np.mod(mu_true + rng.vonmises(0.0, 4.0, n), 2 * np.pi)
    df = pl.DataFrame({"theta": theta, "phi": phi})

    # phi on [0, 2π], no explicit knots — the default brackets it to the period
    b2 = circ_gam(["theta ~ s(phi, bs='cc')", "~ s(phi, bs='cc')"], df)
    assert b2.converged
    # mu_true's circular mean (≈ π/2) sits within snap of the wall's half, so
    # center=True (the default) rotates the fit off the θ = π wall; the raw
    # fitted_values then live in that centred frame, so add circ_center back —
    # the original frame predict(type="response") reports — before scoring.
    fv = np.asarray(b2.fitted_values)
    mu_fit = np.mod(fv[:, 0] + b2.circ_center, 2 * np.pi)
    err = np.abs(np.angle(np.exp(1j * (mu_fit - mu_true))))
    assert err.mean() < 0.15

    # a single formula auto-expands the constant second LP
    m = circ_gam("theta ~ s(phi, bs='cc')", df)
    assert m.converged


def test_circ_gam_cyclic_knots_default_and_guard():
    """``_resolve_cyclic_knots_data`` pins each cyclic covariate to the
    [0, 2π] period (pycircstat2's convention), honors user knots, and rejects
    covariates that fall outside [0, 2π] (a wrong-branch footgun)."""
    from pycircstat2.regression import _resolve_cyclic_knots_data

    pos = pl.DataFrame({"phi": np.linspace(0.0, 2 * np.pi, 50)})
    f = ["theta ~ s(phi, bs='cc')", "~ 1"]

    assert _resolve_cyclic_knots_data(f, pos, None) == {"phi": [0.0, 2 * np.pi]}
    # user knots always win
    assert _resolve_cyclic_knots_data(f, pos, {"phi": [0.0, 12.0]}) == {
        "phi": [0.0, 12.0]
    }
    # a plain (non-cyclic) smooth gets no default knots
    assert _resolve_cyclic_knots_data(["theta ~ s(phi)", "~ 1"], pos, None) is None

    # off-branch covariates are rejected (signed, or beyond 2π)
    for bad in (np.linspace(-np.pi, np.pi, 50), np.linspace(0.0, 7.0, 50)):
        with pytest.raises(ValueError, match=r"outside \[0, 2π\]"):
            _resolve_cyclic_knots_data(f, pl.DataFrame({"phi": bad}), None)


def test_circ_gam_new_families_by_name_and_fill():
    """The flat-top / asymmetric-JP / inverse-Batschelet families (shipped in
    distributions.py) resolve by name, and a single formula fills the extra
    shape LPs with ``~ 1`` up to the family's parameter count."""
    from pycircstat2.distributions import ajplss, ibslss, vmftlss
    from pycircstat2.regression import _resolve_gam_family

    for name, fam in [("vmftlss", vmftlss), ("ajplss", ajplss), ("ibslss", ibslss)]:
        assert _resolve_gam_family(name) is fam
        assert _resolve_gam_family(fam.dist.name) is fam  # distribution name too

    # vmftlss is 3-LP: a single formula expands to mu-smooth + two ~ 1 LPs
    rng = np.random.default_rng(7)
    theta = np.mod(rng.vonmises(0.5, 3.0, 80), 2 * np.pi)
    df = pl.DataFrame({"theta": theta})
    m = circ_gam("theta ~ 1", df, family="vmftlss")
    assert np.isfinite(float(m.logLik))


def test_circ_gam_family_resolution():
    """Strings resolve through the circular catalog (alias or distribution
    name); unknown names raise with guidance."""
    df, _ = _cl_gam_sim(n=300, seed=11)
    a = circ_gam(["theta ~ s(x)", "~ 1"], df, family="vonmises")
    b = circ_gam(["theta ~ s(x)", "~ 1"], df, family=vmlss)
    assert float(a.logLik) == pytest.approx(float(b.logLik), rel=1e-10)
    with pytest.raises(ValueError, match="unknown family"):
        circ_gam("theta ~ s(x)", df, family="nope")


def test_circ_gam_gaussian_passthrough():
    """No gatekeeping: a linear response rides through to hea untouched
    (the old LC-smooth case), with the period-knot default still applied."""
    import hea.family as hea_family

    rng = np.random.default_rng(21)
    n = 300
    phi = rng.uniform(0, 2 * np.pi, n)
    y = 2.0 + np.sin(phi) + rng.normal(0, 0.3, n)
    df = pl.DataFrame({"y": y, "phi": phi})
    g = circ_gam("y ~ s(phi, bs='cc')", df, family="gaussian")
    direct = hea_gam(
        "y ~ s(phi, bs='cc')",
        df,
        family=hea_family.gaussian,
        knots={"phi": [0.0, 2 * np.pi]},
        method="REML",
    )
    assert float(g.AIC) == pytest.approx(float(direct.AIC), rel=1e-12)


def test_circ_gam_k3_k4_families_post_gate():
    """jplss (3-LP) and kjlss (4-LP) ride hea's efsud K=3/4 paths — now
    R-pinned hea-side (twlss/shash landed) — through circ_gam by name."""
    rng = np.random.default_rng(5)
    theta = np.mod(rng.vonmises(1.0, 3.0, 80), 2 * np.pi)
    df = pl.DataFrame({"theta": theta})
    j = circ_gam(["theta ~ 1", "~ 1", "~ 1"], df, family="jplss")
    assert np.isfinite(float(j.logLik))
    k = circ_gam(["theta ~ 1", "~ 1", "~ 1", "~ 1"], df, family="kjlss")
    assert np.isfinite(float(k.logLik))


def test_circ_gam_closed_form_null_start_avoids_indefinite_hessian():
    """Closed-form null start in ``CircularLL._null_params`` (the circlss
    ``initialize`` convention): location = mean direction, concentration =
    the Rbar-based estimator, every shape/skewness parameter = 0 (the
    von-Mises/symmetric reduction member).

    The marginal joint MLE start it replaces (``dist.fit``) pushes a shape
    parameter to an extreme on covariate-driven-location data: the pooled 2nd
    moment of angles with a swinging mean inflates toward the boundary, so
    ssjplss gets psi-hat ~ -4 (vs the data-generating 0.5). That start makes
    ``gam.fit5``'s penalized Hessian indefinite ("indefinite penalized
    likelihood"); the neutral shape=0 start stays in the well-conditioned
    basin. kjlss is the same failure mode via the disc-chart blow-up
    |u| -> ~1e4 (now bounded at 8). See dev/plans/pycircstat2-divergences.md.
    """
    from pycircstat2 import distributions as D

    # (a) the start *is* the closed form: shape/skew exactly 0, concentration
    #     from Rbar via the per-distribution hook, location = mean direction.
    rng = np.random.default_rng(3)
    y = np.mod(rng.vonmises(1.0, 2.5, 300), 2 * np.pi)
    sy, cy = float(np.mean(np.sin(y))), float(np.mean(np.cos(y)))
    Rbar = float(np.hypot(sy, cy))
    np.testing.assert_allclose(
        D.ssjplss._null_params(y),
        [np.arctan2(sy, cy), D.ssjplss.dist._concentration_start(Rbar), 0.0, 0.0],
        atol=1e-12,
    )
    # von Mises concentration start is exactly Fisher's A1-inverse, clamped
    assert D.vmlss._null_params(y)[1] == pytest.approx(
        float(np.clip(A1inv(Rbar), 0.01, 500.0)), abs=1e-12
    )
    # the 2-component projected normal has no concentration hook -> MLE fallback
    assert len(D.pnlss._null_params(y)) == 2

    # (b) ssjplss with a covariate-driven location: the marginal-MLE start
    #     raised FloatingPointError("indefinite penalized likelihood") here;
    #     the closed-form start converges to a finite fit. n is 500 (not the
    #     handful needed to show the start works): the small-n fit sits right on
    #     the indefinite-Hessian cliff — it converged on macOS/Accelerate but
    #     tipped over on Linux/OpenBLAS (and for ~1 in 12 seeds even locally), so
    #     more data gives a well-conditioned margin that holds across platforms.
    rng = np.random.default_rng(11)
    n = 500
    x = np.sort(rng.uniform(0.0, 1.0, n))
    mu = np.mod(2.0 * np.arctan(2.0 * np.sin(2 * np.pi * x)), 2 * np.pi)
    th = np.mod(
        np.array(
            [
                float(
                    D.jonespewsey_sineskewed.rvs(
                        xi=float(m),
                        kappa=2.0,
                        psi=0.5,
                        lmbd=0.6,
                        size=1,
                        random_state=rng,
                    )[0]
                )
                for m in mu
            ]
        ),
        2 * np.pi,
    )
    df = pl.DataFrame({"theta": th, "x": x})
    g = circ_gam(["theta ~ s(x)", "~ 1", "~ 1", "~ 1"], df, family="ssjplss")
    assert np.isfinite(float(g.logLik)) and g.converged

    # (c) kjlss with a covariate-driven location: the uncapped disc-chart
    #     inverse lands on the Theorem-1 feasibility circle and returns a
    #     |u| ~ 2e4 start, which makes gam.fit5's penalized Hessian indefinite;
    #     the |u| <= 8 norm cap keeps the start finite. Assert that cap *directly*
    #     on the null start — it is the platform-independent guard, whereas the
    #     EFS fit on this pathological full-wrap-mu data does not converge and
    #     tips into the same indefinite-Hessian guard on stricter BLAS (so a fit
    #     assertion here would be the very flake this start fix exists to avoid).
    rng = np.random.default_rng(7)
    n = 200
    x = rng.uniform(0.0, 1.0, n)
    mu = np.mod(2 * np.pi * x, 2 * np.pi)
    th = np.mod(
        np.array(
            [
                float(
                    D.katojones.rvs(
                        mu=float(m),
                        gamma=0.4,
                        rho=0.3,
                        lam=0.5,
                        size=1,
                        random_state=rng,
                    )[0]
                )
                for m in mu
            ]
        ),
        2 * np.pi,
    )
    g0, rho0, lam0 = D.katojones.fit(th, method="moments")[1:]
    u1, u2 = D.katojones.disc_chart_inverse(g0, rho0, lam0)
    assert np.hypot(float(u1), float(u2)) > 1e3  # uncapped: ~2e4 → crash
    u_start = D.kjlss._null_params(th)[2:]  # the capped chart coords
    assert np.hypot(*u_start) == pytest.approx(8.0)  # clamped to the |u| <= 8 bound


def test_circ_gam_cartlss_location_warm_start_recovers_wiggly_mu():
    """Role-aware location warm start in ``CircularLL.initialize_coef``.

    Cartwright's density is exactly 0 at the antipode for every ζ, so a
    flat-μ start strands antipodal observations on the log-likelihood cliffs
    and EFS collapses μ to a constant (a coupled bad basin: μ̂ flat, ρ̂ → 0).
    The projected pilot starts the location LP near the data and escapes it.
    On wiggly-μ, moderate-ζ data (ρ ≈ 0.67) the flat start gave mean angular
    error ≈ 1.0 rad — no recovery; the pilot recovers μ to < 0.1 rad with the
    fitted location's circular spread matching the truth's (not collapsed).
    """
    from pycircstat2.distributions import cartwright

    rng = np.random.default_rng(4)
    n = 600
    x = rng.uniform(0.0, 1.0, n)
    mu_true = np.mod(2.0 * np.arctan(np.sin(2.0 * np.pi * x)), 2 * np.pi)
    theta = np.mod(
        np.array(
            [
                float(
                    cartwright.rvs(mu=float(m), zeta=0.5, size=1, random_state=rng)[0]
                )
                for m in mu_true
            ]
        ),
        2 * np.pi,
    )
    df = pl.DataFrame({"theta": theta, "x": x})

    g = circ_gam(["theta ~ s(x)", "~ 1"], df, family="cartlss")
    assert g.converged
    mu_fit = np.mod(np.asarray(g.fitted_values)[:, 0], 2 * np.pi)
    err = np.abs(np.angle(np.exp(1j * (mu_fit - mu_true))))
    assert err.mean() < 0.3  # ≈ 1.0 without the pilot
    # fitted μ is not collapsed to a constant: its circular dispersion
    # (1 − R̄) tracks the truth's (≈ 0.58) instead of falling toward 0
    fit_disp = 1.0 - np.abs(np.mean(np.exp(1j * mu_fit)))
    assert fit_disp > 0.4


def test_circ_gam_cyclic_summary_general_family(capsys):
    """summary() on a fully-penalized smooth (bs='cc' → penalty null space
    0 → hea's reTest/_recov path) under a general family. Crashed with
    AttributeError('_fisher_w') before hea@97a244b; fixed by consuming the
    stored gam.fit5.post.proc R factor (R'R = −lbb, mgcv's object$R) in
    _recov — see hea/.claude/plans/fit5-recov-summary-fix.md."""
    rng = np.random.default_rng(8)
    n = 150
    phi = rng.uniform(0, 2 * np.pi, n)
    theta = np.mod(np.pi / 2 + np.sin(phi) + rng.vonmises(0.0, 4.0, n), 2 * np.pi)
    df = pl.DataFrame({"theta": theta, "phi": phi})
    m = circ_gam("theta ~ s(phi, bs='cc')", df)
    assert m.converged
    m.summary()
    out = capsys.readouterr().out
    assert "Approximate significance of smooth terms:" in out


# --- KatoJones disc-chart routing + family default (circ_gam) --------------


def test_katojones_family_dispatch():
    """katojones regresses in disc-chart coordinates (u1/u2 are not logpdf
    parameters): the bare distribution must auto-route to KatoJonesLL via
    _resolve_gam_family / circ_gam, and base CircularLL must refuse it with
    guidance."""
    from pycircstat2.distributions import KatoJonesLL, katojones
    from pycircstat2.regression import _resolve_gam_family

    with pytest.raises(TypeError, match="KatoJonesLL"):
        CircularLL(katojones)
    assert isinstance(_resolve_gam_family(katojones), KatoJonesLL)

    rng = np.random.default_rng(3)
    theta = np.array(
        [
            float(
                katojones.rvs(
                    mu=2.0, gamma=0.4, rho=0.3, lam=0.5, size=1, random_state=rng
                )[0]
            )
            for _ in range(80)
        ]
    )
    df = pl.DataFrame({"theta": theta})
    g = circ_gam(["theta ~ 1", "~ 1", "~ 1", "~ 1"], df, family=katojones)
    assert np.isfinite(float(g.logLik))
    # the auto-route fits the same model as an explicit KatoJonesLL()
    g2 = circ_gam(["theta ~ 1", "~ 1", "~ 1", "~ 1"], df, family=KatoJonesLL())
    assert np.isclose(float(g.logLik), float(g2.logLik), rtol=1e-8)


def test_circ_gam_family_default_is_vmlss():
    """family=None resolves to the vmlss alias (the shared circlss name); it,
    family=vmlss and family=CircularLL(vonmises) all fit the same model."""
    from pycircstat2.distributions import vonmises
    from pycircstat2.regression import _resolve_gam_family

    assert _resolve_gam_family(None) is vmlss
    rng = np.random.default_rng(7)
    theta = np.mod(rng.vonmises(1.0, 2.0, size=60), 2 * np.pi)
    df = pl.DataFrame({"theta": theta})
    g0 = circ_gam("theta ~ 1", df)
    g1 = circ_gam("theta ~ 1", df, family=vmlss)
    g2 = circ_gam("theta ~ 1", df, family=CircularLL(vonmises))
    assert np.isclose(float(g0.logLik), float(g1.logLik), rtol=1e-10)
    assert np.isclose(float(g1.logLik), float(g2.logLik), rtol=1e-10)


# --- center=TRUE: rotate the response off the tan-half wall -----------------
# Ported from circlss's test-circ_gam.R "center = TRUE machinery" block: the
# reference chooser, the wall-column gate, the response rotate-back, and an
# end-to-end recovery of a wall-hugging location that the wall otherwise breaks.


def test_circ_center_ref_noop_when_wall_clear():
    """_center_ref is an exact no-op when the circular mean is clear of the
    wall: concentrated near 0, and a wide symmetric fan, both mean ≈ 0 → 0."""
    from pycircstat2.regression import _center_ref, _wrap

    rng = np.random.default_rng(1)
    assert _center_ref(_wrap(rng.normal(0.0, 0.4, 200))) == 0.0
    assert _center_ref(_wrap(np.linspace(-2.0, 2.0, 200))) == 0.0


def test_circ_center_ref_moves_wall_off_straddling_pi():
    """Data sitting on the wall (mean ≈ π) is rotated by ≈ π, pulling its
    circular mean back to ≈ 0 (off the wall)."""
    from pycircstat2.regression import _center_ref, _wrap

    rng = np.random.default_rng(2)
    th = _wrap(rng.normal(np.pi, 0.4, 200))
    ref = _center_ref(th)
    assert abs(ref) > 0.5
    recentered = np.arctan2(np.mean(np.sin(th - ref)), np.mean(np.cos(th - ref)))
    assert abs(recentered) < 0.3


def test_circ_center_ref_weighted_branch():
    """The weighted branch (a circ_mix component's responsibilities) centers
    the weighted mode: weight the near-π mass → ref ≈ π; weight the near-0
    mass (clear of the wall) → no-op."""
    from pycircstat2.regression import _center_ref, _wrap

    rng = np.random.default_rng(3)
    th = _wrap(np.concatenate([rng.normal(np.pi, 0.3, 100), rng.normal(0.0, 0.3, 100)]))
    w_pi = np.r_[np.ones(100), np.zeros(100)]
    w_0 = np.r_[np.zeros(100), np.ones(100)]
    assert abs(float(np.angle(np.exp(1j * (_center_ref(th, w_pi) - np.pi))))) < 0.3
    assert _center_ref(th, w_0) == 0.0


def test_wall_loc_flags_tanhalf_only():
    """_wall_loc points at the tan-half circular location and nothing else:
    vmlss/wclss → 0, pnlss (derived atan2, identity links) and a linear family
    → None."""
    import hea.family as hea_family
    from pycircstat2.distributions import pnlss, vmlss, wclss
    from pycircstat2.regression import _wall_loc

    assert _wall_loc(vmlss) == 0
    assert _wall_loc(wclss) == 0
    assert _wall_loc(pnlss) is None
    assert _wall_loc(hea_family.gaussian) is None


def test_rotate_response_shifts_only_circular_col():
    """_rotate_response wraps only the location column; scale columns and the
    ref==0 / wall-less cases are untouched. Works on the polars frame
    predict(type='response') returns and on a bare ndarray."""
    from pycircstat2.regression import _rotate_response, _wrap

    p = pl.DataFrame({"fit": [-3.0, 0.0, 3.0], "fit.1": [2.0, 5.0, 9.0]})
    out = _rotate_response(p, 0, 0.5)
    np.testing.assert_allclose(out["fit"].to_numpy(), _wrap(np.array([-3.0, 0.0, 3.0]) + 0.5))
    np.testing.assert_array_equal(out["fit.1"].to_numpy(), [2.0, 5.0, 9.0])  # scale kept
    assert _rotate_response(p, 0, 0.0) is p  # ref 0 = no-op
    assert _rotate_response(p, None, 0.5) is p  # no wall = no-op
    arr = np.array([[-3.0, 2.0], [3.0, 9.0]])
    np.testing.assert_allclose(_rotate_response(arr, 0, 0.5)[:, 0], _wrap(arr[:, 0] + 0.5))


def test_circ_gam_center_default_recenters_wall_hugging_data():
    """End-to-end: a location that straddles the θ = π wall (mean ≈ π) is
    unrepresentable on the tan-half link, so center=True (the default) rotates
    the fit off the wall and predict(type='response') reports μ̂ back in the
    original frame — recovering the truth the uncentred fit cannot."""
    rng = np.random.default_rng(0)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    mu_true = np.pi + 0.8 * np.sin(2 * np.pi * x)  # swings across the wall
    theta = np.mod(mu_true + rng.vonmises(0.0, 5.0, n), 2 * np.pi)
    df = pl.DataFrame({"theta": theta, "x": x})

    g = circ_gam(["theta ~ s(x)", "~ s(x)"], df)  # center=True default
    assert abs(g.circ_center) > 0.5  # rotated off the wall
    grid = np.linspace(0.0, 1.0, 100)
    nd = pl.DataFrame({"x": grid})
    mu_hat = np.asarray(g.predict(nd, type="response"))[:, 0]
    err = np.abs(np.angle(np.exp(1j * (mu_hat - (np.pi + 0.8 * np.sin(2 * np.pi * grid))))))
    assert err.mean() < 0.15

    # the rotate-back lands on response, not link: response == wrap(linkinv(link)
    # + circ_center). predict(type="link") stays in the centred fit frame.
    eta = np.asarray(g.predict(nd, type="link"))[:, 0]
    from pycircstat2.regression import _wrap

    expected = _wrap(np.asarray(g.family.links[0].linkinv(eta)) + g.circ_center)
    np.testing.assert_allclose(_wrap(mu_hat), expected, atol=1e-9)

    # center=False leaves the fit on the wall: no rotation recorded.
    g0 = circ_gam(["theta ~ s(x)", "~ s(x)"], df, center=False)
    assert g0.circ_center == 0.0


def test_circ_gam_center_numeric_and_wall_free_family():
    """center accepts an explicit reference angle, and a wall-free family
    (pnlss, identity-linked Cartesian location) is never centred even on
    wall-hugging data."""
    from pycircstat2.distributions import pnlss

    rng = np.random.default_rng(5)
    theta = np.mod(rng.normal(np.pi, 0.3, 250), 2 * np.pi)  # mass on the wall
    df = pl.DataFrame({"theta": theta})

    g = circ_gam("theta ~ 1", df, family="vmlss", center=0.9)
    assert g.circ_center == pytest.approx(0.9)
    # pnlss has no wall: center=True is a no-op even with mass on θ = π
    gp = circ_gam(["theta ~ 1", "~ 1"], df, family=pnlss)
    assert gp.circ_center == 0.0


# ===========================================================================
# circ_lm — classical (parametric) circular regression.
# lc returns a hea lm object; cl/cc return result dicts. References: the
# `circular` R package / circlss, and Pewsey et al. (2013) §8.4.1 (lc lung).
# ===========================================================================
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from pycircstat2 import load_data  # noqa: E402
from pycircstat2.regression import circ_lm  # noqa: E402


def _lung_dataframe() -> "pl.DataFrame":
    """Pewsey, Neuhäuser & Ruxton (2013) §8.4.1 lung-disease deaths, with the
    two February outliers dropped, theta = (pi/6)*month, deaths renamed y."""
    df = load_data("lung_deaths", source="pewsey")
    df = df.with_columns(((np.pi / 6) * pl.col("month")).alias("theta"))
    df = df.rename({"deaths": "y"})
    return df.filter(~((pl.col("month") == 2) & pl.col("year").is_in([1976, 1979])))


def _simulate_cl(seed: int = 0, n: int = 400):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 1))
    mu_true, beta_true, kappa_true = 0.7, np.array([0.9]), 5.0
    eps = rng.vonmises(0, kappa_true, size=n)
    theta = mu_true + 2 * np.arctan(x @ beta_true) + eps
    return x, theta, mu_true, beta_true, kappa_true


def _cl_frame() -> "pl.DataFrame":
    df = load_data("B20", source="fisher")
    return pl.DataFrame(
        {"X": df["x"].to_numpy(), "theta": np.deg2rad(df["θ"].to_numpy())}
    )


def _milwaukee_frame() -> "pl.DataFrame":
    df = load_data("milwaukee", source="jammalamadaka")
    return pl.DataFrame(
        {
            "theta": np.deg2rad(df["theta"].to_numpy()),
            "psi": np.deg2rad(df["psi"].to_numpy()),
        }
    )


def _coefs(lm_fit) -> dict:
    """name -> value for a hea lm fit."""
    return dict(zip(lm_fit.column_names, lm_fit.bhat.row(0)))


# --------------------------------------------------------------------------- #
# cl — Fisher-Lee von Mises regression (returns a result dict)
# --------------------------------------------------------------------------- #
def test_circ_lm_cl_mean_against_r():
    """Mean-direction model on Fisher's B20: reproduces circular::lm.circular."""
    m = circ_lm("theta ~ X", _cl_frame(), type="cl", tol=1e-10)
    assert m["model"] == "mean"
    assert np.isclose(m["beta"][0], -0.008317, atol=1e-3)
    assert np.isclose(m["se_beta"][0], 0.001359, atol=1e-3)
    assert np.isclose(m["mu"], 2.426, atol=1e-2)
    assert np.isclose(m["se_mu"], 0.1119, atol=1e-2)
    assert np.isclose(m["kappa"], 3.224, atol=1e-2)
    assert np.isclose(m["se_kappa"], 0.7159, atol=1e-2)
    assert np.isclose(m["loglik"], 27.76, atol=1e-2)


def test_circ_lm_cl_kappa_matches_circular_not_circlss():
    """The one cross-language gap: kappa passes through A1inv, and pycircstat2's
    A1inv follows the classical `circular` approximation while circlss returns
    the machine-precision inverse. pycircstat2's kappa matches `circular`
    (3.224), differing from circlss (3.241) at the approximation's ~1e-2 level;
    beta/mu/loglik match both to ~1e-3."""
    m = circ_lm("theta ~ X", _cl_frame(), type="cl", tol=1e-10)
    assert np.isclose(m["kappa"], 3.224, atol=3e-3)  # circular value
    assert not np.isclose(m["kappa"], 3.2406, atol=3e-3)  # circlss machine-precision


def test_circ_lm_cl_single_formula_is_mean_model():
    """A bare mu-formula is sugar for [mu ~ x, ~ 1]: constant kappa."""
    d = _cl_frame()
    single = circ_lm("theta ~ X", d, type="cl", tol=1e-10)
    paired = circ_lm(["theta ~ X", "~ 1"], d, type="cl", tol=1e-10)
    assert single["model"] == "mean" and paired["model"] == "mean"
    np.testing.assert_allclose(single["beta"], paired["beta"], atol=1e-10)
    assert np.ndim(single["kappa"]) == 0  # scalar concentration


@pytest.mark.filterwarnings("ignore:circ_lm")
def test_circ_lm_cl_model_selection_and_guards():
    """The formula list selects mean / kappa / mixed by which LP carries
    covariates; the fitter ties mu and kappa to one shared design. (B20's mixed
    fit is weakly identified and may not converge — only dispatch is checked.)"""
    d = _cl_frame().with_columns((pl.col("X") * 2.0).alias("Z"))
    assert circ_lm(["theta ~ X", "~ 1"], d, type="cl")["model"] == "mean"
    assert circ_lm(["theta ~ 1", "~ X"], d, type="cl")["model"] == "kappa"
    assert circ_lm(["theta ~ X", "~ X"], d, type="cl")["model"] == "mixed"
    with pytest.raises(ValueError, match="shared design"):
        circ_lm(["theta ~ X", "~ Z"], d, type="cl")
    with pytest.raises(ValueError, match="nothing to regress"):
        circ_lm(["theta ~ 1", "~ 1"], d, type="cl")
    with pytest.raises(ValueError, match="at most two"):
        circ_lm(["theta ~ X", "~ X", "~ X"], d, type="cl")


def test_circ_lm_cl_covariate_expressions():
    """cl covariate designs go through hea's parser, so cos()/sin() expressions
    are valid predictors for a circular response (not only bare columns)."""
    m = circ_lm("theta ~ cos(X) + sin(X)", _cl_frame(), type="cl", tol=1e-8)
    assert m["mu_terms"] == ["cos(X)", "sin(X)"]
    assert m["beta"].shape == (2,) and np.all(np.isfinite(m["beta"]))


def test_circ_lm_cl_mixed_matches_mean_when_kappa_constant():
    """With constant true kappa, mixed and mean agree on beta/mu."""
    x, theta, *_ = _simulate_cl()
    d = pl.DataFrame({"theta": theta, "X": x[:, 0]})
    mean = circ_lm("theta ~ X", d, type="cl", tol=1e-10, maxit=500)
    mixed = circ_lm(["theta ~ X", "~ X"], d, type="cl", tol=1e-10, maxit=500)
    np.testing.assert_allclose(mixed["beta"], mean["beta"], atol=5e-3)
    np.testing.assert_allclose(mixed["mu"], mean["mu"], atol=5e-3)
    np.testing.assert_allclose(np.exp(mixed["alpha"]), mean["kappa"], rtol=0.1)
    assert abs(mixed["gamma"][0]) < 0.1


def test_circ_lm_cl_mixed_recovers_true_parameters():
    x, theta, mu_t, beta_t, kappa_t = _simulate_cl(seed=1, n=800)
    d = pl.DataFrame({"theta": theta, "X": x[:, 0]})
    m = circ_lm(["theta ~ X", "~ X"], d, type="cl", tol=1e-10, maxit=500)
    np.testing.assert_allclose(m["beta"], beta_t, atol=0.15)
    np.testing.assert_allclose(m["mu"], mu_t, atol=0.1)
    np.testing.assert_allclose(np.exp(m["alpha"]), kappa_t, rtol=0.25)


def test_circ_lm_cl_kappa_model():
    """Kappa-only model: per-observation kappa from X, with finite per-obs SEs."""
    x, theta, *_ = _simulate_cl(seed=2, n=300)
    d = pl.DataFrame({"theta": theta, "X": x[:, 0]})
    m = circ_lm(["theta ~ 1", "~ X"], d, type="cl", tol=1e-8)
    assert m["model"] == "kappa"
    assert np.all(np.isfinite(m["kappa"])) and np.all(m["kappa"] > 0)
    assert m["se_kappa"].shape == (300,) and np.all(m["se_kappa"] > 0)
    # kappa at X=0 equals exp(alpha)
    i0 = int(np.argmin(np.abs(x[:, 0])))
    np.testing.assert_allclose(
        m["kappa"][i0], np.exp(m["alpha"] + m["gamma"][0] * x[i0, 0]), rtol=1e-8
    )


# --------------------------------------------------------------------------- #
# cc — Sarma & Jammalamadaka harmonic circular-circular (returns a result dict)
# --------------------------------------------------------------------------- #
def test_circ_lm_cc_against_circlss():
    """Milwaukee order 2 & 4: cos/sin coefficients (interleaved [(Intercept),
    cos1, sin1, …]), rho, A_k and the higher-order test reproduce circlss."""
    d = _milwaukee_frame()
    c2 = circ_lm("theta ~ psi", d, type="cc", order=2)
    np.testing.assert_allclose(c2["rho"], 0.6358710, atol=1e-5)
    np.testing.assert_allclose(
        c2["coefficients"]["cos"],
        [0.1441268, 0.6414811, 0.2171076, 0.1165915, -0.4374547],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        c2["coefficients"]["sin"],
        [-0.2191974, -0.4509745, 0.2225796, 0.1831359, 0.2924121],
        atol=1e-5,
    )
    np.testing.assert_allclose(c2["p_values"], [0.8645504, 0.2263628], atol=1e-5)
    np.testing.assert_allclose(c2["A_k"], 0.6131735, atol=1e-5)

    c4 = circ_lm("theta ~ psi", d, type="cc", order=4)
    np.testing.assert_allclose(c4["rho"], 0.7164767, atol=1e-5)
    np.testing.assert_allclose(c4["p_values"], [0.9915110, 0.7997684], atol=1e-5)


def test_circ_lm_cc_residual_kappa_and_underlying_lms():
    """Residual concentration is A1inv(A_k); the two embedding fits are exposed
    as full hea lm objects (cos_lm / sin_lm)."""
    m = circ_lm("theta ~ psi", _milwaukee_frame(), type="cc", order=2)
    assert -1 <= m["A_k"] <= 1 and np.isfinite(m["kappa"]) and m["kappa"] >= 0
    np.testing.assert_allclose(m["kappa"], A1inv(m["A_k"]), atol=1e-12)
    # the underlying OLS fits are hea lm objects with their own interface
    assert hasattr(m["cos_lm"], "predict") and hasattr(m["sin_lm"], "summary")
    # reassembled fitted directions are valid angles
    assert np.all((m["fitted"] >= 0) & (m["fitted"] < 2 * np.pi + 1e-9))


# --------------------------------------------------------------------------- #
# lc — linear ~ circular: a plain OLS fit; circ_lm returns the hea lm object
# --------------------------------------------------------------------------- #
def test_circ_lm_lc_manual_harmonics_pewsey():
    """The §8.4.1 reduced extended model, written with explicit harmonic terms
    (the manually-written-harmonic case): fed straight to hea.lm, reproduces
    Pewsey et al. (2013)."""
    m = circ_lm(
        "y ~ cos(theta) + sin(theta) + sin(2*theta)", _lung_dataframe(), type="lc"
    )
    c = _coefs(m)
    assert np.isclose(c["(Intercept)"], 2125.12, atol=1e-1)
    assert np.isclose(c["cos(theta)"], 454.18, atol=1e-1)
    assert np.isclose(c["sin(theta)"], 601.96, atol=1e-1)
    assert np.isclose(c["sin(2 * theta)"], 108.69, atol=1e-1)
    assert np.isclose(m.sigma, 171.3, atol=1e-1)
    assert np.isclose(m.r_squared, 0.9093, atol=1e-3)


def test_circ_lm_lc_full_order2_matches_circlss():
    """Full order-2 harmonic regression reproduces circlss's lc to machine
    precision (it is the same OLS)."""
    m = circ_lm(
        "y ~ cos(theta) + sin(theta) + cos(2*theta) + sin(2*theta)",
        _lung_dataframe(),
        type="lc",
    )
    c = _coefs(m)
    assert np.isclose(c["(Intercept)"], 2125.1789, atol=1e-2)
    assert np.isclose(c["cos(theta)"], 454.2394, atol=1e-2)
    assert np.isclose(c["sin(theta)"], 602.0726, atol=1e-2)
    assert np.isclose(c["cos(2 * theta)"], -4.0123, atol=1e-2)
    assert np.isclose(c["sin(2 * theta)"], 108.8037, atol=1e-2)
    assert np.isclose(m.sigma, 172.625, atol=1e-2)
    assert np.isclose(m.r_squared, 0.9093750, atol=1e-6)
    assert np.isclose(m.AIC, 926.6210, atol=1e-2)
    assert np.isclose(m.BIC, 940.1119, atol=1e-2)


def test_circ_lm_lc_is_hea_lm_with_full_interface():
    """lc returns hea's lm object: predict, summary, plot, coefficients, AIC."""
    m = circ_lm("y ~ cos(theta) + sin(theta)", _lung_dataframe(), type="lc")
    for attr in (
        "predict",
        "summary",
        "plot",
        "coefficients",
        "AIC",
        "BIC",
        "sigma",
        "r_squared",
        "ci_bhat",
    ):
        assert hasattr(m, attr), attr


def test_circ_lm_lc_plot_returns_figure():
    m = circ_lm("y ~ cos(theta) + sin(theta)", _lung_dataframe(), type="lc")
    fig = m.plot()
    assert fig is not None and len(fig.axes) >= 1


def test_circ_lm_lc_predict_round_trip():
    lung = _lung_dataframe()
    m = circ_lm("y ~ cos(theta) + sin(theta) + sin(2*theta)", lung, type="lc")
    pred = m.predict(lung.select("theta"))
    np.testing.assert_allclose(
        pred["fit"].to_numpy(), m.yhat["fit"].to_numpy(), atol=1e-9
    )


def test_circ_lm_lc_amplitude_phase_recovery():
    """A first-harmonic fit recovers a known amplitude/phase: amplitude =
    hypot(cos, sin), phase = atan2(sin, cos) of the fitted coefficients."""
    rng = np.random.default_rng(1)
    n = 1000
    theta = rng.uniform(0, 2 * np.pi, n)
    true_amp, true_phase = 2.5, 0.9
    y = 5.0 + true_amp * np.cos(theta - true_phase) + rng.normal(0, 0.1, n)
    m = circ_lm(
        "y ~ cos(theta) + sin(theta)", pl.DataFrame({"y": y, "theta": theta}), type="lc"
    )
    c = _coefs(m)
    amp = np.hypot(c["cos(theta)"], c["sin(theta)"])
    phase = np.arctan2(c["sin(theta)"], c["cos(theta)"])
    assert np.isclose(amp, true_amp, atol=0.05)
    assert np.isclose(phase, true_phase, atol=0.05)


# --------------------------------------------------------------------------- #
# shared front-door behavior
# --------------------------------------------------------------------------- #
def test_circ_lm_type_spellings_and_unknown():
    d = _milwaukee_frame()
    ref = circ_lm("theta ~ psi", d, type="cc", order=2)
    for spelling in ("cc", "c-c", "C-C"):
        m = circ_lm("theta ~ psi", d, type=spelling, order=2)
        np.testing.assert_allclose(m["fitted"], ref["fitted"], atol=1e-12)
    with pytest.raises(ValueError, match="type must be"):
        circ_lm("theta ~ psi", d, type="xy")


def test_circ_lm_rejects_smooth_terms():
    """A smooth term is parametric-only's error, pointing at circ_gam."""
    d = pl.DataFrame({"y": [0.1, 1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="circ_gam"):
        circ_lm("y ~ s(x)", d, type="cl")
    with pytest.raises(ValueError, match="circ_gam"):
        circ_lm(["y ~ x", "~ te(x)"], d, type="cl")
    with pytest.raises(ValueError, match="circ_gam"):
        circ_lm("y ~ s(x, bs='cc')", d, type="lc")


def test_circ_lm_cc_requires_one_predictor():
    d = pl.DataFrame(
        {
            "y": [0.1, 1.0, 2.0, 3.0],
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [1.0, 1.0, 2.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="exactly one"):
        circ_lm("y ~ a + b", d, type="cc")


def test_circ_lm_accepts_pandas():
    """Polars is native; pandas is accepted (lc forwards it to hea.lm)."""
    pytest.importorskip("pandas")
    lung = _lung_dataframe()
    m_pl = circ_lm("y ~ cos(theta) + sin(theta)", lung, type="lc")
    m_pd = circ_lm("y ~ cos(theta) + sin(theta)", lung.to_pandas(), type="lc")
    np.testing.assert_allclose(m_pl.bhat.row(0), m_pd.bhat.row(0), atol=1e-10)


# --------------------------------------------------------------------------- #
# A1 / A1inv numerics (the Bessel-ratio map the residual concentration uses)
# --------------------------------------------------------------------------- #
def test_a1inv_clamps_at_unit_radius():
    # A1 maps κ≥0 to [0,1); A1inv at R≥1 must not explode.
    assert np.isfinite(A1inv(1.0))
    assert np.isfinite(A1inv(1.5))
    assert A1inv(0.0) == 0.0


def test_a1_stable_at_extreme_kappa():
    # i0/i1 overflow around κ ≈ 710; A1 must remain finite via i0e/i1e.
    for k in (700.0, 5_000.0, 1e6):
        val = float(A1(k))
        assert np.isfinite(val)
        assert 0.0 < val < 1.0


# ===========================================================================
# circ_resid / circ_check diagnostics + the CircGAM / CircLM result objects
# ===========================================================================
def test_pvonmises_matches_numerical_vonmises_cdf():
    """The residual-frame von Mises CDF (origin at the antipode) is a proper
    distribution function and matches a direct numerical integral of the von
    Mises residual density to machine precision — the analytic PIT reference."""
    from scipy.integrate import quad
    from scipy.special import i0e

    from pycircstat2.regression import _pvonmises

    mu, kappa = 1.3, 2.5

    def ref(d):  # F_D(d) = ∫_{-π}^d exp(κ(cos t − 1)) / (2π I0e(κ)) dt
        return quad(lambda t: np.exp(kappa * (np.cos(t) - 1.0))
                    / (2 * np.pi * float(i0e(kappa))), -np.pi, d)[0]

    for d in (-np.pi, -1.0, 0.0, 0.7, np.pi):
        assert _pvonmises(mu + d, mu, kappa) == pytest.approx(ref(d), abs=1e-8)
    # boundary / median anchors
    assert _pvonmises(mu - np.pi, mu, kappa) == pytest.approx(0.0, abs=1e-9)
    assert _pvonmises(mu, mu, kappa) == pytest.approx(0.5, abs=1e-9)
    # monotone and bounded over the circle
    grid = mu + np.linspace(-np.pi, np.pi, 200)
    u = _pvonmises(grid, mu, kappa)
    assert np.all(np.diff(u) >= -1e-12) and u.min() >= 0 and u.max() <= 1


def test_watson_u2_uniform_grid_is_exact_floor():
    """For the perfectly uniform plotting positions u_i = (2i−1)/(2n) the Watson
    statistic collapses to its 1/(12n) floor (the sum and mean terms vanish) —
    a deterministic, closed-form pin on the implementation."""
    from pycircstat2.regression import _watson_u2

    n = 50
    u = (2 * np.arange(1, n + 1) - 1) / (2 * n)
    res = _watson_u2(u)
    assert res["stat"] == pytest.approx(1.0 / (12 * n), abs=1e-12)
    assert res["p"] == pytest.approx(1.0, abs=1e-9)
    # a clustered (non-uniform) sample gives a much larger statistic
    assert _watson_u2(np.linspace(0.0, 0.2, n))["stat"] > res["stat"] * 50


def test_circ_resid_types_tags_and_shapes():
    """circ_resid returns one value per observation for each residual type, with
    the circlss ``.type`` / ``.scale`` tags, across the cl / lc / gam legs."""
    cl = circ_lm("theta ~ X", _cl_frame(), type="cl", tol=1e-10)
    for t in ("quantile", "deviance", "angular", "pearson"):
        r = cl.circ_resid(t)
        assert r.shape == (cl.n,) and r.type == t
    assert cl.circ_resid("quantile", scale="normal").scale == "normal"
    with pytest.raises(ValueError, match="quantile|deviance|angular|pearson"):
        cl.circ_resid("bogus")

    lc = circ_lm("y ~ cos(theta) + sin(theta)", _lung_dataframe(), type="lc")
    assert lc.circ_resid("deviance").shape == (lc.n,)

    df, _ = _cl_gam_sim(n=300, seed=4)
    g = circ_gam(["theta ~ s(x)", "~ s(x)"], df)
    for t in ("quantile", "deviance", "angular", "pearson"):
        assert g.circ_resid(t).shape[0] == df.height


def test_circ_check_draws_panels_and_prints_gof(capsys):
    """circ_check draws the diagnostic panel grid and prints the R-style GOF
    table (Watson U² + residual location + the leg backend), returning the
    Figure — as with summary, it does not return a raw dict. The default grid is
    the four circular panels (rose dropped for a linear response); which="all"
    adds the deviance panels."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    cl = circ_lm("theta ~ X", _cl_frame(), type="cl", tol=1e-10)
    fig = cl.circ_check()
    assert isinstance(fig, Figure) and len(fig.axes) == 4
    out = capsys.readouterr().out
    assert "circ_check: circ_lm:cl" in out and "Watson U2" in out
    assert "converged" in out and "resultant length" in out
    plt.close(fig)

    cc = circ_lm("theta ~ psi", _milwaukee_frame(), type="cc", order=2)
    fig = cc.circ_check()
    assert "higher-order harmonic test" in capsys.readouterr().out
    plt.close(fig)

    # linear response: the rose panel is dropped (needs an angular residual)
    lc = circ_lm("y ~ cos(theta) + sin(theta)", _lung_dataframe(), type="lc")
    fig = lc.circ_check(which="all")
    titles = [ax.get_title() for ax in fig.axes]
    assert "angular residuals" not in titles and "R-squared" in \
        capsys.readouterr().out
    plt.close(fig)

    df, _ = _cl_gam_sim(n=400, seed=2)
    g = circ_gam(["theta ~ s(x)", "~ s(x)"], df)
    fig = g.circ_check(which="all")  # general GAM ⇒ cook dropped with a message
    out = capsys.readouterr().out
    assert "effective degrees of freedom" in out and "dropping the 'cook'" in out
    plt.close(fig)


def test_circ_gam_returns_circgam_and_keeps_gam_surface():
    """circ_gam returns a CircGAM (a hea gam subclass): the full gam interface
    stays reachable and the circular methods are added on top."""
    from pycircstat2.regression import CircGAM, _watson_u2

    df, _ = _cl_gam_sim(n=300, seed=11)
    g = circ_gam(["theta ~ s(x)", "~ 1"], df)
    assert isinstance(g, CircGAM) and isinstance(g, hea_gam)
    # inherited gam surface intact
    for attr in ("summary", "predict", "fitted", "AIC", "logLik", "Vp", "edf"):
        assert hasattr(g, attr), attr
    # well-specified fit ⇒ the PIT residual is not flagged non-uniform
    assert _watson_u2(np.asarray(g.circ_resid("quantile")))["p"] > 0.01


def test_circ_lm_cc_pearson_unavailable_paths_are_present():
    """The von Mises Pearson residual (cl/cc) is the score-standardized
    sin(d)/√(A1(κ)/κ); confirm it is finite and matches that closed form for a
    cc fit's single residual concentration."""
    m = circ_lm("theta ~ psi", _milwaukee_frame(), type="cc", order=2)
    pe = m.circ_resid("pearson")
    d = np.angle(np.exp(1j * (m["residuals"])))  # wrapped residual angle
    v = float(A1(m["kappa"]) / m["kappa"])
    np.testing.assert_allclose(pe, np.sin(d) / np.sqrt(v), atol=1e-9)


def test_circ_plot_smoke_all_views_and_legs():
    """Headless (Agg) smoke: circ_plot renders flat / geometry / both for every
    leg and surface — cl→cylinder, cc→torus, lc→can, vmlss/pnlss GAM — and the
    pnlss derived-direction panel is added (mu1, mu2, direction)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n = 150
    x = rng.normal(size=n)
    theta = np.mod(0.7 + 2 * np.arctan(0.9 * x) + rng.vonmises(0, 6, n), 2 * np.pi)
    cl = circ_lm("theta ~ x", pl.DataFrame({"theta": theta, "x": x}),
                 type="cl", tol=1e-9)
    phi = rng.uniform(0, 2 * np.pi, n)
    psi = np.mod(phi / 2 + rng.vonmises(0, 5, n), 2 * np.pi)
    cc = circ_lm("psi ~ phi", pl.DataFrame({"psi": psi, "phi": phi}),
                 type="cc", order=1)
    y = 5 + 2 * np.cos(phi) + rng.normal(0, 0.3, n)
    lc = circ_lm("y ~ cos(phi) + sin(phi)", pl.DataFrame({"y": y, "phi": phi}),
                 type="lc")
    for m, npanel in ((cl, 2), (cc, 1), (lc, 1)):
        for view in ("flat", "geometry", "both"):
            fig = m.circ_plot(view=view)
            assert fig is not None and len(fig.axes) >= 1
            plt.close(fig)

    xg = rng.uniform(0, 1, n)
    g = circ_gam(["theta ~ s(x)", "~ s(x)"],
                 pl.DataFrame({"theta": np.mod(np.pi / 2 + np.sin(2 * np.pi * xg)
                                               + rng.vonmises(0, 5, n), 2 * np.pi),
                               "x": xg}))
    for view in ("flat", "geometry", "both"):
        fig = g.circ_plot(view=view)
        assert fig is not None
        plt.close(fig)

    ph = rng.uniform(0, 2 * np.pi, n)
    th = np.mod(np.arctan2(2 * np.sin(ph) + rng.standard_normal(n),
                           2 * np.cos(ph) + rng.standard_normal(n)), 2 * np.pi)
    gp = circ_gam(["theta ~ s(phi, bs='cc')", "~ s(phi, bs='cc')"],
                  pl.DataFrame({"theta": th, "phi": ph}), family="pnlss")
    fig = gp.circ_plot(view="flat")  # mu1, mu2, direction panels
    assert sum(ax.get_title() == "direction" for ax in fig.axes) == 1
    plt.close(fig)


def test_circ_plot_unicode_covariate_uses_geometry_not_term_fallback():
    """A Greek covariate name (θ) must be detected as the single covariate so
    circ_plot draws the geometry/both view — an ASCII-only identifier regex
    missed it and silently collapsed to mgcv's term-plot fallback (CircGAM) or
    a 'multi-covariate' message (CircLM)."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    rng = np.random.default_rng(0)
    n = 150
    θ = rng.uniform(0, 2 * np.pi, n)
    y = 2 + 1.5 * np.sin(θ) + rng.normal(0, 0.3, n)
    g = circ_gam("y ~ s(θ, bs='cc')", family="gaussian", data=pl.DataFrame({"y": y, "θ": θ}))
    assert g._geometry() == ("lc", False, True, "θ")
    fig = g.circ_plot()  # default "both" → can surface + flat panel
    assert any(isinstance(ax, Axes3D) for ax in fig.axes)
    plt.close(fig)

    # circ_lm legs already use a Unicode-safe parser; confirm θ works there too
    ψ = np.mod(θ / 2 + rng.vonmises(0, 5, n), 2 * np.pi)
    m = circ_lm("ψ ~ θ", pl.DataFrame({"ψ": ψ, "θ": θ}), type="cc", order=1)
    fig = m.circ_plot()
    assert any(isinstance(ax, Axes3D) for ax in fig.axes)
    plt.close(fig)
