import re
import warnings
from itertools import combinations_with_replacement
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from hea.family import GeneralFamily, gamlss_etamu, gamlss_gH, trind_generator
from hea.models import gam as _hea_gam, lm as _hea_lm
from scipy.special import i0e
from scipy.stats import chi2, norm, t as student_t

from .distributions import get_link, katojones, vonmises
from .utils import A1, A1inv, A1prime, significance_code

__all__ = ["CircularLL", "KatoJonesLL", "CLRegression", "CCRegression",
           "LCRegression"]


def _to_polars(data) -> "pl.DataFrame":
    """Coerce a DataFrame to polars.

    Polars frames pass through; pandas frames are accepted as a soft
    convenience and converted via ``pl.from_pandas`` (which imports pandas
    lazily — pandas is never a hard dependency, since a pandas input can only
    exist if pandas is already installed).
    """
    if isinstance(data, pl.DataFrame):
        return data
    if type(data).__module__.startswith("pandas"):
        return pl.from_pandas(data)
    raise TypeError(
        f"`data` must be a polars (or pandas) DataFrame; got {type(data).__name__}"
    )


def _safe_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ rhs


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def _ravel(v) -> np.ndarray:
    """Flatten an ``hea.lm`` output (polars frame or ndarray) to 1-D float."""
    if isinstance(v, pl.DataFrame):
        return v.to_numpy().ravel()
    return np.asarray(v, dtype=float).ravel()


def _harmonic_block_order(by_name: dict, feats: list, order: int) -> np.ndarray:
    """Reorder per-coefficient values keyed by hea coefficient name into the
    legacy CCRegression design order ``[intercept | cos-block | sin-block]``,
    each block feature-major then harmonic-minor. ``by_name`` maps a name
    (``"(Intercept)"`` or ``harmonic(<var>, k=…, period=…)cos{j}``) → value; the
    ``cos{j}``/``sin{j}`` suffix makes the reorder robust to hea's interleaved
    column order. (``_LC_HARMONIC_COEF_RE`` is defined in the LC markers below.)
    """
    lut: dict = {}
    intercept = 0.0
    for name, val in by_name.items():
        if name == "(Intercept)":
            intercept = float(val)
            continue
        m = _LC_HARMONIC_COEF_RE.match(name)
        if m:
            lut[(m.group("var"), m.group("trig"), int(m.group("order")))] = float(val)
    out = [intercept]
    out += [lut[(f, "cos", k)] for f in feats for k in range(1, order + 1)]
    out += [lut[(f, "sin", k)] for f in feats for k in range(1, order + 1)]
    return np.asarray(out, dtype=float)


# --- backend dispatch: parametric (hea.lm) vs smooth (hea.gam) ---------------
# A smooth term — s()/te()/ti()/t2(), mgcv's smooth constructors — routes the
# fit to ``hea.models.gam`` (a penalized REML/GCV smooth); a purely parametric
# RHS (``harmonic()`` / ``cos()+sin()`` / linear) stays on ``hea.models.lm``.
# GAM is a *basis/backend axis, not a taxonomy axis*: a cyclic smooth is the
# nonparametric counterpart of a harmonic on the same (response, predictor)
# category, so it folds into LC/CC rather than a separate class. The ``\b``
# keeps the trailing ``s(`` of ``cos(``/``sin(`` from matching the ``s`` smooth
# constructor (no word boundary between ``o`` and ``s``).
_SMOOTH_RE = re.compile(r"\b(?:s|te|ti|t2)\s*\(")
_SMOOTH_VAR_RE = re.compile(r"\b(?:s|te|ti|t2)\s*\(\s*([^\W\d_]\w*)")


def _has_smooth(formula: str) -> bool:
    """True if the formula RHS contains a smooth term (→ gam backend)."""
    rhs = formula.split("~", 1)[1] if "~" in formula else formula
    return bool(_SMOOTH_RE.search(rhs))


def _smooth_vars(rhs: str) -> List[str]:
    """First-argument variable of each smooth term, in formula order."""
    return [m.group(1) for m in _SMOOTH_VAR_RE.finditer(rhs)]


# Cyclic smooth bases (mgcv): bs='cc' (cyclic cubic) / bs='cp' (cyclic p-spline).
# Their boundary knots set the *period*; for circular predictors that is 2π, so
# LC/CC supply it by default. hea stays general — mgcv defaults cyclic knots to
# the data range, which collapses f(0)=f(period) for angles (the lung-deaths
# Dec≡Jan bug). This default is pycircstat2's circular knowledge, not hea's.
_SMOOTH_TERM_RE = re.compile(r"\b(?:s|te|ti|t2)\s*\(([^)]*)\)")
_CYCLIC_BS_RE = re.compile(r"""bs\s*=\s*['"](?:cc|cp)['"]""")
_TERM_VAR_RE = re.compile(r"\s*([^\W\d_]\w*)")


def _cyclic_smooth_vars(rhs: str) -> List[str]:
    """Variables of cyclic smooths (``bs='cc'``/``'cp'``) in a formula RHS."""
    out = []
    for m in _SMOOTH_TERM_RE.finditer(rhs):
        body = m.group(1)
        if _CYCLIC_BS_RE.search(body):
            vm = _TERM_VAR_RE.match(body)
            if vm:
                out.append(vm.group(1))
    return out


def _resolve_cyclic_knots(formula: str, user_knots: Optional[dict]) -> Optional[dict]:
    """Default each cyclic smooth's boundary knots to the circular period
    ``[0, 2π]`` so callers need not spell it out; explicit ``user_knots`` win
    per variable. Returns ``None`` when there is nothing to set.
    """
    rhs = formula.split("~", 1)[1] if "~" in formula else formula
    merged = {v: [0.0, 2 * np.pi] for v in _cyclic_smooth_vars(rhs)}
    if user_knots:
        merged.update(user_knots)
    return merged or None


def _gam_predict_ci(gam_fit, newdata: "pl.DataFrame", level: float):
    """Point prediction + Wald confidence band from an ``hea.gam`` fit.

    Returns ``(yhat, lo, hi)`` on the response scale; ``lo``/``hi`` are ``None``
    when ``level`` is falsy. The band is ``fit ± z·se.fit`` with the Gaussian
    quantile — mgcv's default across-the-function smooth confidence interval.
    """
    pred = gam_fit.predict(newdata=newdata, se_fit=bool(level))
    yhat = np.asarray(pred["fit"].to_numpy(), dtype=float)
    if not level:
        return yhat, None, None
    se = np.asarray(pred["se.fit"].to_numpy(), dtype=float)
    z = float(norm.ppf(0.5 + level / 2.0))
    return yhat, yhat - z * se, yhat + z * se


def _two_panel_axes(plt, figsize, polar: bool, axes):
    """Create or validate the (overlay, residual) axis pair for a fit plot.

    When the caller supplies no axes, the overlay (left) axis is made polar
    iff ``polar``; the residual (right) axis is always cartesian. A
    caller-provided ``axes`` pair is used as-is (the caller owns its
    projection).
    """
    if axes is None:
        fig = plt.figure(figsize=figsize or (11, 4.5))
        ax0 = fig.add_subplot(1, 2, 1, projection="polar" if polar else None)
        ax1 = fig.add_subplot(1, 2, 2)
        return fig, ax0, ax1
    axes = list(axes)
    if len(axes) != 2:
        raise ValueError("`axes` must be a sequence of length 2.")
    return axes[0].figure, axes[0], axes[1]


# --- the hea bridge: a circular distribution as an mgcv-style general family --


class CircularLL(GeneralFamily):
    """A regression-ready circular distribution as a hea/mgcv **general
    family** — the Phase-2 bridge of ``dev/plans/circular_gam_integration.md``.

    Mirrors ``hea.family.gaulss``: one linear predictor per modelable
    distribution parameter, in the order the distribution declares them
    (von Mises: μ via ``tanhalf``, log κ via ``log``; wrapped Cauchy: μ, logit
    ρ; projected normal: identity μ₁, μ₂). ``ll()`` fills the packed
    per-datum derivative arrays ``l1..l4`` from the distribution's contract
    methods (``dlogpdf``..``d4logpdf`` — derivatives w.r.t. the *distribution
    parameters*, never pre-chained through links) and delegates the link
    chain rule and gradient/Hessian assembly to ``gamlss_etamu`` /
    ``gamlss_gH``. hea never learns "circular"; pycircstat2 never learns
    penalties or REML.

        gam(["theta ~ s(x, bs='cc')",   # LP1 → μ      (tanhalf)
             "      ~ s(z)"],           # LP2 → log κ  (log)
            data, family=CircularLL(vonmises), method="REML")

    Note the mgcv parameterization: every LP's intercept lives *inside* the
    link — μ = 2·atan(β₀ + …) — unlike the fisher-lee ``CLRegression``'s
    Fisher–Lee offset ``μ₀ + 2·atan(Xβ)``. The two coincide for
    intercept-only models; with covariates they are different (both valid)
    parameterizations, so compare fitted curves, not coefficients.
    """

    scale_known = True
    n_theta = 0

    def __init__(self, dist, links=None):
        roles = getattr(dist, "param_roles", None)
        if not roles:
            raise TypeError(
                f"{getattr(dist, 'name', dist)!r} is not regression-ready: it "
                "declares no `param_roles` overlay (see the Phase 1 contract)."
            )
        self.dist = dist
        self.params = list(roles)  # book-named, declaration order = LP order
        self.n_lp = len(self.params)
        # The base class assumes the LP coordinates *are* the distribution's
        # own logpdf parameters (its ll/fit seams pass them straight
        # through). A family that regresses in transformed coordinates —
        # katojones declares the §3.4 chart pair u1/u2, which logpdf does
        # not accept — needs its dedicated subclass; fail fast here instead
        # of deep in scipy's argument parsing on the first ll() call.
        if type(self) is CircularLL:
            shapes = getattr(dist, "shapes", None) or ""
            shape_names = {s.strip() for s in shapes.split(",") if s.strip()}
            missing = [p for p in self.params if p not in shape_names]
            if shape_names and missing:
                hint = _LL_BY_DIST.get(getattr(dist, "name", None))
                raise TypeError(
                    f"{getattr(dist, 'name', dist)!r} declares LP "
                    f"coordinates {missing} that are not parameters of its "
                    "logpdf — it regresses in transformed coordinates"
                    + (f"; use {hint.__name__}() instead of CircularLL"
                       if hint else "")
                    + "."
                )
        if links is None:
            links = [get_link(dist.link_for(p)) for p in self.params]
        else:
            if len(links) != self.n_lp:
                raise ValueError(
                    f"expected {self.n_lp} links (one per LP), got {len(links)}"
                )
            links = [get_link(lnk) for lnk in links]
        # Contract depth gates the outer-Newton mode hea may use:
        # l4 → full Newton (2), l3 → gradient-only outer (1), l2 → EFS (0).
        self.available_derivs = (
            2 if hasattr(dist, "d4logpdf")
            else 1 if hasattr(dist, "d3logpdf")
            else 0
        )
        self.tri = trind_generator(self.n_lp)
        self.name = f"CircularLL({getattr(dist, 'name', 'dist')})"
        super().__init__(links)

    def _etas(self, X, coef, jj, offset):
        etas = []
        for j in range(self.n_lp):
            eta = X[:, jj[j]] @ coef[jj[j]]
            if offset is not None and len(offset) > j and offset[j] is not None:
                eta = eta + offset[j]
            etas.append(eta)
        return etas

    def _param_values(self, etas):
        """Inverse-link each LP into a book-named parameter dict. A tanhalf
        LP is a principal-branch angle in (−π, π); wrap it to [0, 2π) so the
        distribution's own domain checks pass (every tanhalf consumer is
        2π-periodic in that parameter, so this changes nothing else)."""
        out = {}
        for name, link, eta in zip(self.params, self.links, etas):
            val = link.linkinv(eta)
            if link.name == "tanhalf":
                val = np.mod(val, 2.0 * np.pi)
            out[name] = val
        return out

    def _loglik_values(self, y, params):
        """Per-datum log-likelihood at LP-named parameter values. The seam a
        subclass overrides when its LP coordinates are not the
        distribution's own (KatoJonesLL: chart → book translation)."""
        return np.asarray(self.dist.logpdf(y, **params), dtype=float)

    def _null_params(self, y):
        """Intercept-only parameter estimates in LP coordinates,
        declaration-ordered — the distribution's own ``fit`` for every
        family whose LP coordinates are the book parameters."""
        return np.atleast_1d(self.dist.fit(y)).astype(float)

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None) -> dict:
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        etas = self._etas(X, coef, jj, offset)
        params = self._param_values(etas)

        l0 = self._loglik_values(y, params)
        ret: dict = {"l": float(np.sum(l0)), "l0": l0}
        if deriv == 0:
            return ret

        names = self.params
        shape = y.shape
        d1 = self.dist.dlogpdf(y, **params)
        d2 = self.dist.d2logpdf(y, **params)
        l1 = np.column_stack([np.broadcast_to(d1[p], shape) for p in names])
        l2 = np.column_stack(
            [np.broadcast_to(d2[k], shape)
             for k in combinations_with_replacement(names, 2)]
        )
        ig1 = np.column_stack(
            [link.mu_eta(eta) for link, eta in zip(self.links, etas)]
        )
        g2 = np.column_stack(
            [link.d2link(params[name])
             for link, name in zip(self.links, names)]
        )
        l3 = l4 = g3 = g4 = None
        if deriv > 1:
            d3 = self.dist.d3logpdf(y, **params)
            l3 = np.column_stack(
                [np.broadcast_to(d3[k], shape)
                 for k in combinations_with_replacement(names, 3)]
            )
            g3 = np.column_stack(
                [link.d3link(params[name])
                 for link, name in zip(self.links, names)]
            )
        if deriv > 3:
            d4 = self.dist.d4logpdf(y, **params)
            l4 = np.column_stack(
                [np.broadcast_to(d4[k], shape)
                 for k in combinations_with_replacement(names, 4)]
            )
            g4 = np.column_stack(
                [link.d4link(params[name])
                 for link, name in zip(self.links, names)]
            )

        tri = self.tri
        de = gamlss_etamu(l1, l2, l3, l4, ig1, g2, g3, g4,
                          tri["i2"], tri["i3"], tri["i4"], deriv - 1)
        gh = gamlss_gH(X, jj, de["l1"], de["l2"], tri["i2"],
                       l3=de["l3"], i3=tri["i3"], l4=de["l4"], i4=tri["i4"],
                       d1b=d1b, d2b=d2b, deriv=deriv - 1, fh=fh, D=D)
        ret.update(gh)
        return ret

    def initialize_coef(self, y, X, lpi, E=None, offset=None,
                        use_unscaled: bool = False) -> np.ndarray:
        """Null-model start, mgcv-flavoured but circular-safe: fit the
        *distribution itself* to ``y`` (its own ``fit`` — analytic or
        moment-seeded MLE), then least-squares each LP onto the constant
        target ``link(param̂)`` with the penalty root ``E`` stacked as a
        regularizer. gaulss regresses transformed *data* per LP and needs
        pen.reg's edf search; a constant target has no df to overfit, so the
        plain stacked solve serves both ``use_unscaled`` branches.
        """
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        n, p = X.shape
        if E is None:
            E = np.zeros((0, p))
        param_hat = self._null_params(y)
        start = np.zeros(p)
        for j, (link, par0) in enumerate(zip(self.links, param_hat)):
            # clip guards the tanhalf pole at μ̂ ≡ π (η → ∞)
            eta0 = float(np.clip(link.link(float(par0)), -1e6, 1e6))
            target = np.full(n, eta0)
            if offset is not None and len(offset) > j and offset[j] is not None:
                target = target - offset[j]
            cols = jj[j]
            xa = np.vstack([X[:, cols], E[:, cols]])
            ta = np.concatenate([target, np.zeros(E.shape[0])])
            b, *_ = np.linalg.lstsq(xa, ta, rcond=None)
            b[~np.isfinite(b)] = 0.0
            start[cols] = b
        return start

    def _fitted_direction(self, fitted):
        """Fitted mean direction from the (n, n_lp) parameter matrix —
        role-aware: a single location parameter *is* the angle; the projected
        normal's two Cartesian location components combine via atan2."""
        loc = self.dist.params_by_role().get("location", [])
        idx = [self.params.index(nm) for nm in loc]
        if len(idx) == 1:
            return np.asarray(fitted)[:, idx[0]]
        if len(idx) == 2:
            f = np.asarray(fitted)
            return np.arctan2(f[:, idx[1]], f[:, idx[0]])
        raise NotImplementedError(
            f"{self.name}: cannot derive a direction from location "
            f"parameters {loc!r}"
        )

    def postproc(self, y, fitted) -> dict:
        """Null deviance for the summary's "deviance explained": the
        distribution's own intercept-only fit pushed through the same
        deviance-residual convention as :meth:`residuals` (twice the
        log-likelihood gap to the fitted-mode saturated reference)."""
        y = np.asarray(y, dtype=float)
        par0 = self._null_params(y)
        fitted0 = np.broadcast_to(par0, (y.shape[0], self.n_lp))
        r0 = self.residuals(y, fitted0, type="deviance")
        return {"null_deviance": float(np.sum(r0 * r0))}

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """Angular residuals. ``response`` = the wrapped difference
        ``y − μ̂(x)`` in (−π, π]; ``deviance``/``pearson`` =
        ``sign·√(2·max(ℓ_mode − ℓ, 0))`` with the saturated value taken at
        the fitted direction — the density mode for every symmetric family;
        for the skewed ones (sine-skewed JP, Kato–Jones) the fitted
        direction is a mode *anchor* rather than the argmax, so the inner
        clip at 0 keeps the convention well-defined there.
        ``fitted`` is the (n, n_lp) matrix of inverse-linked parameters."""
        y = np.asarray(y, dtype=float)
        fitted = np.asarray(fitted, dtype=float)
        mu_hat = self._fitted_direction(fitted)
        rsd = np.angle(np.exp(1j * (y - mu_hat)))
        if type == "response":
            return rsd
        params = {
            name: (np.mod(fitted[:, j], 2.0 * np.pi)
                   if self.links[j].name == "tanhalf" else fitted[:, j])
            for j, name in enumerate(self.params)
        }
        l_obs = self._loglik_values(y, params)
        l_sat = self._loglik_values(np.mod(mu_hat, 2.0 * np.pi), params)
        return np.sign(rsd) * np.sqrt(2.0 * np.clip(l_sat - l_obs, 0.0, None))

    def __repr__(self):
        links = ", ".join(repr(lnk.name) for lnk in self.links)
        return f"{self.name} (links: {links})"


class KatoJonesLL(CircularLL):
    """Kato–Jones (2015) as a general family in **disc-chart coordinates**
    (μ, γ, u₁, u₂) — the §3.4 design of ``dev/plans/
    distribution-validation.md``.

    The four LPs are μ (tanhalf), γ (logit) and the unconstrained chart pair
    u₁, u₂ (identity); the chart ``(a, b) = (γ, 0) + (1−γ)·u/√(1+‖u‖²)``
    keeps the Cartesian shape pair (a, b) = (ρ cos λ, ρ sin λ) strictly
    inside the Theorem-1 feasible disc for *every* coefficient vector, so
    the coupled constraint never reaches hea. ``u ≡ 0`` recovers the wrapped
    Cauchy WC(μ, γ) exactly — an intercept-only model for both u-LPs is the
    natural reduced model.

    The distribution's contract derivatives are already chart-coordinate
    (``katojones.dlogpdf``/``d2logpdf`` chain the Cartesian scores through
    the chart), so this subclass only owns the two seams where book
    parameters appear: per-datum log-likelihood values (γ, u → ρ, λ
    translation) and the null start (moments fit + closed-form chart
    inverse).

        gam(["theta ~ s(x, bs='cc')",   # LP1 → μ        (tanhalf)
             "      ~ z",               # LP2 → logit γ  (logit)
             "      ~ 1",               # LP3 → u₁       (identity)
             "      ~ 1"],              # LP4 → u₂       (identity)
            data, family=KatoJonesLL(), method="REML")
    """

    def __init__(self, links=None):
        super().__init__(katojones, links)

    @staticmethod
    def _book_params(params):
        """Chart-named LP values → the distribution's book parameters."""
        a, b = katojones.disc_chart(params["gamma"], params["u1"],
                                    params["u2"])
        return {
            "mu": params["mu"],
            "gamma": params["gamma"],
            "rho": np.hypot(a, b),
            "lam": np.mod(np.arctan2(b, a), 2.0 * np.pi),
        }

    def _loglik_values(self, y, params):
        return np.asarray(
            self.dist.logpdf(y, **self._book_params(params)), dtype=float
        )

    def _null_params(self, y):
        mu0, g0, rho0, lam0 = self.dist.fit(y, method="moments")
        u1, u2 = self.dist.disc_chart_inverse(g0, rho0, lam0)
        return np.array([mu0, g0, float(u1), float(u2)])


# Distributions whose regression (LP) coordinates are not their own logpdf
# parameters, keyed by distribution name → the dedicated family class.
# Consulted by `_circular_family` (the `CLRegression` auto-route) and by the
# `CircularLL.__init__` fail-fast guard's error hint.
_LL_BY_DIST = {"katojones": KatoJonesLL}


def _circular_family(dist) -> CircularLL:
    """Wrap a regression-ready distribution in its general-family class:
    plain :class:`CircularLL` for every family whose LP coordinates are its
    own parameters, the dedicated subclass otherwise (``katojones`` →
    :class:`KatoJonesLL`, which owns the disc-chart translation)."""
    cls = _LL_BY_DIST.get(getattr(dist, "name", None))
    return cls() if cls is not None else CircularLL(dist)


class CLRegression:
    """
    Circular-Linear Regression.

    Two backends behind **one formula grammar**, selected with ``backend=``:

    - **"gam"** (default): ``hea.gam`` with a :class:`CircularLL` general
      family — one formula per linear predictor in the mgcv parameterization
      (intercept *inside* the link). Handles any regression-ready ``family=``,
      penalized smooths (``s()``/``te()``), REML, and **different covariates
      per LP**. ``CLRegression(["θ ~ s(x)", "~ s(z)"], data)`` smooths μ(x)
      and log κ(z) jointly. Use ``family=projectednormal`` when μ(x) must
      sweep the full circle (the tanhalf link cannot cross ±π).
    - **"fisher-lee"**: the von Mises MLE — ``μ_i = μ₀ + 2·atan(x_iᵀβ)``
      (offset *outside* the link; Fisher & Lee 1992 / Fisher 1993 §6.4),
      ``κ_i = exp(α + x_iᵀγ)``. Fast, dependency-light, R-validated. It ties
      μ and κ to one shared design, so it expresses mean / kappa / mixed
      (which LP carries covariates) but **not** different covariates per LP.

    **Formula grammar (both backends):** a string ``"θ ~ rhs"`` is sugar for
    ``["θ ~ rhs", "~ 1"]`` (a mean model — μ on the covariates, constant κ);
    a list ``[μ-formula, logκ-formula]`` models each linear predictor.

    Dispatch: a **smooth term forces "gam"**; ``theta=``/``X=`` arrays and the
    deprecated ``model_type=`` alias imply ``"fisher-lee"``; otherwise the
    default is ``"gam"``. The two parameterizations differ with covariates
    (intercept inside vs outside the link) — they coincide for intercept-only
    models, otherwise compare fitted *curves*, not coefficients.

    Parameters
    ----------
    formula : str or list of str, optional
        ``"θ ~ rhs"`` (≡ ``["θ ~ rhs", "~ 1"]``, a mean model) or a
        ``[μ-formula, logκ-formula]`` list. Smooth terms require
        ``backend="gam"``.
    data : polars.DataFrame, optional
        A polars (or pandas) DataFrame containing the response and predictors.
    theta : np.ndarray, optional
        Circular response in radians (fisher-lee array interface; with ``X``).
    X : np.ndarray, optional
        Predictor array (fisher-lee array interface; with ``theta``).
    backend : {"gam", "fisher-lee"}, optional
        Which engine to use. Default ``"gam"`` (``"fisher-lee"`` for the array
        interface and the legacy ``model_type=`` alias). A smooth term forces
        ``"gam"``.
    model_type : str, optional
        **Deprecated** — pass a formula list instead (``["θ ~ X", "~ 1"]`` =
        mean, ``["θ ~ 1", "~ X"]`` = kappa, ``["θ ~ X", "~ X"]`` = mixed).
        Still accepted for the array/single-formula forms; implies
        ``backend="fisher-lee"``. One of 'mean', 'kappa', 'mixed'.
    beta0 : np.ndarray, optional
        Initial values for the beta coefficients.
    alpha0 : float, optional
        Initial value for the intercept.
    gamma0 : np.ndarray, optional
        Initial values for the gamma coefficients.
    tol : float, optional
        Convergence tolerance for the optimization.
    max_iter : int, optional
        Maximum number of iterations for the optimization.
    verbose : bool, optional
        Whether to print optimization progress.

    Attributes
    ----------
    result : dict
        A dictionary containing the estimated coefficients and other statistics.

        - beta : np.ndarray
            Estimated beta coefficients for the mean direction. Used by
            'mean' and 'mixed' models; zero for 'kappa'.
        - alpha : float
            Estimated intercept for the concentration parameter.
        - gamma : np.ndarray
            Estimated coefficients for the concentration parameter.
        - mu : float
            Estimated mean direction of the circular response.
        - kappa : float or np.ndarray
            Concentration parameter. Scalar for 'mean'; n-element array
            of per-observation values κ_i = exp(α + X_iᵀγ) for 'kappa'
            and 'mixed'.
        - log_likelihood : float
            Log-likelihood of the model.

    Methods
    -------
    summary()
        Print the coefficient table, mean direction, concentration, and fit
        metrics.
    predict(X_new)
        Predict mean direction at new X (constant μ for ``model_type='kappa'``).
    predict_kappa(X_new)
        Predict per-observation κ̂(X) for ``model_type`` in
        ``{'kappa', 'mixed'}``.
    plot(figsize=None, n_curve=200, axes=None)
        Two-panel diagnostic figure (fit overlay / κ curve / residuals,
        depending on ``model_type`` and dimensionality).
    AIC(), BIC()
        Information criteria for the fitted model.

    Notes
    -----
    The 'mean' branch is ported from ``lm.circular.cl`` in the ``circular``
    R package (Agostinelli & Lund); SE formulas follow Fisher (1993)
    eq. 6.62-6.64. The 'kappa' and 'mixed' branches extend that framework
    to model the concentration as a log-linear function of predictors,
    following Fisher (1993) §6.4.3-§6.4.4 (eq. 6.81, 6.82, 6.86, 6.87).
    Per-observation SE for κ̂_i uses the delta method on (α̂, γ̂).

    References
    ----------
    - Fisher, N. I. (1993). Statistical analysis of circular data. Cambridge University Press.
    - Pewsey, A., Neuhäuser, M., & Ruxton, G. D. (2014) Circular Statistics in R. Oxford University Press.
    """

    def __init__(
        self,
        formula: Union[str, List[str], None] = None,
        data: Optional[pl.DataFrame] = None,
        theta: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        model_type: Optional[str] = None,
        beta0: Union[np.ndarray, None] = None,
        alpha0: Union[float, None] = None,
        gamma0: Union[np.ndarray, None] = None,
        tol: float = 1e-8,
        max_iter: int = 100,
        verbose: bool = False,
        *,
        backend: Optional[str] = None,
        family=None,
        knots: Optional[dict] = None,
        method: str = "REML",
        **gam_kwargs,
    ):
        self.verbose = verbose
        self.tol = tol
        self.max_iter = max_iter

        # --- backend resolution (plan §5). One formula grammar, two engines:
        #   • "fisher-lee" — von Mises MLE with the tan-half link and the
        #     offset *outside* (μ = μ₀ + 2·atan(Xβ); Fisher & Lee 1992 /
        #     Fisher 1993 §6.4). Fast, R-validated; ties μ and κ to one shared
        #     design (mean / kappa / mixed = which LP carries covariates).
        #   • "gam" — hea general family (CircularLL) in the mgcv
        #     parameterization (intercept *inside* the link); any
        #     regression-ready family, smooths, REML, different covariates
        #     per LP.
        # Default: gam. A smooth term forces gam. theta=/X= arrays and the
        # legacy (deprecated) model_type= alias both imply fisher-lee.
        if isinstance(formula, (list, tuple)):
            has_smooth = any(_has_smooth(str(f)) for f in formula)
        elif isinstance(formula, str):
            has_smooth = _has_smooth(formula)
        else:
            has_smooth = False
        arrays = theta is not None or X is not None

        if model_type is not None:
            if backend not in (None, "fisher-lee"):
                raise ValueError(
                    "model_type= is a fisher-lee alias; drop it to use "
                    "backend='gam' (model each LP via the formula list)."
                )
            backend = "fisher-lee"
        elif arrays:
            backend = backend or "fisher-lee"
            if backend != "fisher-lee":
                raise ValueError(
                    "theta=/X= arrays use backend='fisher-lee'; pass a "
                    "formula + data for backend='gam'."
                )
        else:
            backend = backend or "gam"

        if backend not in ("fisher-lee", "gam"):
            raise ValueError(f"backend must be 'fisher-lee' or 'gam'; got {backend!r}.")
        if has_smooth and backend == "fisher-lee":
            raise ValueError(
                "smooth terms (s()/te()/…) require backend='gam'; the "
                "fisher-lee backend has no penalized smoother."
            )
        self.backend = backend

        if backend == "gam":
            if data is None:
                raise ValueError("The gam backend requires a formula + `data`.")
            self.model_type = None  # the formula list plays this role
            self._init_gam(formula, data, family, knots, method, gam_kwargs)
            return

        if family is not None or knots is not None or gam_kwargs:
            raise ValueError("family=/knots=/gam options apply only to backend='gam'.")
        self.gam_fit = None
        self.family = None
        self._init_fisher_lee(formula, data, theta, X, model_type, beta0, alpha0, gamma0)

    def _init_fisher_lee(self, formula, data, theta, X, model_type,
                         beta0, alpha0, gamma0):
        """Set up and fit the Fisher–Lee von Mises backend from any accepted
        input shape: ``theta``/``X`` arrays, a ``[μ-formula, logκ-formula]``
        list, or a single μ-formula (≡ a mean model). The legacy
        ``model_type`` still selects mean/kappa/mixed for the array and
        single-formula forms; for a formula list it is read off which LP
        carries covariates."""
        # Location link from the contract (vonmises declares tanhalf via its
        # role overlay); μ_i = μ₀ + linkinv(x_iᵀβ).
        self._mu_link = get_link(vonmises.link_for("mu"))

        if isinstance(formula, (list, tuple)):
            theta_arr, X_arr, feature_names, mt = self._parse_lp_formulas(formula, data)
        elif formula is not None and data is not None:
            theta_arr, X_arr, feature_names = self._parse_formula(formula, data)
            mt = model_type or "mean"  # a bare μ-formula is a mean model
        elif theta is not None and X is not None:
            theta_arr, X_arr, feature_names = theta, X, None
            mt = model_type or "mixed"  # historic array default
        else:
            raise ValueError(
                "Provide a formula + data, a formula list, or theta and X."
            )

        if mt not in ("mean", "kappa", "mixed"):
            raise ValueError("model_type must be 'mean', 'kappa', or 'mixed'.")
        self.model_type = mt

        self.theta, self.X = self._prepare_design(theta_arr, X_arr)
        self.feature_names = feature_names or [f"x{i}" for i in range(self.X.shape[1])]

        p = self.X.shape[1]
        self.alpha = float(alpha0) if alpha0 is not None else 0.0
        self.beta = self._coerce_vector(beta0, p, name="beta")
        self.gamma = self._coerce_vector(gamma0, p, name="gamma")
        self.result = self._fit()

    def _parse_lp_formulas(self, formulas, data):
        """Parse a fisher-lee ``[μ-formula, logκ-formula]`` list into
        ``(theta, X, feature_names, model_type)``. The model type follows
        which LP carries covariates; since the backend ties both LPs to one
        shared design, two non-empty LPs must name the same predictors (else
        use ``backend='gam'``)."""
        if len(formulas) != 2:
            raise ValueError(
                "the fisher-lee backend expects a [μ-formula, logκ-formula] "
                f"list of length 2 (got {len(formulas)})."
            )
        data = _to_polars(data)
        cols = set(data.columns)
        mu_f, kappa_f = formulas
        response = mu_f.split("~", 1)[0].strip()
        if not response:
            raise ValueError(f"First formula must name the response: {mu_f!r}")
        theta = data[response].to_numpy()
        mu_cov = self._formula_covariates(mu_f, cols)
        kappa_cov = self._formula_covariates(kappa_f, cols)

        if mu_cov and not kappa_cov:
            return theta, data[mu_cov].to_numpy(), mu_cov, "mean"
        if kappa_cov and not mu_cov:
            return theta, data[kappa_cov].to_numpy(), kappa_cov, "kappa"
        if mu_cov and kappa_cov:
            if set(mu_cov) != set(kappa_cov):
                raise ValueError(
                    "the fisher-lee backend ties μ and κ to one shared design, "
                    f"but μ uses {mu_cov} and κ uses {kappa_cov}. Use "
                    "backend='gam' for different covariates per predictor."
                )
            return theta, data[mu_cov].to_numpy(), mu_cov, "mixed"
        raise ValueError("no predictors in either formula; nothing to regress.")

    @staticmethod
    def _formula_covariates(formula: str, data_cols) -> List[str]:
        """Covariate column names on a formula RHS (``~ 1`` / ``~`` ⇒ none)."""
        rhs = formula.split("~", 1)[1] if "~" in formula else formula
        out: List[str] = []
        for term in rhs.split("+"):
            term = term.strip()
            if term in ("", "1"):
                continue
            if term not in data_cols:
                raise ValueError(
                    f"unknown predictor {term!r}; columns are {sorted(data_cols)}."
                )
            out.append(term)
        return out

    def _init_gam(self, formula, data, family, knots, method, gam_kwargs):
        """Fit the distributional gam backend: one formula per linear
        predictor, ``hea.gam`` + a :class:`CircularLL` family, REML by
        default. A single formula gets a constant second LP (``"~ 1"``) —
        smooth μ(x), constant concentration."""
        if family is None:
            family = CircularLL(vonmises)
        elif not isinstance(family, CircularLL):
            # a regression-ready distribution: wrap in its family class
            # (katojones auto-routes to KatoJonesLL via _circular_family)
            family = _circular_family(family)
        formulas = list(formula) if isinstance(formula, (list, tuple)) else [formula]
        if len(formulas) == 1 and family.n_lp == 2:
            formulas.append("~ 1")
        if len(formulas) != family.n_lp:
            raise ValueError(
                f"{family.name} has {family.n_lp} linear predictors; got "
                f"{len(formulas)} formulas."
            )
        self.formula = formulas
        self.family = family
        self.data = _to_polars(data)
        response = formulas[0].split("~", 1)[0].strip()
        if not response:
            raise ValueError(f"First formula must name the response: {formulas[0]!r}")
        self.feature_names = None
        self.theta = np.mod(
            np.asarray(self.data[response].to_numpy(), dtype=float), 2.0 * np.pi
        )

        # Cyclic smooths (bs='cc'/'cp') on circular predictors default their
        # boundary knots to the period [0, 2π] — same circular knowledge the
        # LC/CC backends supply; explicit user knots win per variable.
        merged: dict = {}
        for f in formulas:
            merged.update(_resolve_cyclic_knots(f, None) or {})
        if knots:
            merged.update(knots)

        self.gam_fit = _hea_gam(
            formulas,
            self.data,
            family=family,
            knots=merged or None,
            method=method,
            **gam_kwargs,
        )
        self.result = self._build_result_gam()

    def _build_result_gam(self) -> dict:
        """Result dict for the distributional gam backend. Keys shared with
        the fisher-lee path keep their meaning (``mu``/``kappa`` become
        per-observation fitted values; ``log_likelihood`` is the penalized
        fit's log-likelihood); smooth-specific summaries are added."""
        g = self.gam_fit
        fam = self.family
        fv = np.asarray(g.fitted_values, dtype=float)
        mu = np.mod(fam._fitted_direction(fv), 2.0 * np.pi)
        conc = fam.dist.params_by_role().get("concentration", [])
        kappa = fv[:, fam.params.index(conc[0])] if conc else None
        out = {
            "backend": "gam",
            "param_names": list(fam.params),
            "fitted_params": fv,
            "mu": mu,
            "kappa": kappa,
            "coefficients": dict(zip(g.bhat.columns, g.bhat.row(0))),
            "log_likelihood": float(g.logLik),
            "aic": float(g.AIC),
            "bic": float(g.BIC),
            "reml": float(g.REML_criterion),
            "edf_total": float(g.edf_total),
            "residuals": np.asarray(g.residuals, dtype=float),
            "n": fv.shape[0],
        }
        edf_by = getattr(g, "edf_by_smooth", None)
        if edf_by is not None:
            out["edf_by_smooth"] = {k: float(v) for k, v in dict(edf_by).items()}
        return out

    def predict_params(self, data) -> dict:
        """Per-observation fitted distribution parameters at new data
        (gam backend only): a book-named dict, one entry per linear
        predictor, on the parameter scale (tanhalf angles wrapped to
        ``[0, 2π)``). ``data`` may be a DataFrame or a design array."""
        if self.backend != "gam":
            raise ValueError(
                "predict_params() is for the distributional gam backend; the "
                "fisher-lee path exposes predict()/predict_kappa()."
            )
        new = self._gam_newdata(data)
        pred = self.gam_fit.predict(newdata=new)
        cols = list(pred.columns)
        out = {}
        for j, name in enumerate(self.family.params):
            v = np.asarray(pred[cols[j]].to_numpy(), dtype=float)
            if self.family.links[j].name == "tanhalf":
                v = np.mod(v, 2.0 * np.pi)
            out[name] = v
        return out

    @staticmethod
    def _coerce_vector(vec: Optional[np.ndarray], length: int, name: str) -> np.ndarray:
        if vec is None:
            return np.zeros(length, dtype=float)
        arr = np.asarray(vec, dtype=float).reshape(-1)
        if arr.size != length:
            raise ValueError(f"Initial {name} must have length {length} (got {arr.size}).")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Initial {name} contains non-finite values.")
        return arr

    @staticmethod
    def _prepare_design(theta: Iterable[float], X: Iterable[Iterable[float]]) -> Tuple[np.ndarray, np.ndarray]:
        theta_arr = np.asarray(theta, dtype=float).reshape(-1)
        if theta_arr.size == 0:
            raise ValueError("`theta` must contain at least one observation.")
        if not np.all(np.isfinite(theta_arr)):
            raise ValueError("`theta` contains non-finite values.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.ndim != 2:
            raise ValueError("`X` must be convertible to a 2D numeric array.")
        if X_arr.shape[0] != theta_arr.size:
            raise ValueError("`theta` and `X` must have matching numbers of rows.")
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("`X` contains non-finite values.")
        return theta_arr, X_arr

    def _parse_formula(
        self, formula: str, data: pl.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        data = _to_polars(data)
        parts = formula.split("~")
        if len(parts) != 2:
            raise ValueError(
                f"Formula must contain exactly one '~'; got: {formula!r}"
            )
        theta_col, x_cols = parts
        theta_series = data[theta_col.strip()]
        if theta_series.is_null().any():
            raise ValueError("Response column contains missing values.")
        theta = theta_series.to_numpy()
        x_cols = [col.strip() for col in x_cols.split("+") if col.strip()]
        if not x_cols:
            raise ValueError(f"No predictors found in formula: {formula!r}")
        X_df = data[x_cols]
        if X_df.null_count().to_numpy().any():
            raise ValueError("Predictor columns contain missing values.")
        X = X_df.to_numpy()
        return theta, X, x_cols

    @staticmethod
    def _safe_exp_kappa(eta: np.ndarray) -> np.ndarray:
        # Bound the log-concentration to avoid exp overflow during iterations.
        # exp(±50) ≈ {5e21, 2e-22}, comfortably finite.
        return np.exp(np.clip(eta, -50.0, 50.0))

    @staticmethod
    def _log_i0(kappa: np.ndarray) -> np.ndarray:
        # log I_0(κ) computed via the exponentially scaled Bessel to stay finite
        # for large κ (raw i0 overflows around κ ≈ 710).
        return np.asarray(kappa) + np.log(i0e(kappa))

    @staticmethod
    def _delta_se_kappa(
        kappa: np.ndarray, X1: np.ndarray, cov_alpha_gamma: np.ndarray
    ) -> np.ndarray:
        # κ_i = exp(α + X_iᵀ γ); ∂κ_i/∂(α,γ) = κ_i · z_i with z_i = [1, X_i].
        # Var(κ_i) ≈ κ_i² · z_iᵀ Σ z_i (delta method).
        z_cov = X1 @ cov_alpha_gamma
        quad = np.einsum("ij,ij->i", z_cov, X1)
        var_kappa = (kappa**2) * np.clip(quad, 0.0, None)
        return np.sqrt(var_kappa)

    def _fit(self):
        theta = self.theta
        n = len(theta)
        X = self.X
        X1 = np.column_stack((np.ones(n), X))  # Add intercept
        beta, alpha, gamma = self.beta, self.alpha, self.gamma
        diff = self.tol + 1
        log_likelihood_old = -np.inf

        # Tiny ridge added to the normal-equation LHS to keep solves finite
        # when XtX is near-singular. Hoisted out of the loop body.
        ridge_X = 1e-8 * np.eye(X.shape[1])
        ridge_X1 = 1e-8 * np.eye(X1.shape[1])

        for iter_count in range(self.max_iter):
            if self.model_type == "mean":
                # Step 1: Compute mu and kappa
                raw_deviation = theta - self._mu_link.linkinv(X @ beta)
                S = np.mean(np.sin(raw_deviation))
                C = np.mean(np.cos(raw_deviation))
                R = np.hypot(S, C)
                kappa = float(A1inv(R))
                mu = np.arctan2(S, C)

                # Step 2: Update beta. Score from the von Mises regression
                # contract (vonmises.dlogpdf — the §3.2 lift); the IRLS weight
                # is the *expected* (Fisher) information −E[∂²_{μμ}ℓ] = κ A1(κ),
                # distinct from the observed d2logpdf. G = ∂μ/∂β chains the
                # link's mu_eta through the design.
                G = self._mu_link.mu_eta(X @ beta)[:, None] * X
                weight = float(kappa * A1(kappa))
                u = vonmises.dlogpdf(raw_deviation, mu, kappa)["mu"]
                XtX = G.T @ G
                rhs = G.T @ u + weight * XtX @ beta
                mat = weight * XtX + ridge_X
                beta_new = _safe_solve(mat, rhs)
                alpha_new, gamma_new = alpha, gamma

                # Log-likelihood
                log_likelihood = -n * float(self._log_i0(kappa)) + kappa * np.sum(
                    np.cos(raw_deviation - mu)
                )

            elif self.model_type == "kappa":
                # Step 1: Compute mu and kappa
                kappa = self._safe_exp_kappa(alpha + X @ gamma)
                S = float(np.sum(kappa * np.sin(theta)))
                C = float(np.sum(kappa * np.cos(theta)))
                mu = np.arctan2(S, C)

                # Step 2: Update gamma. κ-score from the contract
                # (vonmises.dlogpdf — cos(θ−μ) − A1(κ)); the IRLS weight uses
                # the expected info A1'(κ). Floor A1'(κ) to keep the step finite
                # when some κ_i are large (A1'(κ) → 0 as κ → ∞ ⇒ y_gamma blows up).
                a1_prime = np.maximum(A1prime(kappa), 1e-12)
                residuals_gamma = vonmises.dlogpdf(theta, mu, kappa)["kappa"]
                y_gamma = residuals_gamma / (a1_prime * kappa)
                weights = (kappa**2) * a1_prime
                XtWX = X1.T @ (weights[:, None] * X1)
                XtWy = X1.T @ (weights * y_gamma)
                update = _safe_solve(XtWX + ridge_X1, XtWy)
                alpha_new = alpha + update[0]
                gamma_new = gamma + update[1:]
                beta_new = beta
                # Log-likelihood
                log_likelihood = -np.sum(self._log_i0(kappa)) + np.sum(
                    kappa * np.cos(theta - mu)
                )

            elif self.model_type == "mixed":
                # Step 1: Compute mu and kappa
                kappa = self._safe_exp_kappa(alpha + X @ gamma)
                raw_deviation = theta - self._mu_link.linkinv(X @ beta)
                S = np.sum(kappa * np.sin(raw_deviation))
                C = np.sum(kappa * np.cos(raw_deviation))
                mu = np.arctan2(S, C)

                # Both scores from the contract (vonmises.dlogpdf) at the
                # de-trended angle; the IRLS weights are the expected (Fisher)
                # information κ A1(κ) for β and A1'(κ) for γ.
                score = vonmises.dlogpdf(raw_deviation, mu, kappa)

                # Step 2: Update beta — Fisher scoring step from current β.
                # Score s(β) = Gᵀ (κ ⊙ sin(rdev − μ)); info I(β) = Gᵀ diag(κ A1(κ)) G,
                # with G = ∂μ/∂β = mu_eta(Xβ)·X. β_new solves I β_new = I β + s.
                G = self._mu_link.mu_eta(X @ beta)[:, None] * X
                weights_beta = kappa * A1(kappa)
                XtWX_beta = G.T @ (weights_beta[:, None] * G)
                u_beta = score["mu"]
                rhs_beta = G.T @ u_beta + XtWX_beta @ beta
                beta_new = _safe_solve(XtWX_beta + ridge_X, rhs_beta)

                # Step 3: Update gamma
                a1_prime = np.maximum(A1prime(kappa), 1e-12)
                residuals_gamma = score["kappa"]
                y_gamma = residuals_gamma / (a1_prime * kappa)
                weights_gamma = (kappa**2) * a1_prime
                XtWX = X1.T @ (weights_gamma[:, None] * X1)
                XtWy = X1.T @ (weights_gamma * y_gamma)
                update = _safe_solve(XtWX + ridge_X1, XtWy)
                alpha_new = alpha + update[0]
                gamma_new = gamma + update[1:]

                # Log-likelihood
                log_likelihood = -np.sum(self._log_i0(kappa)) + np.sum(
                    kappa * np.cos(raw_deviation - mu)
                )

            # Convergence check
            diff = np.abs(log_likelihood - log_likelihood_old)
            if self.verbose:
                print(
                    f"Iteration {iter_count + 1}: Log-Likelihood = {log_likelihood:.5f}, diff = {diff:.2e}"
                )
            if diff < self.tol:
                break

            beta, alpha, gamma = beta_new, alpha_new, gamma_new
            log_likelihood_old = log_likelihood
        else:
            warnings.warn(
                f"CLRegression did not converge in {self.max_iter} iterations "
                f"(last diff={diff:.2e}, tol={self.tol:.2e}).",
                RuntimeWarning,
                stacklevel=2,
            )

        result = {
            "beta": beta,
            "alpha": alpha,
            "gamma": gamma,
            "mu": mu,
            "kappa": kappa,
            "log_likelihood": log_likelihood,
        }

        se_result = self._compute_standard_errors(result)

        result.update(se_result)

        return result

    def _compute_standard_errors(self, result):
        """
        Compute standard errors for the parameters based on the fitted model.
        """
        theta = self.theta
        X = self.X
        n = len(theta)
        kappa = result["kappa"]
        beta = result["beta"]

        se_results = {}

        if self.model_type == "mean":
            # Mean Direction Model
            G = self._mu_link.mu_eta(X @ beta)[:, None] * X
            weight = float(kappa * A1(kappa))
            XtAX = weight * (G.T @ G)
            cov_beta = _safe_inverse(XtAX)
            se_beta = np.sqrt(np.diag(cov_beta))

            denom_mu = max((n - X.shape[1]) * kappa * A1(kappa), 1e-12)
            se_mu = 1 / np.sqrt(denom_mu)
            denom_kappa = n * (1 - A1(kappa) ** 2 - A1(kappa) / kappa)
            se_kappa = np.sqrt(1 / max(denom_kappa, 1e-12))

            se_results.update(
                {
                    "se_beta": se_beta,
                    "se_mu": se_mu,
                    "se_kappa": se_kappa,
                }
            )

        elif self.model_type == "kappa":
            # Concentration Parameter Model
            X1 = np.column_stack((np.ones(n), X))  # Add intercept
            weights = (kappa**2) * A1prime(kappa)
            XtWX = X1.T @ (weights[:, None] * X1)

            cov_gamma_alpha = _safe_inverse(XtWX)
            se_alpha = np.sqrt(cov_gamma_alpha[0, 0])
            se_gamma = np.sqrt(np.diag(cov_gamma_alpha[1:, 1:]))

            # Fisher (1993), eq. 6.82: σ̂_μ = (Σ κ̂_i A1(κ̂_i) − 1/2)^(−1/2).
            denom_mu = max(float(np.sum(kappa * A1(kappa))) - 0.5, 1e-12)
            se_mu = 1 / np.sqrt(denom_mu)

            se_kappa = self._delta_se_kappa(kappa, X1, cov_gamma_alpha)

            se_results.update(
                {
                    "se_alpha": se_alpha,
                    "se_gamma": se_gamma,
                    "se_mu": se_mu,
                    "se_kappa": se_kappa,
                }
            )

        elif self.model_type == "mixed":
            # Mixed Model
            G = self._mu_link.mu_eta(X @ beta)[:, None] * X
            weights_beta = kappa * A1(kappa)
            XtGKGX = G.T @ (weights_beta[:, None] * G)

            cov_beta = _safe_inverse(XtGKGX)
            se_beta = np.sqrt(np.diag(cov_beta))

            X1 = np.column_stack((np.ones(n), X))  # Add intercept
            weights_gamma = (kappa**2) * A1prime(kappa)
            XtWX_gamma = X1.T @ (weights_gamma[:, None] * X1)

            cov_gamma_alpha = _safe_inverse(XtWX_gamma)
            se_alpha = np.sqrt(cov_gamma_alpha[0, 0])
            se_gamma = np.sqrt(np.diag(cov_gamma_alpha[1:, 1:]))

            # Fisher (1993), eq. 6.82: σ̂_μ = (Σ κ̂_i A1(κ̂_i) − 1/2)^(−1/2).
            denom_mu = max(float(np.sum(kappa * A1(kappa))) - 0.5, 1e-12)
            se_mu = 1 / np.sqrt(denom_mu)
            se_kappa = self._delta_se_kappa(kappa, X1, cov_gamma_alpha)
            se_results.update(
                {
                    "se_beta": se_beta,
                    "se_alpha": se_alpha,
                    "se_gamma": se_gamma,
                    "se_mu": se_mu,
                    "se_kappa": se_kappa,
                }
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return se_results

    def AIC(self):
        """
        Calculate Akaike Information Criterion (AIC).
        """
        if self.result is None:
            raise ValueError("Model must be fitted before calculating AIC.")

        if self.backend == "gam":
            # hea's general-family AIC (edf-corrected, mgcv convention).
            return self.result["aic"]

        log_likelihood = self.result["log_likelihood"]
        if self.model_type == "mean":
            n_params = len(self.result["beta"])  # Only beta
        elif self.model_type == "kappa":
            n_params = 1 + len(self.result["gamma"])  # alpha + gamma
        elif self.model_type == "mixed":
            n_params = (
                1 + len(self.result["beta"]) + len(self.result["gamma"])
            )  # alpha + beta + gamma
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return -2 * log_likelihood + 2 * n_params

    def BIC(self):
        """
        Calculate Bayesian Information Criterion (BIC).
        """
        if self.result is None:
            raise ValueError("Model must be fitted before calculating BIC.")

        if self.backend == "gam":
            return self.result["bic"]

        log_likelihood = self.result["log_likelihood"]
        n = len(self.theta)
        if self.model_type == "mean":
            n_params = len(self.result["beta"])  # Only beta
        elif self.model_type == "kappa":
            n_params = 1 + len(self.result["gamma"])  # alpha + gamma
        elif self.model_type == "mixed":
            n_params = (
                1 + len(self.result["beta"]) + len(self.result["gamma"])
            )  # alpha + beta + gamma
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return -2 * log_likelihood + n_params * np.log(n)

    def _gam_newdata(self, data):
        """Coerce gam-backend predict input to a DataFrame. A DataFrame passes
        through; a numpy array maps its columns to the formula predictor
        variables in order (mirroring the fisher-lee array interface), so
        ``predict(array)`` works on both backends."""
        if isinstance(data, pl.DataFrame) or type(data).__module__.startswith("pandas"):
            return _to_polars(data)
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        variables = self._gam_variables()
        if arr.shape[1] != len(variables):
            raise ValueError(
                f"expected {len(variables)} predictor column(s) {variables} for "
                f"the gam backend; got an array with {arr.shape[1]}."
            )
        return pl.DataFrame({v: arr[:, i] for i, v in enumerate(variables)})

    def _fl_design(self, X_new) -> np.ndarray:
        """Coerce fisher-lee predict input to the design array. An array is
        used as-is; a DataFrame's ``feature_names`` columns are extracted (in
        the model's order), so ``predict(df)`` works on both backends."""
        if isinstance(X_new, pl.DataFrame) or type(X_new).__module__.startswith("pandas"):
            dfn = _to_polars(X_new)
            missing = [c for c in self.feature_names if c not in dfn.columns]
            if missing:
                raise ValueError(
                    f"missing predictor column(s) {missing}; frame has {dfn.columns}."
                )
            arr = dfn.select(self.feature_names).to_numpy().astype(float)
        else:
            arr = np.asarray(X_new, dtype=float)
            if arr.ndim == 1:
                arr = arr[:, None]
        if not np.all(np.isfinite(arr)):
            raise ValueError("`X_new` contains non-finite values.")
        if arr.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"Expected {self.X.shape[1]} predictors, received {arr.shape[1]}."
            )
        return arr

    def predict(self, X_new):
        """
        Predict circular response values (fitted mean direction) for new
        predictor values.

        Parameters
        ----------
        X_new: array-like or DataFrame
            New predictor data. Both backends accept either a design array
            (columns = predictors in model order) or a DataFrame carrying the
            predictor columns.

        Returns
        -------
        theta_new: array-like, shape(n_samples, )
            New circular response values.
        """
        if self.result is None:
            raise ValueError("Model must be fitted before making predictions.")

        if self.backend == "gam":
            params = self.predict_params(X_new)
            fv = np.column_stack([params[nm] for nm in self.family.params])
            return np.mod(self.family._fitted_direction(fv), 2.0 * np.pi)

        X_arr = self._fl_design(X_new)
        mu = self.result["mu"]
        if self.model_type == "kappa":
            # Conditional mean is constant μ (β is not part of the model).
            return np.full(X_arr.shape[0], np.mod(mu, 2 * np.pi))

        beta = self.result.get("beta")
        if beta is None or np.any(~np.isfinite(beta)):
            raise ValueError("Model does not contain beta coefficients for prediction.")
        return np.mod(mu + self._mu_link.linkinv(X_arr @ beta), 2 * np.pi)

    def predict_kappa(self, X_new) -> np.ndarray:
        """Predict per-observation concentration κ_i = exp(α + X_iᵀγ).

        Only meaningful for ``model_type`` in ``{"kappa", "mixed"}``; for the
        ``"mean"`` model the concentration is a single scalar already in
        ``self.result["kappa"]``.

        Parameters
        ----------
        X_new : array-like, shape (n_samples, n_features) or (n_features,)
            New predictor data.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self.backend == "gam":
            conc = self.family.dist.params_by_role().get("concentration", [])
            if not conc:
                raise ValueError(
                    f"{self.family.name} has no 'concentration' parameter; "
                    "use predict_params() for the fitted parameters."
                )
            return self.predict_params(X_new)[conc[0]]

        if self.model_type == "mean":
            raise ValueError(
                "predict_kappa() is for model_type in {'kappa', 'mixed'}; "
                "the 'mean' model has a scalar κ in result['kappa']."
            )
        return self._predict_kappa(self._fl_design(X_new))

    def _predict_kappa(self, X_arr: np.ndarray) -> np.ndarray:
        """Internal: numpy-only κ̂(X) without input validation."""
        alpha = self.result["alpha"]
        gamma = self.result["gamma"]
        eta = alpha + X_arr @ gamma
        return np.exp(np.clip(eta, -50.0, 50.0))

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        n_curve: int = 200,
        ci: bool = True,
        pi: bool = True,
        level: float = 0.95,
        axes=None,
    ):
        """Two-panel diagnostic figure, laid out the same way for both
        backends (single shared predictor):

        - **Left — fit overlay:** data (replicated at θ and θ+2π) and the
          fitted μ̂(x) curve, with two optional bands — a ``level`` CI band
          ``μ̂ ± z·se`` (``ci``) and a ±1 circular-SD *dispersion* band from
          the fitted concentration (``pi``, ``circ-SD = √(−2 ln R̂(x))`` with
          ``R̂ = A1(κ̂)`` for von Mises, ``ρ̂`` for wrapped Cauchy). For the
          fisher-lee ``"kappa"`` model the curve is the constant μ line.
        - **Right — fitted concentration vs residuals:** the κ̂(x) curve when
          concentration depends on x (fisher-lee kappa/mixed; a gam κ LP with
          covariates), else residuals vs x (constant-κ models). The projected
          normal shows ‖μ̂(x)‖ as its concentration proxy.

        Multi-predictor fits drop the overlay for a residuals-vs-fitted /
        residual-histogram pair.

        The CI band is exact (joint covariance) for the gam backend and a
        μ₀ ⟂ β delta-method approximation for fisher-lee; the gam CI band is
        available only when the direction is a single fitted parameter (von
        Mises, wrapped Cauchy — not the projected normal's atan2 of two LPs).
        ``ci``/``pi``/``level`` apply to both backends.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if self.backend == "gam":
            return self._plot_gam(plt, figsize, n_curve, ci, pi, level, axes)

        n_features = self.X.shape[1]
        is_1d = n_features == 1

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize or (11, 5))
        else:
            axes = list(axes)
            if len(axes) != 2:
                raise ValueError("`axes` must be a sequence of length 2.")
            fig = axes[0].figure

        if not is_1d:
            self._plot_residual_diagnostic(axes)
            fig.tight_layout()
            return fig

        x_data = self.X[:, 0]
        theta_data = np.mod(self.theta, 2 * np.pi)
        feature_label = self.feature_names[0]

        x_grid = np.linspace(x_data.min(), x_data.max(), n_curve)
        x_grid_2d = x_grid[:, None]

        ax = axes[0]
        if self.model_type in ("mean", "mixed"):
            mu = self.result["mu"]
            beta = self.result["beta"]
            eta = x_grid * beta[0]
            curve = np.mod(mu + self._mu_link.linkinv(eta), 2 * np.pi)
            jumps = np.where(np.abs(np.diff(curve)) > np.pi)[0]
            # CI band on μ̂(X): delta method through the link, treating μ₀ and β
            # as independent (the fit estimates them in separate steps and stores
            # no cross-covariance — an approximation, unlike the gam backend's
            # full-covariance se). ∂μ̂/∂β = mu_eta(βX)·X. Dispersion band from κ̂.
            se_dir = None
            if self.result.get("se_beta") is not None and self.result.get("se_mu") is not None:
                g = self._mu_link.mu_eta(eta) * x_grid
                se_dir = np.sqrt(self.result["se_mu"] ** 2 + (g * self.result["se_beta"][0]) ** 2)
            kappa = self.result["kappa"] if self.model_type == "mean" else self._predict_kappa(x_grid_2d)
            circ_sd = np.sqrt(-2.0 * np.log(np.clip(A1(np.asarray(kappa, dtype=float)), 1e-12, 1.0)))
            self._fill_mu_bands(ax, x_grid, curve, jumps, ci=ci, pi=pi, level=level,
                                se_dir=se_dir, circ_sd=circ_sd)
            curve_plot = curve.astype(float).copy()
            curve_plot[jumps] = np.nan
            ax.plot(x_grid, curve_plot, color="C1", lw=2, label="fit")
            ax.plot(x_grid, curve_plot + 2 * np.pi, color="C1", lw=2)
        else:  # kappa-only: conditional mean is the constant μ.
            mu = self.result["mu"]
            curve = np.full_like(x_grid, np.mod(mu, 2 * np.pi))
            se_dir = (np.full_like(x_grid, self.result["se_mu"])
                      if self.result.get("se_mu") is not None else None)
            kappa = self._predict_kappa(x_grid_2d)
            circ_sd = np.sqrt(-2.0 * np.log(np.clip(A1(kappa), 1e-12, 1.0)))
            self._fill_mu_bands(ax, x_grid, curve, np.array([], dtype=int),
                                ci=ci, pi=pi, level=level, se_dir=se_dir, circ_sd=circ_sd)
            ax.axhline(mu, color="C1", lw=2, label=f"μ = {mu:.3f}")
            ax.axhline(mu + 2 * np.pi, color="C1", lw=2)

        ax.scatter(x_data, theta_data, color="C0", s=20, alpha=0.6, edgecolors="none", label="data")
        ax.scatter(x_data, theta_data + 2 * np.pi, color="C0", s=20, alpha=0.6, edgecolors="none")
        ax.set_ylim(0, 4 * np.pi)
        ax.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        ax.set_yticklabels(["0", "π", "2π", "3π", "4π"])
        ax.set_xlabel(feature_label)
        ax.set_ylabel("θ")
        ax.set_title("Fit overlay")
        ax.legend(loc="best", frameon=False)

        ax = axes[1]
        # Right panel = fitted concentration when κ depends on X (kappa AND
        # mixed both model κ(X) = exp(α + Xγ)); residuals when κ is constant
        # (mean). Same rule the gam overlay uses, so all models read alike.
        if self.model_type in ("kappa", "mixed"):
            kappa_curve = self._predict_kappa(x_grid_2d)
            ax.plot(x_grid, kappa_curve, color="C1", lw=2)
            ax.set_ylabel("κ̂(X) = exp(α + Xγ)")
            ax.set_title("Fitted concentration")
        else:
            residuals = np.angle(np.exp(1j * (self.theta - self._fitted_mean())))
            ax.scatter(x_data, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_ylabel("Residual (rad)")
            ax.set_title("Residuals vs X")
        ax.set_xlabel(feature_label)

        fig.tight_layout()
        return fig

    def _fitted_mean(self) -> np.ndarray:
        """Conditional mean angle at the training data (constant μ for
        kappa-only; per-observation fitted direction for the gam backend)."""
        if self.backend == "gam":
            return self.result["mu"]
        mu = self.result["mu"]
        if self.model_type == "kappa":
            return np.full(self.theta.shape, mu)
        beta = self.result["beta"]
        return mu + self._mu_link.linkinv(self.X @ beta)

    def _gam_variables(self) -> List[str]:
        """Data columns referenced by the gam formulas' RHSs, formula order."""
        cols = set(self.data.columns)
        seen: List[str] = []
        for f in self.formula:
            rhs = f.split("~", 1)[1] if "~" in f else f
            for m in re.finditer(r"[A-Za-z_]\w*", rhs):
                v = m.group(0)
                if v in cols and v not in seen:
                    seen.append(v)
        return seen

    def _fill_mu_bands(self, ax, x_grid, curve, jumps, *, ci, pi, level,
                       se_dir=None, circ_sd=None) -> None:
        """Draw ±band(s) around a (possibly wrapping) μ̂(x) curve on the
        ``[0, 4π]`` overlay, shared by the fisher-lee and gam backends: a ±1
        circular-SD *dispersion* band (``pi``, wide, from ``circ_sd``) and/or
        a ``level`` *confidence* band (``ci``, narrow, from ``se_dir``), each
        replicated at +2π and broken at the same wrap points as the curve.
        A band whose width array is ``None`` is skipped."""
        def _fill(delta, color, alpha, label):
            lo = (curve - delta).astype(float)
            hi = (curve + delta).astype(float)
            lo[jumps] = np.nan
            hi[jumps] = np.nan
            ax.fill_between(x_grid, lo, hi, color=color, alpha=alpha, lw=0, label=label)
            ax.fill_between(x_grid, lo + 2 * np.pi, hi + 2 * np.pi, color=color, alpha=alpha, lw=0)

        if pi and circ_sd is not None:
            _fill(np.broadcast_to(circ_sd, x_grid.shape), "C1", 0.12, "±1 circ-SD")
        if ci and se_dir is not None:
            z = float(norm.ppf(0.5 + level / 2.0))
            _fill(z * np.broadcast_to(se_dir, x_grid.shape), "C3", 0.25,
                  f"{int(round(level * 100))}% CI")

    def _gam_direction_se(self, grid_df) -> Optional[np.ndarray]:
        """Delta-method se of the fitted mean direction on the grid, in
        radians — hea's ``se.fit`` for the location LP. Returns ``None`` when
        the direction is not a single fitted parameter (e.g. projected
        normal, where it is ``atan2`` of two LPs and would need a joint delta
        method)."""
        loc = self.family.dist.params_by_role().get("location", [])
        if len(loc) != 1:
            return None
        li = self.family.params.index(loc[0])
        pred = self.gam_fit.predict(newdata=grid_df, se_fit=True)
        cols = pred.columns  # [fit_0..fit_{k-1}, se_0..se_{k-1}]
        return np.asarray(
            pred[cols[self.family.n_lp + li]].to_numpy(), dtype=float
        )

    def _gam_resultant_length(self, params) -> Optional[np.ndarray]:
        """Mean resultant length R̂(x) from the fitted concentration, for the
        ±1 circular-SD dispersion band. ``A1(κ̂)`` for a κ-concentration
        (von Mises), ``ρ̂`` itself for a ρ-concentration (wrapped Cauchy);
        ``None`` when there is no concentration role mapping known."""
        conc = self.family.dist.params_by_role().get("concentration", [])
        if not conc:
            return None
        name = conc[0]
        if name == "kappa":
            return A1(np.asarray(params[name], dtype=float))
        if name == "rho":
            return np.asarray(params[name], dtype=float)
        return None

    def _plot_gam(self, plt, figsize, n_curve, ci, pi, level, axes):
        """Two-panel figure for the distributional gam backend.

        One shared predictor: fit overlay (data + μ̂(x) curve, replicated at
        +2π) with an optional confidence band (``ci``) and ±1 circular-SD
        dispersion band (``pi``), and the fitted concentration curve — κ̂(x)
        for a concentration family, ‖μ̂(x)‖ for the projected normal.
        Multiple predictors: the residual diagnostic pair instead.
        """
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize or (11, 5))
        else:
            axes = list(axes)
            if len(axes) != 2:
                raise ValueError("`axes` must be a sequence of length 2.")
            fig = axes[0].figure

        variables = self._gam_variables()
        if len(variables) != 1:
            self._plot_residual_diagnostic(axes)
            fig.tight_layout()
            return fig

        var = variables[0]
        x_data = np.asarray(self.data[var].to_numpy(), dtype=float)
        theta_data = np.mod(self.theta, 2 * np.pi)
        x_grid = np.linspace(x_data.min(), x_data.max(), n_curve)
        grid_df = pl.DataFrame({var: x_grid})
        params = self.predict_params(grid_df)
        fv = np.column_stack([params[nm] for nm in self.family.params])
        curve = np.mod(self.family._fitted_direction(fv), 2 * np.pi)
        jumps = np.where(np.abs(np.diff(curve)) > np.pi)[0]

        ax = axes[0]
        # Bands first (under the curve and data). Dispersion (wide) below CI.
        R = self._gam_resultant_length(params) if pi else None
        circ_sd = None if R is None else np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0)))
        se_dir = self._gam_direction_se(grid_df) if ci else None
        self._fill_mu_bands(ax, x_grid, curve, jumps, ci=ci, pi=pi, level=level,
                            se_dir=se_dir, circ_sd=circ_sd)

        curve_plot = curve.astype(float).copy()
        curve_plot[jumps] = np.nan
        ax.plot(x_grid, curve_plot, color="C1", lw=2, label="fit")
        ax.plot(x_grid, curve_plot + 2 * np.pi, color="C1", lw=2)
        ax.scatter(x_data, theta_data, color="C0", s=20, alpha=0.6, edgecolors="none", label="data")
        ax.scatter(x_data, theta_data + 2 * np.pi, color="C0", s=20, alpha=0.6, edgecolors="none")
        ax.set_ylim(0, 4 * np.pi)
        ax.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        ax.set_yticklabels(["0", "π", "2π", "3π", "4π"])
        ax.set_xlabel(var)
        ax.set_ylabel("θ")
        ax.set_title(f"Fit overlay ({self.family.name})")
        ax.legend(loc="best", frameon=False)

        # Right panel = the fitted concentration when it depends on the
        # predictor; residuals when it is constant (an intercept-only κ LP).
        # κ̂(X) for a concentration family, ‖μ̂(X)‖ for the projected normal.
        # Same rule as the fisher-lee overlay, so all CL plots read alike.
        ax = axes[1]
        roles = self.family.dist.params_by_role()
        conc = roles.get("concentration", [])
        loc = roles.get("location", [])
        if conc:
            proxy, label, title = params[conc[0]], f"{conc[0]}̂({var})", "Fitted concentration"
        elif len(loc) == 2:
            proxy = np.hypot(params[loc[0]], params[loc[1]])
            label, title = f"‖μ̂‖({var})", "Fitted resultant length (concentration proxy)"
        else:
            proxy = None
        varies = proxy is not None and np.ptp(proxy) > 1e-8 * max(1.0, float(np.mean(np.abs(proxy))))
        if varies:
            ax.plot(x_grid, proxy, color="C1", lw=2)
            ax.set_ylabel(label)
            ax.set_title(title)
        else:
            residuals = np.angle(np.exp(1j * (self.theta - self._fitted_mean())))
            ax.scatter(x_data, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_ylabel("Residual (rad)")
            ax.set_title("Residuals vs " + var)
        ax.set_xlabel(var)

        fig.tight_layout()
        return fig

    def _plot_residual_diagnostic(self, axes) -> None:
        residuals = np.angle(np.exp(1j * (self.theta - self._fitted_mean())))
        fitted = np.mod(self._fitted_mean(), 2 * np.pi)
        ax = axes[0]
        ax.scatter(fitted, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel("Fitted θ (rad)")
        ax.set_ylabel("Residual (rad)")
        ax.set_title("Residuals vs fitted")

        ax = axes[1]
        ax.hist(residuals, bins=20, color="C0", alpha=0.7, edgecolor="black")
        ax.axvline(0.0, color="k", lw=0.5)
        ax.set_xlabel("Residual (rad)")
        ax.set_ylabel("Count")
        ax.set_title("Residual histogram")

    @staticmethod
    def _two_sided_p(t_value: float) -> float:
        if np.isnan(t_value):
            return np.nan
        return float(2.0 * norm.sf(np.abs(t_value)))

    def summary(self):
        if self.result is None:
            raise ValueError("Model must be fitted before summarizing.")

        if self.backend == "gam":
            # hea.gam.summary() prints the mgcv-style block (parametric
            # coefficients + approximate smooth significance); prepend the
            # LP ↔ parameter mapping and append the fitted-parameter ranges.
            print("\nCircular-Linear Regression — distributional gam backend")
            print(f"Family: {self.family!r}")
            for f, nm in zip(self.formula, self.family.params):
                print(f"  LP[{nm}]: {f.strip()}")
            print()
            self.gam_fit.summary()
            fv = self.result["fitted_params"]
            print("\nFitted parameters (per observation):")
            for j, nm in enumerate(self.family.params):
                col = fv[:, j]
                print(
                    f"  {nm:<6s} min {np.min(col): .4f}   "
                    f"median {np.median(col): .4f}   max {np.max(col): .4f}"
                )
            return

        # Title based on model type
        if self.model_type == "mean":
            print("\nCircular Regression for the Mean Direction\n")
        elif self.model_type == "kappa":
            print("\nCircular Regression for the Concentration Parameter\n")
        elif self.model_type == "mixed":
            print("\nMixed Circular-Linear Regression\n")

        # Call
        print("Call:")
        print(f"  CLRegression(model_type='{self.model_type}')\n")

        # Coefficients for mean direction (Beta)
        se_beta = self.result.get("se_beta")
        if (
            self.model_type in ["mean", "mixed"]
            and self.result.get("beta") is not None
            and se_beta is not None
        ):
            print("Coefficients for Mean Direction (Beta):\n")
            print(
                f"{'':<5} {'Estimate':<12} {'Std. Error':<12} {'t value':<10} {'Pr(>|t|)'}"
            )
            for i, coef in enumerate(self.result["beta"]):
                se_val = se_beta[i]
                t_value = coef / se_val if se_val else np.nan
                p_value = self._two_sided_p(t_value)
                print(
                    f"β{i:<3} {coef:<12.5f} {se_val:<12.5f} {t_value:<10.2f} {p_value:<12.5f}{significance_code(p_value):<3}"
                )

        # Coefficients for concentration parameter (Gamma)
        se_gamma = self.result.get("se_gamma")
        se_alpha = self.result.get("se_alpha")
        if (
            self.model_type in ["kappa", "mixed"]
            and self.result.get("gamma") is not None
            and se_gamma is not None
            and se_alpha is not None
        ):
            print("\nCoefficients for Concentration (Gamma):\n")
            print(
                f"{'':<5} {'Estimate':<12} {'Std. Error':<12} {'t value':<10} {'Pr(>|t|)':<12}"
            )
            # Report alpha as the first coefficient
            alpha = self.result["alpha"]
            t_value_alpha = alpha / se_alpha if se_alpha else np.nan
            p_value_alpha = self._two_sided_p(t_value_alpha)
            print(
                f"α{'':<5} {alpha:<12.5f} {se_alpha:<12.5f} {t_value_alpha:<10.2f} {p_value_alpha:<12.5f}{significance_code(p_value_alpha)}"
            )
            for i, coef in enumerate(self.result["gamma"]):
                se_val = se_gamma[i]
                t_value = coef / se_val if se_val else np.nan
                p_value = self._two_sided_p(t_value)
                print(
                    f"γ{i:<5} {coef:<12.5f} {se_val:<12.5f} {t_value:<10.2f} {p_value:<12.5f}{significance_code(p_value)}"
                )

        # Summary for mu and kappa
        print("\nSummary:")
        print("  Mean Direction (mu) in radians:")
        mu = self.result["mu"]
        se_mu = self.result.get("se_mu")
        if se_mu is not None:
            print(f"    μ: {mu:.5f} (SE: {se_mu:.5f})")
        else:
            print(f"    μ: {mu:.5f}")

        print("\n  Concentration Parameter (kappa):")
        kappa = self.result["kappa"]
        se_kappa = self.result.get("se_kappa")
        if isinstance(kappa, np.ndarray):
            print("    Index    kappa        Std. Error")
            for i, k in enumerate(kappa, start=1):
                se_val = se_kappa[i - 1] if se_kappa is not None else float("nan")
                print(f"    [{i}]    {k:>10.5f}    {se_val:>10.5f}")
            # Per-obs κ_i are correlated (shared α, γ), so averaging individual
            # SEs is not the SE of the mean — report only the point estimate.
            print(f"    Mean:    {np.mean(kappa):.5f}")
        else:
            if se_kappa is not None:
                print(f"    κ: {kappa:.5f} (SE: {se_kappa:.5f})")
            else:
                print(f"    κ: {kappa:.5f}")

        # Summary for model fit metrics
        print("\nModel Fit Metrics:\n")
        print(f"{'Metric':<12} {'Value':<12}")
        log_likelihood = self.result.get("log_likelihood", float("nan"))
        nll = -log_likelihood  # Negative log-likelihood
        print(f"{'nLL':<12} {nll:<12.5f}")
        print(f"{'AIC':<12} {self.AIC():<12.5f}")
        print(f"{'BIC':<12} {self.BIC():<12.5f}")

        # Notes
        print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        print("p-values are approximated using the normal distribution.\n")


class CCRegression:
    """
    Circular-Circular Regression.

    Fits a circular response to circular predictors. A harmonic formula
    (``"theta ~ psi"``, ``order=K``) uses two ``hea.lm`` fits; a **smooth**
    formula (``"theta ~ s(psi, bs='cc')"``) dispatches to two ``hea.gam`` fits
    on the cos/sin embedding — the nonparametric counterpart.

    Parameters
    ----------
    theta : np.ndarray
        A numpy array of circular response values in radians.
    x : np.ndarray
        A numpy array of circular predictor values in radians.
    order : int, optional
        Order of harmonics to include in the model (default is 1).
        Parametric backend only.
    level : float, optional
        Significance level for testing higher-order terms (default is 0.05).
    knots : dict, optional
        Smooth-backend only. Per-variable boundary knots for ``hea.gam``;
        cyclic smooths (``bs='cc'``/``'cp'``) default to the period
        ``[0, 2π]`` (override via this argument).
    method : str, optional
        Smooth-backend only. Smoothing-parameter selection for ``hea.gam``
        (default ``"REML"``).

    Attributes
    ----------
    rho : float
        Circular correlation coefficient.
    fitted : np.ndarray
        Fitted values of the circular response in radians.
    residuals : np.ndarray
        Residuals of the circular response in radians.
    coefficients : dict
        Coefficients of the cos and sin terms for each harmonic order.
    p_values : np.ndarray
        P-values for higher-order terms.
    kappa : float
        Concentration of the residuals, A1⁻¹(mean cos(residuals)).
    A_k : float
        Mean cosine of the residuals (input to A1⁻¹).
    message : str
        Message indicating the significance of higher-order terms.

    Methods
    -------
    summary()
        Print the harmonic coefficient table, ρ, residual κ, and the test
        of higher-order terms.
    predict(x)
        Predict the circular response at new ``x``.
    plot(figsize=None, n_curve=200, axes=None, band=True, polar=False)
        Two-panel diagnostic figure (fit overlay ± circular-SD band, or a
        polar fit clock, for 1-D ``x``; residuals vs fitted + histogram
        for multi-D).

    Notes
    -----
    The implementation is ported from the ``lm.circular.cc`` in the
    ``circular`` R package (Agostinelli & Lund).

    References
    ----------
    - Jammalamadaka, S. R., & Sengupta, A. (2001) Topics in Circular Statistics. World Scientific.
    - Pewsey, A., Neuhäuser, M., & Ruxton, G. D. (2014) Circular Statistics in R. Oxford University Press.
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pl.DataFrame] = None,
        theta: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        order: int = 1,
        level: float = 0.05,
        *,
        knots: Optional[dict] = None,
        method: str = "REML",
        **gam_kwargs,
    ):
        self.formula = formula
        # Backend dispatch: a smooth term (s()/te()/…) in the formula → two
        # cyclic-smooth gam fits on the cos/sin embedding; otherwise the
        # parametric harmonic path (two hea.lm fits). Same embedding + circular
        # reassembly either way — gam is a basis swap, not a new model.
        self.backend = "gam" if (formula and _has_smooth(formula)) else "lm"
        self._knots = knots
        self._method = method
        self._gam_kwargs = gam_kwargs

        if self.backend == "gam":
            if data is None:
                raise ValueError(
                    "The smooth (gam) backend requires a formula and data."
                )
            self.data = _to_polars(data)
            self._gam_response = formula.split("~", 1)[0].strip()
            self._gam_rhs = formula.split("~", 1)[1].strip()
            self.feature_names = _smooth_vars(self._gam_rhs)
            if not self.feature_names:
                raise ValueError(
                    f"No smooth predictor found in formula: {formula!r}"
                )
            # Default cyclic-smooth knots to the circular period [0, 2π].
            self._knots = _resolve_cyclic_knots(self._gam_rhs, knots)
            theta_arr = self.data[self._gam_response].to_numpy()
            x_arr = self.data[self.feature_names].to_numpy()
            self.theta = self._validate_input(theta_arr)
            self.x = self._validate_input(x_arr)
            if self.x.ndim == 1:
                self.x = self.x[:, None]
        elif formula and data is not None:
            if knots is not None or gam_kwargs:
                raise ValueError(
                    "knots=/gam options only apply to smooth formulas "
                    "(s()/te()/…); this harmonic formula uses hea.lm."
                )
            theta_arr, x_arr, self.feature_names = self._parse_formula(formula, data)
            self.theta = self._validate_input(theta_arr)
            self.x = self._validate_input(x_arr)
            if self.x.ndim == 1:
                self.x = self.x[:, None]
        elif theta is not None and x is not None:
            if knots is not None or gam_kwargs:
                raise ValueError(
                    "knots=/gam options require a formula with a smooth term."
                )
            self.theta = self._validate_input(theta)
            self.x = self._validate_input(x)
            if self.x.ndim == 1:
                self.x = self.x[:, None]
            self.feature_names = [f"x{i}" for i in range(self.x.shape[1])]
        else:
            raise ValueError("Provide either a formula + data or theta and x.")

        if self.theta.ndim != 1:
            raise ValueError(
                f"`theta` must be 1-dimensional (got shape {self.theta.shape})."
            )
        if self.theta.size != self.x.shape[0]:
            raise ValueError("`theta` and `x` must have matching numbers of rows.")

        self.order = order
        self.level = level

        if self.order < 1:
            raise ValueError("`order` must be a positive integer.")
        if not (0 < self.level < 1):
            raise ValueError("`level` must lie between 0 and 1.")

        # The harmonic order / observation-count check is specific to the
        # parametric backend; the smooth backend's complexity is set by edf,
        # not by `order`.
        if self.backend == "lm":
            n_params = 1 + 2 * self.x.shape[1] * self.order
            if self.theta.size <= n_params:
                raise ValueError(
                    f"order={self.order} requires more than {n_params} observations "
                    f"(got {self.theta.size}); reduce `order` or provide more data."
                )

        # Fit the model
        self.result = self._fit()

    @staticmethod
    def _validate_input(arr: np.ndarray) -> np.ndarray:
        """Validate angular input and wrap to ``[0, 2π)``.

        The model is 2π-periodic, so values are normalised modulo ``2π``.
        Input is expected to be in radians; degrees would silently wrap to
        the wrong range (e.g. 360° → 360 mod 2π ≈ 5.97 rad ≈ 342°).
        """
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.ndim == 0:
            raise ValueError("Input must be at least one-dimensional.")
        if not np.all(np.isfinite(arr_np)):
            raise ValueError("Circular input contains non-finite values.")
        if arr_np.size and float(np.max(np.abs(arr_np))) > 4 * np.pi:
            warnings.warn(
                "Circular input contains values with |x| > 4π; expected "
                "radians. Degree-valued input will be silently wrapped "
                "modulo 2π and produce incorrect results — use np.deg2rad.",
                UserWarning,
                stacklevel=3,
            )
        return np.mod(arr_np, 2 * np.pi)

    def _parse_formula(
        self, formula: str, data: pl.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        data = _to_polars(data)
        parts = formula.split("~")
        if len(parts) != 2:
            raise ValueError(
                f"Formula must contain exactly one '~'; got: {formula!r}"
            )
        theta_col, x_cols = parts
        theta = data[theta_col.strip()].to_numpy()
        x_cols = [col.strip() for col in x_cols.split("+") if col.strip()]
        if not x_cols:
            raise ValueError(f"No predictors found in formula: {formula!r}")
        X = data[x_cols].to_numpy()
        return theta, X, x_cols

    def _fit(self):
        if self.backend == "gam":
            return self._fit_gam()
        n = self.x.shape[0]
        order = self.order
        n_features = self.x.shape[1]
        feats = list(self.feature_names)

        # (feature, harmonic) labels for each cos/sin block column.
        cos_labels: List[Tuple[int, int]] = [
            (j, k) for j in range(n_features) for k in range(1, order + 1)
        ]
        sin_labels = list(cos_labels)

        Y_cos = np.cos(self.theta)
        Y_sin = np.sin(self.theta)

        # Two OLS fits via hea.lm on a harmonic design (period=2π — x is angular):
        # the cos/sin embedding of the circular response. β/SE are reordered from
        # hea's interleaved columns into the legacy [intercept | cos-block |
        # sin-block] order via the cos{j}/sin{j} suffixes. The circular reassembly
        # (μ̂, ρ, residual κ̂, higher-order test) stays here in pycircstat2.
        df_fit = pl.DataFrame(
            {**{f: self.x[:, i] for i, f in enumerate(feats)},
             "cos_t": Y_cos, "sin_t": Y_sin}
        )
        rhs = " + ".join(
            f"harmonic({f}, k={order}, period={2 * np.pi})" for f in feats
        )
        self._lm_cos = _hea_lm(f"cos_t ~ {rhs}", df_fit)
        self._lm_sin = _hea_lm(f"sin_t ~ {rhs}", df_fit)
        self._feature_cols = feats

        beta_cos = _harmonic_block_order(
            dict(zip(self._lm_cos.bhat.columns, self._lm_cos.bhat.row(0))), feats, order
        )
        beta_sin = _harmonic_block_order(
            dict(zip(self._lm_sin.bhat.columns, self._lm_sin.bhat.row(0))), feats, order
        )

        cos_fit = _ravel(self._lm_cos.yhat)
        sin_fit = _ravel(self._lm_sin.yhat)
        fitted = np.mod(np.arctan2(sin_fit, cos_fit), 2 * np.pi)

        # Residuals (angular for diagnostics + raw OLS residuals on cos/sin)
        residuals = np.angle(np.exp(1j * (self.theta - fitted)))
        residual_cos = _ravel(self._lm_cos.residuals)
        residual_sin = _ravel(self._lm_sin.residuals)

        # Circular correlation coefficient
        rho = float(np.clip(np.sqrt(np.mean(cos_fit**2 + sin_fit**2)), 0.0, 1.0))

        # Per-coefficient SEs straight from each lm's covariance (V_bhat =
        # σ̂²(XᵀX)⁻¹), reordered to the same block order as the coefficients.
        def _se_by_name(lm_fit):
            cov = np.asarray(lm_fit.V_bhat, dtype=float)
            return {nm: float(np.sqrt(max(cov[i, i], 0.0)))
                    for i, nm in enumerate(lm_fit.column_names)}

        se_beta_cos = _harmonic_block_order(_se_by_name(self._lm_cos), feats, order)
        se_beta_sin = _harmonic_block_order(_se_by_name(self._lm_sin), feats, order)
        df_resid = max(n - (1 + 2 * n_features * order), 1)

        # Higher-order test (Jammalamadaka & Sengupta 2001): do the order+1
        # harmonics add signal? Uses the fitted design's hat matrix (projection
        # is column-order-invariant, so hea's interleaved X is fine) + residuals.
        higher_order_cos = []
        higher_order_sin = []
        for j in range(n_features):
            x_col = self.x[:, j]
            higher_order_cos.append(np.cos((order + 1) * x_col))
            higher_order_sin.append(np.sin((order + 1) * x_col))
        if higher_order_cos:
            W = np.column_stack(higher_order_cos + higher_order_sin)
        else:
            W = np.empty((n, 0))

        if W.size:
            X = np.asarray(self._lm_cos.X.to_numpy(), dtype=float)
            M = X @ _safe_inverse(X.T @ X) @ X.T
            N = W @ _safe_inverse(W.T @ (np.eye(n) - M) @ W) @ W.T

            denom_cos = float(residual_cos @ residual_cos)
            denom_sin = float(residual_sin @ residual_sin)
            adj = max(n - (2 * order + 1), 1)
            T1 = (
                adj
                * float(residual_cos @ N @ residual_cos)
                / max(denom_cos, 1e-12)
            )
            T2 = (
                adj
                * float(residual_sin @ N @ residual_sin)
                / max(denom_sin, 1e-12)
            )

            p1 = 1 - chi2.cdf(T1, W.shape[1])
            p2 = 1 - chi2.cdf(T2, W.shape[1])
            p_values = np.array([p1, p2], dtype=float)
        else:
            p_values = np.array([np.nan, np.nan], dtype=float)

        # Message about higher-order terms
        if np.all(np.isnan(p_values)):
            message = "No additional harmonics available for testing."
        elif np.all(p_values > self.level):
            message = (
                f"Higher-order terms are not significant at the {self.level} level."
            )
        else:
            message = f"Higher-order terms are significant at the {self.level} level."

        # Residual concentration (R parity): A1inv of the mean cosine of residuals.
        A_k = float(np.mean(np.cos(residuals)))
        if A_k < 0:
            warnings.warn(
                f"Mean residual cosine A_k={A_k:.4f} is negative — residuals "
                "are systematically anti-aligned with the fitted direction. "
                "κ has been clamped to 0; check for sign errors or model "
                "misspecification.",
                UserWarning,
                stacklevel=3,
            )
        kappa_residual = float(A1inv(A_k))

        return {
            "rho": rho,
            "fitted": fitted,
            "residuals": residuals,
            "coefficients": {
                "cos": beta_cos,
                "sin": beta_sin,
            },
            "se_coefficients": {
                "cos": se_beta_cos,
                "sin": se_beta_sin,
            },
            "df_resid": df_resid,
            "cos_labels": cos_labels,
            "sin_labels": sin_labels,
            "p_values": p_values,
            "A_k": A_k,
            "kappa": kappa_residual,
            "message": message,
        }

    def _fit_gam(self) -> dict:
        """Smooth (gam) backend: two cyclic-smooth Gaussian gam fits on the
        cos/sin embedding of the circular response, reassembled into
        ``μ̂ = arctan2(sin_fit, cos_fit)`` with a residual concentration κ̂.

        Identical embedding to the parametric path, with penalized smooths
        instead of harmonics; ``knots``/``method``/options pass straight to
        ``hea.gam``. The higher-order harmonic χ² test does not apply (smooth
        complexity is selected by REML/GCV), so ``p_values`` are NaN.
        """
        Y_cos = np.cos(self.theta)
        Y_sin = np.sin(self.theta)
        df_fit = self.data.with_columns(
            pl.Series("cos_t", Y_cos), pl.Series("sin_t", Y_sin)
        )
        self._gam_cos = _hea_gam(
            f"cos_t ~ {self._gam_rhs}", df_fit,
            knots=self._knots, method=self._method, **self._gam_kwargs,
        )
        self._gam_sin = _hea_gam(
            f"sin_t ~ {self._gam_rhs}", df_fit,
            knots=self._knots, method=self._method, **self._gam_kwargs,
        )

        cos_fit = np.asarray(self._gam_cos.fitted_values, dtype=float)
        sin_fit = np.asarray(self._gam_sin.fitted_values, dtype=float)
        fitted = np.mod(np.arctan2(sin_fit, cos_fit), 2 * np.pi)
        residuals = np.angle(np.exp(1j * (self.theta - fitted)))

        rho = float(np.clip(np.sqrt(np.mean(cos_fit**2 + sin_fit**2)), 0.0, 1.0))

        A_k = float(np.mean(np.cos(residuals)))
        if A_k < 0:
            warnings.warn(
                f"Mean residual cosine A_k={A_k:.4f} is negative — residuals "
                "are systematically anti-aligned with the fitted direction. "
                "κ has been clamped to 0; check for sign errors or model "
                "misspecification.",
                UserWarning,
                stacklevel=3,
            )
        kappa_residual = float(A1inv(A_k))

        return {
            "rho": rho,
            "fitted": fitted,
            "residuals": residuals,
            "coefficients": None,
            "se_coefficients": None,
            "edf_total": {
                "cos": float(self._gam_cos.edf_total),
                "sin": float(self._gam_sin.edf_total),
            },
            "edf_by_smooth": {
                "cos": {k: float(v) for k, v in dict(self._gam_cos.edf_by_smooth).items()},
                "sin": {k: float(v) for k, v in dict(self._gam_sin.edf_by_smooth).items()},
            },
            "deviance_explained": {
                "cos": float(self._gam_cos.deviance_explained),
                "sin": float(self._gam_sin.deviance_explained),
            },
            "p_values": np.array([np.nan, np.nan], dtype=float),
            "A_k": A_k,
            "kappa": kappa_residual,
            "message": (
                "Smooth (gam) backend: the higher-order harmonic test does not "
                "apply; smoothness is selected by REML/GCV per coordinate."
            ),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the circular response at new predictor values.

        Parameters
        ----------
        x : array-like, shape (n,) or (n, n_features)
            New predictor values in radians. For multi-feature models the
            second axis must match ``self.x.shape[1]``.

        Returns
        -------
        np.ndarray, shape (n,)
            Predicted angles wrapped to ``[0, 2π)``.
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr[:, None]
        if x_arr.shape[1] != self.x.shape[1]:
            raise ValueError(
                f"Expected {self.x.shape[1]} predictor column(s); received "
                f"{x_arr.shape[1]}."
            )
        x_arr = np.mod(x_arr, 2 * np.pi)
        cos_pred, sin_pred = self._predict_embedding(x_arr)
        return np.mod(np.arctan2(sin_pred, cos_pred), 2 * np.pi)

    def _predict_embedding(
        self, x_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicted cos/sin embedding components at predictor rows.

        The reassembled direction is ``arctan2(sin, cos)``; the vector
        length ``hypot(cos, sin)`` estimates the conditional resultant
        ``ρ̂(x)`` — the model's conditional concentration (Jammalamadaka &
        Sengupta 2001), which ``arctan2`` discards.
        """
        if self.backend == "gam":
            newdata = pl.DataFrame(
                {f: x_arr[:, i] for i, f in enumerate(self.feature_names)}
            )
            cos_pred = self._gam_cos.predict(newdata=newdata)["fit"].to_numpy()
            sin_pred = self._gam_sin.predict(newdata=newdata)["fit"].to_numpy()
            return (
                np.asarray(cos_pred, dtype=float),
                np.asarray(sin_pred, dtype=float),
            )
        newdata = pl.DataFrame(
            {f: x_arr[:, i] for i, f in enumerate(self._feature_cols)}
        )
        return (
            _ravel(self._lm_cos.predict(newdata=newdata)),
            _ravel(self._lm_sin.predict(newdata=newdata)),
        )

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        n_curve: int = 200,
        axes=None,
        *,
        band: bool = True,
        polar: bool = False,
    ):
        """Two-panel diagnostic figure.

        For a single circular predictor, the left panel is a fit overlay
        with both data and curve replicated at ``θ`` and ``θ + 2π`` (Pewsey
        Fig 6.10 convention) so the wrap-around does not visually break the
        relationship; the right panel shows the wrapped residuals against
        the predictor.

        The fitted direction ``μ̂(x) = arctan2(ŝ, ĉ)`` discards the length
        of the fitted (cos, sin) vector, so by default the overlay also
        shades ``μ̂ ± 1`` circular standard deviation ``√(−2 ln ρ̂(x))``
        implied by the conditional resultant ``ρ̂(x) = ‖(ĉ, ŝ)‖``. This is
        a model-implied *spread* band, not a confidence band for ``μ̂``: it
        widens (clipped at π) where ``ρ̂ → 0`` and the direction becomes
        ill-defined, and it is what visually separates the lm and gam
        backends — their ``μ̂`` curves often almost coincide while their
        shrunken amplitudes differ.

        For multiple circular predictors, the left panel shows residuals
        vs the fitted angle and the right panel a residual histogram.

        Parameters
        ----------
        figsize : tuple, optional
            Matplotlib figure size; defaults to ``(11, 5)``.
        n_curve : int
            Number of grid points used to draw the fitted curve.
        axes : sequence of matplotlib Axes, optional
            Two pre-existing axes to draw into (the caller owns their
            projections). If omitted, a fresh figure is created.
        band : bool
            Shade the ``μ̂ ± 1`` circular-SD band on the cartesian overlay
            (single predictor only). Default True.
        polar : bool
            Single predictor only: replace the cartesian overlay with a
            compass clock — each rim position is a predictor angle ``x``
            with an arrow pointing in the fitted direction ``μ̂(x)`` and
            length ``ρ̂(x)``, so a vanishing arrow marks an ill-defined
            direction. Data are fixed-length arrows just outside the rim.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        n_features = self.x.shape[1]
        if polar and n_features != 1:
            raise ValueError("polar=True requires a single circular predictor.")

        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize or (11, 5))
        else:
            axes = list(axes)
            if len(axes) != 2:
                raise ValueError("`axes` must be a sequence of length 2.")
            fig = axes[0].figure

        if n_features == 1:
            x_data = self.x[:, 0]
            theta_data = self.theta
            residuals = self.result["residuals"]

            if polar:
                self._draw_polar_clock(axes[0], x_data, theta_data)
            else:
                x_grid = np.linspace(0.0, 2 * np.pi, n_curve)
                cos_fit, sin_fit = self._predict_embedding(x_grid[:, None])
                # Unwrap the fitted direction into a continuous curve (the
                # fit is smooth on the circle; only the flat chart has a
                # 0/2π seam), then draw it at every 2π offset that crosses
                # the doubled [0, 4π] window — branches run off the panel
                # edges instead of breaking mid-panel.
                theta_curve = np.unwrap(np.arctan2(sin_fit, cos_fit))
                # Circular SD implied by the conditional resultant; π
                # ("any direction") once ρ̂ falls below exp(−π²/2).
                rho_grid = np.clip(np.hypot(cos_fit, sin_fit), 1e-9, 1.0)
                sd_grid = np.minimum(np.sqrt(-2.0 * np.log(rho_grid)), np.pi)

                ax = axes[0]
                two_pi = 2 * np.pi
                lo = float(np.min(theta_curve - (sd_grid if band else 0.0)))
                hi = float(np.max(theta_curve + (sd_grid if band else 0.0)))
                k_min = int(np.ceil((0.0 - hi) / two_pi))
                k_max = int(np.floor((4 * np.pi - lo) / two_pi))
                for k in range(k_min, k_max + 1):
                    offset = k * two_pi
                    if band:
                        ax.fill_between(
                            x_grid,
                            theta_curve - sd_grid + offset,
                            theta_curve + sd_grid + offset,
                            color="C1",
                            alpha=0.18,
                            linewidth=0,
                            label="μ̂ ± 1 circular SD" if k == k_min else None,
                        )
                    ax.plot(
                        x_grid,
                        theta_curve + offset,
                        color="C1",
                        lw=2,
                        label="fit" if k == k_min else None,
                    )
                ax.scatter(x_data, theta_data, color="C0", s=20, alpha=0.6, edgecolors="none", label="data")
                ax.scatter(x_data, theta_data + 2 * np.pi, color="C0", s=20, alpha=0.6, edgecolors="none")
                ax.set_xlim(0, 2 * np.pi)
                ax.set_ylim(0, 4 * np.pi)
                ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
                ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
                ax.set_yticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
                ax.set_yticklabels(["0", "π", "2π", "3π", "4π"])
                ax.set_xlabel(self.feature_names[0])
                ax.set_ylabel("θ")
                ax.set_title("Fit overlay")
                ax.legend(loc="best", frameon=False)

            ax = axes[1]
            ax.scatter(x_data, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
            ax.set_xlabel(self.feature_names[0])
            ax.set_ylabel("Residual (rad)")
            ax.set_title("Residuals vs predictor")
        else:
            residuals = self.result["residuals"]
            fitted = self.result["fitted"]

            ax = axes[0]
            ax.scatter(fitted, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
            ax.axhline(0.0, color="k", lw=0.5)
            ax.set_xlabel("Fitted θ (rad)")
            ax.set_ylabel("Residual (rad)")
            ax.set_title("Residuals vs fitted")

            ax = axes[1]
            ax.hist(residuals, bins=20, color="C0", alpha=0.7, edgecolor="black")
            ax.axvline(0.0, color="k", lw=0.5)
            ax.set_xlabel("Residual (rad)")
            ax.set_ylabel("Count")
            ax.set_title("Residual histogram")

        fig.tight_layout()
        return fig

    def _draw_polar_clock(
        self,
        ax,
        x_data: np.ndarray,
        theta_data: np.ndarray,
        n_arrow: int = 36,
    ) -> None:
        """Compass-frame fit panel for a single circular predictor.

        Each rim position is a predictor angle ``x``; the arrow anchored
        there points in the fitted direction ``μ̂(x)`` with length
        ``ρ̂(x)``, so a vanishing arrow marks an ill-defined direction.
        Observations are fixed-length arrows just outside the rim at their
        predictor angle, pointing in their observed direction.
        """
        grid = np.linspace(0.0, 2 * np.pi, n_arrow, endpoint=False)
        cos_fit, sin_fit = self._predict_embedding(grid[:, None])
        mu = np.arctan2(sin_fit, cos_fit)
        rho = np.clip(np.hypot(cos_fit, sin_fit), 0.0, 1.0)

        rim = np.linspace(0.0, 2 * np.pi, 256)
        ax.plot(np.cos(rim), np.sin(rim), color="0.85", lw=1.0, zorder=1)

        fit_scale = 0.5  # display length of a ρ̂ = 1 arrow
        ax.quiver(
            np.cos(grid), np.sin(grid),
            fit_scale * rho * np.cos(mu), fit_scale * rho * np.sin(mu),
            angles="xy", scale_units="xy", scale=1.0,
            color="C1", width=0.008, zorder=3,
            label="fit: μ̂(x), length ρ̂(x)",
        )
        data_anchor, data_len = 1.12, 0.18
        ax.quiver(
            data_anchor * np.cos(x_data), data_anchor * np.sin(x_data),
            data_len * np.cos(theta_data), data_len * np.sin(theta_data),
            angles="xy", scale_units="xy", scale=1.0,
            color="C0", alpha=0.6, width=0.005, zorder=2,
            label="data: θ at rim x",
        )
        for ang, lab in ((0.0, "0"), (np.pi / 2, "π/2"), (np.pi, "π"), (3 * np.pi / 2, "3π/2")):
            ax.text(
                1.45 * np.cos(ang), 1.45 * np.sin(ang), lab,
                ha="center", va="center", color="0.4",
            )
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_axis_off()
        ax.set_title("Fit clock: arrow = μ̂(x), length = ρ̂(x)")
        ax.legend(loc="lower left", frameon=False, fontsize=8)

    def summary(self):
        """
        Print a summary of the regression results.
        """
        if self.backend == "gam":
            return self._summary_gam()
        print("\nCircular-Circular Regression\n")
        print(f"Circular Correlation Coefficient (rho): {self.result['rho']:.5f}")
        print(f"Mean Residual Cosine (A_k):             {self.result['A_k']:.5f}")
        print(f"Residual Concentration (kappa):         {self.result['kappa']:.5f}\n")

        cos_coeffs = self.result["coefficients"]["cos"]
        sin_coeffs = self.result["coefficients"]["sin"]
        se_cos = self.result["se_coefficients"]["cos"]
        se_sin = self.result["se_coefficients"]["sin"]
        df_resid = self.result["df_resid"]
        cos_labels = self.result.get("cos_labels", [])
        sin_labels = self.result.get("sin_labels", [])

        intercept_label = "(Intercept)"
        cos_label_strs = [f"cos(x{f + 1},k={k})" for (f, k) in cos_labels]
        sin_label_strs = [f"sin(x{f + 1},k={k})" for (f, k) in sin_labels]
        row_labels = [intercept_label, *cos_label_strs, *sin_label_strs]
        label_width = max(12, *(len(s) for s in row_labels))

        def _print_block(title: str, coefs: np.ndarray, ses: np.ndarray) -> None:
            print(f"{title}:\n")
            print(
                f"{'':<{label_width}} {'Estimate':<12} {'Std. Error':<12} "
                f"{'t value':<10} {'Pr(>|t|)':<12}"
            )
            for label, coef, se_val in zip(row_labels, coefs, ses):
                t_val = coef / se_val if se_val else np.nan
                if np.isnan(t_val):
                    p_val = np.nan
                else:
                    p_val = float(2.0 * student_t.sf(np.abs(t_val), df=df_resid))
                print(
                    f"{label:<{label_width}} {coef:<12.5f} {se_val:<12.5f} "
                    f"{t_val:<10.2f} {p_val:<12.5f}{significance_code(p_val)}"
                )
            print()

        _print_block("Coefficients (Cosine Model)", cos_coeffs, se_cos)
        _print_block("Coefficients (Sine Model)", sin_coeffs, se_sin)

        # Higher-order test (parity with R's lm.circular.cc): jointly tests
        # whether the order+1 cos/sin pair adds explanatory power, separately
        # for the cosine and sine sub-models.
        p1, p2 = self.result["p_values"]
        print("Higher-Order Terms Test:\n")
        print(f"{'':<{label_width}} {'Pr(>χ²)':<12}")
        print(f"{'cosine model':<{label_width}} {p1:<12.5f}{significance_code(p1)}")
        print(f"{'sine model':<{label_width}} {p2:<12.5f}{significance_code(p2)}")

        print(f"\n{self.result['message']}")
        print(
            "\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        )
        print(
            "Per-coefficient p-values use the t distribution; the higher-order "
            "test uses χ² (Jammalamadaka & Sengupta 2001).\n"
        )

    def _summary_gam(self) -> None:
        """Summary for the smooth (gam) backend: circular correlation, residual
        concentration, and the per-coordinate smooth effective dof / deviance
        explained (no harmonic coefficient table, no higher-order χ² test)."""
        print("\nCircular-Circular Regression (smooth / gam backend)\n")
        print(f"Circular Correlation Coefficient (rho): {self.result['rho']:.5f}")
        print(f"Mean Residual Cosine (A_k):             {self.result['A_k']:.5f}")
        print(f"Residual Concentration (kappa):         {self.result['kappa']:.5f}\n")

        print("Per-coordinate smooths:\n")
        for coord in ("cos", "sin"):
            edf = self.result["edf_by_smooth"][coord]
            dev = self.result["deviance_explained"][coord]
            terms = ", ".join(f"{name} edf={val:.2f}" for name, val in edf.items())
            print(f"  {coord}(θ):  {terms};  deviance explained {dev * 100:.1f}%")

        print(f"\n{self.result['message']}\n")


# Markers used by LCRegression's formula parser.
# `[^\W\d_]\w*` matches a Python-style identifier including Unicode letters
# (e.g. Greek `θ`), while still forbidding a leading digit.
_LC_IDENT = r"[^\W\d_]\w*"
# Accept both `harmonic(theta, k=K)` and `harmonic(theta, K)`.
_LC_HARMONIC_RE = re.compile(
    rf"harmonic\s*\(\s*({_LC_IDENT})\s*(?:,\s*(?:k\s*=\s*)?(\d+)\s*)?\)"
)
_LC_UNSUPPORTED_RE = re.compile(r"\b(skew|flat)\s*\(")
# Coefficient-name pattern as emitted by hea's ``harmonic()`` term, e.g.
# "harmonic(theta, 2, period = 6.283185307179586)cos1". The (var, order, cos|sin)
# structure is read straight off the wrapper + the ``cos{j}``/``sin{j}`` suffix —
# no fragile multiplier parsing.
_LC_HARMONIC_COEF_RE = re.compile(
    rf"^harmonic\(\s*(?P<var>{_LC_IDENT})\s*,\s*(?:k\s*=\s*)?\d+\s*,\s*period\s*=\s*[^)]+\)"
    rf"(?P<trig>cos|sin)(?P<order>\d+)$"
)
# Explicit cos/sin coefficient names, for formulas that write the trig terms
# directly ("y ~ cos(theta) + sin(theta)") instead of via harmonic():
# "cos(theta)", "sin(2 * theta)", "cos(theta * 2)" — multiplier on either side.
_LC_TRIG_RE = re.compile(
    rf"^(cos|sin)\(\s*"
    rf"(?:(?P<lmult>\d+)\s*\*\s*(?P<lvar>{_LC_IDENT})"
    rf"|(?P<rvar>{_LC_IDENT})(?:\s*\*\s*(?P<rmult>\d+))?)"
    rf"\s*\)$"
)


class LCRegression:
    """
    Linear–Circular Regression.

    Models a linear response Y as a function of a circular regressor θ
    (in radians). Backed by ``hea.lm``.

    Formula syntax
    --------------
    The right-hand side accepts either a marker that expands to a Fourier
    basis, or fully explicit ``cos(...) / sin(...)`` terms (or both).

    - ``"y ~ harmonic(theta)"`` — basic cosine model (Pewsey et al. 2014, §8.4.1)
    - ``"y ~ harmonic(theta, k=K)"`` — extended model with K harmonics (§8.4.2)
    - ``"y ~ cos(theta) + sin(theta) + cos(3*theta) + sin(3*theta)"`` —
      fully explicit; useful for non-contiguous harmonic orders
    - ``"y ~ harmonic(theta, k=2) + temperature"`` — mix marker with extra
      linear covariates
    - ``"y ~ s(theta, bs='cc')"`` — a **smooth** term dispatches the fit to
      ``hea.models.gam`` (penalized spline) instead of ``hea.lm``; a cyclic
      basis's period defaults to ``[0, 2π]`` (override via ``knots=``).

    Markers ``skew(theta)`` and ``flat(theta)`` are reserved for the
    nonlinear models in §8.4.3 / §8.4.4 and currently raise
    ``NotImplementedError`` (they need a nonlinear least-squares backend
    that ``hea`` does not yet provide).

    Parameters
    ----------
    formula : str
        R-style formula. See above.
    data : pandas.DataFrame or polars.DataFrame
        Input data. Pandas inputs are converted to polars internally.
    knots : dict, optional
        Smooth-backend only. Per-variable boundary knots forwarded to
        ``hea.gam``. Cyclic smooths (``bs='cc'``/``'cp'``) default to the
        circular period ``[0, 2π]`` — pass this only to override (e.g. a
        different period). Rejected for parametric formulas.
    method : str, optional
        Smooth-backend only. Smoothing-parameter selection for ``hea.gam``
        (default ``"REML"``).

    Attributes
    ----------
    formula : str
        The original formula passed in.
    expanded_formula : str
        Formula after marker expansion, as actually fit by ``hea.lm``.
    lm_fit : hea.lm
        The underlying linear-model fit. Use it for diagnostics
        (``.plot()``, ``.summary()``, ``.r_squared``, etc.).
    result : dict
        - coefficients : dict of {name: value} from the linear fit
        - harmonics : list of dicts, one per matched ``cos(k·θ)/sin(k·θ)``
          pair, each with ``variable``, ``k``, ``cos_coef``, ``sin_coef``,
          ``amplitude``, ``phase``, ``se_amplitude``, ``se_phase`` (the
          last two via the delta method on the (cos, sin) covariance).
        - sigma, r_squared, aic, bic : scalars
        - fitted, residuals : np.ndarray

    Notes
    -----
    The harmonic-pair detector recognises only **integer** multipliers,
    written on either side of ``*`` (e.g. ``cos(theta)``, ``cos(2*theta)``,
    ``cos(theta*2)``). A term like ``cos(0.5*theta)`` is treated as a regular
    linear predictor and won't appear in ``result['harmonics']``.

    References
    ----------
    Pewsey, A., Neuhäuser, M., Ruxton, G. D. (2014). *Circular Statistics
    in R*. Oxford University Press, §8.4.
    """

    def __init__(
        self,
        formula: str,
        data: pl.DataFrame,
        *,
        knots: Optional[dict] = None,
        method: str = "REML",
        **gam_kwargs,
    ):
        if not isinstance(formula, str) or "~" not in formula:
            raise ValueError(
                f"Formula must be a string containing '~'; got {formula!r}"
            )

        self.formula = formula
        self.response = formula.split("~", 1)[0].strip()
        self.data = _to_polars(data)
        self.expanded_formula = self._expand_formula(formula)

        # Backend dispatch: a smooth term (s()/te()/…) → penalized hea.gam;
        # a parametric RHS (harmonic()/cos()+sin()) → hea.lm. The circular
        # semantics (amplitude/phase for lm; the cyclic smooth for gam) and a
        # single circular-aware plot()/summary() wrap whichever backend ran.
        self.backend = "gam" if _has_smooth(self.expanded_formula) else "lm"
        if self.backend == "gam":
            self.lm_fit = None
            self.gam_fit = _hea_gam(
                self.expanded_formula,
                self.data,
                knots=_resolve_cyclic_knots(self.expanded_formula, knots),
                method=method,
                **gam_kwargs,
            )
        else:
            if knots is not None or gam_kwargs:
                raise ValueError(
                    "knots=/gam options only apply to smooth formulas "
                    "(s()/te()/…); this parametric formula uses hea.lm."
                )
            self.gam_fit = None
            self.lm_fit = _hea_lm(self.expanded_formula, self.data)
        self.result = self._build_result()

    @staticmethod
    def _expand_formula(formula: str) -> str:
        lhs, _, rhs = formula.partition("~")
        if _LC_UNSUPPORTED_RE.search(rhs):
            raise NotImplementedError(
                "skew() and flat() markers require a nonlinear least-squares "
                "backend (hea.nls), which is not yet available."
            )

        def _expand(match: "re.Match[str]") -> str:
            col = match.group(1)
            k = int(match.group(2)) if match.group(2) else 1
            if k < 1:
                raise ValueError(f"harmonic(..., k={k}): k must be a positive integer.")
            # Expand to explicit cos/sin terms (cos(θ) + sin(θ) + cos(2 * θ) + …)
            # so the fitted coefficient names read cleanly in lm.summary() —
            # ``cos(θ)``, ``sin(2 * θ)`` — rather than hea's verbose
            # ``harmonic(θ, k=…, period=…)cos1`` labels. The fit is identical
            # either way (same cos/sin design); only the names differ.
            terms = []
            for j in range(1, k + 1):
                factor = col if j == 1 else f"{j} * {col}"
                terms.append(f"cos({factor})")
                terms.append(f"sin({factor})")
            return " + ".join(terms)

        expanded_rhs = _LC_HARMONIC_RE.sub(_expand, rhs)
        return f"{lhs.strip()} ~ {expanded_rhs.strip()}"

    def _build_result(self) -> dict:
        if self.backend == "gam":
            return self._build_result_gam()
        bhat_df = self.lm_fit.bhat
        coef_names = list(bhat_df.columns)
        coef_values = list(bhat_df.row(0))
        coefficients = dict(zip(coef_names, coef_values))

        cov = np.asarray(self.lm_fit.V_bhat, dtype=float)
        column_names = list(self.lm_fit.column_names)
        name_to_idx = {n: i for i, n in enumerate(column_names)}

        # Group cos/sin terms by (variable, multiplier).
        groups: dict = {}
        for name, value in coefficients.items():
            m = _LC_HARMONIC_COEF_RE.match(name)
            if m:
                func, var, k = m.group("trig"), m.group("var"), int(m.group("order"))
            else:
                # explicit cos()/sin() terms written directly in the formula
                m = _LC_TRIG_RE.match(name)
                if not m:
                    continue
                func = m.group(1)
                if m.group("lvar") is not None:
                    var, k = m.group("lvar"), int(m.group("lmult"))
                else:
                    var = m.group("rvar")
                    k = int(m.group("rmult")) if m.group("rmult") else 1
            slot = groups.setdefault((var, k), {})
            slot[func] = (name, value)

        harmonics = []
        for (var, k), pair in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            cos_entry = pair.get("cos")
            sin_entry = pair.get("sin")
            cos_val = cos_entry[1] if cos_entry else None
            sin_val = sin_entry[1] if sin_entry else None

            amplitude = phase = se_amp = se_phase = None
            if cos_val is not None and sin_val is not None:
                amplitude = float(np.hypot(cos_val, sin_val))
                phase = float(np.arctan2(sin_val, cos_val))
                # Delta-method SEs from the (c, s) covariance block.
                ic = name_to_idx.get(cos_entry[0])
                isn = name_to_idx.get(sin_entry[0])
                if ic is not None and isn is not None and amplitude > 0:
                    var_c = cov[ic, ic]
                    var_s = cov[isn, isn]
                    cov_cs = cov[ic, isn]
                    r2 = amplitude ** 2
                    var_amp = (
                        cos_val ** 2 * var_c
                        + 2 * cos_val * sin_val * cov_cs
                        + sin_val ** 2 * var_s
                    ) / r2
                    var_phase = (
                        sin_val ** 2 * var_c
                        - 2 * cos_val * sin_val * cov_cs
                        + cos_val ** 2 * var_s
                    ) / (r2 ** 2)
                    se_amp = float(np.sqrt(max(var_amp, 0.0)))
                    se_phase = float(np.sqrt(max(var_phase, 0.0)))

            harmonics.append(
                {
                    "variable": var,
                    "k": k,
                    "cos_coef": cos_val,
                    "sin_coef": sin_val,
                    "amplitude": amplitude,
                    "phase": phase,
                    "se_amplitude": se_amp,
                    "se_phase": se_phase,
                }
            )

        residuals = self.lm_fit.residuals
        if isinstance(residuals, pl.DataFrame):
            residuals = residuals.to_numpy().ravel()
        fitted = self.lm_fit.yhat
        if isinstance(fitted, pl.DataFrame):
            fitted = fitted.to_numpy().ravel()

        return {
            "coefficients": coefficients,
            "harmonics": harmonics,
            "sigma": float(self.lm_fit.sigma),
            "r_squared": float(self.lm_fit.r_squared),
            "aic": float(self.lm_fit.AIC),
            "bic": float(self.lm_fit.BIC),
            "fitted": np.asarray(fitted, dtype=float),
            "residuals": np.asarray(residuals, dtype=float),
        }

    def _build_result_gam(self) -> dict:
        """Result dict for the smooth (gam) backend.

        Mirrors the lm result's common keys (``fitted``/``residuals``/fit
        metrics) so downstream code is backend-agnostic, and adds the smooth
        summaries (``edf_total``, ``edf_by_smooth``, ``deviance_explained``).
        There is no harmonic amplitude/phase decomposition — that is specific
        to the parametric backend — so ``harmonics`` is empty.
        """
        g = self.gam_fit
        return {
            "coefficients": dict(zip(g.bhat.columns, g.bhat.row(0))),
            "harmonics": [],
            "edf_total": float(g.edf_total),
            "edf_by_smooth": {k: float(v) for k, v in dict(g.edf_by_smooth).items()},
            "sigma": float(g.sigma),
            "r_squared": float(g.r_squared),
            "deviance_explained": float(g.deviance_explained),
            "aic": float(g.AIC),
            "bic": float(g.BIC),
            "fitted": np.asarray(g.fitted_values, dtype=float),
            "residuals": np.asarray(g.residuals, dtype=float),
        }

    def predict(
        self, data: pl.DataFrame
    ) -> np.ndarray:
        """Predict the linear response for new values of the regressors."""
        new = _to_polars(data)
        if self.backend == "gam":
            out = self.gam_fit.predict(newdata=new)["fit"].to_numpy()
            return np.asarray(out, dtype=float)
        out = self.lm_fit.predict(newdata=new)
        if isinstance(out, pl.DataFrame):
            out = out.to_numpy().ravel()
        return np.asarray(out, dtype=float)

    def summary(self) -> None:
        """Print a full diagnostic summary.

        Reuses ``hea.lm.summary()`` for the standard regression block
        (residual quantiles, coefficient table with SEs/CIs/t/p, fit metrics)
        and appends a harmonic-decomposition table with delta-method SEs and
        95% CIs for each cos/sin amplitude and phase.
        """
        print("\nLinear-Circular Regression")
        if self.expanded_formula != self.formula:
            print(f"User formula:     {self.formula}")
            print(f"Expanded formula: {self.expanded_formula}")
        print()

        if self.backend == "gam":
            # hea.gam.summary() prints the mgcv-style block itself (parametric
            # coefficients + approximate smooth significance with edf + deviance
            # explained). No harmonic table — the smooth replaces the harmonics.
            self.gam_fit.summary()
            return

        # hea.models.lm.summary() returns a SummaryLm whose __repr__ is the
        # R-style regression block (it no longer prints to stdout itself).
        print(self.lm_fit.summary())

        if self.result["harmonics"]:
            self._print_harmonic_table()

    def _print_harmonic_table(self) -> None:
        z = float(norm.ppf(0.975))

        def _fmt(value):
            return "n/a" if value is None else f"{value:.4f}"

        def _ci(value, se):
            if value is None or se is None:
                return "n/a"
            return f"[{value - z * se:.4f}, {value + z * se:.4f}]"

        rows = []
        for h in self.result["harmonics"]:
            amp, ph = h["amplitude"], h["phase"]
            se_a, se_p = h["se_amplitude"], h["se_phase"]
            label = f"{h['variable']}, k={h['k']}"
            rows.append(
                (label, _fmt(amp), _fmt(se_a), _ci(amp, se_a), _fmt(ph), _fmt(se_p), _ci(ph, se_p))
            )

        headers = (
            "term",
            "amplitude",
            "SE",
            "CI[2.5%, 97.5%]",
            "phase",
            "SE",
            "CI[2.5%, 97.5%]",
        )
        widths = [
            max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)
        ]

        print("\nHarmonic decomposition:")
        line = "  ".join(f"{h:<{w}s}" for h, w in zip(headers, widths))
        print(line)
        print("-" * len(line))
        for r in rows:
            print("  ".join(f"{c:<{w}s}" for c, w in zip(r, widths)))
        print(
            "Phase in radians; SEs and CIs from the delta method on (cos, sin) "
            "coefficients.\n"
        )

    def plot(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        n_curve: int = 200,
        ci: bool = True,
        pi: bool = False,
        level: float = 0.95,
        polar: bool = False,
        axes=None,
    ):
        """Two-panel diagnostic figure.

        Left:  scatter (θ, y) with the fitted curve over the data's θ range,
               optionally with confidence and/or prediction bands.
        Right: residuals vs fitted values.

        For models with extra non-circular covariates (e.g. ``y ~
        harmonic(θ) + temperature``), the curve is drawn fixing those
        covariates at their column means.

        Parameters
        ----------
        figsize : tuple, optional
            Matplotlib figure size; defaults to ``(11, 4.5)``.
        n_curve : int
            Number of θ points used to draw the fitted curve.
        ci, pi : bool
            Whether to shade a confidence band (``ci``) and/or prediction
            band (``pi``) at the requested ``level``. The prediction band
            (``pi``) is only available for the parametric (lm) backend.
        level : float
            Coverage probability for the bands (default 0.95).
        polar : bool
            Draw the left panel on polar axes (θ as angle, response as
            radius) — natural for a circular predictor over its full period.
        axes : sequence of matplotlib Axes, optional
            Two pre-existing axes to draw into. If omitted, a fresh figure
            is created.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        if self.backend == "gam":
            return self._plot_gam(
                figsize=figsize,
                n_curve=n_curve,
                ci=ci,
                level=level,
                polar=polar,
                axes=axes,
            )

        if not self.result["harmonics"]:
            raise ValueError(
                "plot() requires at least one matched cos/sin pair "
                "(harmonic decomposition)."
            )

        theta_var = self.result["harmonics"][0]["variable"]
        theta_data = self.data[theta_var].to_numpy()
        y_data = self.data[self.response].to_numpy()
        fitted = self.result["fitted"]
        residuals = self.result["residuals"]

        # Build a θ grid spanning the data; hold any other covariates at their mean.
        t_lo = float(min(theta_data.min(), 0.0))
        t_hi = float(max(theta_data.max(), 2 * np.pi))
        theta_grid = np.linspace(t_lo, t_hi, n_curve)
        grid: dict = {theta_var: theta_grid}
        for col in self.data.columns:
            if col in (theta_var, self.response):
                continue
            series = self.data[col]
            if series.dtype.is_numeric():
                grid[col] = np.full(n_curve, float(series.mean()))
            else:
                # Hold non-numeric columns at their mode.
                grid[col] = [series.mode()[0]] * n_curve
        grid_df = pl.DataFrame(grid)

        yhat_df = self.lm_fit.predict(grid_df)
        yhat = np.asarray(yhat_df.to_numpy()).ravel()
        alpha = 1.0 - level
        ci_lo = ci_hi = pi_lo = pi_hi = None
        if ci:
            arr = self.lm_fit.compute_ci_yhat(yhat=yhat_df, Xnew=grid_df, alpha=alpha).to_numpy()
            ci_lo, ci_hi = arr[:, 0], arr[:, 1]
        if pi:
            arr = self.lm_fit.compute_pi_yhat(yhat=yhat_df, Xnew=grid_df, alpha=alpha).to_numpy()
            pi_lo, pi_hi = arr[:, 0], arr[:, 1]

        bands = []
        if pi_lo is not None:
            bands.append((pi_lo, pi_hi, 0.15, f"{int(level * 100)}% PI"))
        if ci_lo is not None:
            bands.append((ci_lo, ci_hi, 0.30, f"{int(level * 100)}% CI"))

        fig, ax0, ax1 = _two_panel_axes(plt, figsize, polar, axes)
        self._draw_fit_panel(
            ax0, theta_var, theta_data, y_data, theta_grid, yhat, bands, polar
        )

        ax1.scatter(fitted, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
        ax1.axhline(0.0, color="k", lw=0.5)
        ax1.set_xlabel("Fitted")
        ax1.set_ylabel("Residual")
        ax1.set_title("Residuals vs fitted")

        fig.tight_layout()
        return fig

    def _draw_fit_panel(
        self, ax, theta_var, theta_data, y_data, theta_grid, yhat, bands, polar
    ) -> None:
        """Draw the left 'Fit overlay' panel shared by both backends.

        Data scatter + fitted response-scale curve, with optional shaded
        ``bands`` — each a ``(lo, hi, alpha, label)`` tuple, drawn in order so
        a wider PI sits under a narrower CI. Cartesian by default; ``polar``
        puts θ on the angular axis and the response on the radius.
        """
        for lo, hi, band_alpha, label in bands:
            ax.fill_between(theta_grid, lo, hi, color="C1", alpha=band_alpha, label=label)
        ax.plot(theta_grid, yhat, color="C1", lw=2, label="fit")
        ax.scatter(
            theta_data, y_data, color="C0", s=20, alpha=0.6,
            edgecolors="none", label="data",
        )
        ax.set_title("Fit overlay")
        ax.legend(loc="best", frameon=False)
        if polar:
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            ax.set_xticklabels(["0", "π/2", "π", "3π/2"])
        else:
            ax.set_xlabel(theta_var)
            ax.set_ylabel(self.response)
            # If the grid covers a full 2π span, mark the canonical ticks.
            if float(theta_grid.min()) <= 0 and float(theta_grid.max()) >= 2 * np.pi:
                ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
                ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])

    def _gam_predictor(self) -> str:
        """The (first) smooth predictor variable — the θ axis for plotting."""
        vs = _smooth_vars(self.expanded_formula.split("~", 1)[1])
        if not vs:
            raise ValueError("No smooth predictor found in the formula.")
        return vs[0]

    def _plot_gam(self, figsize, n_curve, ci, level, polar, axes):
        """Circular-aware overlay for the smooth (gam) backend.

        The response-scale smooth is drawn over the raw scatter on a
        period-aware grid ``[0, 2π]`` (a cyclic smooth is defined over the full
        period), with a Wald confidence band from ``se.fit``; the right panel
        is residuals vs fitted. Extra covariates are held at their column mean
        (numeric) or mode (categorical).
        """
        import matplotlib.pyplot as plt

        theta_var = self._gam_predictor()
        theta_data = self.data[theta_var].to_numpy()
        y_data = self.data[self.response].to_numpy()
        fitted = self.result["fitted"]
        residuals = self.result["residuals"]

        theta_grid = np.linspace(0.0, 2 * np.pi, n_curve)
        grid: dict = {theta_var: theta_grid}
        for col in self.data.columns:
            if col in (theta_var, self.response):
                continue
            series = self.data[col]
            if series.dtype.is_numeric():
                grid[col] = np.full(n_curve, float(series.mean()))
            else:
                grid[col] = [series.mode()[0]] * n_curve
        grid_df = pl.DataFrame(grid)

        yhat, ci_lo, ci_hi = _gam_predict_ci(
            self.gam_fit, grid_df, level if ci else 0.0
        )
        bands = []
        if ci_lo is not None:
            bands.append((ci_lo, ci_hi, 0.30, f"{int(level * 100)}% CI"))

        fig, ax0, ax1 = _two_panel_axes(plt, figsize, polar, axes)
        self._draw_fit_panel(
            ax0, theta_var, theta_data, y_data, theta_grid, yhat, bands, polar
        )

        ax1.scatter(fitted, residuals, color="C0", s=20, alpha=0.6, edgecolors="none")
        ax1.axhline(0.0, color="k", lw=0.5)
        ax1.set_xlabel("Fitted")
        ax1.set_ylabel("Residual")
        ax1.set_title("Residuals vs fitted")

        fig.tight_layout()
        return fig
