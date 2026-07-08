import types
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.special import beta as beta_fn
from scipy.special import (
    gamma,
    i0e,
    i1e,
    ndtr,
    ndtri,
    ive,
    betainc,
    betaincinv,
    gammaln,
    digamma,
    expit,
    log_ndtr,
    logsumexp,
    owens_t,
    polygamma,
    roots_legendre,
)
from hea.family import (
    GeneralFamily,
    IdentityLink,
    Link,
    LogitLink,
    LogLink,
    gamlss_etamu,
    gamlss_gH,
    trind_generator,
)
from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from .descriptive import circ_kappa, circ_mean_and_r
from .utils import A1, A1inv, A1prime, A1prime2, A1prime3, angmod

__all__ = [
    "TanHalfLink",
    "TanhLink",
    "LogitHalfLink",
    "get_link",
    "circularuniform",
    "triangular",
    "cardioid",
    "cartwright",
    "wrapnorm",
    "wrapcauchy",
    "vonmises",
    "projectednormal",
    "vonmises_flattopped",
    "jonespewsey",
    "jonespewsey_sineskewed",
    "jonespewsey_asym",
    "inverse_batschelet",
    "wrapstable",
    "katojones",
    # regression bridge — general-family classes + the *lss instance
    # aliases (names mirror the circlss R package)
    "CircularLL",
    "KatoJonesLL",
    "cardlss",
    "cartlss",
    "wnlss",
    "wclss",
    "vmlss",
    "pnlss",
    "vmftlss",
    "jplss",
    "ssjplss",
    "ajplss",
    "ibslss",
    "kjlss",
]

INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

_VMFT_MIN_GRID = 512
_VMFT_MAX_GRID = 8192
_VMFT_GRID_BASE = 64.0
_VMFT_GRID_SHARPNESS = 12.0
_VMFT_KAPPA_TOL = 1e-9
_VMFT_KAPPA_UPPER = 1e3
_VMFT_ACCEPT_EPS = 1e-12
_VMFT_NEWTON_MAXITER = 50
_VMFT_NEWTON_TOL = 1e-12
_VMFT_NEWTON_WIDTH_TOL = 1e-10

_INVBAT_KAPPA_TOL = 1e-9
_INVBAT_KAPPA_UPPER = 700.0
_INVBAT_NUMERIC_GRID = 4096
# Coarser grid for the *derivative* normalizer: the FD'd log-c gradient/Hessian
# only feeds EFS, and the central difference cancels the grid quadrature error
# because the integrand's peak (where B(u)=0) does not move with κ — the same
# grid nodes sample it on both sides of the difference. Benchmarked across the
# whole domain (κ up to the 700 clip, |λ|≤0.95): the FD grad/Hessian is
# grid-independent from 4096 down to 256 (residual ~3e-8 grad / ~1e-5 Hess is
# pure FD-step truncation), so 256 runs ~16x lighter than the 4096 value grid
# with no accuracy cost.
_INVBAT_DERIV_GRID = 256
_INVBAT_NU_TOL = 1e-12
_INVBAT_LMBDA_TOL = 1e-12
_INVBAT_MIN_GRID = 512
_INVBAT_MAX_GRID = 8192
_INVBAT_NEWTON_MAXITER = 60
_INVBAT_NEWTON_TOL = 1e-12
_INVBAT_NEWTON_WIDTH_TOL = 1e-10
_INVBAT_ENV_MIN_KAPPA = 1e-6

_WRAPSTABLE_PDF_TOL = 1e-12
_WRAPSTABLE_CDF_TOL = 1e-12
_WRAPSTABLE_ALPHA_TOL = 1e-10
_WRAPSTABLE_MAX_TERMS = 20000
_WRAPSTABLE_NEWTON_MAXITER = 60
_WRAPSTABLE_NEWTON_TOL = 1e-12
_WRAPSTABLE_NEWTON_WIDTH_TOL = 1e-10

_KJ_CDF_TOL = 1e-12
_KJ_MAX_TERMS = 5000
_KJ_GAMMA_TOL = 1e-12
_KJ_NEWTON_MAXITER = 60
_KJ_NEWTON_TOL = 1e-12
_KJ_NEWTON_WIDTH_TOL = 1e-10

OPTIMIZERS = [
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-constr",
    "dogleg",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
]


class _RegressionReady:
    """Regression overlay for a circular distribution (read by the regression
    engine only; descriptive use never sees it).

    A distribution opts in by composing this mixin and declaring two class
    dicts in **book-named** parameters:

    - ``param_roles``  : ``{book_name -> role}``  (e.g. ``{"mu": "location"}``)
    - ``default_links`` : ``{role -> link_name}`` (e.g. ``{"location": "tanhalf"}``,
      resolved to a link object by :func:`get_link` below)

    The mixin derives the inverse views for free — nothing is renamed; values
    always flow in the distribution's own book-named dicts. Roles exist only so
    a generic engine can find the mean-direction vs concentration vs shape
    parameters and attach each one's default link.

    Derivatives (optional, tiered) keep book names too:

    - ``dlogpdf(x, **params) -> {name: ∂logpdf/∂name}``                  (l1)
    - ``d2logpdf(x, **params) -> {(name_i, name_j): ∂²logpdf/∂i∂j}``     (l2)

    only the unique unordered pairs are stored in ``d2logpdf``.
    """

    param_roles: dict = {}
    default_links: dict = {}
    # Optional size-aware MAP degeneracy guard (read by CircularLL, applied only
    # in a reweighted circ_mix M-step): a tuple of `_degen_*` kernels naming the
    # degeneracy-prone natural parameter(s). Empty -> no guard (a standalone fit is
    # unaffected regardless).
    degen_penalty: tuple = ()

    @classmethod
    def params_by_role(cls) -> dict:
        """``{role -> [book_name, ...]}`` — one-to-many (e.g. several shape
        params), insertion-ordered by ``param_roles``."""
        out: dict = {}
        for name, role in cls.param_roles.items():
            out.setdefault(role, []).append(name)
        return out

    @classmethod
    def link_for(cls, name: str) -> str:
        """Default link name for a parameter, via its role."""
        return cls.default_links[cls.param_roles[name]]


# --- links (the contract's other half: resolving ``default_links`` names) ----
# A link maps a parameter onto the unconstrained linear-predictor scale,
# η = g(param). Every link resolved here is a ``hea.family.Link``, so one
# authored object feeds the hea general-family bridge: ``CircularLL`` hands it
# to ``hea.gam``, whose ``gamlss_etamu`` chain rule consumes ``link`` (g),
# ``linkinv`` (η → parameter), ``mu_eta`` (∂param/∂η) and
# ``d2link``/``d3link``/``d4link`` (g″, g‴, g⁗ — derivatives w.r.t. the
# *parameter*, mgcv convention). hea's catalog already carries those hooks for
# log/logit/identity, so only the circular-specific tan-half link — which hea
# must not learn — is authored here. (``circ_lm``'s parametric von Mises fit
# inlines the tan-half link directly, so it consults no link object.)


class TanHalfLink(Link):
    r"""Fisher–Lee tan-half link for a circular location parameter.

    $$\eta = g(\mu) = \tan(\mu/2), \qquad
      \mu = g^{-1}(\eta) = 2\arctan(\eta) \in (-\pi, \pi).$$

    Maps the principal angle branch monotonically onto ℝ; a CL model offsets
    it by the intercept direction, ``μ_i = μ₀ + 2 arctan(x_iᵀβ)`` (Fisher &
    Lee 1992). ``tan(μ/2)`` is 2π-periodic in μ, so ``link`` returns the
    principal-branch η for any angle parameterization; the only singularity
    is ``μ ≡ π (mod 2π)``, the antipode of the offset.

    Derivatives w.r.t. μ, in ``t = tan(μ/2)`` (so ``dt/dμ = (1+t²)/2``):

    $$g' = \tfrac{1+t^2}{2},\quad g'' = \tfrac{t(1+t^2)}{2},\quad
      g''' = \tfrac{(1+t^2)(1+3t^2)}{4},\quad
      g'''' = \tfrac{t(1+t^2)(2+3t^2)}{2}.$$
    """

    name = "tanhalf"

    def link(self, mu):
        return np.tan(0.5 * np.asarray(mu, dtype=float))

    def linkinv(self, eta):
        return 2.0 * np.arctan(np.asarray(eta, dtype=float))

    def mu_eta(self, eta):
        eta = np.asarray(eta, dtype=float)
        return 2.0 / (1.0 + eta * eta)

    def d2link(self, mu):
        t = np.tan(0.5 * np.asarray(mu, dtype=float))
        return 0.5 * t * (1.0 + t * t)

    def d3link(self, mu):
        t2 = np.tan(0.5 * np.asarray(mu, dtype=float)) ** 2
        return 0.25 * (1.0 + t2) * (1.0 + 3.0 * t2)

    def d4link(self, mu):
        t = np.tan(0.5 * np.asarray(mu, dtype=float))
        t2 = t * t
        return 0.5 * t * (1.0 + t2) * (2.0 + 3.0 * t2)


class TanhLink(Link):
    r"""Tanh link for a (−1, 1)-bounded shape parameter (the sine-skew λ).

    $$\eta = g(\lambda) = \operatorname{atanh}(\lambda), \qquad
      \lambda = \tanh(\eta) \in (-1, 1).$$

    Derivatives w.r.t. λ:

    $$g' = \frac{1}{1-\lambda^2},\quad g'' = \frac{2\lambda}{(1-\lambda^2)^2},
      \quad g''' = \frac{2+6\lambda^2}{(1-\lambda^2)^3},\quad
      g'''' = \frac{24\lambda(1+\lambda^2)}{(1-\lambda^2)^4}.$$
    """

    name = "tanh"

    def link(self, mu):
        return np.arctanh(np.asarray(mu, dtype=float))

    def linkinv(self, eta):
        # clamped inside the open interval (mgcv convention) — at λ = ±1 a
        # sine-skewed density touches 0 and its logpdf has no finite limit
        eps = np.finfo(float).eps
        return np.clip(np.tanh(np.asarray(eta, dtype=float)),
                       -1.0 + eps, 1.0 - eps)

    def mu_eta(self, eta):
        # sech²η = 4e^{-2|η|}/(1+e^{-2|η|})², overflow-safe; floored at eps
        a = np.exp(-2.0 * np.abs(np.asarray(eta, dtype=float)))
        return np.maximum(4.0 * a / (1.0 + a) ** 2, np.finfo(float).eps)

    def d2link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 2.0 * mu / (1.0 - mu * mu) ** 2

    def d3link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return (2.0 + 6.0 * mu * mu) / (1.0 - mu * mu) ** 3

    def d4link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 24.0 * mu * (1.0 + mu * mu) / (1.0 - mu * mu) ** 4


class LogitHalfLink(Link):
    r"""Scaled logit for a (0, ½)-bounded concentration (the cardioid ρ).

    $$\eta = g(\rho) = \log\frac{\rho}{\tfrac12-\rho}, \qquad
      \rho = \tfrac12\,\operatorname{expit}(\eta) \in (0, \tfrac12).$$

    The general scaled logit on (0, c): a subclass overriding ``hi`` serves
    any upper bound c.
    """

    name = "logit_half"
    hi = 0.5

    def link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return np.log(mu / (self.hi - mu))

    def linkinv(self, eta):
        # clamped to (hi·eps, hi·(1−eps)) — at ρ = ½ the cardioid density
        # touches 0 at the antimode (logpdf → −∞ there)
        eps = np.finfo(float).eps
        return self.hi * np.clip(expit(np.asarray(eta, dtype=float)),
                                 eps, 1.0 - eps)

    def mu_eta(self, eta):
        # hi·e^{-|η|}/(1+e^{-|η|})², overflow-safe; floored at eps
        a = np.exp(-np.abs(np.asarray(eta, dtype=float)))
        return np.maximum(self.hi * a / (1.0 + a) ** 2, np.finfo(float).eps)

    def d2link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 1.0 / (self.hi - mu) ** 2 - 1.0 / mu**2

    def d3link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 2.0 / (self.hi - mu) ** 3 + 2.0 / mu**3

    def d4link(self, mu):
        mu = np.asarray(mu, dtype=float)
        return 6.0 / (self.hi - mu) ** 4 - 6.0 / mu**4


# Names a distribution may declare in ``default_links``. tanhalf, tanh and
# logit_half are ours (circular/bounded-shape knowledge hea must not learn);
# the rest resolve to hea's implementations (log for κ > 0, logit for a
# (0, 1)-bounded concentration such as the wrapped Cauchy ρ, identity for
# unconstrained shape parameters).
_LINKS = {
    "tanhalf": TanHalfLink,
    "log": LogLink,
    "logit": LogitLink,
    "identity": IdentityLink,
    "tanh": TanhLink,
    "logit_half": LogitHalfLink,
}


def get_link(link: "str | Link") -> Link:
    """Resolve a link declared by the regression overlay to a link object.

    Accepts a name from ``default_links`` (e.g. ``vonmises.link_for("mu")``)
    or an already-constructed ``hea.family.Link``, which passes through so a
    custom link can be handed anywhere a name is accepted.
    """
    if isinstance(link, Link):
        return link
    if isinstance(link, str) and link in _LINKS:
        return _LINKS[link]()
    raise ValueError(f"unknown link {link!r}; available: {sorted(_LINKS)}")


def _inverse_cdf_knots(phi, cumulative, min_step=1e-14):
    """Strictly-increasing (cdf, angle) knots for an inverse-cdf
    interpolant, greedily thinned to steps ≥ ``min_step``: in dead tails a
    tabulated cdf grows by denormal amounts per node (the pdf is floored at
    ``np.finfo(float).tiny``), so dφ/dq overflows and Pchip's derivative
    screen rejects the work arrays ("``dydx`` must contain only finite
    values" — the κ ≳ 200 vmft crash, and
    its inverse-Batschelet twin at κ ≳ 400). Quantile accuracy is
    unaffected for any q the thinned knots can distinguish; the forced
    q = 1 endpoint keeps the domain closed (its gap is ≥ ~1e-16, because
    doubles within eps of 1 collapse to 1). Returns ``(values, angles)``,
    or ``None`` when fewer than two knots survive."""
    unique_vals, unique_idx = np.unique(cumulative, return_index=True)
    if unique_vals.size < 2:
        return None
    sel = [0]
    last = unique_vals[0]
    for i in range(1, unique_vals.size):
        if unique_vals[i] - last >= min_step:
            sel.append(i)
            last = unique_vals[i]
    if sel[-1] != unique_vals.size - 1:
        sel.append(unique_vals.size - 1)
    keep = np.asarray(sel, dtype=int)
    return unique_vals[keep], np.asarray(phi, dtype=float)[unique_idx[keep]]


def _as_scalar_param(value, dist_name):
    """Collapse a shape parameter to a float, tolerating arrays that repeat a
    single value — scipy's ``cdf`` machinery broadcasts shape parameters to
    the shape of ``x`` before ``_cdf`` is called, so a scalar parameter can
    arrive as a constant array."""
    arr = np.asarray(value, dtype=float)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    first = float(arr.flat[0])
    if np.all((arr == first) | (np.isnan(arr) & np.isnan(first))):
        return first
    raise ValueError(f"{dist_name} parameters must be scalar-valued.")


# --- the hea bridge: a circular distribution as an mgcv-style general family --
# param_roles/default_links/dlogpdf..d4logpdf — the family contract — live in
# this module, and CircularLL is its assembled reader; regression.py re-exports
# both classes, so either import site works.


# --- size-aware MAP-penalty degeneracy guard (circ_mix M-step only) -----------
# A finite mixture's likelihood is unbounded: a reweighted component raises it by
# concentrating onto its responsibility-weighted subset — concentration runs off
# (kappa -> Inf) or a bounded shape/peakedness parameter is driven onto its
# singular boundary (cardioid rho -> 1/2, the tanh-linked nu/lmbd -> +/-1, the
# Kato-Jones disc edge), where the penalized Hessian goes indefinite. The guard
# adds to each reweighted M-step a penalty pulling the degeneracy-prone NATURAL
# parameter toward the family's diffuse/reduced model, with strength
# lambda_k = c / N_k that VANISHES as the component grows (N_k the effective size
# sum_i gamma_ik) — the dual of c diffuse pseudo-observations, so a well-populated
# component is unaffected and a collapsing one cannot reach the boundary. The
# gamlss_etamu chain rule carries the natural-coord penalty to eta; it is folded
# into the derivative blocks BEFORE the wt-scaling, so it is automatically
# responsibility-weighted, and `l0` (the per-datum density the E-step reads) is
# left untouched. It activates only when a circ_mix M-step has set
# `family.map_lambda > 0`; for a standalone circ_gam `map_lambda` is None, so
# `_degen_active` is False and the fit is byte-for-byte unchanged. A family
# declares its degeneracy-prone parameter(s) + kernel(s) via the `degen_penalty`
# class attribute on its distribution (mirroring `param_roles`); a family with no
# boundary pathology omits it (Cartwright). Port of circlss's internal-degen.R.

_DEGEN_EPS = float(np.finfo(float).eps)


class _DegenKernel:
    """One penalty kernel rho(v) on a single natural parameter, carrying its
    value and first two derivatives (orders 0-2 are all the Newton/EFS families
    need; rho3 = rho4 = 0 for every kernel here). ``param`` is the book name of
    the parameter it acts on; ``scale`` (baked into the closures) is a
    family-declared relative strength, so the one global ``c`` stays the default."""

    __slots__ = ("param", "rho0", "rho1", "rho2")

    def __init__(self, param, rho0, rho1, rho2):
        self.param = param
        self.rho0, self.rho1, self.rho2 = rho0, rho1, rho2


def _degen_linear(param, scale=1.0):
    """rho(v) = scale*v — exponential prior toward 0 for an unbounded-above
    concentration (kappa, log link): a constant gradient pull, no curvature
    (rho2 = 0), right where the data Hessian flattens (kappa -> Inf) rather than
    going singular."""
    s = float(scale)
    return _DegenKernel(
        param,
        lambda v: s * v,
        lambda v: np.full(np.shape(v), s, dtype=float),
        lambda v: np.zeros(np.shape(v), dtype=float),
    )


def _degen_ridge(param, scale=1.0):
    """rho(v) = scale*v^2 — Gaussian prior toward 0 for an unbounded shape
    coordinate whose diffuse value is 0 (jplss psi -> von Mises, pnlss mu ->
    uniform radius, kjlss u -> wrapped Cauchy); the positive curvature also
    stabilizes the Hessian."""
    s = float(scale)
    return _DegenKernel(
        param,
        lambda v: s * v * v,
        lambda v: 2.0 * s * v,
        lambda v: np.full(np.shape(v), 2.0 * s, dtype=float),
    )


def _degen_boundary_upper(param, vmax, scale=1.0):
    """rho(v) = -scale*log(1 - v/vmax) on v in [0, vmax) — diverges as v -> vmax,
    so a small component cannot push the parameter onto the wall (cardioid
    rho -> 1/2, wrapped normal rho -> 1). Pulls toward 0 (uniform); an eps floor
    on the denominator keeps a parameter sitting on the wall finite."""
    s, vm = float(scale), float(vmax)

    def z(v):
        return np.maximum(1.0 - np.asarray(v, dtype=float) / vm, _DEGEN_EPS)

    return _DegenKernel(
        param,
        lambda v: -s * np.log(z(v)),
        lambda v: s * (1.0 / vm) / z(v),
        lambda v: s * (1.0 / vm ** 2) / (z(v) * z(v)),
    )


def _degen_boundary_sym(param, vmax, scale=1.0):
    """rho(v) = -scale*log(1 - (v/vmax)^2) on v in (-vmax, vmax) — diverges as
    v -> +/-vmax, pulling toward 0 (the von Mises / symmetric member). For the
    tanh-linked peakedness/skewness parameters (nu, lmbd) whose crash is the
    +/-1 edge. Eps-floored denominator."""
    s, vm = float(scale), float(vmax)

    def z(v):
        return np.maximum(1.0 - (np.asarray(v, dtype=float) / vm) ** 2, _DEGEN_EPS)

    return _DegenKernel(
        param,
        lambda v: -s * np.log(z(v)),
        lambda v: s * (2.0 * v / vm ** 2) / z(v),
        lambda v: s * ((2.0 / vm ** 2) / z(v) + (4.0 * v ** 2 / vm ** 4) / (z(v) * z(v))),
    )


def _degen_active(family) -> bool:
    """True only when a circ_mix M-step has set a positive ``map_lambda`` AND the
    family declares a ``degen`` spec; None/absent for a standalone circ_gam, so a
    non-mixture fit is byte-for-byte unchanged."""
    lam = getattr(family, "map_lambda", None)
    return bool(getattr(family, "degen", None)) and (
        lam is not None and bool(np.isfinite(lam)) and float(lam) > 0.0
    )


def _lss_map_penalty(family, params: dict, lam: float) -> dict:
    """The MAP penalty's contribution to ``(l0, l1, l2)`` in NATURAL coordinates,
    ready to ADD to the family's per-datum blocks before the wt-scaling. Negation
    and ``lam``-scaling are baked in; the penalty is separable across parameters,
    so it touches the l1 gradient entries and the l2 DIAGONAL only, placed via the
    family's own trind index. ``params`` is the LP-space (inverse-linked)
    parameter dict."""
    n_lp = family.n_lp
    i2 = family.tri["i2"]
    n = int(np.atleast_1d(np.asarray(params[family.degen[0].param])).shape[0])
    dl0 = np.zeros(n)
    dl1 = np.zeros((n, n_lp))
    dl2 = np.zeros((n, n_lp * (n_lp + 1) // 2))
    for k in family.degen:
        j = family.params.index(k.param)
        v = np.broadcast_to(np.asarray(params[k.param], dtype=float), (n,))
        dl0 = dl0 - lam * k.rho0(v)
        dl1[:, j] = dl1[:, j] - lam * k.rho1(v)
        cjj = int(i2[j, j])
        dl2[:, cjj] = dl2[:, cjj] - lam * k.rho2(v)
    return {"l0": dl0, "l1": dl1, "l2": dl2}


class CircularLL(GeneralFamily):
    """A regression-ready circular distribution as a hea/mgcv **general
    family**.

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

    The primary spelling is the module-level ``*lss`` alias of each
    regression-ready distribution (``vmlss``, ``wclss``, … ``kjlss`` — names
    shared with the circlss R package), a pre-built instance of this class:

        gam(["theta ~ s(x, bs='cc')",   # LP1 → μ      (tanhalf)
             "      ~ s(z)"],           # LP2 → log κ  (log)
            data, family=vmlss, method="REML")

    Construct directly — ``CircularLL(dist, links=[...])`` — or *call* an
    alias — ``vmlss(links=[...])``, which returns a fresh configured
    instance — only when overriding the declared default links.

    Note the mgcv parameterization: every LP's intercept lives *inside* the
    link — μ = 2·atan(β₀ + …) — unlike ``circ_lm(type="cl")``'s
    Fisher–Lee offset ``μ₀ + 2·atan(Xβ)``. The two coincide for
    intercept-only models; with covariates they are different (both valid)
    parameterizations, so compare fitted curves, not coefficients.
    """

    scale_known = True
    n_theta = 0

    def __init__(self, dist, links=None, name=None):
        roles = getattr(dist, "param_roles", None)
        if not roles:
            raise TypeError(
                f"{getattr(dist, 'name', dist)!r} is not regression-ready: it "
                "declares no `param_roles` overlay."
            )
        self.dist = dist
        self.params = list(roles)  # book-named, declaration order = LP order
        self.n_lp = len(self.params)
        # The base class assumes the LP coordinates *are* the distribution's
        # own logpdf parameters (its ll/fit seams pass them straight
        # through). A family that regresses in transformed coordinates —
        # katojones declares the chart pair u1/u2, which logpdf does
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
        # Size-aware MAP degeneracy guard (circ_mix M-step only). The distribution
        # declares which natural parameter(s) degenerate and toward what via its
        # `degen_penalty` overlay; `map_lambda` is set per-fit by a circ_mix M-step
        # (None here -> `_degen_active` False -> a standalone circ_gam is
        # byte-for-byte unchanged). A clone (`__call__`) re-reads the spec from the
        # same distribution and resets `map_lambda`, as a fresh family should.
        self.degen = tuple(getattr(dist, "degen_penalty", ()) or ())
        self.map_lambda = None
        # `name` is what fitted summaries print; the *lss aliases set it to
        # the cross-language family name (vmlss(…).name == "vmlss" ==
        # circlss's family$family) so differential tests diff clean.
        self.name = (name if name is not None
                     else f"CircularLL({getattr(dist, 'name', 'dist')})")
        super().__init__(links)

    def __call__(self, links=None, name=None):
        """A fresh family of the same class over the same distribution —
        the family-side twin of the scipy freeze idiom (calling
        ``vonmises(mu, kappa)`` freezes a distribution; calling
        ``vmlss(links=[...])`` configures a family). The instance itself is
        never mutated, so the module-level aliases stay pristine; ``name``
        defaults to this instance's (clones keep reporting "vmlss"). Also
        makes the R parens spelling ``family=vmlss()`` valid verbatim."""
        if name is None:
            name = self.name
        if type(self) is CircularLL:
            return CircularLL(self.dist, links, name=name)
        # baked-in-dist subclass (KatoJonesLL convention): (links=, name=)
        return type(self)(links=links, name=name)

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
        """Intercept-only start in LP coordinates, declaration-ordered.

        The neutral closed-form start of the circlss ``initialize`` convention:
        location = mean direction, every concentration parameter = its
        ``Rbar``-based moment/A1-inverse estimator (the per-distribution
        ``_concentration_start`` hook), every shape/skewness parameter = 0 (the
        symmetric / von-Mises reduction member). Optimizer-free and bit-exact
        across languages and — unlike the marginal joint MLE (``dist.fit``) —
        it never lets a covariate-distorted pooled moment push a shape
        parameter to an extreme where ``gam.fit5``'s penalized Hessian goes
        indefinite. Both starts reach the same optimum (only the EFS path
        differs).

        Feeds both the EFS start (via :meth:`initialize_coef`, which overrides
        the single location with the projected pilot) and ``postproc``'s null
        deviance, so this is also the null-deviance reference — matching
        circlss's ``postproc``. Families that declare no ``_concentration_start``
        (the 2-component projected normal: pure location, no concentration or
        shape, no boundary pathology) keep the marginal ``fit``.
        """
        roles = self.dist.param_roles
        conc_start = getattr(self.dist, "_concentration_start", None)
        locs = [p for p in self.params if roles[p] == "location"]
        if conc_start is None or len(locs) != 1:
            return np.atleast_1d(self.dist.fit(y)).astype(float)
        y = np.asarray(y, dtype=float)
        sy, cy = float(np.mean(np.sin(y))), float(np.mean(np.cos(y)))
        mu0 = float(np.arctan2(sy, cy))
        Rbar = float(np.hypot(sy, cy))
        out = []
        for p in self.params:
            role = roles[p]
            if role == "location":
                out.append(mu0)
            elif role == "concentration":
                out.append(float(conc_start(Rbar)))
            else:  # shape / skewness -> reduction member
                out.append(0.0)
        return np.asarray(out, dtype=float)

    def ll(self, y, X, coef, wt=None, *, lpi, offset=None, deriv: int = 0,
           d1b=None, d2b=None, fh=None, D=None) -> dict:
        # Prior weights as a likelihood weight. mgcv's gamlss families receive
        # `wt` but drop it; circlss makes it count and so do we: a weighted
        # log-likelihood scales the objective and every per-observation
        # derivative row by wt, so weighting a row by w is identical to
        # duplicating that row w times -- which is what lets a weighted fit
        # (gam(weights=), e.g. a finite-mixture EM M-step) reach the weighted
        # MLE. hea passes ones when the user gave no gam(weights=). `l0` itself
        # stays UNWEIGHTED: it is the per-observation log-density a downstream
        # E-step reads (only the scalar objective `l` and the derivative blocks
        # are weighted).
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        coef = np.asarray(coef, dtype=float)
        wt = (np.ones(y.shape, dtype=float) if wt is None
              else np.broadcast_to(np.asarray(wt, dtype=float), y.shape))
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        etas = self._etas(X, coef, jj, offset)
        params = self._param_values(etas)

        l0 = self._loglik_values(y, params)
        # size-aware MAP degeneracy penalty (circ_mix M-step only; inert for a
        # standalone circ_gam, where map_lambda is None). Penalizes the scalar
        # objective here and the natural derivative blocks below (before the
        # wt-scaling); the returned `l0` (the E-step density) stays unpenalized.
        pen = (_lss_map_penalty(self, params, self.map_lambda)
               if _degen_active(self) else None)
        lval = float(np.sum(wt * l0))
        if pen is not None:
            lval += float(np.sum(wt * pen["l0"]))
        ret: dict = {"l": lval, "l0": l0}
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

        # Fold the MAP degeneracy penalty into the natural derivative blocks
        # before the wt-scaling, so it is responsibility-weighted like every other
        # row; only orders 0-2 are nonzero (rho3 = rho4 = 0), so l3/l4 are untouched.
        if pen is not None:
            l1 = l1 + pen["l1"]
            l2 = l2 + pen["l2"]

        # Scale the derivative blocks by the prior weights before the chain rule:
        # gamlss_etamu is linear per row in (l1..l4), so weighting these inputs
        # equals weighting the eta-space derivatives gamlss_gH assembles. Absent
        # higher orders stay None.
        w = wt[:, None]
        l1 = l1 * w
        l2 = l2 * w
        if l3 is not None:
            l3 = l3 * w
        if l4 is not None:
            l4 = l4 * w

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
        """Null-model start, mgcv-flavoured but circular-safe, **role-aware**.

        Every concentration/shape LP is least-squares fit onto the *constant*
        target ``link(param̂)`` — the distribution's own intercept-only ``fit``
        — with the penalty root ``E`` stacked as a regularizer. A constant has
        no df to overfit, so the plain stacked solve serves both
        ``use_unscaled`` branches.

        A **single circular location** LP (every tan-half family: vM/WC/WN/
        cardioid/Cartwright/JP/ssJP) instead gets a *data-following* projected
        pilot: penalized-smooth ``cos y`` and ``sin y`` over that LP's design
        columns, recombine ``μ̂ = atan2(ŝ, ĉ)``, and start from ``link(μ̂)``.
        A flat-μ start strands antipodal observations on the log-likelihood
        cliffs of any density with a zero on the circle — Cartwright's
        ``(1+cos)^{1/ζ}`` is exactly 0 at the antipode for every ζ, where
        ``∂ℓ/∂μ = tan(d/2)/ζ → ∞`` — so EFS oversmooths μ to a constant and
        inflates the scale to absorb the apparent diffuseness, a coupled bad
        basin (μ̂ flat, ρ̂ → 0). The projected pilot starts near the data and
        sidesteps it; it strictly improves the start for the other tan-half
        families too. The projected normal's two identity-linked Cartesian
        location components have no antipode zero, so a 2-component location
        falls through to the constant start (unchanged).
        """
        y = np.asarray(y, dtype=float)
        X = np.asarray(X, dtype=float)
        jj = [np.asarray(ix, dtype=int) for ix in lpi]
        n, p = X.shape
        if E is None:
            E = np.zeros((0, p))

        def stacked_solve(cols, target):
            xa = np.vstack([X[:, cols], E[:, cols]])
            ta = np.concatenate([target, np.zeros(E.shape[0])])
            b, *_ = np.linalg.lstsq(xa, ta, rcond=None)
            b[~np.isfinite(b)] = 0.0
            return b

        loc = self.dist.params_by_role().get("location", [])
        loc_idx = [self.params.index(nm) for nm in loc]
        # projected pilot only for a SINGLE circular location (tanhalf); the
        # 2-component projected normal keeps the constant start (no antipode
        # zero), so it is left out
        single_loc = loc_idx[0] if len(loc_idx) == 1 else None

        start = np.zeros(p)
        if single_loc is not None:
            cols = jj[single_loc]
            chat = X[:, cols] @ stacked_solve(cols, np.cos(y))
            shat = X[:, cols] @ stacked_solve(cols, np.sin(y))
            muhat = np.arctan2(shat, chat)
            # clip guards the tanhalf pole at μ̂ ≡ π (η → ∞)
            target = np.clip(self.links[single_loc].link(muhat), -1e6, 1e6)
            if (offset is not None and len(offset) > single_loc
                    and offset[single_loc] is not None):
                target = target - offset[single_loc]
            start[cols] = stacked_solve(cols, target)

        param_hat = self._null_params(y)
        for j, (link, par0) in enumerate(zip(self.links, param_hat)):
            if j == single_loc:
                continue
            # clip guards the tanhalf pole at μ̂ ≡ π (η → ∞)
            eta0 = float(np.clip(link.link(float(par0)), -1e6, 1e6))
            target = np.full(n, eta0)
            if offset is not None and len(offset) > j and offset[j] is not None:
                target = target - offset[j]
            start[jj[j]] = stacked_solve(jj[j], target)
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

    def postproc(self, y, fitted=None, prior_weights=None,
                 linear_predictors=None, offset=None, intercept=True) -> dict:
        """Null deviance for the summary's "deviance explained": the
        distribution's own intercept-only fit pushed through the same
        deviance-residual convention as :meth:`residuals` (twice the
        log-likelihood gap to the fitted-mode saturated reference).

        The signature binds both hea calling conventions: 0.1.4 passes
        ``(y, fitted)`` positionally; later hea passes mgcv's full hook —
        ``postproc(y, prior_weights=, fitted=, linear_predictors=,
        offset=, intercept=)`` — by keyword. Only ``y`` enters the
        computation (the null model is refit from scratch), so the
        remaining arguments are accepted and ignored."""
        y = np.asarray(y, dtype=float)
        par0 = self._null_params(y)
        fitted0 = np.broadcast_to(par0, (y.shape[0], self.n_lp))
        r0 = self.residuals(y, fitted0, type="deviance")
        return {"null_deviance": float(np.sum(r0 * r0))}

    def residuals(self, y, fitted, type: str = "deviance") -> np.ndarray:
        """Angular residuals, ``fitted`` the (n, n_lp) inverse-linked matrix.

        - ``response``: the wrapped difference ``y − μ̂(x)`` in (−π, π].
        - ``pearson``: the score-standardized ``sin(y − μ̂)/√(Var sin)``, with
          ``Var(sin(y − μ̂)) = (1 − α₂)/2`` from the family's centered second
          cosine moment (:meth:`_pearson_var`). Families with no single
          circular location (the projected normal's Cartesian pair) have no
          such standardization and alias the deviance residual.
        - ``deviance``: the signed root of twice the log-likelihood gap to the
          **saturated** reference — the maximum achievable per-observation
          log-density (:meth:`_saturated_loglik`), the density peak rather
          than the value at the location anchor, so a skewed family whose mode
          is off the anchor still gets a well-posed residual.
        """
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
        if type == "pearson":
            v = self._pearson_var(params)
            if v is not None:
                return np.sin(rsd) / np.sqrt(np.clip(v, 1e-12, None))
            # no single circular location (e.g. pnlss's Cartesian pair): there
            # is no sin-residual standardization, so alias the deviance
            # residual — matching circlss's pnlss convention.
        l_obs = self._loglik_values(y, params)
        l_sat = self._saturated_loglik(params, mu_hat, l_obs)
        return np.sign(rsd) * np.sqrt(2.0 * np.clip(l_sat - l_obs, 0.0, None))

    def _shape_groups(self, params, loc_name):
        """Distinct rows of the non-location (shape/concentration) parameters,
        as ``(unique_rows, inverse_index, shape_names, n)``. The centered
        moment and the density peak both depend only on these — the location
        merely slides the density rigidly — so each is computed once per
        distinct shape and broadcast back over the inverse index."""
        shape_names = [p for p in self.params if p != loc_name]
        n = int(np.atleast_1d(np.asarray(params[loc_name])).shape[0])
        cols = [np.broadcast_to(np.asarray(params[p], dtype=float), (n,))
                for p in shape_names]
        if cols:
            uniq, inv = np.unique(np.column_stack(cols), axis=0,
                                  return_inverse=True)
            inv = np.asarray(inv).ravel()
        else:
            uniq, inv = np.zeros((1, 0)), np.zeros(n, dtype=int)
        return uniq, inv, shape_names, n

    def _pearson_var(self, params):
        """Per-observation ``Var(sin(y − μ̂)) = (1 − α₂)/2`` for the circular
        Pearson residual, ``α₂`` the centered second cosine moment read from
        the distribution's own :meth:`trig_moment` (location set to 0 so the
        moment is centered; location-invariant ⇒ once per distinct shape).
        Returns ``None`` when the family has no single circular location (the
        projected normal's Cartesian pair) ⇒ the caller aliases the deviance
        residual, as circlss does for pnlss. Exact closed forms fall out:
        von Mises → ``A1(κ)/κ``, wrapped Cauchy → ``(1 − ρ²)/2``."""
        locs = self.dist.params_by_role().get("location", [])
        if len(locs) != 1:
            return None
        loc_name = locs[0]
        uniq, inv, shape_names, _ = self._shape_groups(params, loc_name)
        v = np.empty(len(uniq))
        for i, row in enumerate(uniq):
            kw = {loc_name: 0.0}
            kw.update({nm: float(row[k]) for k, nm in enumerate(shape_names)})
            a2 = float(np.real(self.dist.trig_moment(2, **kw)))
            v[i] = 0.5 * (1.0 - a2)
        return v[inv]

    def _saturated_loglik(self, params, mu_hat, l_obs):
        """The saturated per-observation log-likelihood for the deviance
        residual: the maximum achievable log-density. A single-location family
        has a location-invariant peak height, so it is the grid maximum of the
        log-density (:meth:`_peak_loglik`); the value at the location anchor
        and at the datum bound it, keeping symmetric families bit-exact (their
        mode *is* the location) and the reference never below the attained
        value. A multi-location family (projected normal) is unimodal at the
        fitted direction, so the anchor value is already the peak."""
        l_loc = self._loglik_values(np.mod(mu_hat, 2.0 * np.pi), params)
        locs = self.dist.params_by_role().get("location", [])
        if len(locs) == 1:
            return np.maximum(np.maximum(l_loc, self._peak_loglik(params, locs[0])),
                              l_obs)
        return np.maximum(l_loc, l_obs)

    def _peak_loglik(self, params, loc_name, ngrid: int = 1024):
        """Grid maximum (parabola-refined) of the log-density over θ ∈ [0, 2π):
        the location-invariant peak log-density, evaluated once per distinct
        shape and broadcast back."""
        uniq, inv, shape_names, _ = self._shape_groups(params, loc_name)
        grid = np.linspace(0.0, 2.0 * np.pi, ngrid, endpoint=False)
        peaks = np.empty(len(uniq))
        for i, row in enumerate(uniq):
            pr = {loc_name: np.zeros(ngrid)}
            pr.update({nm: np.full(ngrid, float(row[k]))
                       for k, nm in enumerate(shape_names)})
            ll = np.asarray(self._loglik_values(grid, pr), dtype=float)
            j = int(np.argmax(ll))
            y0, y1, y2 = ll[(j - 1) % ngrid], ll[j], ll[(j + 1) % ngrid]
            denom = y0 - 2.0 * y1 + y2
            # parabolic vertex value (>= the grid max when denom < 0, i.e. a
            # concave peak); fall back to the grid max on a flat/degenerate run
            peaks[i] = y1 - (y2 - y0) ** 2 / (4.0 * denom) if denom < 0 else y1
        return peaks[inv]

    def __repr__(self):
        links = ", ".join(repr(lnk.name) for lnk in self.links)
        return f"{self.name} (links: {links})"


class KatoJonesLL(CircularLL):
    """Kato–Jones (2015) as a general family in **disc-chart coordinates**
    (μ, γ, u₁, u₂).

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
    inverse). ``kjlss`` is the pre-built alias instance — the primary
    spelling:

        gam(["theta ~ s(x, bs='cc')",   # LP1 → μ        (tanhalf)
             "      ~ z",               # LP2 → logit γ  (logit)
             "      ~ 1",               # LP3 → u₁       (identity)
             "      ~ 1"],              # LP4 → u₂       (identity)
            data, family=kjlss, method="REML")
    """

    def __init__(self, links=None, name=None):
        super().__init__(katojones, links, name=name)

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

    def _pearson_var(self, params):
        """Kato–Jones Pearson variance: the wrapped-Cauchy first-moment scale
        ``(1 − γ²)/2`` (γ the fitted concentration LP), matching circlss. The
        chart-coordinate LPs (γ, u₁, u₂) are not book parameters of the
        distribution, so the generic centered-moment path cannot apply here."""
        gamma = np.asarray(params["gamma"], dtype=float)
        return 0.5 * (1.0 - gamma * gamma)

    def _null_params(self, y):
        mu0, g0, rho0, lam0 = self.dist.fit(y, method="moments")
        u1, u2 = self.dist.disc_chart_inverse(g0, rho0, lam0)
        # Cap the chart-coordinate magnitude. disc_chart_inverse keeps the disc
        # coordinate finite (|v| <= vmax) but u = v/sqrt(1-|v|^2) can still
        # reach ~1e4 when the marginal moment fit lands on the Theorem-1
        # feasibility circle — which it does whenever mu is covariate-driven
        # (the pooled 2nd moment of angles with a swinging mean inflates to the
        # boundary). |u| ~ 1e4 makes gam.fit5's penalized Hessian indefinite
        # ("indefinite penalized likelihood"). |u| <= 8 (the circlss
        # ``initialize`` bound) is solver-agnostic and harmless when the fit is
        # sane — the start only needs the right basin, EFS refines from there.
        u = np.array([float(u1), float(u2)])
        nrm = float(np.hypot(*u))
        if nrm > 8.0:
            u *= 8.0 / nrm
        return np.array([mu0, g0, u[0], u[1]])


# Distributions whose regression (LP) coordinates are not their own logpdf
# parameters, keyed by distribution name → the dedicated family class.
# Consulted by `_circular_family` (the `circ_gam` family auto-route) and by the
# `CircularLL.__init__` fail-fast guard's error hint.
_LL_BY_DIST = {"katojones": KatoJonesLL}


def _circular_family(dist) -> CircularLL:
    """Wrap a regression-ready distribution in its general-family class:
    plain :class:`CircularLL` for every family whose LP coordinates are its
    own parameters, the dedicated subclass otherwise (``katojones`` →
    :class:`KatoJonesLL`, which owns the disc-chart translation)."""
    cls = _LL_BY_DIST.get(getattr(dist, "name", None))
    return cls() if cls is not None else CircularLL(dist)


class CircularContinuous(rv_continuous):
    """
    Base class for circular distributions with fixed loc=0 and scale=1.

    Notes
    -----
    - ``loc``/``scale`` are intentionally removed from the user-facing API; the
      named parameters of each distribution already encode location/shape.
    - Circular data are assumed to be pre-wrapped to ``[0, 2π)``; for
      convenience every public method wraps angular inputs to that support.
    - Cumulative methods remain circular/periodic under this wrapping
      (``cdf(θ + 2π) == cdf(θ)``).
    """

    _loc_default = 0.0
    _scale_default = 1.0

    def __init__(
        self,
        momtype=1,
        a=None,
        b=None,
        *,
        support=None,
        xtol=1e-14,
        badvalue=None,
        name=None,
        longname=None,
        shapes=None,
        seed=None,
    ):
        if support is not None:
            support_a, support_b = support
            if a is None:
                a = support_a
            if b is None:
                b = support_b
        if a is None:
            a = 0.0
        if b is None:
            b = 2 * np.pi

        super().__init__(
            momtype=momtype,
            a=a,
            b=b,
            xtol=xtol,
            badvalue=badvalue,
            name=name,
            longname=longname,
            shapes=shapes,
            seed=seed,
        )

        self._circular_arg_wrapped = False
        self._wrap_arg_parsers()
        self._lower_bound, self._period = self._compute_period()
        self._normalization_cache = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_normalization_cache(self):
        cache = getattr(self, "_normalization_cache", None)
        if cache is None:
            cache = {}
            self._normalization_cache = cache
        return cache

    def _clear_normalization_cache(self):
        self._normalization_cache = {}

    def _wrap_arg_parsers(self):
        """Ensure internal arg-parsing keeps loc/scale fixed to defaults."""
        if getattr(self, "_circular_arg_wrapped", False):
            return

        for attr in ("_parse_args", "_parse_args_rvs", "_parse_args_stats"):
            original = getattr(self, attr)

            def wrapper(this, *args, __orig=original, __name=attr, **kwargs):
                clean_kwargs = this._clean_loc_scale_kwargs(kwargs, caller=__name)
                return __orig(*args, **clean_kwargs)

            setattr(self, attr, types.MethodType(wrapper, self))

        self._circular_arg_wrapped = True

    def _compute_period(self):
        try:
            lower = float(self.a)
            upper = float(self.b)
        except (TypeError, ValueError):
            return None, None

        period = upper - lower
        if not np.isfinite(period) or period <= 0:
            return None, None
        return lower, period

    def _wrap_angles(self, values):
        if self._period is None or self._lower_bound is None:
            return values

        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return values

        if arr.size == 0:
            return arr

        wrapped = np.mod(arr - self._lower_bound, self._period) + self._lower_bound
        upper_bound = self._lower_bound + self._period
        if np.isfinite(upper_bound):
            tol = np.finfo(float).eps * max(1.0, abs(upper_bound))
            if np.isscalar(values):
                if np.isclose(values, upper_bound, rtol=0.0, atol=tol):
                    return upper_bound
            else:
                mask = np.isclose(arr, upper_bound, rtol=0.0, atol=tol)
                if np.any(mask):
                    wrapped = wrapped.copy()
                    wrapped[mask] = upper_bound
        if np.isscalar(values):
            return float(wrapped)
        return wrapped

    def _init_rng(self, random_state):
        """
        Normalize the ``random_state`` argument to a NumPy ``Generator``.

        Accepts integers, ``RandomState`` instances, ``Generator`` objects, or
        ``None`` (in which case the distribution's cached generator is used).
        """
        candidate = random_state if random_state is not None else getattr(self, "_random_state", None)

        if isinstance(candidate, np.random.Generator):
            return candidate

        if isinstance(candidate, np.random.RandomState):
            seed = candidate.randint(0, 2**32)
            generator = np.random.default_rng(seed)
            if random_state is None:
                self._random_state = generator
            return generator

        if candidate is None:
            generator = np.random.default_rng()
            self._random_state = generator
            return generator

        try:
            generator = np.random.default_rng(candidate)
        except TypeError as err:  # pragma: no cover - defensive branch
            raise TypeError(
                "random_state must be None, an int seed, RandomState, or Generator."
            ) from err

        if random_state is None:
            self._random_state = generator

        return generator

    def _prepare_call_kwargs(self, kwargs, caller):
        if not kwargs:
            return {}
        return self._clean_loc_scale_kwargs(dict(kwargs), caller=caller)

    def _separate_shape_parameters(self, args, kwargs, caller):
        """
        Split positional/keyword shape parameters from kwargs for functions that
        delegate to SciPy helpers lacking keyword support (e.g. ``expect``).
        """
        if not kwargs:
            return tuple(args), {}

        remaining_kwargs = dict(kwargs)
        shape_args = list(args)

        shapespec = getattr(self, "shapes", None)
        if shapespec:
            shape_names = [name.strip() for name in shapespec.split(",") if name.strip()]
            for idx, name in enumerate(shape_names):
                if name not in remaining_kwargs:
                    continue
                value = remaining_kwargs.pop(name)
                if idx < len(shape_args):
                    existing = shape_args[idx]
                    try:
                        equal = np.allclose(existing, value)
                    except Exception:
                        equal = existing == value
                    if not equal:
                        raise TypeError(
                            f"{self._dist_name(caller)} received conflicting values for `{name}`."
                        )
                else:
                    shape_args.append(value)

        return tuple(shape_args), remaining_kwargs

    def _clean_loc_scale_kwargs(self, kwargs, *, caller):
        if not kwargs:
            return kwargs

        cleaned = kwargs
        mutated = False

        if "loc" in kwargs:
            loc_val = kwargs["loc"]
            if not self._is_default_value(loc_val, self._loc_default):
                raise TypeError(
                    f"{self._dist_name(caller)} does not support a free `loc` parameter."
                )
            cleaned = dict(cleaned) if not mutated else cleaned
            cleaned.pop("loc", None)
            mutated = True

        if "scale" in kwargs:
            scale_val = kwargs["scale"]
            if not self._is_default_value(scale_val, self._scale_default):
                raise TypeError(
                    f"{self._dist_name(caller)} does not support a free `scale` parameter."
                )
            if not mutated:
                cleaned = dict(cleaned)
                mutated = True
            cleaned.pop("scale", None)
            mutated = True

        forbidden_aliases = ("floc", "fscale", "fix_loc", "fix_scale")
        for alias in forbidden_aliases:
            if alias in kwargs:
                raise TypeError(
                    f"{self._dist_name(caller)} does not support `{alias}`; the distribution fixes location/scale."
                )

        return cleaned if mutated else kwargs

    def _is_default_value(self, value, default):
        try:
            arr = np.asarray(value)
        except Exception:  # pragma: no cover - defensive
            return False
        if arr.size == 0:
            return True
        try:
            return np.allclose(arr, default)
        except TypeError:  # pragma: no cover - fallback if casting fails
            return False

    def _dist_name(self, caller: str) -> str:
        dist_name = getattr(self, "name", None)
        if dist_name:
            return f"{dist_name}.{caller}"
        return f"{self.__class__.__name__}.{caller}"

    def _normalization_cache_key(self, *params):
        key_components = []
        for param in params:
            try:
                arr = np.asarray(param, dtype=float)
            except (TypeError, ValueError):
                return None
            if arr.ndim > 1 or arr.size > 1:
                return None
            try:
                scalar = arr.item() if isinstance(arr, np.ndarray) else float(arr)
            except (TypeError, ValueError):
                try:
                    scalar = float(arr)
                except (TypeError, ValueError):
                    return None
            key_components.append(float(scalar))
        return tuple(key_components)

    def _get_cached_normalizer(self, compute, *params):
        key = self._normalization_cache_key(*params)
        if key is None:
            return compute()
        cache = self._get_normalization_cache()
        if key not in cache:
            cache[key] = compute()
        return cache[key]

    def freeze(self, *args, **kwds) -> "CircularContinuousFrozen":
        """
        Return a frozen circular distribution while enforcing fixed loc/scale.
        """
        call_kwargs = self._prepare_call_kwargs(kwds, "freeze")
        return CircularContinuousFrozen(self, *args, **call_kwargs)

    __call__ = freeze

    # ------------------------------------------------------------------
    # Public overrides
    # ------------------------------------------------------------------
    def pdf(self, x, *args, **kwargs):
        call_kwargs = self._prepare_call_kwargs(kwargs, "pdf")
        return super().pdf(self._wrap_angles(x), *args, **call_kwargs)

    def logpdf(self, x, *args, **kwargs):
        call_kwargs = self._prepare_call_kwargs(kwargs, "logpdf")
        return super().logpdf(self._wrap_angles(x), *args, **call_kwargs)

    def cdf(self, x, *args, **kwargs):
        call_kwargs = self._prepare_call_kwargs(kwargs, "cdf")
        return super().cdf(self._wrap_angles(x), *args, **call_kwargs)

    def logcdf(self, x, *args, **kwargs):
        call_kwargs = self._prepare_call_kwargs(kwargs, "logcdf")
        return super().logcdf(self._wrap_angles(x), *args, **call_kwargs)

    def sf(self, x, *args, **kwargs):
        call_kwargs = self._prepare_call_kwargs(kwargs, "sf")
        return super().sf(self._wrap_angles(x), *args, **call_kwargs)

    def logsf(self, x, *args, **kwargs):
        call_kwargs = self._prepare_call_kwargs(kwargs, "logsf")
        return super().logsf(self._wrap_angles(x), *args, **call_kwargs)

    def nnlf(self, theta, x):
        return super().nnlf(theta, self._wrap_angles(x))

    def fit(self, data, *args, **kwds):
        kwds = self._sanitize_fit_kwargs(kwds)
        wrapped_data = self._wrap_angles(data)
        return super().fit(wrapped_data, *args, **kwds)

    def fit_loc_scale(self, *args, **kwargs):  # pragma: no cover - API guard
        raise NotImplementedError(
            "Circular distributions have fixed location and scale; use `fit` for shape parameters only."
        )

    def _sanitize_fit_kwargs(self, kwds):
        if not kwds:
            kwds = {}
        else:
            kwds = dict(kwds)

        # Reject attempts to seed loc/scale with non-default values.
        for key, default in (("loc", self._loc_default), ("scale", self._scale_default)):
            if key in kwds:
                if not self._is_default_value(kwds[key], default):
                    raise TypeError(
                        f"{self._dist_name('fit')} fixes `{key}` to {default}; remove the argument."
                    )
                kwds.pop(key)

        for key in ("fix_loc", "fix_scale"):
            if key in kwds:
                raise TypeError(
                    f"{self._dist_name('fit')} does not expose `{key}`; the distribution is already fixed."
                )

        for key, default in (("floc", self._loc_default), ("fscale", self._scale_default)):
            if key in kwds:
                if not self._is_default_value(kwds[key], default):
                    raise TypeError(
                        f"{self._dist_name('fit')} requires `{key}` == {default}."
                    )
                kwds.pop(key)

        kwds["floc"] = self._loc_default
        kwds["fscale"] = self._scale_default
        return kwds

    def _attach_methods(self):  # pragma: no cover - mirrors parent for pickling
        super()._attach_methods()
        # Reapply wrappers; _attach_methods is used during unpickling.
        self._circular_arg_wrapped = False
        self._wrap_arg_parsers()

    def _wrap_direction(self, angle: float) -> float:
        """
        Wrap a direction onto the distribution's support if known, otherwise [0, 2π).
        """
        if self._lower_bound is not None and self._period is not None:
            return float(self._wrap_angles(angle))
        return float(angmod(angle))

    # ------------------------------------------------------------------
    # Numeric integration helpers
    # ------------------------------------------------------------------
    def _cdf_integral(
        self,
        x,
        integrand,
        params,
        *,
        lower=None,
        upper=None,
        epsabs=1e-9,
        epsrel=1e-9,
        limit=200,
    ):
        """
        Numerically integrate a one-dimensional PDF to obtain CDF values.

        Evaluates the cumulative integral of ``integrand`` from ``lower`` to each
        point in ``x``, reusing work across sorted evaluation points to minimise
        the number of quadrature calls.
        """
        if np.isscalar(x):
            x_vals = np.array([float(x)], dtype=float)
            scalar_input = True
        else:
            x_arr = np.asarray(x, dtype=float)
            x_vals = x_arr.ravel()
            scalar_input = False
            original_shape = x_arr.shape

        if x_vals.size == 0:
            if scalar_input:
                return float()
            return np.empty(original_shape, dtype=float)

        params = tuple(params)
        lower_bound = float(self.a if lower is None else lower)
        upper_bound = float(self.b if upper is None else upper)

        def scalar_integrand(value, *args):
            out = integrand(value, *args)
            arr = np.asarray(out, dtype=float)
            if arr.ndim == 0:
                return float(arr)
            return float(arr.reshape(-1)[0])

        results = np.zeros_like(x_vals, dtype=float)
        sorted_indices = np.argsort(x_vals, kind="mergesort")
        sorted_vals = x_vals[sorted_indices]

        cumulative = 0.0
        current = lower_bound

        for order_idx, orig_idx in enumerate(sorted_indices):
            value = float(sorted_vals[order_idx])

            if not np.isfinite(value):
                results[orig_idx] = np.nan
                continue

            if value <= lower_bound:
                results[orig_idx] = 0.0
                continue

            clipped = min(value, upper_bound)
            if clipped > current + 1e-15:
                segment, _ = quad(
                    scalar_integrand,
                    current,
                    clipped,
                    args=params,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                )
                cumulative += segment
                current = clipped

            if value >= upper_bound:
                cumulative = 1.0
                current = upper_bound
                results[orig_idx] = 1.0
            else:
                results[orig_idx] = cumulative

        results = np.clip(results, 0.0, 1.0)

        if scalar_input:
            return float(results[0])
        return results.reshape(original_shape)

    def _cdf_from_pdf(self, x, *params, **quad_kwargs):
        """Convenience wrapper around `_cdf_integral` using ``self._pdf``."""
        return self._cdf_integral(
            x,
            self._pdf,
            params,
            lower=quad_kwargs.pop("lower", None),
            upper=quad_kwargs.pop("upper", None),
            epsabs=quad_kwargs.pop("epsabs", 1e-9),
            epsrel=quad_kwargs.pop("epsrel", 1e-9),
            limit=quad_kwargs.pop("limit", 200),
        )

    # ------------------------------------------------------------------
    # Circular descriptive helpers
    # ------------------------------------------------------------------
    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """
        Circular (trigonometric) moment m_p = E[e^{i p Θ}] = C_p + i S_p.

        Falls back to numeric evaluation via ``self.expect``; subclasses may
        override with closed-form expressions.
        """
        shape_args, non_shape_kwargs = self._separate_shape_parameters(args, kwargs, "trig_moment")
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        C_p = float(
            np.asarray(self.expect(lambda x: np.cos(p * x), args=shape_args, **call_kwargs))
        )
        S_p = float(
            np.asarray(self.expect(lambda x: np.sin(p * x), args=shape_args, **call_kwargs))
        )
        return complex(C_p, S_p)

    def r(self, *args, **kwargs) -> float:
        """Mean resultant length R = |m₁|."""
        m1 = self.trig_moment(1, *args, **kwargs)
        return float(np.clip(abs(m1), 0.0, 1.0))

    def mean(self, *args, **kwargs) -> float:
        """Circular mean direction μ = arg(m₁).

        Returns ``nan`` when the mean resultant length R = |m₁| is ≈ 0
        (within 1e-12): for an isotropic law — the circular uniform, or
        any family at its uniform limit — every direction is equally
        central, so no mean direction exists. This is the standard
        circular-statistics convention, not an error.
        """
        m1 = self.trig_moment(1, *args, **kwargs)
        R = np.clip(abs(m1), 0.0, 1.0)
        if np.isclose(R, 0.0, atol=1e-12):
            return float("nan")
        return self._wrap_direction(np.angle(m1))

    def median(self, *args, **kwargs) -> float:
        """Circular median (50% quantile)."""
        call_kwargs = self._prepare_call_kwargs(kwargs, "median")
        return float(super().ppf(0.5, *args, **call_kwargs))

    def var(self, *args, **kwargs) -> float:
        """Circular variance V = 1 - R."""
        return float(1.0 - self.r(*args, **kwargs))

    def std(self, *args, **kwargs) -> float:
        """Circular standard deviation s = sqrt(-2 ln R)."""
        R = np.clip(self.r(*args, **kwargs), 0.0, 1.0)
        if np.isclose(R, 0.0, atol=1e-12):
            return float("inf")
        return float(np.sqrt(max(0.0, -2.0 * np.log(np.clip(R, np.finfo(float).tiny, 1.0)))))

    def dispersion(self, *args, **kwargs) -> float:
        """Circular dispersion δ̂ = (1 - ρ₂) / (2 ρ₁²)."""
        m1 = self.trig_moment(1, *args, **kwargs)
        r1 = np.clip(abs(m1), 0.0, 1.0)
        if np.isclose(r1, 0.0, atol=1e-12):
            return float("inf")
        m2 = self.trig_moment(2, *args, **kwargs)
        r2 = np.clip(abs(m2), 0.0, 1.0)
        return float((1.0 - r2) / (2.0 * r1 * r1))

    def skewness(self, *args, **kwargs) -> float:
        """Pewsey-style circular skewness."""
        m1 = self.trig_moment(1, *args, **kwargs)
        u1 = np.angle(m1)
        r1 = np.clip(abs(m1), 0.0, 1.0)
        m2 = self.trig_moment(2, *args, **kwargs)
        u2 = np.angle(m2)
        r2 = np.clip(abs(m2), 0.0, 1.0)

        denom_base = max(0.0, 1.0 - r1)
        if np.isclose(denom_base, 0.0, atol=1e-12):
            return float("nan")
        denom = denom_base**1.5
        return float((r2 * np.sin(u2 - 2.0 * u1)) / denom)

    def kurtosis(self, *args, **kwargs) -> float:
        """Pewsey-style circular kurtosis."""
        m1 = self.trig_moment(1, *args, **kwargs)
        u1 = np.angle(m1)
        r1 = np.clip(abs(m1), 0.0, 1.0)
        m2 = self.trig_moment(2, *args, **kwargs)
        u2 = np.angle(m2)
        r2 = np.clip(abs(m2), 0.0, 1.0)

        denom_base = max(0.0, 1.0 - r1)
        if np.isclose(denom_base, 0.0, atol=1e-12):
            return float("nan")
        denom = denom_base**2
        return float((r2 * np.cos(u2 - 2.0 * u1) - r1**4) / denom)

    def stats(self, *args, **kwargs):
        """Convenience bundle of circular descriptive statistics."""
        m1 = self.trig_moment(1, *args, **kwargs)
        r1 = np.clip(abs(m1), 0.0, 1.0)
        u1 = np.angle(m1)

        r1_is_zero = np.isclose(r1, 0.0, atol=1e-12)
        mean_val = float("nan") if r1_is_zero else self._wrap_direction(u1)

        m2 = self.trig_moment(2, *args, **kwargs)
        r2 = np.clip(abs(m2), 0.0, 1.0)
        u2 = np.angle(m2)

        denom_base = max(0.0, 1.0 - r1)
        if np.isclose(denom_base, 0.0, atol=1e-12):
            skew = float("nan")
            kurt = float("nan")
        else:
            skew = float((r2 * np.sin(u2 - 2.0 * u1)) / (denom_base**1.5))
            kurt = float((r2 * np.cos(u2 - 2.0 * u1) - r1**4) / (denom_base**2))

        std_val = float("inf") if r1_is_zero else float(
            np.sqrt(max(0.0, -2.0 * np.log(np.clip(r1, np.finfo(float).tiny, 1.0))))
        )
        dispersion_val = float("inf") if r1_is_zero else float((1.0 - r2) / (2.0 * r1 * r1))

        return {
            "mean": mean_val,
            "median": self.median(*args, **kwargs),
            "r": float(r1),
            "var": float(1.0 - r1),
            "std": std_val,
            "dispersion": dispersion_val,
            "skewness": skew,
            "kurtosis": kurt,
        }


class CircularContinuousFrozen(rv_continuous_frozen):
    """Frozen circular distribution exposing circular descriptive helpers."""

    def _call_dist_method(self, name, *args, **kwargs):
        call_kwargs = dict(self.kwds)
        call_kwargs.update(kwargs)
        call_args = self.args + args
        return getattr(self.dist, name)(*call_args, **call_kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        call_kwargs = dict(self.kwds)
        call_kwargs.update(kwargs)
        call_args = self.args + args
        return self.dist.trig_moment(p, *call_args, **call_kwargs)

    def r(self, *args, **kwargs) -> float:
        return self._call_dist_method("r", *args, **kwargs)

    def dispersion(self, *args, **kwargs) -> float:
        return self._call_dist_method("dispersion", *args, **kwargs)

    def skewness(self, *args, **kwargs) -> float:
        return self._call_dist_method("skewness", *args, **kwargs)

    def kurtosis(self, *args, **kwargs) -> float:
        return self._call_dist_method("kurtosis", *args, **kwargs)

    def stats(self, *args, **kwargs):
        return self._call_dist_method("stats", *args, **kwargs)


############################
## Symmetric Distribtions ##
############################


class circularuniform_gen(CircularContinuous):
    """Continuous Circular Uniform Distribution

    ![circularuniform](../images/circ-mod-circularuniform.png)

    Methods
    -------
    pdf(x)
        Probability density function.

    logpdf(x)
        Logarithm of the probability density function.

    cdf(x)
        Cumulative distribution function.

    ppf(q)
        Percent-point function (inverse of CDF).

    rvs(size, random_state)
        Random variates.
    """

    def _pdf(self, x):
        return 1 / (2 * np.pi)

    def pdf(self, x, *args, **kwargs):
        r"""
        Probability density function of the Circular Uniform distribution.

        $$
        f(\theta) = \frac{1}{2\pi}
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """
        return super().pdf(x, *args, **kwargs)

    def _logpdf(self, x):
        return np.full_like(np.asarray(x, dtype=float), -np.log(2.0 * np.pi))

    def logpdf(self, x, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the Circular
        Uniform distribution: the constant $-\log 2\pi$.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, *args, **kwargs)

    def _cdf(self, x):
        return x / (2 * np.pi)

    def cdf(self, x, *args, **kwargs):
        r"""
        Cumulative distribution function of the Circular Uniform distribution.

        $$
        F(\theta) = \frac{\theta}{2\pi}
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.

        Returns
        -------
        cdf_values : array_like
            Cumulative distribution function evaluated at `x`.
        """
        return super().cdf(x, *args, **kwargs)

    def _ppf(self, q):
        return 2 * np.pi * q

    def ppf(self, q, *args, **kwargs):
        r"""
        Percent-point function (inverse of the CDF) of the Circular Uniform distribution.

        $$
        Q(q) = F^{-1}(q) = 2\pi q, \space 0 \leq q \leq 1
        $$

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate.

        Returns
        -------
        ppf_values : array_like
            Values at the given quantiles.
        """
        return super().ppf(q, *args, **kwargs)

    def _rvs(self, size=None, random_state=None):
        rng = self._init_rng(random_state)
        return rng.uniform(0.0, 2 * np.pi, size=size)

    def rvs(self, size=None, random_state=None):
        """
        Random variate generation for the circular uniform distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Number of samples to draw. If ``None`` (default), return a single value.
        random_state : np.random.Generator, np.random.RandomState, or None, optional
            Random number generator to use. If ``None``, fall back to the
            distribution's internal generator.

        Returns
        -------
        samples : ndarray or float
            Samples drawn uniformly from the interval ``[0, 2π)``.
        """
        return self._rvs(size=size, random_state=random_state)

    def fit(self, data):
        """
        The circular uniform distribution has no free parameters to estimate,
        so calling ``fit`` is undefined. A ``NotImplementedError`` is raised to
        signal that users should rely on descriptive helpers (e.g.,
        ``circ_mean_and_r``) instead of maximum-likelihood fitting.
        """
        raise NotImplementedError(
            "circularuniform.fit() is undefined: the distribution has no parameters to estimate."
        )


circularuniform = circularuniform_gen(name="circularuniform")


class triangular_gen(CircularContinuous):
    """Triangular Distribution

    ![triangular](../images/circ-mod-triangular.png)

    Methods
    -------
    pdf(x, rho)
        Probability density function.

    logpdf(x, rho)
        Logarithm of the probability density function.

    cdf(x, rho)
        Cumulative distribution function.

    ppf(q, rho)
        Closed-form quantile (inverse CDF).

    rvs(rho, size=None, random_state=None)
        Random variates via inverse-transform using the closed-form quantile.

    fit(data, *, weights=None, method="mle" | "moments", ...)
        Fit the distribution to the data and return the parameter (rho).

    Notes
    -----
    Implementation based on Section 2.2.3 of Jammalamadaka & SenGupta (2001)
    """

    def _argcheck(self, rho):
        rho_arr = np.asarray(rho, dtype=float)
        return (rho_arr >= 0.0) & (rho_arr <= 4.0 / np.pi**2)

    def _pdf(self, x, rho):
        return (
            (4 - np.pi**2.0 * rho + 2.0 * np.pi * rho * np.abs(np.pi - x)) / 8.0 / np.pi
        )

    def pdf(self, x, rho, *args, **kwargs):
        r"""
        Probability density function of the Triangular distribution.

        $$
        f(\theta) = \frac{4 - \pi^2 \rho + 2\pi \rho |\pi - \theta|}{8\pi}
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        rho : float
            Concentratio parameter, 0 <= rho <= 4/pi^2.

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """

        return super().pdf(x, rho, *args, **kwargs)

    def _logpdf(self, x, rho):
        # the piecewise-linear density never underflows, so the log of the
        # closed form is already the stable log-density (kinks unchanged);
        # −inf only at the honest zero (x = π at ρ = 4/π²)
        with np.errstate(divide="ignore"):
            return np.log(self._pdf(x, rho))

    def logpdf(self, x, rho, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the Triangular
        distribution: the log of the closed form, $-\infty$ only where the
        density is genuinely zero (the antipode at $\rho = 4/\pi^2$).

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.
        rho : float
            Concentration parameter, 0 <= rho <= 4/pi^2.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, rho, *args, **kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """Closed-form trigonometric moment (J&S §2.2.3): the mode-at-0
        triangular law has m_p = ρ/p² for odd p and 0 for even p (real —
        the density is symmetric about 0)."""
        shape_args, non_shape_kwargs = self._separate_shape_parameters(
            args, kwargs, "trig_moment"
        )
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        (rho,) = (float(np.asarray(v, dtype=float))
                  for v in self._parse_args(*shape_args, **call_kwargs)[0])

        if not np.isscalar(p):
            raise ValueError("`p` must be an integer scalar.")
        if int(round(p)) != p:
            raise ValueError("`p` must be an integer.")
        k = abs(int(round(p)))
        if k == 0:
            return complex(1.0, 0.0)
        if k % 2 == 0:
            return complex(0.0, 0.0)
        return complex(rho / (k * k), 0.0)

    def _cdf(self, x, rho):
        x_arr = np.asarray(x, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)
        x_b, rho_b = np.broadcast_arrays(x_arr, rho_arr)

        result = np.zeros_like(x_b, dtype=float)

        # lower branch: 0 <= x <= pi
        mask_lower = (x_b >= 0.0) & (x_b <= np.pi)
        if np.any(mask_lower):
            xl = x_b[mask_lower]
            rl = rho_b[mask_lower]
            result[mask_lower] = ((4 + np.pi**2 * rl) * xl - np.pi * rl * xl**2) / (
                8 * np.pi
            )

        # upper branch: pi < x < 2pi
        mask_upper = (x_b > np.pi) & (x_b < 2 * np.pi)
        if np.any(mask_upper):
            xu = x_b[mask_upper]
            ru = rho_b[mask_upper]
            result[mask_upper] = 0.5 + (
                (4 - 3 * np.pi**2 * ru) * (xu - np.pi)
                + np.pi * ru * (xu**2 - np.pi**2)
            ) / (8 * np.pi)

        # upper tail: x >= 2pi
        result = np.where(x_b >= 2 * np.pi, 1.0, result)

        if np.ndim(result) == 0:
            return float(result)
        return result

    def cdf(self, x, rho, *args, **kwargs):
        r"""
        Cumulative distribution function of the circular triangular distribution on $[0, 2\pi)$.

        $$
        F(\theta;\,\rho)=
        \begin{cases}
        \dfrac{(4+\pi^2\rho)\,\theta - \pi\rho\,\theta^2}{8\pi}, & 0 \le \theta \le \pi,\\[6pt]
        \dfrac{1}{2} + \dfrac{(4 - 3\pi^2\rho)\,(\theta-\pi) + \pi\rho\,(\theta^2-\pi^2)}{8\pi},
            & \pi < \theta < 2\pi.
        \end{cases}
        $$

        (With $F(\theta)=0$ for $\theta<0$ and $F(\theta)=1$ for $\theta\ge 2\pi$.)

        Parameters
        ----------
        x : array_like
            Angles in radians on $[0, 2\pi)$.
        rho : float
            Concentration parameter, $0 \le \rho \le 4/\pi^2$.

        Returns
        -------
        cdf_values : array_like
            Cumulative distribution function evaluated at `x`.
        """
        return super().cdf(x, rho, *args, **kwargs)

    def _ppf(self, q, rho):
        q_arr = np.asarray(q, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)
        q_b, rho_b = np.broadcast_arrays(q_arr, rho_arr)

        result = np.empty_like(q_b, dtype=float)

        mask_zero = np.isclose(rho_b, 0.0, atol=1e-12)
        if np.any(mask_zero):
            result[mask_zero] = q_b[mask_zero] * (2 * np.pi)

        mask_general = ~mask_zero
        if np.any(mask_general):
            q_g = q_b[mask_general]
            rho_g = rho_b[mask_general]

            a_left = rho_g
            b_left = -(4 + np.pi**2 * rho_g) / np.pi
            a_right = rho_g
            b_right = (4 - 3 * np.pi**2 * rho_g) / np.pi

            res_general = np.empty_like(q_g, dtype=float)
            mask_left = q_g <= 0.5

            if np.any(mask_left):
                c_left = 8 * q_g[mask_left]
                disc_left = np.clip(
                    b_left[mask_left] ** 2 - 4 * a_left[mask_left] * c_left,
                    0.0,
                    None,
                )
                res_general[mask_left] = (
                    -b_left[mask_left] - np.sqrt(disc_left)
                ) / (2 * a_left[mask_left])

            if np.any(~mask_left):
                c_right = 2 * np.pi**2 * rho_g[~mask_left] - 8 * q_g[~mask_left]
                disc_right = np.clip(
                    b_right[~mask_left] ** 2 - 4 * a_right[~mask_left] * c_right,
                    0.0,
                    None,
                )
                res_general[~mask_left] = (
                    -b_right[~mask_left] + np.sqrt(disc_right)
                ) / (2 * a_right[~mask_left])

            result[mask_general] = res_general

        np.clip(result, 0.0, 2 * np.pi - np.finfo(float).eps, out=result)
        if result.ndim == 0:
            return float(result)
        return result

    def ppf(self, q, rho, *args, **kwargs):
        r"""
        Percent-point function (quantile) of the circular triangular distribution on $[0, 2\pi)$.

        For $\rho=0$ (circular uniform):

        $$
        \operatorname{PPF}(q;0)=2\pi q.
        $$

        For $\rho>0$:

        $$
        \operatorname{PPF}(q;\rho)=
        \begin{cases}
        \dfrac{1}{2\rho}\!\left(\dfrac{4+\pi^2\rho}{\pi}
        - \sqrt{\left(\dfrac{4+\pi^2\rho}{\pi}\right)^{\!2} - 32\rho\,q}\right),
        & 0 \le q \le \tfrac{1}{2}, \\[10pt]
        \pi + \dfrac{-\,(4-\pi^2\rho) + \sqrt{(4-\pi^2\rho)^{2} + 32\pi^{2}\rho\,(q-\tfrac{1}{2})}}
        {2\pi\rho},
        & \tfrac{1}{2} < q < 1.
        \end{cases}
        $$

        Parameters
        ----------
        q : array_like
            Quantiles in $[0, 1]$.
        rho : float
            Concentration parameter, $0 \le \rho \le 4/\pi^2$.

        Returns
        -------
        ppf_values : array_like
            Quantiles (angles in radians on $[0, 2\pi)$).
        """
        return super().ppf(q, rho, *args, **kwargs)

    def _rvs(self, rho, size=None, random_state=None):
        rng = self._init_rng(random_state)
        u = rng.uniform(0.0, 1.0, size=size)
        samples = self._ppf(u, rho)
        if np.isscalar(samples):
            return float(samples)
        return np.asarray(samples, dtype=float)

    def rvs(self, rho=None, size=None, random_state=None):
        r"""
        Random variates from the circular triangular distribution on $[0, 2\pi)$.

        Sampling uses **inverse-transform** with the closed-form quantile:
        let $U \sim \mathrm{Unif}(0,1)$ and set $\theta = \operatorname{PPF}(U;\rho)$, where

        - For $\rho = 0$ (circular uniform):

        $$
        \theta = 2\pi U.
        $$

        - For $\rho > 0$ (piecewise quadratic inverse):

        $$
        \theta =
        \begin{cases}
            \dfrac{1}{2\rho}\!\left(\dfrac{4+\pi^2\rho}{\pi}
            - \sqrt{\left(\dfrac{4+\pi^2\rho}{\pi}\right)^{\!2} - 32\rho\,U}\right),
            & 0 \le U \le \tfrac{1}{2}, \\[10pt]
            \pi + \dfrac{-\,(4-\pi^2\rho) + \sqrt{(4-\pi^2\rho)^{2} + 32\pi^{2}\rho\,(U-\tfrac{1}{2})}}
            {2\pi\rho},
            & \tfrac{1}{2} < U < 1.
        \end{cases}
        $$

        Parameters
        ----------
        rho : float, optional
            Concentration, $0 \le \rho \le 4/\pi^2$. Supply explicitly or by
            freezing the distribution.
        size : int or tuple of ints, optional
            Output shape. If ``None`` (default), return a single scalar.
        random_state : int, numpy.random.Generator, numpy.random.RandomState, optional
            PRNG seed or generator. If ``None``, use the distribution's internal RNG.

        Returns
        -------
        samples : ndarray or float
            Angles in radians on $[0, 2\pi)$, with shape ``size``.

        Notes
        -----
        This is equivalent in law to R's **circular** `rtriangular` after
        shifting its output by $+\pi$ modulo $2\pi$.
        """
        rho_val = getattr(self, "rho", None) if rho is None else rho
        if rho_val is None:
            raise ValueError("'rho' must be provided.")
        return self._rvs(rho_val, size=size, random_state=random_state)

    def fit(self, data, *, weights=None, method="mle", return_info=False,
            optimizer=None):
        r"""
        Estimate the concentration parameter $\rho$ of the circular triangular law on $[0,2\pi)$.

        Methods
        -------

        mle (default):
            maximize the log-likelihood. This solves the 1-D score equation
            $\sum_i \frac{c_i}{4+\rho\,c_i}=0$ with $c_i = 2\pi\,|\,\pi-x_i\,| - \pi^2$.
            Unique solution in $[0, 4/\pi^2)$ or at a boundary.
        moments :
            closed-form $\hat\rho = \max\{0, \min\{4/\pi^2,\ \overline{\cos x}\}\}$.

        Parameters
        ----------
        data : array_like
            Sample angles (radians). Values are wrapped to $[0, 2\pi)$ internally.
        weights : array_like, optional
            Nonnegative sample weights. Broadcastable to `data`. Interpreted as frequencies.
        method : {"mle","moments"}, optional
            Estimation method (see above).
        return_info : bool, optional
            If True, also return a dict with diagnostics (loglik, se, n_effective, method).
        optimizer : str, optional
            Accepted for cross-family ``fit`` signature uniformity and
            ignored: the MLE is the exact root of a strictly monotone 1-D
            score equation (bracketed Brent), so there is no optimizer to
            choose.

        Returns
        -------
        rho_hat : float
            Estimated concentration $\hat\rho \in [0, 4/\pi^2]$.
        info : dict, optional
            Returned only if `return_info=True`. Contains keys:
            {"loglik", "se", "n_effective", "method", "converged"}.

        Notes
        -----
        For this distribution $\mathbb{E}[\cos \Theta]=\rho$, so the method-of-moments
        estimator is simply the (weighted) mean of $\cos x$ clipped to $[0,4/\pi^2]$.
        The MLE solves a strictly monotone score equation, so bracketing root-finding
        is robust and $O(n)$ per evaluation.
        """
        del optimizer  # signature uniformity only — exact 1-D root inside
        x = np.asarray(data, dtype=float)
        x = np.mod(x, 2*np.pi)

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("weights must be nonnegative")
            w = np.broadcast_to(w, x.shape).astype(float, copy=False)

        # Effective sample size for diagnostics
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("sum of weights must be positive")
        w_norm = w / w_sum
        n_eff = w_sum**2 / np.sum(w**2)  # Kish effective n

        # Method-of-moments (always available; used as fallback/initial intuition)
        r_bar = float(np.sum(w_norm * np.cos(x)))
        rho_mom = float(np.clip(r_bar, 0.0, 4/np.pi**2))

        if method == "moments":
            rho_hat = rho_mom
            # log-likelihood at MoM (for info only)
            y = np.abs(np.pi - x)
            ll = float(np.sum(w * np.log(4 - np.pi**2 * rho_hat + 2*np.pi * rho_hat * y)) - w_sum*np.log(8*np.pi))
            # observed Fisher info for SE
            c = 2*np.pi*y - np.pi**2
            info_obs = float(np.sum(w * (c**2) / (4 + rho_hat*c)**2))
            se = (1.0 / np.sqrt(info_obs)) if info_obs > 0 else np.nan
            if return_info:
                return rho_hat, {"loglik": ll, "se": se, "n_effective": n_eff, "method": "moments", "converged": True}
            return rho_hat

        # --- MLE via monotone root of the score ---
        y = np.abs(np.pi - x)                 # in [0, π]
        c = 2*np.pi*y - np.pi**2              # in [-π^2, π^2]

        def score(rho):
            return float(np.sum(w * (c / (4.0 + rho * c))))

        # Bracket: score(ρ) is strictly decreasing on [0, ρ_max)
        rho_lo = 0.0
        rho_hi = float(4/np.pi**2) - 1e-12

        s_lo = score(rho_lo)  # = (1/4) * sum w*c
        if s_lo <= 0:         # likelihood decreasing at 0 → boundary optimum
            rho_hat = 0.0
            converged = True
        else:
            s_hi = score(rho_hi)  # tends negative if any y_i≈0
            if s_hi >= 0:
                # all mass far from π (extreme case) → boundary at ρ_max
                rho_hat = rho_hi
                converged = True
            else:
                # Unique root inside (0, ρ_max)
                rho_hat = float(brentq(score, rho_lo, rho_hi, xtol=1e-12, rtol=1e-12, maxiter=256))
                converged = True

        # Diagnostics
        ll = float(np.sum(w * np.log(4 - np.pi**2 * rho_hat + 2*np.pi * rho_hat * y)) - w_sum*np.log(8*np.pi))
        info_obs = float(np.sum(w * (c**2) / (4 + rho_hat*c)**2))
        se = (1.0 / np.sqrt(info_obs)) if info_obs > 0 else np.nan

        if return_info:
            return rho_hat, {"loglik": ll, "se": se, "n_effective": n_eff, "method": "mle", "converged": converged, "optimizer": "brentq"}
        return rho_hat


triangular = triangular_gen(name="triangular")


class cardioid_gen(_RegressionReady, CircularContinuous):
    r"""Cardioid (cosine) Distribution

    ![cardioid](../images/circ-mod-cardioid.png)

    A cosine-modulated perturbation of the circular uniform law with support on
    ``[0, 2π)``. The mean direction ``mu`` controls location, while the mean
    resultant length ``rho`` (bounded by 0.5) governs concentration. Closed-form
    expressions are used for the PDF and CDF, and quantiles are obtained by
    solving ``F(theta; mu, rho) = q`` with a safeguarded Halley--Newton iteration
    shared by ``ppf`` and ``rvs``.

    Methods
    -------
    pdf(x, mu, rho)
        Probability density function.
    logpdf(x, mu, rho)
        Logarithm of the probability density function.
    cdf(x, mu, rho)
        Cumulative distribution function.
    ppf(q, mu, rho)
        Percent-point function (inverse CDF).
    rvs(mu, rho, size=None, random_state=None)
        Random variates via inverse transform using the quantile solver.
    fit(data, *args, **kwargs)
        Estimate ``(mu, rho)`` via method-of-moments or maximum likelihood.

    Notes
    -----
    Implementation based on Section 4.3.4 of Pewsey et al. (2013).
    """

    # --- regression overlay (read by the regression engine
    # only). Book names mu/rho are preserved; ρ is the mean resultant length,
    # bounded in (0, ½), so its default link is the scaled logit. ---
    param_roles = {"mu": "location", "rho": "concentration"}
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): keep rho off its 1/2 wall, where 1/P^2 blows the Hessian up.
    degen_penalty = (_degen_boundary_upper("rho", 0.5),)
    default_links = {"location": "tanhalf", "concentration": "logit_half"}

    def _concentration_start(self, Rbar):
        """Closed-form concentration start from the mean resultant ``Rbar``
        for the regression null model: ``rho = Rbar`` (the cardioid moment
        estimator, E[cos(y-mu)] = rho), clamped strictly below the 1/2 bound —
        the circlss ``initialize`` convention (see CircularLL._null_params)."""
        return float(np.clip(Rbar, 0.01, 0.49))

    # The log-density is ℓ = log P − log 2π with P = 1 + 2ρ cos(θ−μ): the
    # same −log-quadratic pattern as the wrapped Cauchy's −log D, but P is
    # *linear* in ρ (P_ρρ = 0), so the derivative table below is even
    # sparser. All pure numpy, broadcasting over per-observation parameter
    # arrays. Likelihood hazard: at
    # ρ → ½ the density touches 0 at the antimode (θ−μ = π), where ℓ and
    # every derivative diverge — the logit_half link keeps ρ interior, but
    # data at the antimode still produce −∞/large scores.

    def dlogpdf(self, x, mu, rho):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        $$\frac{\partial\ell}{\partial\mu} = \frac{2\rho\sin(\theta-\mu)}{P},
          \qquad
          \frac{\partial\ell}{\partial\rho} = \frac{2\cos(\theta-\mu)}{P},
          \qquad P = 1 + 2\rho\cos(\theta-\mu).$$

        Vectorizes over per-observation ``mu``/``rho`` arrays. Returns a
        book-named dict ``{"mu": …, "rho": …}``.
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        rho = np.asarray(rho, dtype=float)
        d = x - mu
        s, c = np.sin(d), np.cos(d)
        w = 1.0 / (1.0 + 2.0 * rho * c)
        return {"mu": 2.0 * rho * s * w, "rho": 2.0 * c * w}

    def d2logpdf(self, x, mu, rho):
        r"""Second derivatives of ``logpdf`` (l2) — unique unordered pairs.

        $$\ell_{\mu\mu} = -\frac{2\rho(cP + 2\rho s^2)}{P^2},\quad
          \ell_{\mu\rho} = \frac{2s}{P^2},\quad
          \ell_{\rho\rho} = -\frac{4c^2}{P^2},$$

        with $s = \sin(\theta-\mu)$, $c = \cos(\theta-\mu)$ (the ℓ_{μρ}
        numerator collapses because $P - 2\rho c = 1$).
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        rho = np.asarray(rho, dtype=float)
        d = x - mu
        s, c = np.sin(d), np.cos(d)
        P = 1.0 + 2.0 * rho * c
        w2 = 1.0 / (P * P)
        return {
            ("mu", "mu"): -2.0 * rho * (c * P + 2.0 * rho * s * s) * w2,
            ("mu", "rho"): 2.0 * s * w2,
            ("rho", "rho"): -4.0 * c * c * w2,
        }

    def d3logpdf(self, x, mu, rho):
        r"""Third derivatives of ``logpdf`` (l3) — unique unordered triples,
        via ``(\log P)_{abc} = P_{abc}/P − (P_{ab}P_c + P_{ac}P_b + P_{bc}P_a)
        /P² + 2P_aP_bP_c/P³`` (the ``+log P`` twin of the wrapped Cauchy's
        ``−log D`` block, hence the flipped structural signs). ``P = 1 + 2ρ
        cos`` is *linear* in ρ, so every ``P``-partial with two or more
        ρ-derivatives vanishes (``P_{ρρ} = 0``).
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        rho = np.asarray(rho, dtype=float)
        d = x - mu
        s, c = np.sin(d), np.cos(d)
        P = 1.0 + 2.0 * rho * c
        w, w2, w3 = 1.0 / P, 1.0 / P**2, 1.0 / P**3
        Pm, Pr = 2.0 * rho * s, 2.0 * c
        Pmm, Pmr = -2.0 * rho * c, 2.0 * s        # Prr = 0
        Pmmm, Pmmr = -2.0 * rho * s, -2.0 * c     # Pmrr = Prrr = 0
        return {
            ("mu", "mu", "mu"): Pmmm * w - 3.0 * Pmm * Pm * w2 + 2.0 * Pm**3 * w3,
            ("mu", "mu", "rho"): Pmmr * w
            - (Pmm * Pr + 2.0 * Pmr * Pm) * w2
            + 2.0 * Pm * Pm * Pr * w3,
            ("mu", "rho", "rho"): -2.0 * Pmr * Pr * w2 + 2.0 * Pm * Pr * Pr * w3,
            ("rho", "rho", "rho"): 2.0 * Pr**3 * w3,
        }

    def d4logpdf(self, x, mu, rho):
        r"""Fourth derivatives of ``logpdf`` (l4) — unique unordered
        quadruples, from the order-4 ``log P`` partition formula; ``P``'s only
        nonzero quartic-relevant partials are ``P_{μμμμ}`` and ``P_{μμμρ}``
        (linear in ρ). Completes the contract to full-Newton depth (hea
        ``available_derivs = 2``).
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        rho = np.asarray(rho, dtype=float)
        d = x - mu
        s, c = np.sin(d), np.cos(d)
        P = 1.0 + 2.0 * rho * c
        w, w2, w3, w4 = 1.0 / P, 1.0 / P**2, 1.0 / P**3, 1.0 / P**4
        Pm, Pr = 2.0 * rho * s, 2.0 * c
        Pmm, Pmr = -2.0 * rho * c, 2.0 * s
        Pmmm, Pmmr = -2.0 * rho * s, -2.0 * c
        Pmmmm, Pmmmr = 2.0 * rho * c, -2.0 * s
        return {
            ("mu", "mu", "mu", "mu"): Pmmmm * w
            - (4.0 * Pmmm * Pm + 3.0 * Pmm * Pmm) * w2
            + 12.0 * Pmm * Pm * Pm * w3
            - 6.0 * Pm**4 * w4,
            ("mu", "mu", "mu", "rho"): Pmmmr * w
            - (Pmmm * Pr + 3.0 * Pmmr * Pm + 3.0 * Pmm * Pmr) * w2
            + 6.0 * (Pmm * Pm * Pr + Pmr * Pm * Pm) * w3
            - 6.0 * Pm**3 * Pr * w4,
            ("mu", "mu", "rho", "rho"): -(2.0 * Pmmr * Pr + 2.0 * Pmr * Pmr) * w2
            + 2.0 * (Pmm * Pr * Pr + 4.0 * Pmr * Pm * Pr) * w3
            - 6.0 * Pm * Pm * Pr * Pr * w4,
            ("mu", "rho", "rho", "rho"): 6.0 * Pmr * Pr * Pr * w3
            - 6.0 * Pm * Pr**3 * w4,
            ("rho", "rho", "rho", "rho"): -6.0 * Pr**4 * w4,
        }

    def _argcheck(self, mu, rho):
        try:
            mu_arr, rho_arr = np.broadcast_arrays(mu, rho)
        except ValueError:
            return False
        return (
            (mu_arr >= 0.0)
            & (mu_arr <= 2.0 * np.pi)
            & (rho_arr >= 0.0)
            & (rho_arr <= 0.5)
        )

    def _pdf(self, x, mu, rho):
        return (1 + 2 * rho * np.cos(x - mu)) / 2.0 / np.pi

    def pdf(self, x, mu, rho, *args, **kwargs):
        r"""
        Probability density function of the Cardioid distribution.

        $$
        f(\theta) = \frac{1}{2\pi} \left(1 + 2\rho \cos(\theta - \mu)\right), \space \rho \in [0, 1/2]
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Mean resultant length, 0 <= rho <= 0.5.

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """
        return super().pdf(x, mu, rho, *args, **kwargs)

    def _logpdf(self, x, mu, rho):
        # log1p keeps the log-density exact down to the honest zero at the
        # antipode for ρ = 1/2 (the linear-space density never underflows)
        with np.errstate(divide="ignore"):
            return np.log1p(2.0 * rho * np.cos(x - mu)) - np.log(2.0 * np.pi)

    def logpdf(self, x, mu, rho, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the Cardioid
        distribution,

        $$
        \log f(\theta) = \mathrm{log1p}\big(2\rho \cos(\theta - \mu)\big) - \log 2\pi,
        $$

        $-\infty$ only at the honest zero (the antipode at $\rho = 1/2$).

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Mean resultant length, 0 <= rho <= 0.5.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, mu, rho, *args, **kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """Closed-form trigonometric moment (book §4.3.4): the cardioid is
        a pure first-harmonic perturbation, so m₁ = ρ·e^{iμ} and m_p = 0
        for every |p| ≥ 2."""
        shape_args, non_shape_kwargs = self._separate_shape_parameters(
            args, kwargs, "trig_moment"
        )
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        mu, rho = (float(np.asarray(v, dtype=float))
                   for v in self._parse_args(*shape_args, **call_kwargs)[0])

        if not np.isscalar(p):
            raise ValueError("`p` must be an integer scalar.")
        if int(round(p)) != p:
            raise ValueError("`p` must be an integer.")
        k = int(round(p))
        if k == 0:
            return complex(1.0, 0.0)
        if abs(k) > 1:
            return complex(0.0, 0.0)
        value = rho * np.exp(1j * mu)
        return complex(np.conjugate(value)) if k < 0 else complex(value)

    def _cdf(self, x, mu, rho):
        return (x + 2 * rho * (np.sin(x - mu) + np.sin(mu))) / (2 * np.pi)

    def cdf(self, x, mu, rho, *args, **kwargs):
        r"""
        Cumulative distribution function of the Cardioid distribution.

        $$
        F(\theta) = \frac{\theta + 2\rho (\sin(\mu) + \sin(\theta - \mu))}{2\pi}
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Mean resultant length, 0 <= rho <= 0.5.

        Returns
        -------
        cdf_values : array_like
            Cumulative distribution function evaluated at `x`.
        """
        return super().cdf(x, mu, rho, *args, **kwargs)

    def _solve_inverse_cdf(self, probabilities, mu_val, rho_val):
        two_pi = 2.0 * np.pi
        probs = np.asarray(probabilities, dtype=float)

        if probs.size == 0:
            return probs.astype(float)

        sin_mu = np.sin(mu_val)

        if np.isclose(rho_val, 0.0, atol=1e-15):
            result = two_pi * probs
            if result.ndim == 0:
                value = float(result)
                return two_pi if np.isclose(float(probs), 1.0, rtol=0.0, atol=1e-12) else value
            mask_one = np.isclose(probs, 1.0, rtol=0.0, atol=1e-12)
            if np.any(mask_one):
                result = result.copy()
                result[mask_one] = two_pi
            return result

        theta = mu_val + two_pi * (probs - 0.5)
        theta = np.mod(theta, two_pi)
        theta = np.asarray(theta, dtype=float)

        tol = 1e-12
        tiny = 1e-14
        use_halley = rho_val > 0.25
        max_iter = 6 if use_halley else 3

        for iteration in range(max_iter):
            delta = (
                theta + 2.0 * rho_val * (np.sin(theta - mu_val) + sin_mu)
            ) / two_pi - probs

            converged = np.abs(delta) <= tol
            if np.all(converged):
                break

            d1 = (1.0 + 2.0 * rho_val * np.cos(theta - mu_val)) / two_pi
            d2 = (-2.0 * rho_val * np.sin(theta - mu_val)) / two_pi

            step_newton = np.divide(
                delta,
                d1,
                out=np.zeros_like(delta, dtype=float),
                where=np.abs(d1) > tiny,
            )

            if iteration == 0 and use_halley:
                denom = 2.0 * d1**2 - delta * d2
                halley_valid = np.abs(denom) > tiny
                step_halley = np.divide(
                    2.0 * delta * d1,
                    denom,
                    out=np.zeros_like(delta, dtype=float),
                    where=halley_valid,
                )
                step = np.where(halley_valid, step_halley, step_newton)
            else:
                step = step_newton

            step = np.clip(step, -np.pi, np.pi)
            theta = np.where(converged, theta, theta - step)
            theta = np.mod(theta, two_pi)

        delta = (
            theta + 2.0 * rho_val * (np.sin(theta - mu_val) + sin_mu)
        ) / two_pi - probs
        remaining = np.abs(delta) > 10.0 * tol
        if np.any(remaining):
            theta_shape = theta.shape
            theta_flat = theta.reshape(-1)
            probs_flat = probs.reshape(-1)
            remaining_flat = remaining.reshape(-1)
            target = probs_flat[remaining_flat]
            low = np.zeros_like(target)
            high = np.full_like(target, two_pi)
            for _ in range(32):
                mid = 0.5 * (low + high)
                f_mid = (
                    mid + 2.0 * rho_val * (np.sin(mid - mu_val) + sin_mu)
                ) / two_pi
                mask_low = f_mid <= target
                low = np.where(mask_low, mid, low)
                high = np.where(mask_low, high, mid)
            theta_flat[remaining_flat] = 0.5 * (low + high)
            theta = theta_flat.reshape(theta_shape)

        result = np.mod(theta, two_pi)
        if result.ndim == 0:
            value = float(result)
            return two_pi if np.isclose(float(probs), 1.0, rtol=0.0, atol=1e-12) else value

        mask_one = np.isclose(probs, 1.0, rtol=0.0, atol=1e-12)
        if np.any(mask_one):
            result = result.copy()
            result[mask_one] = two_pi
        return result

    def _ppf(self, q, mu, rho):
        mu_arr = np.asarray(mu, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)

        mu_val = float(np.mod(mu_arr.reshape(-1)[0], 2.0 * np.pi))
        rho_val = float(rho_arr.reshape(-1)[0])
        if not (0.0 <= rho_val <= 0.5):
            raise ValueError("`rho` must lie in [0, 0.5].")

        q_arr = np.asarray(q, dtype=float)
        if q_arr.size == 0:
            return q_arr.astype(float)

        flat = q_arr.reshape(-1)
        result = np.full_like(flat, np.nan, dtype=float)
        valid = np.isfinite(flat) & (flat >= 0.0) & (flat <= 1.0)
        if np.any(valid):
            solved = np.asarray(
                self._solve_inverse_cdf(flat[valid], mu_val, rho_val),
                dtype=float,
            ).reshape(-1)
            result[valid] = solved

        result = result.reshape(q_arr.shape)
        if q_arr.ndim == 0:
            return float(result)
        return result

    def ppf(self, q, mu, rho, *args, **kwargs):
        r"""
        Percent-point function (inverse CDF) of the Cardioid distribution.

        The quantile $\theta$ solves

        $$
        F(\theta) = \frac{\theta + 2\rho\bigl(\sin\mu + \sin(\theta - \mu)\bigr)}{2\pi} = q,
        $$

        on the support $[0, 2\pi]$.  The implementation applies a
        Halley--Newton iteration with adaptive clipping and a final bisection
        safeguard, ensuring robustness for large $\rho$ and quantiles
        close to the boundary.  The same solver powers ``rvs``, so sampled
        variates and tabulated quantiles are numerically consistent.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate; finite values in ``[0, 1]`` are supported.
        mu : float
            Mean direction, ``0 <= mu <= 2*pi``.
        rho : float
            Mean resultant length, ``0 <= rho <= 0.5``.

        Returns
        -------
        ppf_values : array_like
            Angles satisfying $F(\theta)=q$. Inputs outside ``[0, 1]`` are
            returned as ``nan``.
        """
        return super().ppf(q, mu, rho, *args, **kwargs)

    def _rvs(self, mu, rho, size=None, random_state=None):
        rng = self._init_rng(random_state)

        mu_arr = np.asarray(mu, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)
        if mu_arr.size != 1 or rho_arr.size != 1:
            raise ValueError("cardioid parameters must be scalar-valued.")

        mu_val = float(np.mod(mu_arr.reshape(-1)[0], 2.0 * np.pi))
        rho_val = float(rho_arr.reshape(-1)[0])
        if not (0.0 <= rho_val <= 0.5):
            raise ValueError("`rho` must lie in [0, 0.5].")

        two_pi = 2.0 * np.pi

        if np.isclose(rho_val, 0.0, atol=1e-15):
            samples = rng.uniform(0.0, two_pi, size=size)
            return float(samples) if np.isscalar(samples) else samples

        u = rng.uniform(0.0, 1.0, size=size)
        samples = self._solve_inverse_cdf(u, mu_val, rho_val)
        return float(samples) if np.isscalar(samples) else np.asarray(samples, dtype=float)


    def rvs(self, mu=None, rho=None, size=None, random_state=None):
        r"""
        Draw random variates from the Cardioid distribution.

        Each sample is obtained by inverse-transform sampling.  For a uniform
        draw $U \sim \mathcal{U}(0, 1)$, the angle $\Theta$
        satisfies

        $$
        \frac{\Theta + 2\rho\bigl(\sin\mu + \sin(\Theta - \mu)\bigr)}{2\pi} = U,
        $$

        and is computed with the safeguarded Halley--Newton solver described in
        ``ppf``.  When $\rho = 0$, the distribution degenerates to the
        circular uniform law and samples are drawn directly from ``[0, 2π)``.

        Parameters
        ----------
        mu : float, optional
            Mean direction, ``0 <= mu <= 2*pi``. Supply explicitly or by
            freezing the distribution.
        rho : float, optional
            Mean resultant length, ``0 <= rho <= 0.5``. Supply explicitly or by
            freezing the distribution.
        size : int or tuple of ints, optional
            Number of samples to draw. ``None`` (default) returns a scalar.
        random_state : np.random.Generator, np.random.RandomState, or None, optional
            Random number generator to use.

        Returns
        -------
        samples : ndarray or float
            Random variates on ``[0, 2π)``.
        """
        mu_val = getattr(self, "mu", None) if mu is None else mu
        rho_val = getattr(self, "rho", None) if rho is None else rho

        if mu_val is None or rho_val is None:
            raise ValueError("Both 'mu' and 'rho' must be provided.")

        return self._rvs(mu_val, rho_val, size=size, random_state=random_state)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        return_info=False,
        optimizer="L-BFGS-B",
        **kwargs,
    ):
        """
        Estimate ``mu`` and ``rho`` for the cardioid distribution.

        Parameters
        ----------
        data : array_like
            Sample angles (radians). Values are wrapped to ``[0, 2π)`` internally.
        weights : array_like, optional
            Non-negative weights/frequencies broadcastable to ``data``.
        method : {\"mle\", \"moments\"}, optional
            Estimation strategy. ``"moments"`` uses the first trigonometric
            moment, ``"mle"`` (default) maximises the weighted log-likelihood.
        return_info : bool, optional
            If True, also return a diagnostic dictionary.
        optimizer : str, optional
            Optimiser passed to ``scipy.optimize.minimize`` when
            ``method="mle"``.
        **kwargs :
            Additional keyword arguments forwarded to the optimiser.
        """
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        x = self._wrap_angles(np.asarray(data, dtype=float))
        if x.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, x.shape).astype(float, copy=False)

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = w_sum**2 / np.sum(w**2)

        mu_mom, r_mom = circ_mean_and_r(alpha=x, w=w)
        if not np.isfinite(mu_mom):
            mu_mom = float(0.0)
        mu_mom = float(np.mod(mu_mom, 2.0 * np.pi))
        rho_mom = float(np.clip(r_mom, 0.0, 0.5))

        def _nll(params):
            mu_param, rho_param = params
            if not (0.0 <= rho_param <= 0.5):
                return np.inf
            cos_term = np.cos(x - mu_param)
            denom = 1.0 + 2.0 * rho_param * cos_term
            if np.any(denom <= 0.0):
                return np.inf
            log_terms = np.log(denom)
            value = -np.sum(w * log_terms) + w_sum * np.log(2.0 * np.pi)
            return float(value)

        def _grad(params):
            mu_param, rho_param = params
            cos_term = np.cos(x - mu_param)
            denom = 1.0 + 2.0 * rho_param * cos_term
            mask_bad = denom <= 0.0
            if np.any(mask_bad):
                return np.array([0.0, 0.0], dtype=float)
            sin_term = np.sin(x - mu_param)
            inv = w / denom
            g_mu = -2.0 * rho_param * np.sum(inv * sin_term)
            g_rho = -2.0 * np.sum(inv * cos_term)
            return np.array([g_mu, g_rho], dtype=float)

        method = method.lower()
        if method not in {"mle", "moments"}:
            raise ValueError("`method` must be either 'mle' or 'moments'.")

        if method == "moments":
            mu_hat = self._wrap_direction(mu_mom)
            rho_hat = rho_mom
            info = {
                "method": "moments",
                "loglik": float(-_nll((mu_hat, rho_hat))),
                "n_effective": float(n_eff),
                "converged": True,
            }
        else:
            if rho_mom <= 1e-12:
                mu_hat = self._wrap_direction(mu_mom)
                rho_hat = 0.0
                info = {
                    "method": "mle",
                    "loglik": float(-_nll((mu_hat, rho_hat))),
                    "n_effective": float(n_eff),
                    "converged": True,
                    "nit": 0,
                    "message": "Degenerate start (rho≈0); returning boundary solution.",
                }
            else:
                init = np.array([mu_mom, rho_mom], dtype=float)
                bounds = [(0.0, 2.0 * np.pi), (0.0, 0.5)]
                result = minimize(
                    _nll,
                    init,
                    method=optimizer,
                    jac=_grad,
                    bounds=bounds,
                    **kwargs,
                )
                if not result.success:
                    raise RuntimeError(
                        f"cardioid.fit(method='mle') failed: {result.message}"
                    )
                mu_hat = self._wrap_direction(float(result.x[0]))
                rho_hat = float(np.clip(result.x[1], 0.0, 0.5))
                info = {
                    "method": "mle",
                    "loglik": float(-result.fun),
                    "n_effective": float(n_eff),
                    "converged": bool(result.success),
                    "nit": result.nit,
                    "grad_norm": float(np.linalg.norm(result.jac))
                    if getattr(result, "jac", None) is not None
                    else np.nan,
                    "optimizer": optimizer,
                }

        estimates = (mu_hat, rho_hat)
        if return_info:
            return estimates, info
        return estimates


cardioid = cardioid_gen(name="cardioid")
cardlss = CircularLL(cardioid, name="cardlss")


class cartwright_gen(_RegressionReady, CircularContinuous):
    """Cartwright's Power-of-Cosine Distribution

    ![cartwright](../images/circ-mod-cartwright.png)


    Methods
    -------
    pdf(x, mu, zeta)
        Probability density function.

    logpdf(x, mu, zeta)
        Logarithm of the probability density function.

    cdf(x, mu, zeta)
        Cumulative distribution function.

    ppf(q, mu, zeta)
        Percent-point function obtained by inverting the regularised incomplete beta.

    rvs(mu, zeta, size=None, random_state=None)
        Random variates via a Beta-to-angle transform consistent with the quantile.

    fit(data, *args, **kwargs)
        Estimate ``(mu, zeta)`` using moments or maximum likelihood.

    Note
    ----
    Implementation based on Section 4.3.5 of Pewsey et al. (2013)
    """

    # --- regression overlay (read by the regression engine
    # only). Book names mu/zeta are preserved; ζ > 0 is an inverse
    # peakedness, so its default link is log. ---
    param_roles = {"mu": "location", "zeta": "concentration"}
    # (no `degen_penalty`: Cartwright's power-of-cosine has a bounded shape with no
    # boundary pathology under reweighting, so the circ_mix guard is a no-op here —
    # matching circlss, where cartlss declares no `degen`.)
    default_links = {"location": "tanhalf", "concentration": "log"}

    def _concentration_start(self, Rbar):
        """Closed-form peakedness start from the mean resultant ``Rbar`` for
        the regression null model: ``zeta = (1 - Rbar)/Rbar`` from the relation
        Rbar = 1/(zeta + 1), Rbar clamped — the circlss ``initialize``
        convention (see CircularLL._null_params)."""
        rb = float(np.clip(Rbar, 0.05, 0.95))
        return (1.0 - rb) / rb

    # The log-density is ℓ = (1/ζ − 1) log 2 + 2 log Γ(1+1/ζ) − log π
    # − log Γ(1+2/ζ) + (1/ζ) L with L = log(1 + cos(θ−μ)), evaluated as
    # log 2 + 2 log|cos((θ−μ)/2)| to avoid the 1+cos cancellation near the
    # antipode. Likelihood hazard: the
    # density is exactly 0 at θ−μ = π for every ζ, where L, tan((θ−μ)/2)
    # and all derivatives diverge. In practice this makes the tanhalf
    # likelihood's generic multimodality bite hard — from the intercept-only
    # null start the fit lands in a wrong basin on roughly half of random
    # datasets. Mitigation (verified in the dev smoke harness): warm-start
    # the location LP from a wrapped-normal pilot fit (whose density has no
    # zeros), ζ from this family's own intercept-only ``fit``.

    def dlogpdf(self, x, mu, zeta):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        $$\frac{\partial\ell}{\partial\mu} = \frac{\tan((\theta-\mu)/2)}{\zeta},
          \qquad
          \frac{\partial\ell}{\partial\zeta} = \frac{-\log 2
          - 2\psi(1+1/\zeta) + 2\psi(1+2/\zeta) - L}{\zeta^2},$$

        with $\psi$ the digamma function (the ζ column matches the analytic
        gradient used by ``fit``). Vectorizes over per-observation
        ``mu``/``zeta`` arrays; returns a book-named dict.
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        zeta = np.asarray(zeta, dtype=float)
        d = x - mu
        t = np.tan(0.5 * d)
        with np.errstate(divide="ignore"):  # L → −∞ at the antipode
            L = np.log(2.0) + 2.0 * np.log(np.abs(np.cos(0.5 * d)))
        inv = 1.0 / zeta
        B = (
            -np.log(2.0)
            - 2.0 * digamma(1.0 + inv)
            + 2.0 * digamma(1.0 + 2.0 * inv)
            - L
        )
        return {"mu": t * inv, "zeta": B * inv * inv}

    def d2logpdf(self, x, mu, zeta):
        r"""Second derivatives of ``logpdf`` (l2) — unique unordered pairs.

        $$\ell_{\mu\mu} = -\frac{1+t^2}{2\zeta},\quad
          \ell_{\mu\zeta} = -\frac{t}{\zeta^2},\quad
          \ell_{\zeta\zeta} = \frac{2\psi_1(1+1/\zeta)
          - 4\psi_1(1+2/\zeta)}{\zeta^4} - \frac{2}{\zeta}\,\ell_\zeta,$$

        with $t = \tan((\theta-\mu)/2)$ and $\psi_1$ the trigamma function.
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        zeta = np.asarray(zeta, dtype=float)
        d = x - mu
        t = np.tan(0.5 * d)
        with np.errstate(divide="ignore"):
            L = np.log(2.0) + 2.0 * np.log(np.abs(np.cos(0.5 * d)))
        inv = 1.0 / zeta
        B = (
            -np.log(2.0)
            - 2.0 * digamma(1.0 + inv)
            + 2.0 * digamma(1.0 + 2.0 * inv)
            - L
        )
        l_zeta = B * inv * inv
        return {
            ("mu", "mu"): -0.5 * (1.0 + t * t) * inv,
            ("mu", "zeta"): -t * inv * inv,
            ("zeta", "zeta"): (
                2.0 * polygamma(1, 1.0 + inv)
                - 4.0 * polygamma(1, 1.0 + 2.0 * inv)
            )
            * inv**4
            - 2.0 * inv * l_zeta,
        }

    @staticmethod
    def _logZ_zeta_deriv(zeta, order):
        r"""``∂^order/∂ζ^order`` of the Cartwright log-normalizer
        ``N(ζ) = (2/ζ−1)log2 + 2 logΓ(1+1/ζ) − logΓ(1+2/ζ) − log π`` for
        ``order ∈ {3, 4}``. The log-Γ terms are ``logΓ(1+a/ζ)`` (``a = 1, 2``);
        Faà di Bruno through ``s = a/ζ`` turns their ζ-derivatives into
        ``polygamma(0..order−1, 1+a/ζ)`` weighted by the ``s``-derivatives, and
        ``∂^k_ζ(2/ζ) = 2(−1)^k k!/ζ^{k+1}``. Verified term-by-term against
        ``d2logpdf`` at order 2."""
        z = np.asarray(zeta, dtype=float)
        fact = {3: -6.0, 4: 24.0}[order]          # (−1)^order · order!
        Pb = 2.0 * fact / z ** (order + 1)        # ∂^order_ζ (2/ζ)

        def F(a):
            s1 = -a / z**2
            s2 = 2.0 * a / z**3
            s3 = -6.0 * a / z**4
            arg = 1.0 + a / z
            g0, g1, g2 = digamma(arg), polygamma(1, arg), polygamma(2, arg)
            if order == 3:
                return g2 * s1**3 + 3.0 * g1 * s1 * s2 + g0 * s3
            s4 = 24.0 * a / z**5
            g3 = polygamma(3, arg)
            return (g3 * s1**4 + 6.0 * g2 * s1 * s1 * s2
                    + g1 * (3.0 * s2 * s2 + 4.0 * s1 * s3) + g0 * s4)

        return np.log(2.0) * Pb + 2.0 * F(1.0) - F(2.0)

    def d3logpdf(self, x, mu, zeta):
        r"""Third derivatives of ``logpdf`` (l3). The log-density separates as
        ``ℓ(μ,ζ) = N(ζ) + (2/ζ)·K`` with ``K = log|cos((θ−μ)/2)|`` carrying all
        the μ-dependence, so ``∂^a_μ∂^b_ζ ℓ = [N^{(b)} if a=0] + P_b·M_a`` with
        ``P_b = ∂^b_ζ(2/ζ)`` and ``M_a = ∂^a_μ K`` (``M_1 = t/2``,
        ``M_2 = −(1+t²)/4``, ``M_3 = t(1+t²)/4``; ``t = tan((θ−μ)/2)``). Only the
        all-ζ corner needs ``N‴`` (``polygamma`` up to order 2)."""
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        zeta = np.asarray(zeta, dtype=float)
        d = x - mu
        t = np.tan(0.5 * d)
        t2 = t * t
        with np.errstate(divide="ignore"):        # K → −∞ at the antipode
            K = np.log(np.abs(np.cos(0.5 * d)))
        M1, M2, M3 = 0.5 * t, -0.25 * (1.0 + t2), 0.25 * t * (1.0 + t2)
        return {
            ("mu", "mu", "mu"): (2.0 / zeta) * M3,
            ("mu", "mu", "zeta"): (-2.0 / zeta**2) * M2,
            ("mu", "zeta", "zeta"): (4.0 / zeta**3) * M1,
            ("zeta", "zeta", "zeta"): self._logZ_zeta_deriv(zeta, 3)
            + (-12.0 / zeta**4) * K,
        }

    def d4logpdf(self, x, mu, zeta):
        r"""Fourth derivatives of ``logpdf`` (l4), from the same separable
        ``ℓ = N(ζ) + (2/ζ)K`` (see ``d3logpdf``): ``M_4 = −(1+3t²)(1+t²)/8`` and
        ``∂^b_ζ(2/ζ) = 2(−1)^b b!/ζ^{b+1}``; the all-ζ corner adds ``N⁗``
        (``polygamma`` up to order 3). Completes the contract to full-Newton
        depth (hea ``available_derivs = 2``)."""
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        zeta = np.asarray(zeta, dtype=float)
        d = x - mu
        t = np.tan(0.5 * d)
        t2 = t * t
        with np.errstate(divide="ignore"):
            K = np.log(np.abs(np.cos(0.5 * d)))
        M1, M2 = 0.5 * t, -0.25 * (1.0 + t2)
        M3, M4 = 0.25 * t * (1.0 + t2), -0.125 * (1.0 + 3.0 * t2) * (1.0 + t2)
        return {
            ("mu", "mu", "mu", "mu"): (2.0 / zeta) * M4,
            ("mu", "mu", "mu", "zeta"): (-2.0 / zeta**2) * M3,
            ("mu", "mu", "zeta", "zeta"): (4.0 / zeta**3) * M2,
            ("mu", "zeta", "zeta", "zeta"): (-12.0 / zeta**4) * M1,
            ("zeta", "zeta", "zeta", "zeta"): self._logZ_zeta_deriv(zeta, 4)
            + (48.0 / zeta**5) * K,
        }

    def _argcheck(self, mu, zeta):
        try:
            mu_arr, zeta_arr = np.broadcast_arrays(mu, zeta)
        except ValueError:
            return False
        return (mu_arr >= 0.0) & (mu_arr <= 2.0 * np.pi) & (zeta_arr > 0.0)

    @staticmethod
    def _moment_r(zeta):
        z = np.asarray(zeta, dtype=float)
        if np.any(z <= 0):
            raise ValueError("`zeta` must be positive.")
        inv = 1.0 / z
        log_term = (-1.0 + 2.0 * inv) * np.log(2.0)
        log_term += np.log(2.0)
        log_term += np.log(inv**2 / (inv + 1.0))
        log_term += gammaln(inv)
        log_term += gammaln(inv + 0.5)
        log_term -= 0.5 * np.log(np.pi)
        log_term -= gammaln(1.0 + 2.0 * inv)
        result = np.exp(log_term)
        return float(result) if np.isscalar(zeta) else result

    def _pdf(self, x, mu, zeta):
        # exp of the gammaln log form: the previous raw-gamma assembly
        # (2^{−1+1/ζ}Γ²(1+1/ζ)/(πΓ(1+2/ζ))·(1+cos φ)^{1/ζ}) returned nan
        # for ζ ≲ 0.008 in-range — Γ(1+2/ζ) overflows at 2/ζ ≳ 170
        return np.exp(self._logpdf(x, mu, zeta))

    def pdf(self, x, mu, zeta, *args, **kwargs):
        r"""
        Probability density function of the Cartwright distribution.

        $$
        f(\theta) = \frac{2^{- 1+1/\zeta} \Gamma^2(1 + 1/\zeta)}{\pi \Gamma(1 + 2/\zeta)} (1 + \cos(\theta - \mu))^{1/\zeta}
        $$

        , where $\Gamma$ is the gamma function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        zeta : float
            Shape parameter, zeta > 0.

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """

        return super().pdf(x, mu, zeta, *args, **kwargs)

    def _logpdf(self, x, mu, zeta):
        # (2/ζ − 1) log 2 + 2 log Γ(1+1/ζ) − log π − log Γ(1+2/ζ)
        # + (2/ζ) log|cos(φ/2)| — the same form the regression l-derivatives
        # differentiate (class comment above); the half-angle factorization
        # avoids the 1+cos cancellation near the antipode, where the density
        # has an honest zero for every ζ
        half = 0.5 * (np.asarray(x, dtype=float) - mu)
        with np.errstate(divide="ignore"):
            log_cos = np.log(np.abs(np.cos(half)))
        return (
            (2.0 / zeta - 1.0) * np.log(2.0)
            + 2.0 * gammaln(1.0 + 1.0 / zeta)
            - gammaln(1.0 + 2.0 / zeta)
            - np.log(np.pi)
            + (2.0 / zeta) * log_cos
        )

    def logpdf(self, x, mu, zeta, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the Cartwright
        distribution,

        $$
        \log f(\theta) = \Big(\tfrac{2}{\zeta} - 1\Big)\log 2
        + 2\log\Gamma\Big(1+\tfrac{1}{\zeta}\Big)
        - \log\Gamma\Big(1+\tfrac{2}{\zeta}\Big) - \log\pi
        + \tfrac{2}{\zeta}\log\Big|\cos\tfrac{\theta-\mu}{2}\Big|,
        $$

        finite for every $\zeta > 0$ (the raw-gamma density overflows for
        $\zeta \lesssim 0.008$) and $-\infty$ only at the honest antipodal
        zero.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        zeta : float
            Shape parameter, zeta > 0.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, mu, zeta, *args, **kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """Closed-form trigonometric moment (book §4.3.5 / J&S): with
        s = 1/ζ the centered moments are the Γ-ratio
        Γ(s+1)²/(Γ(s+1+p)Γ(s+1−p)), evaluated as the overflow-free finite
        product Π_{k=1}^{p} (s−k+1)/(s+k) — whose p = 1 case s/(s+1)
        equals the ``_moment_r`` log form by Legendre duplication."""
        shape_args, non_shape_kwargs = self._separate_shape_parameters(
            args, kwargs, "trig_moment"
        )
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        mu, zeta = (float(np.asarray(v, dtype=float))
                    for v in self._parse_args(*shape_args, **call_kwargs)[0])

        if not np.isscalar(p):
            raise ValueError("`p` must be an integer scalar.")
        if int(round(p)) != p:
            raise ValueError("`p` must be an integer.")
        k = int(round(p))
        if k == 0:
            return complex(1.0, 0.0)
        ak = abs(k)
        s = 1.0 / zeta
        alpha_p = 1.0
        for j in range(1, ak + 1):
            alpha_p *= (s - j + 1.0) / (s + j)
        value = alpha_p * np.exp(1j * ak * mu)
        return complex(np.conjugate(value)) if k < 0 else complex(value)

    @staticmethod
    def _cartwright_cumulative(phi, a, b, half_norm):
        phi_arr = np.asarray(phi, dtype=float)
        scalar_input = np.isscalar(phi_arr)
        phi_vec = np.atleast_1d(phi_arr)
        two_pi = 2.0 * np.pi
        result = np.empty_like(phi_vec, dtype=float)

        mask_lower = phi_vec <= np.pi
        if np.any(mask_lower):
            s_small = np.sin(0.5 * phi_vec[mask_lower]) ** 2
            val = betainc(a, b, np.clip(s_small, 0.0, 1.0))
            result[mask_lower] = half_norm * val

        if np.any(~mask_lower):
            phi_ref = two_pi - phi_vec[~mask_lower]
            s_large = np.sin(0.5 * phi_ref) ** 2
            val = betainc(a, b, np.clip(s_large, 0.0, 1.0))
            result[~mask_lower] = 1.0 - half_norm * val

        if scalar_input:
            return float(result[0])
        return result.reshape(phi_arr.shape)

    def _cdf(self, x, mu, zeta):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        flat = arr.reshape(-1)

        if flat.size == 0:
            return arr.astype(float)

        mu_val = _as_scalar_param(mu, "cartwright")
        zeta_val = _as_scalar_param(zeta, "cartwright")
        if zeta_val <= 0.0:
            raise ValueError("`zeta` must be positive.")

        two_pi = 2.0 * np.pi
        a = 0.5
        b = 1.0 / zeta_val + 0.5
        const = (
            2.0 ** (-1.0 + 1.0 / zeta_val)
            * gamma(1.0 + 1.0 / zeta_val) ** 2
            / (np.pi * gamma(1.0 + 2.0 / zeta_val))
        )
        beta_term = beta_fn(a, b)
        half_norm = const * (2.0 ** (1.0 / zeta_val)) * beta_term  # equals 0.5
        half_norm = float(np.clip(half_norm, np.finfo(float).tiny, None))

        phi_start = (-mu_val) % two_pi
        phi_end = (flat - mu_val) % two_pi

        H_start = self._cartwright_cumulative(np.array([phi_start]), a, b, half_norm)[0]
        H_end = self._cartwright_cumulative(phi_end, a, b, half_norm)

        cdf = np.where(
            phi_end >= phi_start,
            np.clip(H_end - H_start, 0.0, 1.0),
            1.0 - np.clip(H_start - H_end, 0.0, 1.0),
        )
        negative = cdf < 0.0
        if np.any(negative):
            cdf = np.where(negative, cdf + 1.0, cdf)
        cdf = np.clip(cdf, 0.0, 1.0)

        if arr.ndim == 0:
            value = float(cdf[0])
            return 1.0 if float(wrapped) == 2.0 * np.pi else value

        result = cdf.reshape(arr.shape)
        result[arr == 2.0 * np.pi] = 1.0
        return result

    def cdf(self, x, mu, zeta, *args, **kwargs):
        r"""
        Cumulative distribution function of the Cartwright distribution.

        The CDF is evaluated analytically via a beta-function series,
        exploiting the symmetry around the mean direction.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        zeta : float
            Shape parameter, zeta > 0.

        Returns
        -------
        cdf_values : array_like
            Cumulative distribution function evaluated at `x`.
        """
        return super().cdf(x, mu, zeta, *args, **kwargs)

    def _ppf(self, q, mu, zeta):
        mu_arr = np.asarray(mu, dtype=float)
        zeta_arr = np.asarray(zeta, dtype=float)

        mu_val = float(np.mod(mu_arr.reshape(-1)[0], 2.0 * np.pi))
        zeta_val = float(zeta_arr.reshape(-1)[0])
        if zeta_val <= 0.0:
            raise ValueError("`zeta` must be positive.")

        q_arr = np.asarray(q, dtype=float)
        if q_arr.size == 0:
            return q_arr.astype(float)

        two_pi = 2.0 * np.pi
        a = 0.5
        b = 1.0 / zeta_val + 0.5
        const = (
            2.0 ** (-1.0 + 1.0 / zeta_val)
            * gamma(1.0 + 1.0 / zeta_val) ** 2
            / (np.pi * gamma(1.0 + 2.0 / zeta_val))
        )
        half_norm = const * (2.0 ** (1.0 / zeta_val)) * beta_fn(a, b)
        half_norm = float(np.clip(half_norm, np.finfo(float).tiny, None))

        phi_start = (-mu_val) % two_pi
        H_start = self._cartwright_cumulative(np.array([phi_start]), a, b, half_norm)[0]

        flat = q_arr.reshape(-1)
        result = np.full_like(flat, np.nan, dtype=float)

        valid = np.isfinite(flat) & (flat >= 0.0) & (flat <= 1.0)
        if np.any(valid):
            q_valid = flat[valid]

            # Handle exact boundary quantiles explicitly
            close_zero = np.isclose(q_valid, 0.0, rtol=0.0, atol=1e-12)
            close_one = np.isclose(q_valid, 1.0, rtol=0.0, atol=1e-12)

            s = (H_start + q_valid) % 1.0

            phi = np.empty_like(q_valid)
            mask_lower = s <= 0.5

            if np.any(mask_lower):
                u = np.clip(s[mask_lower] / half_norm, 0.0, 1.0)
                t = betaincinv(a, b, np.clip(u, 0.0, 1.0))
                t = np.clip(t, 0.0, 1.0)
                phi[mask_lower] = 2.0 * np.arcsin(np.sqrt(t))

            if np.any(~mask_lower):
                s_upper = s[~mask_lower]
                u = np.clip((1.0 - s_upper) / half_norm, 0.0, 1.0)
                t = betaincinv(a, b, np.clip(u, 0.0, 1.0))
                t = np.clip(t, 0.0, 1.0)
                phi[~mask_lower] = two_pi - 2.0 * np.arcsin(np.sqrt(t))

            theta = (mu_val + phi) % two_pi

            if np.any(close_zero):
                theta[close_zero] = float(np.mod(mu_val + phi_start, two_pi))
            if np.any(close_one):
                theta[close_one] = two_pi

            result[valid] = theta

        result = result.reshape(q_arr.shape)
        if q_arr.ndim == 0:
            return float(result)
        return result

    def ppf(self, q, mu, zeta, *args, **kwargs):
        r"""
        Percent-point function (inverse CDF) of the Cartwright distribution.

        The quantile inversion exploits the beta integral governing the CDF.
        With
        $$
        t = \sin^2\!\left(\tfrac{1}{2}\phi\right), \qquad
        a = \tfrac{1}{2}, \qquad b = \tfrac{1}{\zeta} + \tfrac{1}{2},
        $$
        the cumulative distribution reduces to
        $$
        H(\phi) =
        \begin{cases}
        \tfrac{1}{2} I_t(a, b), & 0 \le \phi \le \pi, \\[6pt]
        1 - \tfrac{1}{2} I_t(a, b), & \pi < \phi < 2\pi,
        \end{cases}
        $$
        where $I_t$ is the regularised incomplete beta function. The inverse
        quantile solves $H(\phi) = s$ via the inverse regularised incomplete
        beta, ``betaincinv``, yielding the exact $O(1)$ mapping used here and in
        ``rvs``.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate (0 <= q <= 1).
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        zeta : float
            Shape parameter, zeta > 0.

        Returns
        -------
        ppf_values : array_like
            Angles corresponding to the given quantiles.
        """
        return super().ppf(q, mu, zeta, *args, **kwargs)

    def _rvs(self, mu, zeta, size=None, random_state=None):
        rng = self._init_rng(random_state)

        mu_arr = np.asarray(mu, dtype=float)
        zeta_arr = np.asarray(zeta, dtype=float)
        if mu_arr.size != 1 or zeta_arr.size != 1:
            raise ValueError("cartwright parameters must be scalar-valued.")

        mu_val = float(np.mod(mu_arr.reshape(-1)[0], 2.0 * np.pi))
        zeta_val = float(zeta_arr.reshape(-1)[0])
        if zeta_val <= 0.0:
            raise ValueError("`zeta` must be positive.")

        shape = ()
        if size is not None:
            if np.isscalar(size):
                shape = (int(size),)
            else:
                shape = tuple(int(dim) for dim in np.atleast_1d(size))

        beta_b = 1.0 / zeta_val + 0.5
        t = rng.beta(0.5, beta_b, size=shape)
        sqrt_t = np.sqrt(t)
        angles = 2.0 * np.arcsin(np.clip(sqrt_t, 0.0, 1.0))

        signs = np.where(rng.random(size=shape) < 0.5, -1.0, 1.0)
        theta = mu_val + signs * angles
        theta = np.mod(theta, 2.0 * np.pi)

        if theta.ndim == 0:
            return float(theta)
        return theta.reshape(shape)

    def rvs(self, mu=None, zeta=None, size=None, random_state=None):
        r"""
        Draw random variates from the Cartwright distribution.

        Sampling follows the same Beta-to-angle transform as the quantile
        function: draw $T \sim \mathrm{Beta}\!\left(\tfrac{1}{2},
        \tfrac{1}{\zeta} + \tfrac{1}{2}\right)$, map it via
        $\phi = 2\arcsin(\sqrt{T})$, then reflect $\phi$ with equal probability
        around $\mu$. This construction keeps ``rvs`` numerically consistent
        with ``ppf``.

        Parameters
        ----------
        mu : float, optional
            Mean direction, ``0 <= mu <= 2*pi``. Supply explicitly or by
            freezing the distribution.
        zeta : float, optional
            Shape parameter, ``zeta > 0``. Supply explicitly or by freezing the
            distribution.
        size : int or tuple of ints, optional
            Number of samples to draw. ``None`` (default) returns a scalar.
        random_state : np.random.Generator, np.random.RandomState, or None, optional
            Random number generator to use.

        Returns
        -------
        samples : ndarray or float
            Random variates on ``[0, 2π)``.
        """
        mu_val = getattr(self, "mu", None) if mu is None else mu
        zeta_val = getattr(self, "zeta", None) if zeta is None else zeta

        if mu_val is None or zeta_val is None:
            raise ValueError("Both 'mu' and 'zeta' must be provided.")

        return self._rvs(mu_val, zeta_val, size=size, random_state=random_state)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        return_info=False,
        optimizer="L-BFGS-B",
        **kwargs,
    ):
        """
        Estimate ``mu`` and ``zeta`` for the Cartwright distribution.

        Parameters
        ----------
        data : array_like
            Sample angles (radians). Values are wrapped to ``[0, 2π)`` internally.
        weights : array_like, optional
            Non-negative weights/frequencies broadcastable to ``data``.
        method : {"mle", "moments"}, optional
            Estimation strategy. "moments" matches the first trigonometric
            moment, "mle" (default) maximises the weighted log-likelihood.
        return_info : bool, optional
            If True, also return a diagnostic dictionary.
        optimizer : str, optional
            Optimiser passed to ``scipy.optimize.minimize`` when
            ``method="mle"``.
        **kwargs :
            Additional keyword arguments forwarded to the optimiser.
        """
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        x = self._wrap_angles(np.asarray(data, dtype=float))
        if x.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, x.shape).astype(float, copy=False)

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = w_sum**2 / np.sum(w**2)

        mu_mom, _ = circ_mean_and_r(alpha=x, w=w)
        if not np.isfinite(mu_mom):
            mu_mom = float(0.0)
        mu_mom = float(np.mod(mu_mom, 2.0 * np.pi))
        delta = (x - mu_mom + np.pi) % (2.0 * np.pi) - np.pi
        sin_half = np.sin(0.5 * delta)
        m_t = float(np.sum(w * sin_half**2) / w_sum)
        m_t = float(np.clip(m_t, 0.0, 0.5 - 1e-12))
        if m_t <= 1e-12:
            zeta_mom = 1e-6
        else:
            denom = max(1e-12, 0.5 - m_t)
            zeta_mom = float(np.clip(m_t / denom, 1e-6, 1e6))

        def log_c(z):
            inv = 1.0 / z
            return (
                (-1.0 + inv) * np.log(2.0)
                + 2.0 * gammaln(1.0 + inv)
                - np.log(np.pi)
                - gammaln(1.0 + 2.0 * inv)
            )

        def nll(params):
            mu_param, zeta_param = params
            if zeta_param <= 0.0:
                return np.inf
            cos_term = np.cos(x - mu_param)
            denom = np.clip(1.0 + cos_term, 1e-15, None)
            sum_log = np.sum(w * np.log(denom))
            ll = w_sum * log_c(zeta_param) + (1.0 / zeta_param) * sum_log
            return float(-ll)

        def grad(params):
            mu_param, zeta_param = params
            cos_term = np.cos(x - mu_param)
            denom = np.clip(1.0 + cos_term, 1e-15, None)
            sin_term = np.sin(x - mu_param)
            sum_log = np.sum(w * np.log(denom))
            grad_mu = -(1.0 / zeta_param) * np.sum(w * sin_term / denom)
            inv = 1.0 / zeta_param
            term = 2.0 * digamma(1.0 + 2.0 * inv) - (
                np.log(2.0) + 2.0 * digamma(1.0 + inv)
            )
            grad_zeta = (sum_log - w_sum * term) / (zeta_param**2)
            return np.array([grad_mu, grad_zeta], dtype=float)

        method = method.lower()
        if method not in {"mle", "moments"}:
            raise ValueError("`method` must be either 'mle' or 'moments'.")

        if method == "moments":
            mu_hat = self._wrap_direction(mu_mom)
            zeta_hat = zeta_mom
            info = {
                "method": "moments",
                "loglik": float(-nll((mu_hat, zeta_hat))),
                "n_effective": float(n_eff),
                "converged": True,
            }
        else:
            mu_init = mu_mom
            zeta_init = zeta_mom if np.isfinite(zeta_mom) else 10.0
            zeta_init = float(np.clip(zeta_init, 1e-3, 1e4))
            bounds = [(0.0, 2.0 * np.pi), (1e-6, 1e6)]
            result = minimize(
                nll,
                np.array([mu_init, zeta_init], dtype=float),
                method=optimizer,
                jac=grad,
                bounds=bounds,
                **kwargs,
            )
            if not result.success:
                raise RuntimeError(
                    f"cartwright.fit(method='mle') failed: {result.message}"
                )
            mu_hat = self._wrap_direction(float(result.x[0]))
            zeta_hat = float(np.clip(result.x[1], 1e-6, 1e6))
            info = {
                "method": "mle",
                "loglik": float(-result.fun),
                "n_effective": float(n_eff),
                "converged": bool(result.success),
                "nit": result.nit,
                "grad_norm": float(np.linalg.norm(result.jac))
                if getattr(result, "jac", None) is not None
                else np.nan,
                "optimizer": optimizer,
            }

        estimates = (mu_hat, zeta_hat)
        if return_info:
            return estimates, info
        return estimates


cartwright = cartwright_gen(name="cartwright")
cartlss = CircularLL(cartwright, name="cartlss")


class wrapnorm_gen(_RegressionReady, CircularContinuous):
    """Wrapped Normal Distribution

    ![wrapnorm](../images/circ-mod-wrapnorm.png)

    Methods
    -------
    pdf(x, mu, rho)
        Probability density function.

    logpdf(x, mu, rho)
        Logarithm of the probability density function.

    cdf(x, mu, rho)
        Cumulative distribution function.

    ppf(q, mu, rho)
        Percent-point function (inverse CDF).

    rvs(mu, rho, size=None, random_state=None)
        Random variates.

    fit(data, *args, **kwargs)
        Estimate ``(mu, rho)`` via method-of-moments or maximum likelihood.

    Examples
    --------
    ```
    from pycircstat2.distributions import wrapnorm
    ```

    Notes
    -----
    Implementation based on Section 4.3.7 of Pewsey et al. (2013)
    """

    # --- regression overlay (read by the regression engine
    # only). Book names mu/rho are preserved; ρ is the mean resultant length,
    # bounded in (0, 1), so its default link is logit. ---
    param_roles = {"mu": "location", "rho": "concentration"}
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): keep the concentration rho off its upper wall (rho -> 1).
    degen_penalty = (_degen_boundary_upper("rho", 1.0),)
    default_links = {"location": "tanhalf", "concentration": "logit"}

    def _concentration_start(self, Rbar):
        """Closed-form concentration start from the mean resultant ``Rbar``
        for the regression null model: ``rho = Rbar`` (the wrapped-normal
        moment estimator, E[cos(y-mu)] = rho), clamped — the circlss
        ``initialize`` convention (see CircularLL._null_params)."""
        return float(np.clip(Rbar, 0.01, 0.95))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._series_window_cache = {}

    def _argcheck(self, mu, rho):
        try:
            mu_arr, rho_arr = np.broadcast_arrays(mu, rho)
        except ValueError:
            return False
        return (
            (mu_arr >= 0.0)
            & (mu_arr <= 2.0 * np.pi)
            & (rho_arr > 0.0)
            & (rho_arr < 1.0)
        )

    # Above this value the Fourier form's terms rho^(p^2) decay too slowly for
    # a short series, and the partial sums cancel catastrophically in the
    # tails (negative densities by rho ~ 0.99; relative tail noise already at
    # rho = 0.9). The Gaussian-image form needs only |k| <= 2 images for
    # rho > 0.35 and is positive term-by-term, so 0.8 sits comfortably inside
    # both branches' exact zones.
    _FOURIER_RHO_MAX = 0.8

    def _pdf(self, x, mu, rho):
        x_b, mu_b, rho_b = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (x, mu, rho))
        )
        shape = x_b.shape
        xf = x_b.reshape(-1)
        mf = mu_b.reshape(-1)
        rf = rho_b.reshape(-1)
        out = np.empty(xf.shape, dtype=float)

        lo = rf <= self._FOURIER_RHO_MAX
        if np.any(lo):
            # Fourier form (book eq. 4.50); truncation error at p = 29 is
            # rho^(900) <= 2e-82 for rho <= 0.8.
            p = np.arange(1.0, 30.0)
            d = xf[lo, None] - mf[lo, None]
            series = np.sum(rf[lo, None] ** (p**2) * np.cos(p * d), axis=1)
            out[lo] = (1.0 + 2.0 * series) / (2.0 * np.pi)
        hi = ~lo
        if np.any(hi):
            # Gaussian-image form (book eq. 4.48) — the same representation
            # the cdf path uses, so pdf == dF/dtheta at high concentration.
            sigma = np.sqrt(-2.0 * np.log(np.clip(rf[hi], None, 1.0 - 1e-15)))
            k = np.arange(-2.0, 3.0)
            z = (xf[hi, None] - mf[hi, None] + 2.0 * np.pi * k) / sigma[:, None]
            out[hi] = np.exp(-0.5 * z**2).sum(axis=1) * INV_SQRT_2PI / sigma
        return out.reshape(shape)

    def pdf(self, x, mu, rho, *args, **kwargs):
        r"""
        Probability density function of the Wrapped Normal distribution.

        $$
        f(\theta) = \frac{1}{2\pi} \left(1 + 2\sum_{p=1}^{\infty} \rho^{p^2} \cos(p(\theta - \mu))\right)
        $$

        evaluated via the Fourier form above for $\rho \le 0.8$ (29 terms,
        truncation $\le 2 \times 10^{-82}$) and via the equivalent wrapped
        Gaussian-image sum for $\rho > 0.8$, where the Fourier partial sums
        lose accuracy.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Mean resultant length, 0 < rho < 1.

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """
        return super().pdf(x, mu, rho, *args, **kwargs)

    def _logpdf(self, x, mu, rho):
        x_b, mu_b, rho_b = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (x, mu, rho))
        )
        shape = x_b.shape
        xf = x_b.reshape(-1)
        mf = mu_b.reshape(-1)
        rf = rho_b.reshape(-1)
        out = np.empty(xf.shape, dtype=float)

        lo = rf <= self._FOURIER_RHO_MAX
        if np.any(lo):
            # the Fourier density never underflows for ρ ≤ 0.8 (antipodal
            # minimum ~2e-6), so log1p of the series is already exact
            p = np.arange(1.0, 30.0)
            d = xf[lo, None] - mf[lo, None]
            series = np.sum(rf[lo, None] ** (p**2) * np.cos(p * d), axis=1)
            out[lo] = np.log1p(2.0 * series) - np.log(2.0 * np.pi)
        hi = ~lo
        if np.any(hi):
            # Gaussian-image branch in log space: logsumexp over the 5
            # images keeps the antipodal tail finite (≈ −π²/(2σ²)) where
            # the linear-space sum underflows from ρ ≈ 0.987
            sigma = np.sqrt(-2.0 * np.log(np.clip(rf[hi], None, 1.0 - 1e-15)))
            k = np.arange(-2.0, 3.0)
            z = (xf[hi, None] - mf[hi, None] + 2.0 * np.pi * k) / sigma[:, None]
            out[hi] = (
                logsumexp(-0.5 * z**2, axis=1)
                - np.log(sigma)
                - 0.5 * np.log(2.0 * np.pi)
            )
        return out.reshape(shape)

    def logpdf(self, x, mu, rho, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the Wrapped Normal
        distribution: ``log1p`` of the Fourier series for $\rho \le 0.8$ and
        a log-sum-exp over the wrapped Gaussian images for $\rho > 0.8$, so
        the log-density stays finite (antipodally $\approx -\pi^2/2\sigma^2$)
        at concentrations where the linear-space density underflows.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Mean resultant length, 0 < rho < 1.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, mu, rho, *args, **kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """Closed-form trigonometric moment (book §4.3.7):
        m_p = ρ^{p²}·e^{ipμ}."""
        shape_args, non_shape_kwargs = self._separate_shape_parameters(
            args, kwargs, "trig_moment"
        )
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        mu, rho = (float(np.asarray(v, dtype=float))
                   for v in self._parse_args(*shape_args, **call_kwargs)[0])

        if not np.isscalar(p):
            raise ValueError("`p` must be an integer scalar.")
        if int(round(p)) != p:
            raise ValueError("`p` must be an integer.")
        k = int(round(p))
        if k == 0:
            return complex(1.0, 0.0)
        ak = abs(k)
        value = (rho ** (ak * ak)) * np.exp(1j * ak * mu)
        return complex(np.conjugate(value)) if k < 0 else complex(value)

    # Derivative methods (the regression contract) reuse the same hybrid
    # split as ``_pdf``, so l1/l2 differentiate exactly the density that
    # ``logpdf`` evaluates. Fourier branch: term-wise series ratios. Gaussian
    # branch: scores of a 5-component wrapped-image mixture as softmax-
    # weighted moments m_j = Σ w_k z_k^j (the e^{-z²/2} weights are
    # max-stabilized, so no underflow at high concentration), chained
    # through σ(ρ) = √(−2 log ρ) with σ' = −1/(ρσ), σ'' = (σ²−1)/(ρ²σ³).

    def _score_terms(self, x, mu, rho, second):
        x_b, mu_b, rho_b = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (x, mu, rho))
        )
        shape = x_b.shape
        xf = x_b.reshape(-1)
        mf = mu_b.reshape(-1)
        rf = rho_b.reshape(-1)
        d1 = {k: np.empty(xf.shape, dtype=float) for k in ("mu", "rho")}
        d2 = (
            {k: np.empty(xf.shape, dtype=float)
             for k in (("mu", "mu"), ("mu", "rho"), ("rho", "rho"))}
            if second
            else None
        )

        lo = rf <= self._FOURIER_RHO_MAX
        if np.any(lo):
            p = np.arange(1.0, 30.0)
            d = xf[lo, None] - mf[lo, None]
            r = rf[lo, None]
            rp = r ** (p**2)
            cs, sn = np.cos(p * d), np.sin(p * d)
            S0 = 1.0 + 2.0 * np.sum(rp * cs, axis=1)
            Smu = 2.0 * np.sum(p * rp * sn, axis=1)
            Srho = 2.0 * np.sum(p**2 * r ** (p**2 - 1.0) * cs, axis=1)
            lmu, lrho = Smu / S0, Srho / S0
            d1["mu"][lo], d1["rho"][lo] = lmu, lrho
            if second:
                Smm = -2.0 * np.sum(p**2 * rp * cs, axis=1)
                Smr = 2.0 * np.sum(p**3 * r ** (p**2 - 1.0) * sn, axis=1)
                # the p = 1 term of S_ρρ has coefficient p²(p²−1) = 0; start
                # at p = 2 so ρ^{p²−2} never sees a negative exponent
                p2 = p[1:]
                Srr = 2.0 * np.sum(
                    p2**2 * (p2**2 - 1.0) * r ** (p2**2 - 2.0) * cs[:, 1:],
                    axis=1,
                )
                d2[("mu", "mu")][lo] = Smm / S0 - lmu * lmu
                d2[("mu", "rho")][lo] = Smr / S0 - lmu * lrho
                d2[("rho", "rho")][lo] = Srr / S0 - lrho * lrho

        hi = ~lo
        if np.any(hi):
            rh = np.clip(rf[hi], None, 1.0 - 1e-15)
            sigma = np.sqrt(-2.0 * np.log(rh))
            k = np.arange(-2.0, 3.0)
            z = (xf[hi, None] - mf[hi, None] + 2.0 * np.pi * k) / sigma[:, None]
            z2 = z * z
            w = np.exp(-0.5 * (z2 - np.min(z2, axis=1, keepdims=True)))
            w /= np.sum(w, axis=1, keepdims=True)
            m1 = np.sum(w * z, axis=1)
            m2 = np.sum(w * z2, axis=1)
            l_sig = (m2 - 1.0) / sigma
            sp = -1.0 / (rh * sigma)  # dσ/dρ
            d1["mu"][hi] = m1 / sigma
            d1["rho"][hi] = l_sig * sp
            if second:
                m3 = np.sum(w * z2 * z, axis=1)
                m4 = np.sum(w * z2 * z2, axis=1)
                s2 = sigma * sigma
                l_mm = (m2 - 1.0 - m1 * m1) / s2
                l_ms = (m3 - 3.0 * m1 - m1 * (m2 - 1.0)) / s2
                l_ss = (m4 - 5.0 * m2 + 2.0 - (m2 - 1.0) ** 2) / s2
                spp = (s2 - 1.0) / (rh * rh * sigma * s2)  # d²σ/dρ²
                d2[("mu", "mu")][hi] = l_mm
                d2[("mu", "rho")][hi] = l_ms * sp
                d2[("rho", "rho")][hi] = l_ss * sp * sp + l_sig * spp

        d1 = {k: v.reshape(shape) for k, v in d1.items()}
        if second:
            d2 = {k: v.reshape(shape) for k, v in d2.items()}
        return d1, d2

    def dlogpdf(self, x, mu, rho):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        Fourier branch (ρ ≤ 0.8): with $S_0 = 1 + 2\sum_p \rho^{p^2}\cos p\phi$,

        $$\frac{\partial\ell}{\partial\mu} =
          \frac{2\sum_p p\,\rho^{p^2}\sin p\phi}{S_0},\qquad
          \frac{\partial\ell}{\partial\rho} =
          \frac{2\sum_p p^2\rho^{p^2-1}\cos p\phi}{S_0}.$$

        Gaussian-image branch (ρ > 0.8): mixture scores via softmax-weighted
        moments, chained through σ(ρ). Vectorizes over per-observation
        ``mu``/``rho`` arrays; returns a book-named dict.
        """
        return self._score_terms(x, mu, rho, second=False)[0]

    def d2logpdf(self, x, mu, rho):
        r"""Second derivatives of ``logpdf`` (l2) — unique unordered pairs,
        as ratio forms $f_{ab}/f - (f_a/f)(f_b/f)$ on the same hybrid split
        as ``_pdf`` (see ``_score_terms``)."""
        return self._score_terms(x, mu, rho, second=True)[1]

    @staticmethod
    def _wrapnorm_cdf_pdf(theta, mu_val, sigma, *, tol=1e-13, max_iter=500):
        theta_arr = np.asarray(theta, dtype=float)
        flat = theta_arr.reshape(-1)
        if flat.size == 0:
            return theta_arr.astype(float), theta_arr.astype(float)

        inv_sigma = 1.0 / sigma
        two_pi = 2.0 * np.pi

        diff = flat - mu_val
        z0 = diff * inv_sigma
        z_ref0 = (-mu_val) * inv_sigma

        cdf = ndtr(z0) - ndtr(z_ref0)
        pdf = INV_SQRT_2PI * inv_sigma * np.exp(-0.5 * z0**2)

        k = 1
        max_contrib = np.inf
        while k <= max_iter and max_contrib > tol:
            shift = two_pi * k

            z_pos = (diff + shift) * inv_sigma
            z_pos_ref = (-mu_val + shift) * inv_sigma
            delta_pos = ndtr(z_pos) - ndtr(z_pos_ref)
            pdf += INV_SQRT_2PI * inv_sigma * np.exp(-0.5 * z_pos**2)

            z_neg = (diff - shift) * inv_sigma
            z_neg_ref = (-mu_val - shift) * inv_sigma
            delta_neg = ndtr(z_neg) - ndtr(z_neg_ref)
            pdf += INV_SQRT_2PI * inv_sigma * np.exp(-0.5 * z_neg**2)

            cdf += delta_pos + delta_neg
            max_contrib = max(
                float(np.max(np.abs(delta_pos))),
                float(np.max(np.abs(delta_neg))),
            )
            if not np.isfinite(max_contrib):
                break
            k += 1

        cdf = np.clip(cdf, 0.0, 1.0)
        pdf = np.clip(pdf, 0.0, None)

        cdf = cdf.reshape(theta_arr.shape)
        pdf = pdf.reshape(theta_arr.shape)
        return cdf, pdf

    def _cdf(self, x, mu, rho):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        flat = arr.reshape(-1)

        if flat.size == 0:
            return arr.astype(float)

        mu_val = _as_scalar_param(mu, "wrapnorm")
        rho_val = _as_scalar_param(rho, "wrapnorm")
        two_pi = 2.0 * np.pi

        if rho_val <= 1e-12:
            uniform = flat / two_pi
            if arr.ndim == 0:
                value = float(uniform[0])
                return 1.0 if float(wrapped) == two_pi else value
            result = uniform.reshape(arr.shape)
            result[arr == two_pi] = 1.0
            return result

        rho_clipped = np.clip(rho_val, np.finfo(float).tiny, 1.0 - 1e-15)
        sigma = float(np.sqrt(-2.0 * np.log(rho_clipped)))

        cdf_flat, _ = self._wrapnorm_cdf_pdf(flat, mu_val, sigma)
        if arr.ndim == 0:
            value = float(cdf_flat.reshape(-1)[0])
            return 1.0 if float(wrapped) == two_pi else value

        result = cdf_flat.reshape(arr.shape)
        result[arr == two_pi] = 1.0
        return result

    def cdf(self, x, mu, rho, *args, **kwargs):
        r"""
        Cumulative distribution function of the Wrapped Normal distribution.

        The CDF is evaluated via the wrapped normal series involving the
        standard normal distribution function.

        $$
        F(\theta) = \sum_{k=-\infty}^{\infty} \left[
            \Phi\left(\frac{\theta - \mu + 2\pi k}{\sigma}\right)
            - \Phi\left(\frac{-\mu + 2\pi k}{\sigma}\right)
        \right], \quad \sigma = \sqrt{-2\log \rho}
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Shape parameter, 0 < rho <= 1.

        Returns
        -------
        cdf_values : array_like
            Cumulative distribution function evaluated at `x`.
        """
        return super().cdf(x, mu, rho, *args, **kwargs)

    def _ppf(self, q, mu, rho):
        mu_arr = np.asarray(mu, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)

        mu_val = float(np.mod(mu_arr.reshape(-1)[0], 2.0 * np.pi))
        rho_val = float(rho_arr.reshape(-1)[0])
        two_pi = 2.0 * np.pi

        q_arr = np.asarray(q, dtype=float)
        flat = q_arr.reshape(-1)
        if flat.size == 0:
            return q_arr.astype(float)

        def _finish(arr):
            reshaped = arr.reshape(q_arr.shape)
            if q_arr.ndim == 0:
                return float(reshaped)
            return reshaped

        result = np.full_like(flat, np.nan, dtype=float)
        valid = np.isfinite(flat)

        if not np.any(valid):
            return _finish(result)

        close_zero = valid & (flat <= 0.0)
        close_one = valid & (flat >= 1.0)
        result[close_zero] = 0.0
        result[close_one] = two_pi

        interior = valid & ~(close_zero | close_one)
        if not np.any(interior):
            return _finish(result)

        flat_interior = flat[interior]

        if rho_val <= 1e-12:
            result[interior] = two_pi * flat_interior
            return _finish(result)

        rho_clipped = np.clip(rho_val, np.finfo(float).tiny, 1.0 - 1e-15)
        sigma = float(np.sqrt(-2.0 * np.log(rho_clipped)))

        if sigma <= 1e-12:
            result[interior] = np.mod(mu_val, two_pi)
            return _finish(result)

        q_sub = flat_interior
        theta = np.clip(two_pi * q_sub, 1e-12, two_pi - 1e-12)
        if sigma < 1.0:
            normal_guess = mu_val + sigma * ndtri(np.clip(q_sub, 1e-12, 1.0 - 1e-12))
            theta = 0.5 * theta + 0.5 * np.mod(normal_guess, two_pi)

        lower = np.zeros_like(theta)
        upper = np.full_like(theta, two_pi)
        tol = 1e-12
        max_iter = 6

        theta_curr = theta
        cdf_vals, pdf_vals = self._wrapnorm_cdf_pdf(theta_curr, mu_val, sigma)
        delta = cdf_vals - q_sub

        for _ in range(max_iter):
            lower = np.where(delta <= 0.0, theta_curr, lower)
            upper = np.where(delta > 0.0, theta_curr, upper)
            if np.max(np.abs(delta)) <= tol:
                break
            denom = np.clip(pdf_vals, 1e-15, None)
            step = np.clip(delta / denom, -np.pi, np.pi)
            theta_next = theta_curr - step
            theta_next = np.where(
                (theta_next <= lower) | (theta_next >= upper),
                0.5 * (lower + upper),
                theta_next,
            )
            theta_next = np.clip(theta_next, 0.0, two_pi)
            theta_curr = theta_next
            cdf_vals, pdf_vals = self._wrapnorm_cdf_pdf(theta_curr, mu_val, sigma)
            delta = cdf_vals - q_sub

        lower = np.where(delta <= 0.0, theta_curr, lower)
        upper = np.where(delta > 0.0, theta_curr, upper)

        mask = np.abs(delta) > tol
        if np.any(mask):
            lower_b = lower.copy()
            upper_b = upper.copy()
            theta_b = theta_curr.copy()
            for _ in range(40):
                if not np.any(mask):
                    break
                mid = 0.5 * (lower_b + upper_b)
                mid_cdf, _ = self._wrapnorm_cdf_pdf(mid, mu_val, sigma)
                delta_mid = mid_cdf - q_sub
                take_upper = (delta_mid > 0.0) & mask
                take_lower = (~take_upper) & mask
                upper_b = np.where(take_upper, mid, upper_b)
                lower_b = np.where(take_lower, mid, lower_b)
                theta_b = np.where(mask, mid, theta_b)
                mask = mask & (np.abs(delta_mid) > tol)
            theta_curr = np.where(mask, 0.5 * (lower_b + upper_b), theta_b)

        theta_curr = np.clip(theta_curr, 0.0, two_pi)
        endpoint_mask = theta_curr >= (two_pi - 1e-12)
        if np.any(endpoint_mask):
            endpoint_value = np.nextafter(two_pi, 0.0)
            theta_curr = np.where(endpoint_mask, endpoint_value, theta_curr)

        result[interior] = theta_curr
        return _finish(result)

    def ppf(self, q, mu, rho, *args, **kwargs):
        r"""
        Percent-point function (inverse CDF) of the Wrapped Normal distribution.

        The quantile is found by inverting the wrapped normal CDF using a
        safeguarded Newton iteration on $[0, 2\pi]$. At each step the algorithm
        evaluates the truncated unwrapped Gaussian series
        $$
        F(\theta)=\sum_{k=-\infty}^{\infty}
        \Bigl[\Phi\!\Bigl(\tfrac{\theta-\mu+2\pi k}{\sigma}\Bigr)
        - \Phi\!\Bigl(\tfrac{-\mu+2\pi k}{\sigma}\Bigr)\Bigr],
        \qquad
        f(\theta)=\sum_{k=-\infty}^{\infty}
        \frac{1}{\sigma}\,\varphi\!\Bigl(\tfrac{\theta-\mu+2\pi k}{\sigma}\Bigr),
        $$
        with $\sigma = \sqrt{-2\log\rho}$, using the CDF residual to update the
        bracket and the PDF as the local slope. A final bisection polish ensures
        robust convergence and keeps the quantile consistent with ``cdf`` and
        ``rvs``.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate (0 <= q <= 1).
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Shape parameter, 0 < rho < 1.

        Returns
        -------
        ppf_values : array_like
            Angles corresponding to the given quantiles.
        """
        return super().ppf(q, mu, rho, *args, **kwargs)

    def _rvs(self, mu, rho, size=None, random_state=None):
        rng = self._init_rng(random_state)

        mu_arr = np.asarray(mu, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)
        if mu_arr.size != 1 or rho_arr.size != 1:
            raise ValueError("wrapnorm parameters must be scalar-valued.")

        mu_val = float(np.mod(mu_arr.reshape(-1)[0], 2.0 * np.pi))
        rho_val = float(np.clip(rho_arr.reshape(-1)[0], np.finfo(float).tiny, 1.0 - 1e-15))

        if rho_val <= 1e-12:
            samples = rng.uniform(0.0, 2.0 * np.pi, size=size)
            return float(samples) if np.isscalar(samples) else samples

        sigma = float(np.sqrt(-2.0 * np.log(rho_val)))
        if sigma < 1e-12:
            if size is None:
                return mu_val
            if np.isscalar(size):
                return np.full((int(size),), mu_val, dtype=float)
            shape = tuple(int(dim) for dim in np.atleast_1d(size))
            return np.full(shape, mu_val, dtype=float)

        samples = rng.normal(loc=mu_val, scale=sigma, size=size)
        wrapped = np.mod(samples, 2.0 * np.pi)
        if np.isscalar(wrapped):
            return float(wrapped)
        return wrapped

    def rvs(self, mu=None, rho=None, size=None, random_state=None):
        r"""
        Draw random variates from the Wrapped Normal distribution.

        Samples are obtained by drawing from $N(\mu, \sigma^2)$ with
        $\sigma = \sqrt{-2\log\rho}$ and wrapping the result modulo $2\pi$.
        This matches the analytic mixture used in ``cdf`` and ``ppf``, keeping
        all three methods numerically consistent.

        Parameters
        ----------
        mu : float, optional
            Mean direction, ``0 <= mu <= 2*pi``. Supply explicitly or by
            freezing the distribution.
        rho : float, optional
            Shape parameter, ``0 < rho < 1``. Supply explicitly or by freezing
            the distribution.
        size : int or tuple of ints, optional
            Number of samples to draw. ``None`` (default) returns a scalar.
        random_state : np.random.Generator, np.random.RandomState, or None, optional
            Random number generator to use.

        Returns
        -------
        samples : ndarray or float
            Random variates on ``[0, 2π)``.
        """
        mu_val = getattr(self, "mu", None) if mu is None else mu
        rho_val = getattr(self, "rho", None) if rho is None else rho

        if mu_val is None or rho_val is None:
            raise ValueError("Both 'mu' and 'rho' must be provided.")

        return self._rvs(mu_val, rho_val, size=size, random_state=random_state)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        return_info=False,
        optimizer="L-BFGS-B",
        **kwargs,
    ):
        """
        Estimate ``mu`` and ``rho`` for the wrapped normal distribution.

        Parameters
        ----------
        data : array_like
            Sample angles (radians). Values are wrapped to ``[0, 2π)`` internally.
        weights : array_like, optional
            Non-negative weights broadcastable to ``data``.
        method : {"moments", "mle"}, optional
            Estimation strategy. ``"moments"`` (aliases: "analytical") returns
            the circular mean and resultant length. ``"mle"`` (alias:
            "numerical") maximises the weighted log-likelihood via numerical
            optimisation.
        return_info : bool, optional
            If True, return a diagnostics dictionary alongside the estimates.
        optimizer : str, optional
            Optimiser passed to ``scipy.optimize.minimize`` when
            ``method="mle"``.
        **kwargs :
            Additional keyword arguments forwarded to the optimiser.
        """
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        x = self._wrap_angles(np.asarray(data, dtype=float)).ravel()
        if x.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, x.shape).astype(float, copy=False).ravel()

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = w_sum**2 / np.sum(w**2)

        mu_mom, rho_mom = circ_mean_and_r(alpha=x, w=w)
        if not np.isfinite(mu_mom):
            mu_mom = float(0.0)
        mu_mom = float(np.mod(mu_mom, 2.0 * np.pi))
        rho_mom = float(np.clip(rho_mom, 1e-9, 1.0 - 1e-9))

        def logpdf_series(mu_param, rho_param):
            rho_val = float(np.clip(rho_param, 1e-12, 1.0 - 1e-12))
            if rho_val <= 1e-8:
                return np.full_like(x, -np.log(2.0 * np.pi), dtype=float)

            sigma = float(np.sqrt(-2.0 * np.log(rho_val)))
            if sigma > 10.0:
                return np.full_like(x, -np.log(2.0 * np.pi), dtype=float)

            two_pi = 2.0 * np.pi
            cache = getattr(self, "_series_window_cache", None)
            if cache is None:
                cache = {}
                self._series_window_cache = cache

            mu_norm = float(np.mod(mu_param, two_pi))
            mu_bucket = int(round(mu_norm / two_pi * 512)) % 512
            rho_bucket = int(round(min(4095.0, -np.log1p(-rho_val) * 64.0)))
            key = (mu_bucket, rho_bucket)

            max_cap = 256
            max_k = cache.get(
                key,
                max(5, int(np.ceil(3.0 * sigma / two_pi)) + 5),
            )

            tail_tol = 1e-10
            while True:
                ks = np.arange(-max_k, max_k + 1, dtype=float)
                diff = x[:, None] - mu_param + two_pi * ks[None, :]
                exponents = -0.5 * (diff / sigma) ** 2
                max_exp = np.max(exponents, axis=1, keepdims=True)
                shifted = np.exp(exponents - max_exp)
                sum_exp = np.sum(shifted, axis=1)
                log_pdf = max_exp.squeeze(1) + np.log(sum_exp)
                log_pdf -= 0.5 * np.log(2.0 * np.pi) + np.log(sigma)

                tail_contrib = float(
                    np.max(shifted[:, (0, -1)] / np.maximum(sum_exp[:, None], 1e-300))
                )
                if tail_contrib <= tail_tol or max_k >= max_cap:
                    cache[key] = max_k
                    return log_pdf.astype(float, copy=False)

                max_k = min(max_cap, max_k + 2)
            return log_pdf

        def nll(params):
            mu_param, rho_param = params
            if not (0.0 <= rho_param < 1.0):
                return np.inf
            log_pdf = logpdf_series(mu_param, rho_param)
            return float(-np.sum(w * log_pdf))

        method_key = method.lower()
        alias = {"analytical": "moments", "numerical": "mle"}
        method_key = alias.get(method_key, method_key)

        if method_key not in {"moments", "mle"}:
            raise ValueError("`method` must be one of {'moments', 'mle', 'analytical', 'numerical'}.")

        if "algorithm" in kwargs:
            optimizer = kwargs.pop("algorithm")

        if method_key == "moments":
            mu_hat = self._wrap_direction(mu_mom)
            rho_hat = rho_mom
            info = {
                "method": "moments",
                "loglik": float(-nll((mu_hat, rho_hat))),
                "n_effective": float(n_eff),
                "converged": True,
            }
        else:
            bounds = [(0.0, 2.0 * np.pi), (1e-9, 1.0 - 1e-9)]
            init = np.array([mu_mom, rho_mom], dtype=float)
            result = minimize(
                nll,
                init,
                method=optimizer,
                bounds=bounds,
                **kwargs,
            )
            if not result.success:
                raise RuntimeError(
                    f"wrapnorm.fit(method='mle') failed: {result.message}"
                )
            mu_hat = self._wrap_direction(float(result.x[0]))
            rho_hat = float(np.clip(result.x[1], 1e-9, 1.0 - 1e-9))
            info = {
                "method": "mle",
                "loglik": float(-result.fun),
                "n_effective": float(n_eff),
                "converged": bool(result.success),
                "nit": result.nit,
                "grad_norm": np.nan,
                "optimizer": optimizer,
            }

        estimates = (mu_hat, rho_hat)
        if return_info:
            return estimates, info
        return estimates


wrapnorm = wrapnorm_gen(name="wrapnorm")
# `wnlss` is the candidate circlss name — wrapped normal is regression-ready
# here.
wnlss = CircularLL(wrapnorm, name="wnlss")


class wrapcauchy_gen(_RegressionReady, CircularContinuous):
    """Wrapped Cauchy Distribution.

    ![wrapcauchy](../images/circ-mod-wrapcauchy.png)

    Methods
    -------
    pdf(x, mu, rho)
        Probability density function.

    logpdf(x, mu, rho)
        Logarithm of the probability density function (exact log1p form).

    cdf(x, mu, rho)
        Cumulative distribution function.

    ppf(q, mu, rho)
        Percent-point function (inverse CDF) via the Möbius mapping.

    rvs(mu, rho, size=None, random_state=None)
        Random variates.

    fit(data, method="analytical", *args, **kwargs)
        Fit the distribution to the data and return the parameters (mu, rho).

    Notes
    -----
    Implementation based on Section 4.3.6 of Pewsey et al. (2013).
    """

    # --- regression overlay (read by the regression engine
    # only). Book names mu/rho are preserved; ρ is the mean resultant length,
    # bounded in (0, 1), so its default link is logit. ---
    param_roles = {"mu": "location", "rho": "concentration"}
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): exponential pull on the concentration rho toward 0 (uniform).
    degen_penalty = (_degen_linear("rho"),)
    default_links = {"location": "tanhalf", "concentration": "logit"}

    def _concentration_start(self, Rbar):
        """Closed-form concentration start from the mean resultant ``Rbar``
        for the regression null model: ``rho = Rbar`` (the wrapped-Cauchy
        moment estimator, E[cos(y-mu)] = rho), clamped — the circlss
        ``initialize`` convention (see CircularLL._null_params)."""
        return float(np.clip(Rbar, 0.01, 0.95))

    # The log-density splits as ℓ = log(1−ρ²) − log 2π − log D with
    # D = 1 + ρ² − 2ρ cos(θ−μ). The derivative methods below differentiate
    # −log D through the multivariate chain rule from D's (sparse) partial
    # table — ∂μ cycles the trig terms; D is quadratic in ρ so D_ρρρ… = 0 —
    # and add the ρ-only derivatives of log(1−ρ²). All pure numpy,
    # broadcasting over per-observation parameter arrays.

    def dlogpdf(self, x, mu, rho):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        $$\frac{\partial\ell}{\partial\mu} = \frac{2\rho\sin(\theta-\mu)}{D},
          \qquad
          \frac{\partial\ell}{\partial\rho} = -\frac{2\rho}{1-\rho^2}
          - \frac{2\rho - 2\cos(\theta-\mu)}{D}.$$

        Vectorizes over per-observation ``mu``/``rho`` arrays. Returns a
        book-named dict ``{"mu": …, "rho": …}``.
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        rho = np.asarray(rho, dtype=float)
        d = x - mu
        s, c = np.sin(d), np.cos(d)
        w = 1.0 / ((1.0 - rho) ** 2 + 4.0 * rho * np.sin(0.5 * d) ** 2)
        return {
            "mu": 2.0 * rho * s * w,
            "rho": -2.0 * rho / (1.0 - rho**2) - (2.0 * rho - 2.0 * c) * w,
        }

    def d2logpdf(self, x, mu, rho):
        r"""Second derivatives of ``logpdf`` (l2), as *observed* derivatives —
        only the unique unordered pairs, from ``(−log D)_{ab} = −D_{ab}/D +
        D_a D_b/D²`` plus ``∂²_ρ\log(1-\rho^2) = -2(1+\rho^2)/(1-\rho^2)^2``.
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        rho = np.asarray(rho, dtype=float)
        d = x - mu
        s, c = np.sin(d), np.cos(d)
        w = 1.0 / ((1.0 - rho) ** 2 + 4.0 * rho * np.sin(0.5 * d) ** 2)
        Dm, Dr = -2.0 * rho * s, 2.0 * rho - 2.0 * c
        w2 = w * w
        return {
            ("mu", "mu"): -2.0 * rho * c * w + Dm * Dm * w2,
            ("mu", "rho"): 2.0 * s * w + Dm * Dr * w2,
            ("rho", "rho"): -2.0 * (1.0 + rho**2) / (1.0 - rho**2) ** 2
            - 2.0 * w
            + Dr * Dr * w2,
        }

    def d3logpdf(self, x, mu, rho):
        r"""Third derivatives of ``logpdf`` (l3) — unique unordered triples,
        via ``(−log D)_{abc} = −D_{abc}/D + (D_{ab}D_c + D_{ac}D_b +
        D_{bc}D_a)/D² − 2D_aD_bD_c/D³`` (terms with vanished ``D``-partials
        dropped) and ``∂³_ρ\log(1-\rho^2) = -4\rho(3+\rho^2)/(1-\rho^2)^3``.
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        rho = np.asarray(rho, dtype=float)
        d = x - mu
        s, c = np.sin(d), np.cos(d)
        w = 1.0 / ((1.0 - rho) ** 2 + 4.0 * rho * np.sin(0.5 * d) ** 2)
        Dm, Dr = -2.0 * rho * s, 2.0 * rho - 2.0 * c
        Dmm, Dmr, Drr = 2.0 * rho * c, -2.0 * s, 2.0
        Dmmm, Dmmr = 2.0 * rho * s, 2.0 * c
        w2, w3 = w * w, w * w * w
        return {
            ("mu", "mu", "mu"): -Dmmm * w
            + 3.0 * Dmm * Dm * w2
            - 2.0 * Dm**3 * w3,
            ("mu", "mu", "rho"): -Dmmr * w
            + (Dmm * Dr + 2.0 * Dmr * Dm) * w2
            - 2.0 * Dm * Dm * Dr * w3,
            ("mu", "rho", "rho"): (2.0 * Dmr * Dr + Drr * Dm) * w2
            - 2.0 * Dm * Dr * Dr * w3,
            ("rho", "rho", "rho"): -4.0 * rho * (3.0 + rho**2)
            / (1.0 - rho**2) ** 3
            + 3.0 * Drr * Dr * w2
            - 2.0 * Dr**3 * w3,
        }

    def d4logpdf(self, x, mu, rho):
        r"""Fourth derivatives of ``logpdf`` (l4) — unique unordered
        quadruples, from the order-4 partition formula for ``−log D``
        (``D``'s only nonzero quartic-relevant partials are
        ``D_{μμμμ}, D_{μμμρ}``; everything with ≥2 ρ-derivatives of ``D``
        beyond ``D_{ρρ}=2`` vanishes) and ``∂⁴_ρ\log(1-\rho^2) =
        -12(1+6\rho^2+\rho^4)/(1-\rho^2)^4``. Completes the contract to
        full-Newton depth (hea ``available_derivs = 2``).
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        rho = np.asarray(rho, dtype=float)
        d = x - mu
        s, c = np.sin(d), np.cos(d)
        w = 1.0 / ((1.0 - rho) ** 2 + 4.0 * rho * np.sin(0.5 * d) ** 2)
        Dm, Dr = -2.0 * rho * s, 2.0 * rho - 2.0 * c
        Dmm, Dmr, Drr = 2.0 * rho * c, -2.0 * s, 2.0
        Dmmm, Dmmr = 2.0 * rho * s, 2.0 * c
        Dmmmm, Dmmmr = -2.0 * rho * c, 2.0 * s
        w2, w3, w4 = w * w, w**3, w**4
        return {
            ("mu", "mu", "mu", "mu"): -Dmmmm * w
            + (4.0 * Dmmm * Dm + 3.0 * Dmm * Dmm) * w2
            - 12.0 * Dmm * Dm * Dm * w3
            + 6.0 * Dm**4 * w4,
            ("mu", "mu", "mu", "rho"): -Dmmmr * w
            + (Dmmm * Dr + 3.0 * Dmmr * Dm + 3.0 * Dmm * Dmr) * w2
            - 6.0 * (Dmm * Dm * Dr + Dmr * Dm * Dm) * w3
            + 6.0 * Dm**3 * Dr * w4,
            ("mu", "mu", "rho", "rho"): (
                2.0 * Dmmr * Dr + Dmm * Drr + 2.0 * Dmr * Dmr
            )
            * w2
            - 2.0 * (Dmm * Dr * Dr + Drr * Dm * Dm + 4.0 * Dmr * Dm * Dr) * w3
            + 6.0 * Dm * Dm * Dr * Dr * w4,
            ("mu", "rho", "rho", "rho"): 3.0 * Dmr * Drr * w2
            - 6.0 * (Drr * Dm * Dr + Dmr * Dr * Dr) * w3
            + 6.0 * Dm * Dr**3 * w4,
            ("rho", "rho", "rho", "rho"): -12.0
            * (1.0 + 6.0 * rho**2 + rho**4)
            / (1.0 - rho**2) ** 4
            + 3.0 * Drr * Drr * w2
            - 12.0 * Drr * Dr * Dr * w3
            + 6.0 * Dr**4 * w4,
        }

    def _argcheck(self, mu, rho):
        try:
            mu_arr, rho_arr = np.broadcast_arrays(mu, rho)
        except ValueError:
            return False
        return (
            (mu_arr >= 0.0)
            & (mu_arr <= 2.0 * np.pi)
            & (rho_arr >= 0.0)
            & (rho_arr < 1.0)
        )

    def _pdf(self, x, mu, rho):
        # (1-rho)^2 + 4*rho*sin^2(d/2) == 1 + rho^2 - 2*rho*cos(d), but does
        # not cancel at the peak as rho -> 1 (the naive form rounds to <= 0
        # for rho > 1 - 1e-8).
        denom = (1 - rho) ** 2 + 4 * rho * np.sin(0.5 * (x - mu)) ** 2
        return (1 - rho) * (1 + rho) / (2 * np.pi * denom)

    def pdf(self, x, mu, rho, *args, **kwargs):
        r"""
        Probability density function of the Wrapped Cauchy distribution.

        $$
        f(\theta) = \frac{1 - \rho^2}{2\pi(1 + \rho^2 - 2\rho \cos(\theta - \mu))}
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Shape parameter, 0 < rho <= 1.

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """
        return super().pdf(x, mu, rho, *args, **kwargs)

    def _logpdf(self, x, mu, rho):
        # exact log form on the same cancellation-free denominator as
        # ``_pdf``: the previous log(clip(pdf, 1e-16)) floored the
        # honest tail once ρ came within a few ulp of 1
        denom = (1 - rho) ** 2 + 4 * rho * np.sin(0.5 * (x - mu)) ** 2
        with np.errstate(divide="ignore"):
            return (
                np.log1p(-rho)
                + np.log1p(rho)
                - np.log(2.0 * np.pi)
                - np.log(denom)
            )

    def logpdf(self, x, mu, rho, *args, **kwargs):
        """
        Logarithm of the probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-PDF.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Mean resultant length, 0 < rho <= 1.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, mu, rho, *args, **kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """Closed-form trigonometric moment (book §4.3.6):
        m_p = ρ^{p}·e^{ipμ}."""
        shape_args, non_shape_kwargs = self._separate_shape_parameters(
            args, kwargs, "trig_moment"
        )
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        mu, rho = (float(np.asarray(v, dtype=float))
                   for v in self._parse_args(*shape_args, **call_kwargs)[0])

        if not np.isscalar(p):
            raise ValueError("`p` must be an integer scalar.")
        if int(round(p)) != p:
            raise ValueError("`p` must be an integer.")
        k = int(round(p))
        if k == 0:
            return complex(1.0, 0.0)
        ak = abs(k)
        value = (rho ** ak) * np.exp(1j * ak * mu)
        return complex(np.conjugate(value)) if k < 0 else complex(value)

    def _cdf(self, x, mu, rho):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        flat = arr.reshape(-1)

        mu_val = _as_scalar_param(mu, "wrapcauchy")
        rho_val = _as_scalar_param(rho, "wrapcauchy")
        rho_val = np.clip(rho_val, np.finfo(float).tiny, 1.0 - 1e-15)

        if flat.size == 0:
            return arr.astype(float)

        two_pi = 2.0 * np.pi
        A = (1.0 + rho_val) / (1.0 - rho_val)

        phi = (flat - mu_val + np.pi) % two_pi - np.pi
        base_phi = (-mu_val + np.pi) % two_pi - np.pi

        angle = np.arctan2(A * np.sin(0.5 * phi), np.cos(0.5 * phi))
        base_angle = np.arctan2(A * np.sin(0.5 * base_phi), np.cos(0.5 * base_phi))

        cdf = 0.5 + angle / np.pi
        base_val = 0.5 + base_angle / np.pi

        diff = cdf - base_val
        diff = np.where(diff < -1e-12, diff + 1.0, diff)
        diff = np.where(diff > 1.0, diff - 1.0, diff)
        cdf = np.clip(diff, 0.0, 1.0)

        if arr.ndim == 0:
            value = float(cdf[0])
            return 1.0 if float(wrapped) == 2.0 * np.pi else value
        reshaped = cdf.reshape(arr.shape)
        reshaped[arr == 2.0 * np.pi] = 1.0
        return reshaped

    def cdf(self, x, mu, rho, *args, **kwargs):
        """
        Cumulative distribution function of the Wrapped Cauchy distribution.

        The CDF is evaluated analytically via the wrapped Cauchy series.
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the CDF.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Shape parameter, 0 < rho <= 1.

        Returns
        -------
        cdf_values : array_like
            CDF evaluated at `x`.
        """
        return super().cdf(x, mu, rho, *args, **kwargs)

    @staticmethod
    def _wrapcauchy_H(phi, A):
        phi_arr = np.asarray(phi, dtype=float)
        angle = np.arctan2(A * np.sin(0.5 * phi_arr), np.cos(0.5 * phi_arr))
        H = 0.5 + angle / np.pi
        return float(H) if np.isscalar(phi) else H

    def _ppf(self, q, mu, rho):
        mu_arr = np.asarray(mu, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)

        mu_val = float(np.mod(mu_arr.reshape(-1)[0], 2.0 * np.pi))
        rho_val = float(rho_arr.reshape(-1)[0])
        if not (0.0 <= rho_val < 1.0):
            raise ValueError("`rho` must lie in [0, 1).")

        q_arr = np.asarray(q, dtype=float)
        flat = q_arr.reshape(-1)
        if flat.size == 0:
            return q_arr.astype(float)

        result = np.full_like(flat, np.nan, dtype=float)

        lower_mask = flat <= 0.0
        upper_mask = flat >= 1.0
        result[lower_mask] = 0.0
        result[upper_mask] = 2.0 * np.pi

        interior = ~(lower_mask | upper_mask)
        if not np.any(interior):
            return result.reshape(q_arr.shape)

        q_int = flat[interior]
        two_pi = 2.0 * np.pi

        if rho_val <= 1e-15:
            result[interior] = (two_pi * q_int) % two_pi
            return result.reshape(q_arr.shape)

        A = (1.0 + rho_val) / (1.0 - rho_val)
        phi0 = (-mu_val + np.pi) % two_pi - np.pi
        H_start = float(self._wrapcauchy_H(phi0, A))

        s = (H_start + q_int) % 1.0
        eps = 1e-15
        alpha = np.pi * (np.clip(s, eps, 1.0 - eps) - 0.5)
        tan_alpha = np.tan(alpha)
        phi = 2.0 * np.arctan(tan_alpha / A)
        theta = (mu_val + phi) % two_pi
        result[interior] = theta

        return result.reshape(q_arr.shape)

    def ppf(self, q, mu, rho, *args, **kwargs):
        r"""
        Percent-point function (inverse CDF) of the Wrapped Cauchy distribution.

        The quantile is obtained by inverting the Möbius form of the CDF:
        $$
        \phi = 2 \arctan\!\left(\frac{\tan\left(\pi (s-\tfrac12)\right)}{A}\right),
        \qquad A=\frac{1+\rho}{1-\rho},
        $$
        where $s = (H(\phi_0) + q) \bmod 1$ and $\phi_0$ is the anchored angle
        at $x=0$. This matches the direct normalised CDF and keeps ``ppf`` in
        sync with ``cdf`` and the Möbius sampler used by ``rvs``.
        """
        return super().ppf(q, mu, rho, *args, **kwargs)

    def _rvs(self, mu, rho, size=None, random_state=None):
        rng = self._init_rng(random_state)

        mu_arr = np.asarray(mu, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)
        if mu_arr.size != 1 or rho_arr.size != 1:
            raise ValueError("wrapcauchy parameters must be scalar-valued.")

        mu_val = float(mu_arr.reshape(-1)[0])
        rho_val = float(rho_arr.reshape(-1)[0])
        two_pi = 2.0 * np.pi

        if np.isclose(rho_val, 0.0, atol=1e-15):
            return rng.uniform(0.0, two_pi, size=size)

        if np.isclose(rho_val, 1.0, atol=1e-15):
            angle = float(np.mod(mu_val, two_pi))
            if size is None:
                return angle
            return np.full(size, angle, dtype=float)

        if size is None:
            target_shape = ()
        elif np.isscalar(size):
            target_shape = (int(size),)
        else:
            target_shape = tuple(int(dim) for dim in np.atleast_1d(size))

        # Möbius transform sampler: exact and numerically stable for rho<1.
        u = rng.uniform(-np.pi, np.pi, size=target_shape)
        z = np.exp(1j * u)
        alpha = rho_val * np.exp(1j * mu_val)
        denom = 1.0 + rho_val * np.exp(-1j * mu_val) * z
        tiny = 1e-15
        mask = np.abs(denom) < tiny
        denom = np.where(mask, tiny, denom)
        w = (z + alpha) / denom
        angles = np.angle(w)
        original_shape = angles.shape

        if np.any(mask):
            # Fallback to tangent sampler for rare near-pole cases.
            count = int(np.count_nonzero(mask))
            fallback_u = rng.uniform(0.0, 1.0, size=count)
            factor = (1.0 + rho_val) / (1.0 - rho_val)
            tan_term = np.tan(np.pi * (fallback_u - 0.5))
            fallback = mu_val + 2.0 * np.arctan(factor * tan_term)
            fallback = np.mod(fallback, two_pi)
            angles_flat = angles.reshape(-1)
            mask_flat = mask.reshape(-1)
            angles_flat[mask_flat] = fallback
            angles = angles_flat.reshape(original_shape)

        theta = np.mod(angles, two_pi)
        if target_shape == ():
            return float(theta)
        return theta.reshape(target_shape)

    def rvs(self, mu=None, rho=None, size=None, random_state=None):
        """
        Draw random variates from the Wrapped Cauchy distribution.

        Parameters
        ----------
        mu : float, optional
            Mean direction, ``0 <= mu <= 2*pi``. Supply explicitly or by
            freezing the distribution.
        rho : float, optional
            Shape parameter, ``0 <= rho < 1``. Supply explicitly or by freezing
            the distribution.
        size : int or tuple of ints, optional
            Number of samples to draw. ``None`` (default) returns a scalar.
        random_state : np.random.Generator, np.random.RandomState, or None, optional
            Random number generator to use.

        Returns
        -------
        samples : ndarray or float
            Random variates on ``[0, 2π)``.
        """
        mu_val = getattr(self, "mu", None) if mu is None else mu
        rho_val = getattr(self, "rho", None) if rho is None else rho

        if mu_val is None or rho_val is None:
            raise ValueError("Both 'mu' and 'rho' must be provided.")

        return self._rvs(mu_val, rho_val, size=size, random_state=random_state)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        return_info=False,
        optimizer="L-BFGS-B",
        **kwargs,
    ):
        """
        Estimate ``mu`` and ``rho`` for the wrapped Cauchy distribution.

        Parameters
        ----------
        data : array_like
            Sample angles (radians). Values are wrapped to ``[0, 2π)`` internally.
        weights : array_like, optional
            Non-negative weights broadcastable to ``data``.
        method : {"moments", "mle"}, optional
            Estimation strategy. ``"moments"`` (alias: "analytical") returns the
            closed-form estimates based on the first trigonometric moment.
            ``"mle"`` (alias: "numerical") maximises the weighted log-likelihood.
        return_info : bool, optional
            If True, also return a diagnostic dictionary.
        optimizer : str, optional
            Optimiser passed to ``scipy.optimize.minimize`` when
            ``method="mle"``.
        **kwargs :
            Additional keyword arguments forwarded to the optimiser.
        """
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        x = self._wrap_angles(np.asarray(data, dtype=float))
        if x.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, x.shape).astype(float, copy=False)

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = w_sum**2 / np.sum(w**2)

        mu_mom, rho_mom = circ_mean_and_r(alpha=x, w=w)
        if not np.isfinite(mu_mom):
            mu_mom = float(0.0)
        mu_mom = float(np.mod(mu_mom, 2.0 * np.pi))
        rho_mom = float(np.clip(rho_mom, 0.0, 1.0 - 1e-12))

        def nll(params):
            mu_param, rho_param = params
            if not (0.0 <= rho_param < 1.0):
                return np.inf
            denom = (1.0 - rho_param) ** 2 + 4.0 * rho_param * np.sin(0.5 * (x - mu_param)) ** 2
            log_pdf = np.log1p(-rho_param**2) - np.log(2.0 * np.pi) - np.log(denom)
            value = -np.sum(w * log_pdf)
            return float(value)

        def grad(params):
            mu_param, rho_param = params
            denom = (1.0 - rho_param) ** 2 + 4.0 * rho_param * np.sin(0.5 * (x - mu_param)) ** 2
            cos_term = np.cos(x - mu_param)
            sin_term = np.sin(x - mu_param)

            inv_denom = w / denom
            g_mu = -2.0 * rho_param * np.sum(inv_denom * sin_term)
            g_rho = (
                w_sum * (2.0 * rho_param / np.clip(1.0 - rho_param**2, 1e-15, None))
                + np.sum(inv_denom * (2.0 * rho_param - 2.0 * cos_term))
            )
            return np.array([g_mu, g_rho], dtype=float)

        method_key = method.lower()
        alias = {"analytical": "moments", "numerical": "mle"}
        method_key = alias.get(method_key, method_key)

        if "algorithm" in kwargs:
            optimizer = kwargs.pop("algorithm")

        if method_key not in {"moments", "mle"}:
            raise ValueError("`method` must be one of {'moments', 'mle', 'analytical', 'numerical'}.")

        if method_key == "moments":
            mu_hat = self._wrap_direction(mu_mom)
            rho_hat = rho_mom
            info = {
                "method": "moments",
                "loglik": float(-nll((mu_hat, rho_hat))),
                "n_effective": float(n_eff),
                "converged": True,
            }
        else:
            bounds = [(0.0, 2.0 * np.pi), (1e-9, 1.0 - 1e-9)]
            init = np.array([mu_mom, max(1e-3, min(rho_mom, 1.0 - 1e-3))], dtype=float)
            result = minimize(
                nll,
                init,
                method=optimizer,
                jac=grad,
                bounds=bounds,
                **kwargs,
            )
            if not result.success:
                raise RuntimeError(f"wrapcauchy.fit(method='mle') failed: {result.message}")
            mu_hat = self._wrap_direction(float(result.x[0]))
            rho_hat = float(np.clip(result.x[1], 1e-9, 1.0 - 1e-9))
            info = {
                "method": "mle",
                "loglik": float(-result.fun),
                "n_effective": float(n_eff),
                "converged": bool(result.success),
                "nit": result.nit,
                "grad_norm": float(np.linalg.norm(result.jac))
                if getattr(result, "jac", None) is not None
                else np.nan,
                "optimizer": optimizer,
            }

        estimates = (mu_hat, rho_hat)
        if return_info:
            return estimates, info
        return estimates


wrapcauchy = wrapcauchy_gen(name="wrapcauchy")
wclss = CircularLL(wrapcauchy, name="wclss")


class vonmises_gen(_RegressionReady, CircularContinuous):
    """Von Mises Distribution

    ![vonmises](../images/circ-mod-vonmises.png)

    Methods
    -------
    pdf(x, mu, kappa)
        Probability density function.

    logpdf(x, mu, kappa)
        Logarithm of the probability density function (scaled-Bessel
        form, exact at every κ).

    cdf(x, mu, kappa)
        Cumulative distribution function.

    ppf(q, mu, kappa)
        Percent-point function (inverse of CDF).

    rvs(mu, kappa, size=None, random_state=None)
        Random variates.

    fit(data, *args, **kwargs)
        Fit the distribution to the data and return the parameters (mu, kappa).

    Examples
    --------
    ```
    from pycircstat2.distributions import vonmises
    ```

    References
    ----------
    - Section 4.3.8 of Pewsey et al. (2013)

    """

    _freeze_doc = """
    Freeze the distribution with specific parameters.

    Parameters
    ----------
    mu : float
        The mean direction of the distribution (0 <= mu <= 2*pi).
    kappa : float
        The concentration parameter of the distribution (kappa >= 0; kappa = 0 is the circular uniform limit).

    Returns
    -------
    rv_frozen : rv_frozen instance
        The frozen distribution instance with fixed parameters.
    """

    # --- regression overlay (read by the regression engine
    # only). Book names mu/kappa are preserved; roles attach the default links. ---
    param_roles = {"mu": "location", "kappa": "concentration"}
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): exponential pull on kappa toward 0 (kappa -> Inf is the soft
    # degeneracy, where the data Hessian flattens rather than going singular).
    degen_penalty = (_degen_linear("kappa"),)
    default_links = {"location": "tanhalf", "concentration": "log"}

    def _concentration_start(self, Rbar):
        """Closed-form concentration start from the mean resultant ``Rbar``
        for the regression null model: the von Mises A1-inverse ``kappa`` (the
        marginal MLE for the symmetric member), clamped — the circlss
        ``initialize`` convention (see CircularLL._null_params)."""
        return float(np.clip(A1inv(Rbar), 0.01, 500.0))

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)

    __call__.__doc__ = _freeze_doc

    def dlogpdf(self, x, mu, kappa):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        For the von Mises log-density
        ``ℓ = κ cos(θ − μ) − log(2π I_0(κ))``:

        $$\frac{\partial\ell}{\partial\mu} = \kappa\sin(\theta-\mu),\qquad
          \frac{\partial\ell}{\partial\kappa} = \cos(\theta-\mu) - A_1(\kappa).$$

        Vectorizes over per-observation ``mu``/``kappa`` arrays. Returns a
        book-named dict ``{"mu": …, "kappa": …}``.
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        kappa = np.asarray(kappa, dtype=float)
        d = x - mu
        return {
            "mu": kappa * np.sin(d),
            "kappa": np.cos(d) - A1(kappa),
        }

    def d2logpdf(self, x, mu, kappa):
        r"""Second derivatives of ``logpdf`` (l2), as *observed* (not Fisher)
        derivatives — only the unique unordered pairs:

        $$\partial^2_{\mu\mu}\ell = -\kappa\cos(\theta-\mu),\quad
          \partial^2_{\mu\kappa}\ell = \sin(\theta-\mu),\quad
          \partial^2_{\kappa\kappa}\ell = -A_1'(\kappa).$$

        (The von Mises Fisher information for ``μ`` is ``κ A_1(κ) =
        −E[∂²_{μμ}ℓ]``; ``circ_lm(type="cl")`` uses that expected form for Fisher
        scoring. The contract exposes the honest observed derivatives, which is
        what a general-likelihood Newton step — and hea's ``gam.fit5`` — want.)
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        kappa = np.asarray(kappa, dtype=float)
        d = x - mu
        return {
            ("mu", "mu"): -kappa * np.cos(d),
            ("mu", "kappa"): np.sin(d),
            ("kappa", "kappa"): -A1prime(kappa),
        }

    def d3logpdf(self, x, mu, kappa):
        r"""Third derivatives of ``logpdf`` (l3) — unique unordered triples.
        ``∂_μ`` cycles the trig terms (``∂_μ cos(θ−μ) = sin(θ−μ)``,
        ``∂_μ sin(θ−μ) = −cos(θ−μ)``); the κ-only direction differentiates
        ``−A_1``:

        $$\partial^3_{\mu\mu\mu}\ell = -\kappa\sin(\theta-\mu),\quad
          \partial^3_{\mu\mu\kappa}\ell = -\cos(\theta-\mu),\quad
          \partial^3_{\mu\kappa\kappa}\ell = 0,\quad
          \partial^3_{\kappa\kappa\kappa}\ell = -A_1''(\kappa).$$

        With l1/l2 this is the depth hea's gradient-outer Newton needs
        (``available_derivs = 1``); see ``d4logpdf`` for the full-Newton tier.
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        kappa = np.asarray(kappa, dtype=float)
        d = x - mu
        return {
            ("mu", "mu", "mu"): -kappa * np.sin(d),
            ("mu", "mu", "kappa"): -np.cos(d),
            ("mu", "kappa", "kappa"): np.zeros(
                np.broadcast_shapes(x.shape, mu.shape, kappa.shape)
            ),
            ("kappa", "kappa", "kappa"): -A1prime2(kappa),
        }

    def d4logpdf(self, x, mu, kappa):
        r"""Fourth derivatives of ``logpdf`` (l4) — unique unordered quadruples:

        $$\partial^4_{\mu\mu\mu\mu}\ell = \kappa\cos(\theta-\mu),\quad
          \partial^4_{\mu\mu\mu\kappa}\ell = -\sin(\theta-\mu),\quad
          \partial^4_{\kappa\kappa\kappa\kappa}\ell = -A_1'''(\kappa),$$

        and the remaining mixed quadruples vanish (``ℓ`` is linear in κ apart
        from ``−log I_0(κ)``, so any term with ≥2 κ-derivatives and ≥1
        μ-derivative is zero). This completes the contract to the depth hea's
        full outer Newton uses (``available_derivs = 2``).
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        kappa = np.asarray(kappa, dtype=float)
        d = x - mu
        zeros = np.zeros(np.broadcast_shapes(x.shape, mu.shape, kappa.shape))
        return {
            ("mu", "mu", "mu", "mu"): kappa * np.cos(d),
            ("mu", "mu", "mu", "kappa"): -np.sin(d),
            ("mu", "mu", "kappa", "kappa"): zeros,
            ("mu", "kappa", "kappa", "kappa"): zeros,
            ("kappa", "kappa", "kappa", "kappa"): -A1prime3(kappa),
        }

    def _argcheck(self, mu, kappa):
        try:
            mu_arr, kappa_arr = np.broadcast_arrays(mu, kappa)
        except ValueError:
            return False
        return (
            (mu_arr >= 0.0)
            & (mu_arr <= 2.0 * np.pi)
            & (kappa_arr >= 0.0)
        )

    def _pdf(self, x, mu, kappa):
        # exponentially-scaled form: e^{κ(cosφ−1)}/(2π·i0e(κ)) — the naive
        # e^{κcosφ}/I₀(κ) pair overflows to nan/inf for κ ≥ 713
        return np.exp(kappa * (np.cos(x - mu) - 1.0)) / (2 * np.pi * i0e(kappa))

    def pdf(self, x, mu, kappa, *args, **kwargs):
        r"""
        Probability density function of the Von Mises distribution.

        $$
        f(\theta) = \frac{e^{\kappa \cos(\theta - \mu)}}{2\pi I_0(\kappa)}
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        mu : float
            The mean direction of the distribution (0 <= mu <= 2*pi).
        kappa : float
            The concentration parameter of the distribution (kappa >= 0; kappa = 0 is the circular uniform limit).

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """
        return super().pdf(x, mu, kappa, *args, **kwargs)

    def _logpdf(self, x, mu, kappa):
        # log(2πI₀(κ)) = log(2π·i0e(κ)) + κ, folded into κ(cosφ − 1) so the
        # log-density stays finite at every κ (κ = 800 at the mode is ≈ +2.4,
        # not −inf)
        return kappa * (np.cos(x - mu) - 1.0) - np.log(2 * np.pi * i0e(kappa))

    def logpdf(self, x, mu, kappa, *args, **kwargs):
        """
        Logarithm of the probability density function of the Von Mises
        distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the logarithm of the probability density function.
        mu : float
            The mean direction of the distribution (0 <= mu <= 2*pi).
        kappa : float
            The concentration parameter of the distribution (kappa >= 0; kappa = 0 is the circular uniform limit).

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, mu, kappa, *args, **kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """Closed-form trigonometric moment (book §4.3.8):
        m_p = (I_p(κ)/I₀(κ))·e^{ipμ}, evaluated with the exponentially
        scaled ``ive`` so the Bessel ratio is exact at every κ (the raw
        ``iv`` pair overflows from κ ≈ 713)."""
        shape_args, non_shape_kwargs = self._separate_shape_parameters(
            args, kwargs, "trig_moment"
        )
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        mu, kappa = (float(np.asarray(v, dtype=float))
                     for v in self._parse_args(*shape_args, **call_kwargs)[0])

        if not np.isscalar(p):
            raise ValueError("`p` must be an integer scalar.")
        if int(round(p)) != p:
            raise ValueError("`p` must be an integer.")
        k = int(round(p))
        if k == 0:
            return complex(1.0, 0.0)
        ak = abs(k)
        ratio = float(ive(ak, kappa) / ive(0, kappa))
        value = ratio * np.exp(1j * ak * mu)
        return complex(np.conjugate(value)) if k < 0 else complex(value)

    def _cdf(self, x, mu, kappa):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        flat = arr.reshape(-1)

        if flat.size == 0:
            return arr.astype(float)

        mu_arr = np.asarray(mu, dtype=float)
        kappa_arr = np.asarray(kappa, dtype=float)

        mu_val = float(mu_arr.reshape(-1)[0])
        if mu_arr.size > 1 and not np.allclose(mu_arr, mu_val, atol=0.0, rtol=0.0):
            raise ValueError("vonmises parameters must be broadcastable scalars.")

        kappa_val = float(kappa_arr.reshape(-1)[0])
        if kappa_arr.size > 1 and not np.allclose(kappa_arr, kappa_val, atol=0.0, rtol=0.0):
            raise ValueError("vonmises parameters must be broadcastable scalars.")
        two_pi = 2.0 * np.pi

        if kappa_val < 1e-9:
            uniform = flat / two_pi
            if arr.ndim == 0:
                value = float(uniform[0])
                return 1.0 if float(wrapped) == two_pi else value
            result = uniform.reshape(arr.shape)
            result[arr == two_pi] = 1.0
            return result

        # Exact feature-scale GL ladder on the centered kernel — the von
        # Mises is the ψ → 0 member of the Jones–Pewsey clan
        # (h = κ cos φ exactly in `_jp_score_terms`), so the cdf
        # machinery applies verbatim: the integrand e^{κ(cos φ − 1)} ≤ 1
        # never overflows at any κ, and the 1/√κ peak panels resolve every
        # representable concentration. This retires both defects of the
        # former Fourier–Bessel series path (raw iv/i0 overflowed at
        # κ ≥ 713, the 500-term cap missed for κ ≳ 3000 — both fell back
        # to per-point quadrature, ~200 ms per call).
        phi_c = (flat - mu_val) % two_pi
        c0 = (-mu_val) % two_pi
        h_q, _ = _jp_cum01(phi_c, kappa_val, 0.0, want_skew=False)
        h_0, _ = _jp_cum01(np.array([c0]), kappa_val, 0.0, want_skew=False)
        diff = h_q - float(h_0[0])
        # wrap is decided by the anchor *positions*, never by the sign of
        # the mass between them — in a concentrated dead zone that mass is
        # float dust of either sign while the true cdf is 0 or 1.
        cdf = np.clip(np.where(phi_c >= c0, diff, diff + 1.0), 0.0, 1.0)

        if arr.ndim == 0:
            value = float(cdf[0])
            return 1.0 if float(wrapped) == two_pi else value

        result = cdf.reshape(arr.shape)
        result[arr == two_pi] = 1.0
        return result

    def cdf(self, x, mu, kappa, *args, **kwargs):
        r"""
        Cumulative distribution function of the Von Mises distribution.

        $$
        F(\theta) = \frac{1}{2 \pi I_0(\kappa)}\int_{0}^{\theta} e^{\kappa \cos(t - \mu)} dt
        $$

        The CDF is evaluated exactly by composite Gauss–Legendre quadrature
        on the feature-scale panel ladder of the centered kernel
        ``e^{κ(cos φ − 1)} ≤ 1`` (cached edge cumulatives plus one partial
        panel per query) — overflow-free and quadrature-exact at every
        representable concentration, including the regression log-link
        regime ``κ ≫ 700`` where Bessel-ratio series fail.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        mu : float
            The mean direction of the distribution (0 <= mu <= 2*pi).
        kappa : float
            The concentration parameter of the distribution (kappa >= 0; kappa = 0 is the circular uniform limit).

        Returns
        -------
        cdf_values : array_like
            Cumulative distribution function evaluated at `x`.
        """
        return super().cdf(x, mu, kappa, *args, **kwargs)

    def _ppf(self, q, mu, kappa):
        mu_arr = np.asarray(mu, dtype=float)
        kappa_arr = np.asarray(kappa, dtype=float)
        
        mu_val = float(np.mod(mu_arr.reshape(-1)[0], 2.0 * np.pi))
        kappa_val = float(kappa_arr.reshape(-1)[0])
        if kappa_val < 0.0:
            raise ValueError("`kappa` must be non-negative.")

        q_arr = np.asarray(q, dtype=float)
        flat = q_arr.reshape(-1)
        if flat.size == 0:
            return q_arr.astype(float)

        result = np.full_like(flat, np.nan, dtype=float)

        lower_mask = flat <= 0.0
        upper_mask = flat >= 1.0
        result[lower_mask] = 0.0
        result[upper_mask] = 2.0 * np.pi

        interior = ~(lower_mask | upper_mask)
        if not np.any(interior):
            return result.reshape(q_arr.shape)

        q_int = flat[interior]
        two_pi = 2.0 * np.pi

        if kappa_val <= 1e-9:
            result[interior] = (two_pi * q_int) % two_pi
            return result.reshape(q_arr.shape)

        # Centered-angle ladder solve (the ψ = 0 member of the JP clan):
        # quantile-table inverse init + bracket-safeguarded Newton on the
        # exact GL-ladder cdf — see `_jp_ppf_ladder`.
        result[interior] = _jp_ppf_ladder(q_int, mu_val, kappa_val, 0.0)
        return result.reshape(q_arr.shape)

    def ppf(self, q, mu, kappa, *args, **kwargs):
        """
        Percent-point function (inverse of the CDF) of the Von Mises distribution.

        Quantiles solve the exact Gauss–Legendre ladder CDF in the centered
        angle by bracket-safeguarded Newton, initialized from the sampler's
        quantile-table inverse, so ``ppf`` stays in exact sync with ``cdf``
        at every representable concentration.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate.
        mu : float
            The mean direction of the distribution (0 <= mu <= 2*pi).
        kappa : float
            The concentration parameter of the distribution (kappa >= 0; kappa = 0 is the circular uniform limit).

        Returns
        -------
        ppf_values : array_like
            Values at the given quantiles.
        """
        return super().ppf(q, mu, kappa, *args, **kwargs)

    def _rvs(self, mu, kappa, size=None, random_state=None):
        rng = self._init_rng(random_state)

        mu_arr = np.asarray(mu, dtype=float)
        kappa_arr = np.asarray(kappa, dtype=float)

        mu_val = float(mu_arr.reshape(-1)[0])
        if mu_arr.size > 1 and not np.allclose(mu_arr, mu_val, atol=0.0, rtol=0.0):
            raise ValueError("vonmises parameters must be broadcastable scalars.")
        mu_val = float(np.mod(mu_val, 2.0 * np.pi))

        kappa_val = float(kappa_arr.reshape(-1)[0])
        if kappa_arr.size > 1 and not np.allclose(kappa_arr, kappa_val, atol=0.0, rtol=0.0):
            raise ValueError("vonmises parameters must be broadcastable scalars.")
        two_pi = 2.0 * np.pi

        if kappa_val <= 1e-9:
            return rng.uniform(0.0, two_pi, size=size)

        a = 1.0 + np.sqrt(1.0 + 4.0 * kappa_val**2)
        b = (a - np.sqrt(2.0 * a)) / (2.0 * kappa_val)
        r = (1.0 + b**2) / (2.0 * b)

        if size is None:
            samples = np.empty(1, dtype=float)
            target_shape = ()
        elif np.isscalar(size):
            samples = np.empty(int(size), dtype=float)
            target_shape = (int(size),)
        else:
            target_shape = tuple(int(s) for s in np.atleast_1d(size))
            samples = np.empty(int(np.prod(target_shape)), dtype=float)

        total = samples.size
        for idx in range(total):
            while True:
                u1 = rng.uniform()
                z = np.cos(np.pi * u1)
                f = (1.0 + r * z) / (r + z)
                c = kappa_val * (r - f)
                u2 = rng.uniform()
                if u2 < c * (2.0 - c) or u2 <= c * np.exp(1.0 - c):
                    break
            u3 = rng.uniform()
            theta = mu_val + np.sign(u3 - 0.5) * np.arccos(f)
            samples[idx] = np.mod(theta, two_pi)

        if target_shape == ():
            return float(samples[0])
        return samples.reshape(target_shape)

    def rvs(self, size=None, random_state=None, *args, **kwargs):
        """
        Draw random variates.

        Parameters
        ----------
        size : int or tuple, optional
            Number of samples to generate.
        random_state : RandomState, optional
            Random number generator instance.

        Returns
        -------
        samples : ndarray
            Random variates.
        """
        mu = getattr(self, "mu", None)
        kappa = getattr(self, "kappa", None)

        mu = kwargs.pop("mu", mu)
        kappa = kwargs.pop("kappa", kappa)

        if mu is None or kappa is None:
            raise ValueError("Both 'mu' and 'kappa' must be provided.")

        return self._rvs(mu, kappa, size=size, random_state=random_state)

    def mean(self, *args, **kwargs):
        """
        Circular mean of the Von Mises distribution.

        Returns
        -------
        mean : float
            The circular mean direction (in radians), equal to `mu`.
            At the uniform limit ``kappa = 0`` (admitted per the book)
            the mean resultant length is 0 and no mean direction exists —
            returns ``nan``, matching the base-class convention.
        """
        (mu, kappa) = self._parse_args(*args, **kwargs)[0]
        if np.isclose(A1(kappa), 0.0, atol=1e-12):
            return float("nan")
        return mu

    def median(self, *args, **kwargs):
        """
        Circular median of the Von Mises distribution.

        Returns
        -------
        median : float
            The circular median direction (in radians), equal to `mu`.
            At the uniform limit ``kappa = 0`` falls back to the
            base-class linearized convention ``ppf(0.5)`` (= π).
        """
        (mu, kappa) = self._parse_args(*args, **kwargs)[0]
        if np.isclose(A1(kappa), 0.0, atol=1e-12):
            return super().median(*args, **kwargs)
        return mu

    def var(self, *args, **kwargs):
        """
        Circular variance of the Von Mises distribution.

        Returns
        -------
        variance : float
            The circular variance, derived from `kappa`.
        """
        (_, kappa) = self._parse_args(*args, **kwargs)[0]
        return 1 - A1(kappa)

    def std(self, *args, **kwargs):
        """
        Circular standard deviation of the Von Mises distribution.

        Returns
        -------
        std : float
            The circular standard deviation, derived from `kappa`;
            ``inf`` at the uniform limit ``kappa = 0`` (R = 0).
        """
        (_, kappa) = self._parse_args(*args, **kwargs)[0]
        r = A1(kappa)
        if np.isclose(r, 0.0, atol=1e-12):
            return float("inf")
        return np.sqrt(-2 * np.log(r))

    def entropy(self, *args, **kwargs):
        """
        Entropy of the Von Mises distribution.

        Returns
        -------
        entropy : float
            The entropy of the distribution.
        """
        # H = log(2πI₀(κ)) − κ·A₁(κ), in scaled form log(2π·i0e(κ)) +
        # κ(1 − A₁(κ)).
        (_, kappa) = self._parse_args(*args, **kwargs)[0]
        return np.log(2 * np.pi * i0e(kappa)) + kappa * (1 - A1(kappa))

    def _nnlf(self, theta, data):
        """
        Custom negative log-likelihood function for the Von Mises distribution.
        """
        mu, kappa = theta

        if not self._argcheck(mu, kappa):
            return np.inf

        log_likelihood = self._logpdf(data, mu, kappa)

        return -np.sum(log_likelihood)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        return_info=False,
        optimizer="L-BFGS-B",
        **kwargs,
    ):
        """
        Estimate ``mu`` and ``kappa`` for the von Mises distribution.

        Parameters
        ----------
        data : array_like
            Sample angles (radians). Values are wrapped to ``[0, 2π)`` internally.
        weights : array_like, optional
            Non-negative weights broadcastable to ``data``.
        method : {"moments", "mle"}, optional
            Estimation strategy. ``"moments"`` (alias ``"analytical"``) returns
            the circular mean together with the standard approximation for
            ``kappa``. ``"mle"`` (alias ``"numerical"``) maximises the weighted
            log-likelihood using a bounded optimiser.
        return_info : bool, optional
            If True, return a diagnostics dictionary alongside the estimates.
        optimizer : str, optional
            Optimiser passed to ``scipy.optimize.minimize`` when
            ``method="mle"``.
        **kwargs :
            Additional keyword arguments forwarded to the optimiser.
        """
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        x = self._wrap_angles(np.asarray(data, dtype=float)).ravel()
        if x.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, x.shape).astype(float, copy=False).ravel()

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = w_sum**2 / np.sum(w**2)

        mu_mom, r_mom = circ_mean_and_r(alpha=x, w=w)
        if not np.isfinite(mu_mom):
            mu_mom = float(0.0)
        mu_mom = float(np.mod(mu_mom, 2.0 * np.pi))
        r_mom = float(np.clip(r_mom, 1e-12, 1.0 - 1e-12))
        n_adjust = int(max(1, round(w_sum)))
        kappa_mom = float(np.clip(circ_kappa(r=r_mom, n=n_adjust), 1e-9, 1e6))

        method_key = method.lower()
        alias = {"analytical": "moments", "numerical": "mle"}
        method_key = alias.get(method_key, method_key)

        if "algorithm" in kwargs:
            optimizer = kwargs.pop("algorithm")

        if method_key not in {"moments", "mle"}:
            raise ValueError("`method` must be one of {'moments', 'mle', 'analytical', 'numerical'}.")

        def nll(params):
            mu_param, kappa_param = params
            if not (kappa_param > 0.0):
                return np.inf
            cos_term = np.cos(x - mu_param)
            sum_cos = np.sum(w * cos_term)
            log_i0_val = kappa_param + np.log(i0e(kappa_param))
            return float(
                -kappa_param * sum_cos + w_sum * (np.log(2.0 * np.pi) + log_i0_val)
            )

        def grad(params):
            mu_param, kappa_param = params
            cos_term = np.cos(x - mu_param)
            sin_term = np.sin(x - mu_param)
            sum_sin = np.sum(w * sin_term)
            sum_cos = np.sum(w * cos_term)
            ratio = A1(kappa_param)
            g_mu = kappa_param * sum_sin
            g_kappa = -sum_cos + w_sum * ratio
            return np.array([g_mu, g_kappa], dtype=float)

        if method_key == "moments":
            mu_hat = self._wrap_direction(mu_mom)
            kappa_hat = kappa_mom
            info = {
                "method": "moments",
                "loglik": float(-nll((mu_hat, kappa_hat))),
                "n_effective": float(n_eff),
                "converged": True,
            }
        else:
            bounds = [(0.0, 2.0 * np.pi), (1e-9, 1e6)]
            init = np.array([mu_mom, kappa_mom], dtype=float)
            result = minimize(
                nll,
                init,
                method=optimizer,
                jac=grad,
                bounds=bounds,
                **kwargs,
            )
            if not result.success:
                raise RuntimeError(
                    f"vonmises.fit(method='mle') failed: {result.message}"
                )
            mu_hat = self._wrap_direction(float(result.x[0]))
            kappa_hat = float(np.clip(result.x[1], 1e-9, 1e6))
            info = {
                "method": "mle",
                "loglik": float(-result.fun),
                "n_effective": float(n_eff),
                "converged": bool(result.success),
                "nit": result.nit,
                "grad_norm": float(np.linalg.norm(result.jac))
                if getattr(result, "jac", None) is not None
                else np.nan,
                "optimizer": optimizer,
            }

        estimates = (mu_hat, kappa_hat)
        if return_info:
            return estimates, info
        return estimates


vonmises = vonmises_gen(name="vonmises")
vmlss = CircularLL(vonmises, name="vmlss")


def _pn_bvn_cdf(h, k, rho, s=None):
    r"""Standard bivariate-normal cdf ``Φ₂(h, k; ρ)`` via Owen (1956).

    ``Φ₂ = ½[Φ(h) + Φ(k)] − T(h, a_h) − T(k, a_k) − δ`` with
    ``a_h = (k − ρh)/(h√(1−ρ²))`` (and symmetrically ``a_k``), where ``T``
    is Owen's T function and ``δ = ½`` iff ``hk < 0`` or
    (``hk = 0`` and ``h + k < 0``). Singular cells are patched explicitly:
    ``√(1−ρ²) = 0`` (comonotone/antimonotone) and ``h = k = 0``
    (``¼ + atan2(ρ, √(1−ρ²))/2π``, i.e. ``¼ + arcsin ρ / 2π``). When
    exactly one of ``h, k`` is zero the Owen term is its limit
    ``T(0, ±∞) = ±¼`` with the sign of the *other* argument — riding IEEE
    division here is wrong, because a caller passing ``-0.0`` flips the
    infinity's sign while δ (which compares ``== 0``) cannot compensate,
    producing O(½) errors.

    ``s`` lets the caller supply ``√(1−ρ²)`` exactly when it is known in a
    cancellation-free form. The projected-normal wedge cdf passes
    ``ρ = −cos θ, s = |sin θ|``: near ``θ = π`` the rounded ρ collapses to
    exactly 1.0 over a ~1e-8-wide stretch of θ (``1 − ζ²/2`` rounds to 1
    for ``ζ < √eps``), so deriving ``s`` from ρ would flatten the cdf
    there, while ``|sin θ| = ζ`` keeps full precision.

    Vectorized; accuracy ~1e-15 against ``scipy.stats.multivariate_normal``
    (Owen's T itself is Patefield–Tandy).
    """
    if s is None:
        s = np.sqrt(np.clip(1.0 - np.asarray(rho, dtype=float) ** 2, 0.0, None))
    h, k, rho, s = np.broadcast_arrays(
        np.asarray(h, dtype=float),
        np.asarray(k, dtype=float),
        np.asarray(rho, dtype=float),
        np.asarray(s, dtype=float),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        a_h = (k - rho * h) / (h * s)
        a_k = (h - rho * k) / (k * s)
    t_h = np.where(
        h == 0.0,
        0.25 * np.sign(k),
        owens_t(h, np.where(np.isnan(a_h), 0.0, a_h)),
    )
    t_k = np.where(
        k == 0.0,
        0.25 * np.sign(h),
        owens_t(k, np.where(np.isnan(a_k), 0.0, a_k)),
    )
    delta = np.where((h * k < 0) | ((h * k == 0) & (h + k < 0)), 0.5, 0.0)
    val = 0.5 * (ndtr(h) + ndtr(k)) - t_h - t_k - delta
    val = np.where((s == 0.0) & (rho > 0.0), ndtr(np.minimum(h, k)), val)
    val = np.where(
        (s == 0.0) & (rho <= 0.0),
        np.clip(ndtr(h) + ndtr(k) - 1.0, 0.0, None),
        val,
    )
    both_zero = (h == 0.0) & (k == 0.0) & (s > 0.0)
    if np.any(both_zero):
        val = np.where(
            both_zero,
            0.25 + np.arctan2(rho, s) / (2.0 * np.pi),
            val,
        )
    return np.clip(val, 0.0, 1.0)


class projectednormal_gen(_RegressionReady, CircularContinuous):
    r"""Projected Normal (angular Gaussian) Distribution ``PN₂(μ, I)``.

    The direction of a bivariate normal: ``Θ = atan2(X₂, X₁)`` with
    ``X ~ N₂((mu1, mu2), I₂)``. Parameterized by the *Cartesian* mean
    components — the mean direction is ``atan2(mu2, mu1)``, ``‖μ‖`` acts as
    the concentration, and ``‖μ‖ = 0`` is the circular uniform. The
    identity-covariance form is the Presnell et al. (1998) regression
    workhorse: both components are modelled linearly in covariates, so the
    regression overlay tags both with role ``"location"`` and identity links
    (two linear predictors in the hea bridge).

    With ``t = μ₁cos θ + μ₂sin θ`` and ``s = −μ₁sin θ + μ₂cos θ`` the density
    factorizes,

    $$f(\theta) = \varphi(s)\,[\varphi(t) + t\,\Phi(t)],$$

    so every mixed (t, s) derivative of ``log f`` vanishes and the l1..l4
    contract methods stay closed-form in one scalar function
    ``G(t) = \log(\varphi(t) + t\Phi(t))`` plus a rotation by θ.

    Methods
    -------
    pdf(x, mu1, mu2)
        Probability density function (closed form).

    logpdf(x, mu1, mu2)
        Logarithm of the probability density function (closed form).

    cdf(x, mu1, mu2)
        Cumulative distribution function (closed form: bivariate-normal
        wedge probability via Owen's T).

    ppf(q, mu1, mu2)
        Percent-point function (bracket-safeguarded Newton on the
        closed-form CDF).

    rvs(mu1, mu2, size=None, random_state=None)
        Random variates by direct projection of bivariate normal draws.

    fit(data, method="mle", ...)
        Maximum-likelihood estimates of ``(mu1, mu2)``.

    Notes
    -----
    General (non-identity) covariance is intentionally out of scope: it is
    not identifiable up to scale, and the regression literature fixes Σ = I.

    References
    ----------
    Presnell, B., Morrison, S. P., & Littell, R. C. (1998). Projected
    multivariate linear models for directional data. *JASA* 93(443).
    """

    # --- regression overlay: one role, two parameters —
    # the documented one-to-many case. Both LPs use the identity link. ---
    param_roles = {"mu1": "location", "mu2": "location"}
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): a radial ridge on the Cartesian mean (mu1, mu2) shrinks the
    # concentration ||mu|| toward 0 (uniform) while preserving the direction.
    degen_penalty = (_degen_ridge("mu1"), _degen_ridge("mu2"))
    default_links = {"location": "identity"}

    @staticmethod
    def _t_s(x, mu1, mu2):
        """Rotated coordinates: t = μᵀu(θ) (radial), s = μᵀu⊥(θ) (tangent)."""
        c, s_ = np.cos(x), np.sin(x)
        return mu1 * c + mu2 * s_, mu2 * c - mu1 * s_

    @staticmethod
    def _mills_inv(t):
        """``R(t) = φ(t)/Φ(t)``, stable for all t via ``log_ndtr``
        (t → −∞: R → |t|; t → +∞: R → 0)."""
        return np.exp(-0.5 * t * t - 0.5 * np.log(2.0 * np.pi) - log_ndtr(t))

    def _argcheck(self, mu1, mu2):
        try:
            mu1_arr, mu2_arr = np.broadcast_arrays(mu1, mu2)
        except ValueError:
            return False
        return np.isfinite(mu1_arr) & np.isfinite(mu2_arr)

    def _logpdf(self, x, mu1, mu2):
        x = np.asarray(x, dtype=float)
        mu1 = np.asarray(mu1, dtype=float)
        mu2 = np.asarray(mu2, dtype=float)
        t, s = self._t_s(x, mu1, mu2)
        # log f = log φ(s) + log(φ(t) + tΦ(t)); the bracket equals Φ(t)(t + R)
        # with R = φ/Φ, and t + R > 0 always — stable at both t extremes.
        return (
            -0.5 * s * s
            - 0.5 * np.log(2.0 * np.pi)
            + log_ndtr(t)
            + np.log(t + self._mills_inv(t))
        )

    def _pdf(self, x, mu1, mu2):
        return np.exp(self._logpdf(x, mu1, mu2))

    def pdf(self, x, mu1, mu2, *args, **kwargs):
        r"""
        Probability density function of the Projected Normal distribution.

        $$
        f(\theta) = \varphi(s)\,[\varphi(t) + t\,\Phi(t)], \qquad
        t = \mu_1\cos\theta + \mu_2\sin\theta,\;
        s = -\mu_1\sin\theta + \mu_2\cos\theta,
        $$

        where $\varphi$/$\Phi$ are the standard normal pdf/cdf.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        mu1 : float
            First Cartesian mean component (any real).
        mu2 : float
            Second Cartesian mean component (any real).

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """
        return super().pdf(x, mu1, mu2, *args, **kwargs)

    def logpdf(self, x, mu1, mu2, *args, **kwargs):
        """
        Logarithm of the probability density function (closed form).

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-PDF.
        mu1, mu2 : float
            Cartesian mean components (any reals).

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, mu1, mu2, *args, **kwargs)

    def _pdf_and_slope(self, x, mu1, mu2):
        """Closed pdf and its θ-derivative: with ℓ(θ) = log f,
        dℓ/dθ = s·(t + G'(t)) where G' = 1/(t + R), so f' = f·s·(t + G')."""
        t, s = self._t_s(x, mu1, mu2)
        f = np.exp(self._logpdf(x, mu1, mu2))
        return f, f * s * (t + 1.0 / (t + self._mills_inv(t)))

    def _wedge_cdf(self, theta, mu1, mu2):
        r"""Closed-form cdf on already-wrapped angles in ``[0, 2π]``.

        ``{Θ ≤ θ}`` is the wedge between the rays at angles 0 and θ, i.e.
        for θ ∈ [0, π] the intersection of the half-planes ``{X₂ ≥ 0}`` and
        ``{X₁ sin θ − X₂ cos θ ≥ 0}``; both events are linear in the
        bivariate normal X, so

        ``F(θ) = Φ₂(μ₂, μ₁ sin θ − μ₂ cos θ; −cos θ)``,

        and for θ ∈ (π, 2π] the complement wedge gives
        ``F(θ) = 1 − Φ₂(−μ₂, −(μ₁ sin θ − μ₂ cos θ); −cos θ)``.

        Within ``1e-5`` of the corners θ ∈ {0, π, 2π} the Owen formula is
        replaced by the exact second-order Taylor expansion of F: doubles
        cannot represent ``1 ∓ cos θ`` below ~√eps while Φ₂'s
        ρ-sensitivity diverges like ``1/√(1−ρ²)`` there, which would
        otherwise dent the local slope by O(pdf) over a ~1e-8-wide stretch
        (Taylor error ≤ W³|f″|/6 ~ 1e-13; seam quantization ~1e-11).
        """
        theta_b, mu1_b, mu2_b = np.broadcast_arrays(
            np.asarray(theta, dtype=float),
            np.asarray(mu1, dtype=float),
            np.asarray(mu2, dtype=float),
        )
        shape = theta_b.shape
        th = theta_b.reshape(-1).astype(float)
        m1 = mu1_b.reshape(-1).astype(float)
        m2 = mu2_b.reshape(-1).astype(float)

        sin_t, cos_t = np.sin(th), np.cos(th)
        k = m1 * sin_t - m2 * cos_t
        rho = -cos_t
        s = np.abs(sin_t)
        val = np.empty_like(th)
        low_m = th <= np.pi
        if np.any(low_m):
            val[low_m] = _pn_bvn_cdf(m2[low_m], k[low_m], rho[low_m], s[low_m])
        high_m = ~low_m
        if np.any(high_m):
            val[high_m] = 1.0 - _pn_bvn_cdf(
                np.negative(m2[high_m]), -k[high_m], rho[high_m], s[high_m]
            )

        corner_w = 1e-5
        two_pi = 2.0 * np.pi
        near_zero = th <= corner_w
        near_pi = np.abs(th - np.pi) <= corner_w
        near_two_pi = (two_pi - th) <= corner_w
        if np.any(near_zero) or np.any(near_two_pi):
            mask = near_zero | near_two_pi
            f0, fp0 = self._pdf_and_slope(0.0, m1[mask], m2[mask])
            zeta = np.where(near_zero[mask], th[mask], th[mask] - two_pi)
            base = np.where(near_zero[mask], 0.0, 1.0)
            val[mask] = np.clip(
                base + zeta * f0 + 0.5 * zeta * zeta * fp0, 0.0, 1.0
            )
        if np.any(near_pi):
            f_pi, fp_pi = self._pdf_and_slope(np.pi, m1[near_pi], m2[near_pi])
            base = _pn_bvn_cdf(
                m2[near_pi],
                m1[near_pi] * np.sin(np.pi) + m2[near_pi],
                1.0,
                np.sin(np.pi),
            )
            zeta = th[near_pi] - np.pi
            val[near_pi] = np.clip(
                base + zeta * f_pi + 0.5 * zeta * zeta * fp_pi, 0.0, 1.0
            )

        return val.reshape(shape)

    def _cdf(self, x, mu1, mu2):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        if arr.size == 0:
            return arr.astype(float)

        mu1_arr = np.asarray(mu1, dtype=float)
        mu2_arr = np.asarray(mu2, dtype=float)
        cdf_vals = self._wedge_cdf(arr, mu1_arr, mu2_arr)

        # _wrap_angles already snaps inputs within eps·2π of the upper
        # endpoint to exactly 2π, so only exact equality is pinned here —
        # an isclose() with default rtol would swallow honest upper-tail
        # values (e.g. F = 1 − 1e-6 lives ~6e-6 below 2π).
        two_pi = 2.0 * np.pi
        if arr.ndim == 0:
            value = float(cdf_vals)
            return 1.0 if float(wrapped) == two_pi else value
        result = np.asarray(cdf_vals, dtype=float)
        result[arr == two_pi] = 1.0
        return result

    def cdf(self, x, mu1, mu2, *args, **kwargs):
        r"""
        Cumulative distribution function of the Projected Normal
        distribution (closed form).

        ``{Θ ≤ θ}`` is a wedge of the plane, so the CDF is a bivariate
        normal orthant probability evaluated via Owen's T function:

        $$
        F(\theta) = \Phi_2\!\left(\mu_2,\;
        \mu_1\sin\theta - \mu_2\cos\theta;\; -\cos\theta\right),
        \qquad \theta \in [0, \pi],
        $$

        and $F(\theta) = 1 - \Phi_2(-\mu_2, -(\mu_1\sin\theta -
        \mu_2\cos\theta); -\cos\theta)$ for $\theta \in (\pi, 2\pi]$.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the CDF.
        mu1 : float
            First Cartesian mean component (any real).
        mu2 : float
            Second Cartesian mean component (any real).

        Returns
        -------
        cdf_values : array_like
            CDF evaluated at `x`.
        """
        return super().cdf(x, mu1, mu2, *args, **kwargs)

    def _ppf(self, q, mu1, mu2):
        q_arr = np.asarray(q, dtype=float)
        mu1_b, mu2_b, q_b = np.broadcast_arrays(
            np.asarray(mu1, dtype=float), np.asarray(mu2, dtype=float), q_arr
        )
        flat = q_b.reshape(-1).astype(float)
        m1 = mu1_b.reshape(-1).astype(float)
        m2 = mu2_b.reshape(-1).astype(float)
        two_pi = 2.0 * np.pi

        if flat.size == 0:
            return q_arr.astype(float)

        def _finish(arr):
            reshaped = arr.reshape(q_b.shape)
            if q_arr.ndim == 0:
                return float(reshaped)
            return reshaped

        result = np.full_like(flat, np.nan, dtype=float)
        valid = np.isfinite(flat)
        close_zero = valid & (flat <= 0.0)
        close_one = valid & (flat >= 1.0)
        result[close_zero] = 0.0
        result[close_one] = two_pi

        interior = valid & ~(close_zero | close_one)
        if not np.any(interior):
            return _finish(result)

        q_sub = flat[interior]
        m1_sub = m1[interior]
        m2_sub = m2[interior]

        gamma_sub = np.hypot(m1_sub, m2_sub)
        uniform = gamma_sub <= 1e-12
        if np.all(uniform):
            result[interior] = two_pi * q_sub
            return _finish(result)

        # Bracket each quantile on a coarse grid of the closed cdf, then
        # polish with bracket-safeguarded Newton (pdf is the derivative).
        grid = np.linspace(0.0, two_pi, 33)
        scalar_params = bool(
            np.all(m1_sub == m1_sub[0]) and np.all(m2_sub == m2_sub[0])
        )
        if scalar_params:
            f_grid = self._wedge_cdf(grid, m1_sub[0], m2_sub[0])
            idx = np.clip(np.sum(f_grid[:, None] <= q_sub[None, :], axis=0), 1, 32)
            f_lo = f_grid[idx - 1]
            f_hi = f_grid[idx]
        else:
            f_grid = self._wedge_cdf(
                grid[:, None], m1_sub[None, :], m2_sub[None, :]
            )
            idx = np.clip(np.sum(f_grid <= q_sub[None, :], axis=0), 1, 32)
            cols = np.arange(q_sub.size)
            f_lo = f_grid[idx - 1, cols]
            f_hi = f_grid[idx, cols]
        lower = grid[idx - 1]
        upper = grid[idx]
        span = np.clip(f_hi - f_lo, 1e-300, None)
        theta_curr = lower + (upper - lower) * np.clip(
            (q_sub - f_lo) / span, 0.0, 1.0
        )

        tol = 1e-15
        max_iter = 12
        delta = self._wedge_cdf(theta_curr, m1_sub, m2_sub) - q_sub
        lower = np.where(delta <= 0.0, theta_curr, lower)
        upper = np.where(delta > 0.0, theta_curr, upper)
        act = np.flatnonzero(np.abs(delta) > tol)
        for _ in range(max_iter):
            if not act.size:
                break
            th_a = theta_curr[act]
            lo_a = lower[act]
            hi_a = upper[act]
            pdf_a = np.exp(self._logpdf(th_a, m1_sub[act], m2_sub[act]))
            step = np.clip(
                delta[act] / np.clip(pdf_a, 1e-300, None), -np.pi, np.pi
            )
            th_n = th_a - step
            th_n = np.where(
                (th_n <= lo_a) | (th_n >= hi_a), 0.5 * (lo_a + hi_a), th_n
            )
            d_n = self._wedge_cdf(th_n, m1_sub[act], m2_sub[act]) - q_sub[act]
            theta_curr[act] = th_n
            delta[act] = d_n
            lower[act] = np.where(d_n <= 0.0, th_n, lo_a)
            upper[act] = np.where(d_n > 0.0, th_n, hi_a)
            act = act[np.abs(d_n) > tol]

        if act.size:
            # bisection cleanup, compressed to the unconverged cells
            lo_u = lower[act]
            hi_u = upper[act]
            m1_u = m1_sub[act]
            m2_u = m2_sub[act]
            q_u = q_sub[act]
            for _ in range(60):
                if np.all(hi_u - lo_u <= 1e-15):
                    break
                mid = 0.5 * (lo_u + hi_u)
                go_up = self._wedge_cdf(mid, m1_u, m2_u) <= q_u
                lo_u = np.where(go_up, mid, lo_u)
                hi_u = np.where(go_up, hi_u, mid)
            theta_curr[act] = 0.5 * (lo_u + hi_u)

        theta_curr = np.where(uniform, two_pi * q_sub, theta_curr)
        theta_curr = np.clip(theta_curr, 0.0, two_pi)
        endpoint_mask = theta_curr >= (two_pi - 1e-12)
        if np.any(endpoint_mask):
            theta_curr = np.where(
                endpoint_mask, np.nextafter(two_pi, 0.0), theta_curr
            )
        result[interior] = theta_curr
        return _finish(result)

    def ppf(self, q, mu1, mu2, *args, **kwargs):
        """
        Percent-point function (inverse CDF) of the Projected Normal
        distribution.

        Quantiles are found by inverting the closed-form Owen's-T CDF with
        a bracket-safeguarded Newton iteration (the closed-form PDF is the
        derivative), so ``ppf`` stays in exact sync with ``cdf``.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate (values in ``[0, 1]``).
        mu1 : float
            First Cartesian mean component (any real).
        mu2 : float
            Second Cartesian mean component (any real).

        Returns
        -------
        ppf_values : array_like
            Angles in ``[0, 2π)`` such that ``cdf(angle) = q``.
        """
        return super().ppf(q, mu1, mu2, *args, **kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """First trigonometric moment in closed form (offset-normal
        resultant): with γ = ‖μ‖,
        m₁ = √(π/8)·γ·e^{−γ²/4}[I₀(γ²/4) + I₁(γ²/4)]·e^{i·atan2(μ₂, μ₁)},
        evaluated with the scaled ``ive`` (the e^{−γ²/4} factor cancels
        exactly). Higher |p| fall back to the quadrature base method."""
        if np.isscalar(p) and int(round(p)) == p and abs(int(round(p))) <= 1:
            shape_args, non_shape_kwargs = self._separate_shape_parameters(
                args, kwargs, "trig_moment"
            )
            call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
            mu1, mu2 = (float(np.asarray(v, dtype=float))
                        for v in self._parse_args(*shape_args, **call_kwargs)[0])
            k = int(round(p))
            if k == 0:
                return complex(1.0, 0.0)
            g2 = mu1 * mu1 + mu2 * mu2
            R = np.sqrt(np.pi / 8.0) * np.sqrt(g2) * (
                ive(0, 0.25 * g2) + ive(1, 0.25 * g2)
            )
            value = R * np.exp(1j * np.arctan2(mu2, mu1))
            return complex(np.conjugate(value)) if k < 0 else complex(value)
        return super().trig_moment(p, *args, **kwargs)

    # --- l1..l4 (the regression contract). ℓ(t, s) = log φ(s) + G(t)
    # separates, so the s-direction contributes ℓ_s = −s, ℓ_ss = −1 and
    # nothing at higher order; the t-direction is G', G'', G''', G'''' with
    # G' = 1/(t + R) and P ≡ R/(t + R); the rotation back to (μ₁, μ₂) only
    # mixes in powers of cos θ / sin θ. ---

    def dlogpdf(self, x, mu1, mu2):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        $$\partial_{\mu_1}\ell = G'(t)\cos\theta + s\sin\theta,\qquad
          \partial_{\mu_2}\ell = G'(t)\sin\theta - s\cos\theta,$$

        with ``G'(t) = Φ(t)/(φ(t) + tΦ(t)) = 1/(t + R(t))``. Vectorizes over
        per-observation ``mu1``/``mu2`` arrays; returns a book-named dict.
        """
        x = np.asarray(x, dtype=float)
        mu1 = np.asarray(mu1, dtype=float)
        mu2 = np.asarray(mu2, dtype=float)
        c, s_ = np.cos(x), np.sin(x)
        t, s = self._t_s(x, mu1, mu2)
        g1 = 1.0 / (t + self._mills_inv(t))
        return {
            "mu1": g1 * c + s * s_,
            "mu2": g1 * s_ - s * c,
        }

    def d2logpdf(self, x, mu1, mu2):
        r"""Second derivatives of ``logpdf`` (l2) — unique unordered pairs.

        In rotated coordinates the Hessian is ``diag(G''(t), −1)`` with
        ``G'' = P − G'²`` and ``P = R/(t+R) = φ/(φ + tΦ)``; rotating back:

        $$\partial^2_{\mu_1\mu_1}\ell = G''c^2 - \tilde s^2,\quad
          \partial^2_{\mu_1\mu_2}\ell = (G'' + 1)\,c\tilde s,\quad
          \partial^2_{\mu_2\mu_2}\ell = G''\tilde s^2 - c^2,$$

        where ``c = cos θ``, ``s̃ = sin θ``.
        """
        x = np.asarray(x, dtype=float)
        mu1 = np.asarray(mu1, dtype=float)
        mu2 = np.asarray(mu2, dtype=float)
        c, s_ = np.cos(x), np.sin(x)
        t, _ = self._t_s(x, mu1, mu2)
        R = self._mills_inv(t)
        g1 = 1.0 / (t + R)
        g2 = R * g1 - g1 * g1
        return {
            ("mu1", "mu1"): g2 * c * c - s_ * s_,
            ("mu1", "mu2"): (g2 + 1.0) * c * s_,
            ("mu2", "mu2"): g2 * s_ * s_ - c * c,
        }

    def d3logpdf(self, x, mu1, mu2):
        r"""Third derivatives of ``logpdf`` (l3) — unique unordered triples.

        Only the radial direction survives at third order
        (``∂³_s log φ(s) = 0``), so every entry is ``G'''(t)`` times the
        matching power of ``cos θ``/``sin θ``:
        ``G''' = −tP − 3PG' + 2G'³``.
        """
        x = np.asarray(x, dtype=float)
        mu1 = np.asarray(mu1, dtype=float)
        mu2 = np.asarray(mu2, dtype=float)
        c, s_ = np.cos(x), np.sin(x)
        t, _ = self._t_s(x, mu1, mu2)
        R = self._mills_inv(t)
        g1 = 1.0 / (t + R)
        P = R * g1
        g3 = -t * P - 3.0 * P * g1 + 2.0 * g1**3
        return {
            ("mu1", "mu1", "mu1"): g3 * c**3,
            ("mu1", "mu1", "mu2"): g3 * c * c * s_,
            ("mu1", "mu2", "mu2"): g3 * c * s_ * s_,
            ("mu2", "mu2", "mu2"): g3 * s_**3,
        }

    def d4logpdf(self, x, mu1, mu2):
        r"""Fourth derivatives of ``logpdf`` (l4) — unique unordered
        quadruples: ``G''''(t)`` times powers of ``cos θ``/``sin θ``, with
        ``G'''' = P(t² − 1) + 4tPG' + 12PG'² − 3P² − 6G'⁴``. Completes the
        contract to full-Newton depth (hea ``available_derivs = 2``).
        """
        x = np.asarray(x, dtype=float)
        mu1 = np.asarray(mu1, dtype=float)
        mu2 = np.asarray(mu2, dtype=float)
        c, s_ = np.cos(x), np.sin(x)
        t, _ = self._t_s(x, mu1, mu2)
        R = self._mills_inv(t)
        g1 = 1.0 / (t + R)
        P = R * g1
        g4 = (
            P * (t * t - 1.0)
            + 4.0 * t * P * g1
            + 12.0 * P * g1 * g1
            - 3.0 * P * P
            - 6.0 * g1**4
        )
        return {
            ("mu1", "mu1", "mu1", "mu1"): g4 * c**4,
            ("mu1", "mu1", "mu1", "mu2"): g4 * c**3 * s_,
            ("mu1", "mu1", "mu2", "mu2"): g4 * c * c * s_ * s_,
            ("mu1", "mu2", "mu2", "mu2"): g4 * c * s_**3,
            ("mu2", "mu2", "mu2", "mu2"): g4 * s_**4,
        }

    def _rvs(self, mu1, mu2, size=None, random_state=None):
        rng = self._init_rng(random_state)

        mu1_arr = np.asarray(mu1, dtype=float)
        mu2_arr = np.asarray(mu2, dtype=float)
        if mu1_arr.size != 1 or mu2_arr.size != 1:
            raise ValueError("projectednormal parameters must be scalar-valued.")
        mu1_val = float(mu1_arr.reshape(-1)[0])
        mu2_val = float(mu2_arr.reshape(-1)[0])

        shape = ()
        if size is not None:
            if np.isscalar(size):
                shape = (int(size),)
            else:
                shape = tuple(int(dim) for dim in np.atleast_1d(size))

        z1 = rng.standard_normal(size=shape)
        z2 = rng.standard_normal(size=shape)
        theta = np.mod(np.arctan2(mu2_val + z2, mu1_val + z1), 2.0 * np.pi)

        if np.ndim(theta) == 0:
            return float(theta)
        return theta.reshape(shape)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        return_info=False,
        optimizer="L-BFGS-B",
        **kwargs,
    ):
        """
        Estimate ``(mu1, mu2)`` for the projected normal distribution.

        Parameters
        ----------
        data : array_like
            Sample angles (radians). Values are wrapped to ``[0, 2π)``
            internally.
        weights : array_like, optional
            Non-negative weights broadcastable to ``data``.
        method : {"mle"}, optional
            Only maximum likelihood is provided (alias: "numerical"): the
            direction of μ̂ has a closed form, but ``‖μ‖`` does not, so the
            weighted log-likelihood is maximised directly with the analytic
            score (``dlogpdf``) as gradient, initialised from the circular
            mean direction and a von Mises-scale concentration.
        return_info : bool, optional
            If True, also return a diagnostic dictionary.
        optimizer : str, optional
            Optimiser passed to ``scipy.optimize.minimize``.
        **kwargs :
            Additional keyword arguments forwarded to the optimiser.
        """
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        x = self._wrap_angles(np.asarray(data, dtype=float))
        if x.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, x.shape).astype(float, copy=False)

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = w_sum**2 / np.sum(w**2)

        method_key = {"numerical": "mle"}.get(method.lower(), method.lower())
        if method_key != "mle":
            raise ValueError("`method` must be 'mle' (alias: 'numerical').")
        if "algorithm" in kwargs:
            optimizer = kwargs.pop("algorithm")

        mu_dir, r_bar = circ_mean_and_r(alpha=x, w=w)
        if not np.isfinite(mu_dir):
            mu_dir = 0.0
        gamma0 = max(A1inv(float(np.clip(r_bar, 0.0, 1.0 - 1e-9))), 1e-3)
        init = np.array(
            [gamma0 * np.cos(mu_dir), gamma0 * np.sin(mu_dir)], dtype=float
        )

        def nll(params):
            return float(-np.sum(w * self._logpdf(x, params[0], params[1])))

        def grad(params):
            score = self.dlogpdf(x, params[0], params[1])
            return -np.array(
                [np.sum(w * score["mu1"]), np.sum(w * score["mu2"])], dtype=float
            )

        result = minimize(nll, init, method=optimizer, jac=grad, **kwargs)
        if not result.success:
            raise RuntimeError(f"projectednormal.fit(method='mle') failed: {result.message}")
        mu1_hat, mu2_hat = (float(v) for v in result.x)

        estimates = (mu1_hat, mu2_hat)
        if return_info:
            info = {
                "method": "mle",
                "loglik": float(-result.fun),
                "n_effective": float(n_eff),
                "converged": bool(result.success),
                "nit": result.nit,
                "optimizer": optimizer,
            }
            return estimates, info
        return estimates


projectednormal = projectednormal_gen(name="projectednormal")
pnlss = CircularLL(projectednormal, name="pnlss")


class vonmises_flattopped_gen(_RegressionReady, CircularContinuous):
    r"""Flat-topped von Mises Distribution

    The Flat-topped von Mises distribution is a modification of the von Mises distribution
    that allows for more flexible peak shapes, including flattened or sharper tops, depending
    on the value of the shape parameter $\nu$.

    ![vonmises-ext](../images/circ-mod-vonmises-flat-topped.png)

    Methods
    -------
    pdf(x, mu, kappa, nu)
        Probability density function.

    logpdf(x, mu, kappa, nu)
        Logarithm of the probability density function.

    cdf(x, mu, kappa, nu)
        Cumulative distribution function.

    ppf(q, mu, kappa, nu)
        Percent-point function (inverse CDF) through the cached table's
        monotone inverse.

    rvs(mu, kappa, nu, size=None, random_state=None)
        Random variates by inverse transform on the cached table.

    fit(data, *, weights=None, method="mle", ...)
        Estimate ``(mu, kappa, nu)`` by moments seed + maximum likelihood.

    Note
    ----
    ``cdf``/``ppf``/``rvs`` take scalar parameters (cached normalization tables
    are built per parameter set); ``pdf``/``logpdf`` and the regression
    derivatives ``dlogpdf``/``d2logpdf`` additionally accept per-observation
    parameter arrays (the regression contract — this is the ``vmftlss``
    location-concentration-shape family). Implementation based on Section
    4.3.10 of Pewsey et al. (2013).
    """

    # --- regression overlay (read by the regression engine
    # only). Book names mu/kappa/nu preserved. The peakedness factor warps
    # *forward* (B = φ + ν sinφ), so unlike inverse_batschelet the score needs
    # no implicit differentiation; B is odd in φ → the density stays symmetric,
    # so ν is a peakedness/flat-top knob (ν>0 sharper, ν<0 flatter), not a skew
    # one. The normalizer Z(κ,ν) depends on both κ and ν, so ℓ_κ and ℓ_ν each
    # carry a grid-expectation term (the jplss `ℓ_κ = h_κ − E[h_κ]` pattern).
    # ν ∈ (−1,1) rides the tanh link. Reduction member ν=0 is plain `vmlss`. ---
    param_roles = {"mu": "location", "kappa": "concentration", "nu": "shape"}
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): kappa toward 0 (linear), peakedness nu off its +/-1 walls.
    degen_penalty = (_degen_linear("kappa"), _degen_boundary_sym("nu", 1.0))
    default_links = {
        "location": "tanhalf",
        "concentration": "log",
        "shape": "tanh",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vmft_table_cache = {}

    def _argcheck(self, mu, kappa, nu):
        try:
            mu_arr, kappa_arr, nu_arr = np.broadcast_arrays(mu, kappa, nu)
        except ValueError:
            return False
        return (
            (mu_arr >= 0.0)
            & (mu_arr <= 2.0 * np.pi)
            & (kappa_arr >= 0.0)
            & (kappa_arr <= _VMFT_KAPPA_UPPER)
            & (nu_arr >= -1.0)
            & (nu_arr <= 1.0)
        )

    def _clear_normalization_cache(self):
        super()._clear_normalization_cache()
        self._vmft_table_cache = {}

    def _concentration_start(self, Rbar):
        """Closed-form concentration start from the mean resultant ``Rbar`` for
        the regression null model: the von Mises A1-inverse ``kappa`` (the ν→0
        reduction member), clamped — the circlss ``initialize`` convention (see
        CircularLL._null_params)."""
        return float(np.clip(A1inv(Rbar), 0.01, 500.0))

    def dlogpdf(self, x, mu, kappa, nu):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        With ``φ = θ − μ``, ``B = φ + ν sinφ`` and ``B_φ = 1 + ν cosφ``, the
        log-density is ``κ cos B − log Z(κ, ν)``:

        $$\ell_\mu = \kappa\sin B\,B_\phi,\quad
          \ell_\kappa = \cos B - \mathbb{E}[\cos B],\quad
          \ell_\nu = -\kappa\sin B\sin\phi - \mathbb{E}[-\kappa\sin B\sin\phi].$$

        ``Z`` is μ-invariant (translation), so ``ℓ_μ`` carries no normalizer
        term; the κ/ν expectations come from :func:`_vmft_logZ_moments_vec`.
        Vectorizes over per-observation arrays; returns a book-named dict.
        """
        x, mu, kappa, nu = (np.asarray(v, dtype=float)
                            for v in (x, mu, kappa, nu))
        phi = x - mu
        s, c = np.sin(phi), np.cos(phi)
        B = phi + nu * s
        sinB, cosB = np.sin(B), np.cos(B)
        Bphi = 1.0 + nu * c
        dk, dnu, *_ = _vmft_logZ_moments_vec(kappa, nu)
        return {
            "mu": kappa * sinB * Bphi,
            "kappa": cosB - dk,
            "nu": -kappa * sinB * s - dnu,
        }

    def d2logpdf(self, x, mu, kappa, nu):
        r"""Second derivatives of ``logpdf`` (l2) — unique unordered pairs.
        The location blocks are pure kernel (``Z`` is μ-free, so ``∂²_{μ·}log
        Z = 0``); the (κ,κ), (κ,ν), (ν,ν) blocks subtract the normalizer
        second derivatives ``∂²log Z = E[h_{ab}] + Cov(h_a, h_b)`` from
        :func:`_vmft_logZ_moments_vec` (kernel ``h_{κκ}=0``, so ``ℓ_{κκ}=
        −Var[\cos B]``)."""
        x, mu, kappa, nu = (np.asarray(v, dtype=float)
                            for v in (x, mu, kappa, nu))
        phi = x - mu
        s, c = np.sin(phi), np.cos(phi)
        B = phi + nu * s
        sinB, cosB = np.sin(B), np.cos(B)
        Bphi = 1.0 + nu * c
        _, _, dkk, dknu, dnunu = _vmft_logZ_moments_vec(kappa, nu)
        return {
            ("mu", "mu"): -kappa * cosB * Bphi * Bphi + kappa * nu * sinB * s,
            ("mu", "kappa"): sinB * Bphi,
            ("mu", "nu"): kappa * (cosB * s * Bphi + sinB * c),
            ("kappa", "kappa"): -dkk,
            ("kappa", "nu"): -sinB * s - dknu,
            ("nu", "nu"): -kappa * cosB * s * s - dnunu,
        }

    def _pdf(self, x, mu, kappa, nu):
        x_arr = np.asarray(x, dtype=float)
        mu_val = _vmft_ensure_scalar(mu, "mu")
        kappa_val = float(np.clip(_vmft_ensure_scalar(kappa, "kappa"), 0.0, _VMFT_KAPPA_UPPER))
        nu_val = _vmft_ensure_scalar(nu, "nu")

        if not np.isfinite(mu_val) or not np.isfinite(kappa_val) or not np.isfinite(nu_val):
            return np.full_like(x_arr, np.nan, dtype=float)

        if kappa_val <= _VMFT_KAPPA_TOL:
            self._c = 1.0 / (2.0 * np.pi)
            return np.full_like(x_arr, self._c, dtype=float)

        table = self._get_vmft_table(kappa_val, nu_val)
        phi = ((x_arr - mu_val + np.pi) % (2.0 * np.pi)) - np.pi
        log_kernel = kappa_val * np.cos(phi + nu_val * np.sin(phi))
        log_pdf = log_kernel + table["log_normalizer"]
        pdf_vals = np.exp(log_pdf)
        self._c = table["normalizer"]
        return pdf_vals

    def pdf(self, x, mu, kappa, nu, *args, **kwargs):
        r"""
        Probability density function of the Flat-topped von Mises distribution.

        $$
        f(\theta) = c \exp(\kappa \cos(\theta - \mu + \nu \sin(\theta - \mu)))
        $$

        , where `c` is the normalizing constant:

        $$
        c = \frac{1}{\int_{-\pi}^{\pi} \exp(\kappa \cos(\theta - \mu + \nu \sin(\theta - \mu))) d\theta}
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF, defined on the interval $[0, 2\pi)$.
        mu : float
            Location parameter, $0 \leq \mu \leq 2\pi$. This is the mean direction when $\nu = 0$.
        kappa : float
            Concentration parameter, $\kappa \geq 0$. Higher values indicate a sharper peak around $\mu$.
        nu : float
            Shape parameter, $-1 \leq \nu \leq 1$. Controls the flattening or sharpening of the peak:
            - $\nu > 0$: sharper peaks.
            - $\nu < 0$: flatter peaks.

        Returns
        -------
        pdf_values : array_like
            Values of the probability density function at the specified points.


        Notes
        -----
        - The normalization constant $c$ is computed numerically, as the integral generally
        does not have a closed-form solution.
        - Special cases:
            - When $\nu = 0$, the distribution reduces to the standard von Mises distribution.
            - When $\kappa = 0$, the distribution becomes uniform on $[0, 2\pi)$.
        """
        mu_val = _vmft_ensure_scalar(mu, "mu")
        kappa_val = float(np.clip(_vmft_ensure_scalar(kappa, "kappa"), 0.0, _VMFT_KAPPA_UPPER))
        nu_val = _vmft_ensure_scalar(nu, "nu")
        return super().pdf(x, mu_val, kappa_val, nu_val, *args, **kwargs)

    def _logpdf(self, x, mu, kappa, nu):
        # the same log-space assembly ``_pdf`` exponentiates (log-kernel +
        # cached table log-normalizer), returned before the exp so the
        # antipodal tail stays finite at concentrations where the density
        # underflows (κ ≳ 360 for ν = 0)
        if any(_vmft_as_scalar(v) is None for v in (mu, kappa, nu)):
            # per-observation parameters — regression contract path
            return _vmft_logpdf_vec(x, mu, kappa, nu)
        x_arr = np.asarray(x, dtype=float)
        mu_val = _vmft_ensure_scalar(mu, "mu")
        kappa_val = float(np.clip(_vmft_ensure_scalar(kappa, "kappa"), 0.0, _VMFT_KAPPA_UPPER))
        nu_val = _vmft_ensure_scalar(nu, "nu")

        if not np.isfinite(mu_val) or not np.isfinite(kappa_val) or not np.isfinite(nu_val):
            return np.full_like(x_arr, np.nan, dtype=float)

        if kappa_val <= _VMFT_KAPPA_TOL:
            return np.full_like(x_arr, -np.log(2.0 * np.pi), dtype=float)

        table = self._get_vmft_table(kappa_val, nu_val)
        phi = ((x_arr - mu_val + np.pi) % (2.0 * np.pi)) - np.pi
        return kappa_val * np.cos(phi + nu_val * np.sin(phi)) + table["log_normalizer"]

    def logpdf(self, x, mu, kappa, nu, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the flat-topped
        von Mises distribution: the log-kernel
        $\kappa\cos(\phi + \nu\sin\phi)$ plus the cached log-normalizer —
        finite across the advertised parameter range, including tails where
        the density itself underflows.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        kappa : float
            Concentration parameter, 0 <= kappa <= 1e3.
        nu : float
            Shape parameter, -1 <= nu <= 1.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        if any(_vmft_as_scalar(v) is None for v in (mu, kappa, nu)):
            # per-observation parameters — regression contract path (the
            # `vmftlss` general family calls `dist.logpdf(y, **array_params)`)
            return _vmft_logpdf_vec(x, mu, kappa, nu)
        mu_val = _vmft_ensure_scalar(mu, "mu")
        kappa_val = float(np.clip(_vmft_ensure_scalar(kappa, "kappa"), 0.0, _VMFT_KAPPA_UPPER))
        nu_val = _vmft_ensure_scalar(nu, "nu")
        return super().logpdf(x, mu_val, kappa_val, nu_val, *args, **kwargs)

    def _cdf(self, x, mu, kappa, nu):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        flat = arr.reshape(-1)

        if flat.size == 0:
            return arr.astype(float)

        mu_val = _vmft_ensure_scalar(mu, "mu")
        kappa_val = float(np.clip(_vmft_ensure_scalar(kappa, "kappa"), 0.0, _VMFT_KAPPA_UPPER))
        nu_val = _vmft_ensure_scalar(nu, "nu")

        if not np.isfinite(mu_val) or not np.isfinite(kappa_val) or not np.isfinite(nu_val):
            return np.full_like(arr, np.nan, dtype=float)

        two_pi = 2.0 * np.pi

        if kappa_val <= _VMFT_KAPPA_TOL:
            cdf_flat = flat / two_pi
        else:
            table = self._get_vmft_table(kappa_val, nu_val)
            phi = ((flat - mu_val + np.pi) % two_pi) - np.pi
            phi_start = ((-mu_val + np.pi) % two_pi) - np.pi
            H = table["cdf_interp"](phi)
            H_start = float(table["cdf_interp"](phi_start))
            cdf_flat = np.where(H < H_start, H - H_start + 1.0, H - H_start)
            cdf_flat = np.clip(cdf_flat, 0.0, 1.0)

        if arr.ndim == 0:
            value = float(cdf_flat[0])
            if np.isclose(float(wrapped), two_pi, rtol=0.0, atol=1e-12):
                return 1.0
            return value

        result = cdf_flat.reshape(arr.shape)
        mask_upper = np.isclose(arr, two_pi, rtol=0.0, atol=1e-12)
        if np.any(mask_upper):
            result = result.copy()
            result[mask_upper] = 1.0
        return result

    def _ppf(self, q, mu, kappa, nu):
        mu_val = _vmft_ensure_scalar(mu, "mu")
        kappa_val = _vmft_ensure_scalar(kappa, "kappa")
        nu_val = _vmft_ensure_scalar(nu, "nu")

        q_arr = np.asarray(q, dtype=float)
        flat = q_arr.reshape(-1)
        if flat.size == 0:
            return q_arr.astype(float)

        if not np.isfinite(mu_val) or not np.isfinite(kappa_val) or not np.isfinite(nu_val):
            return np.full_like(q_arr, np.nan, dtype=float)

        two_pi = 2.0 * np.pi
        result = np.full_like(flat, np.nan, dtype=float)

        valid = np.isfinite(flat) & (flat >= 0.0) & (flat <= 1.0)
        if not np.any(valid):
            shaped = result.reshape(q_arr.shape)
            return float(shaped) if q_arr.ndim == 0 else shaped

        q_valid = flat[valid]
        close_zero = np.isclose(q_valid, 0.0, rtol=0.0, atol=1e-12)
        close_one = np.isclose(q_valid, 1.0, rtol=0.0, atol=1e-12)

        if kappa_val <= _VMFT_KAPPA_TOL:
            theta = (two_pi * q_valid) % two_pi
            if np.any(close_zero):
                theta[close_zero] = 0.0
            if np.any(close_one):
                theta[close_one] = two_pi
            result[valid] = theta
        else:
            table = self._get_vmft_table(kappa_val, nu_val)
            phi_grid = table["phi"]
            cdf_grid = table["cdf"]
            cdf_interp = table["cdf_interp"]
            inv_interp = table["inv_cdf_interp"]

            phi_start = ((-mu_val + np.pi) % two_pi) - np.pi
            H_start = float(cdf_interp(phi_start))

            # Prepare bracket indices for each quantile
            targets = (H_start + q_valid) % 1.0
            phi_guess = (
                inv_interp(targets)
                if inv_interp is not None
                else np.interp(targets, cdf_grid, phi_grid, left=phi_grid[0], right=phi_grid[-1])
            )

            # Vectorized safeguarded Newton on the monotone cdf (mirrors the
            # inverse-Batschelet ppf): every quantile refined together, each
            # with its own grid bracket, converged points frozen. The pdf
            # derivative is the closed-form flat-topped vM kernel, so no
            # interpolator call is needed inside the loop.
            i_hi = np.clip(np.searchsorted(cdf_grid, targets, side="right"),
                           1, len(phi_grid) - 1)
            phi_lo = phi_grid[i_hi - 1].astype(float, copy=True)
            phi_hi = phi_grid[i_hi].astype(float, copy=True)
            phi = np.clip(phi_guess, phi_lo, phi_hi)

            done = np.zeros(q_valid.shape, dtype=bool)
            tiny = np.finfo(float).tiny
            log_norm = table["log_normalizer"]
            for _ in range(_VMFT_NEWTON_MAXITER):
                H_phi = np.asarray(cdf_interp(phi), dtype=float)
                residual = H_phi - targets
                derivative = np.maximum(
                    np.exp(kappa_val * np.cos(phi + nu_val * np.sin(phi)) + log_norm),
                    tiny)

                done |= (np.abs(residual) <= _VMFT_NEWTON_TOL) & (
                    (phi_hi - phi_lo) <= _VMFT_NEWTON_WIDTH_TOL)
                if np.all(done):
                    break

                hi_upd = residual > 0.0
                phi_hi = np.where(hi_upd, np.minimum(phi_hi, phi), phi_hi)
                phi_lo = np.where(~hi_upd, np.maximum(phi_lo, phi), phi_lo)

                cand = phi - residual / derivative
                fallback = ~np.isfinite(cand) | (cand <= phi_lo) | (cand >= phi_hi)
                cand = np.where(fallback, 0.5 * (phi_lo + phi_hi), cand)
                phi = np.where(done, phi, np.clip(cand, phi_lo, phi_hi))

            theta = (mu_val + phi) % two_pi
            theta[close_zero] = 0.0
            theta[close_one] = two_pi
            result[valid] = theta

        shaped = result.reshape(q_arr.shape)
        if q_arr.ndim == 0:
            return float(shaped)
        return shaped

    def ppf(self, q, mu, kappa, nu, *args, **kwargs):
        r"""
        Percent-point function (quantile) of the flat-topped von Mises distribution.

        Quantiles are computed by reusing the cached cumulative table described in
        `cdf`. Starting from the monotone inverse of the tabulated primitive
        $H_{\kappa,\nu}$, the implementation applies up to
        :data:`_VMFT_NEWTON_MAXITER` safeguarded Newton steps with derivative
        $f(\theta) = \exp[\kappa \cos(\phi + \nu \sin \phi)]/Z$ to achieve
        machine-precision agreement (dual stopping on residual and bracket width).
        Boundary quantiles default to the support endpoints $0$ and $2\pi$.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate (0 <= q <= 1).
        mu : float
            Location parameter, $0 \le \mu \le 2\pi$.
        kappa : float
            Concentration parameter, $\kappa \ge 0$.
        nu : float
            Shape parameter, $-1 \le \nu \le 1$.

        Returns
        -------
        ppf_values : array_like
            Angles corresponding to the probabilities in `q`.
        """
        mu_val = _vmft_ensure_scalar(mu, "mu")
        kappa_val = _vmft_ensure_scalar(kappa, "kappa")
        nu_val = _vmft_ensure_scalar(nu, "nu")
        return super().ppf(q, mu_val, kappa_val, nu_val, *args, **kwargs)

    def _rvs(self, mu, kappa, nu, size=None, random_state=None):
        rng = self._init_rng(random_state)

        mu_val = _vmft_ensure_scalar(mu, "mu") % (2.0 * np.pi)
        kappa_val = float(np.clip(_vmft_ensure_scalar(kappa, "kappa"), 0.0, _VMFT_KAPPA_UPPER))
        nu_val = _vmft_ensure_scalar(nu, "nu")

        if not np.isfinite(mu_val) or not np.isfinite(kappa_val) or not np.isfinite(nu_val):
            raise ValueError("`mu`, `kappa`, and `nu` must be finite scalars.")

        if size is None:
            shape = ()
            total = 1
        else:
            if np.isscalar(size):
                shape = (int(size),)
            else:
                shape = tuple(int(dim) for dim in np.atleast_1d(size))
            total = int(np.prod(shape, dtype=int))
            if total < 0:
                raise ValueError("`size` must describe a non-negative number of samples.")
        two_pi = 2.0 * np.pi

        if total == 0:
            empty = np.empty(shape, dtype=float)
            return float(empty) if empty.ndim == 0 else empty

        if kappa_val <= _VMFT_KAPPA_TOL:
            samples = rng.uniform(0.0, two_pi, size=shape)
            if samples.ndim == 0:
                return float(samples)
            return samples

        # Inverse-transform through the spectral cdf table: the centered
        # quantile function maps U(0,1) draws straight to angles (the μ
        # shift is absorbed by uniformity), one vectorized Pchip evaluation
        # per call. This replaced the curvature-matched rejection sampler,
        # whose von Mises envelope κ_e = κ(1+ν)² under-covers a flat-topped
        # target's shoulders — the acceptance multiplier exploded with κ
        # for ν ≠ 0 (12 s for 5 draws at κ = 5, ν = 0.5; an effective hang
        # from κ ≈ 100).
        table = self._get_vmft_table(kappa_val, nu_val)
        inv_interp = table["inv_cdf_interp"]
        u = rng.random(size=total)
        if inv_interp is not None:
            phi = np.clip(np.asarray(inv_interp(u), dtype=float), -np.pi, np.pi)
        else:  # degenerate single-knot cdf: effectively uniform
            phi = (u - 0.5) * two_pi

        samples = np.mod(mu_val + phi, two_pi).reshape(shape)
        if samples.ndim == 0:
            return float(samples)
        return samples

    def rvs(self, mu=None, kappa=None, nu=None, size=None, random_state=None):
        r"""
        Draw random variates from the flat-topped von Mises distribution.

        Sampling is by inverse transform through the same spectral cdf table
        that serves `cdf`/`ppf`: uniform draws are pushed through the
        monotone inverse-cdf interpolant of the centered density and shifted
        by $\mu$ — one vectorized evaluation per call, with cost independent
        of $\kappa$ and $\nu$. (An earlier acceptance–rejection scheme with
        a curvature-matched von Mises envelope degraded catastrophically for
        $\nu \ne 0$ at large $\kappa$: matching the mode curvature
        under-covers a flat-topped density's shoulders, so the envelope
        constant grew without bound.)

        Parameters
        ----------
        mu : float
            Location parameter, $0 \le \mu \le 2\pi$.
        kappa : float
            Concentration parameter, $\kappa \ge 0$.
        nu : float
            Shape parameter, $-1 \le \nu \le 1$.
        size : int or tuple of ints, optional
            Output shape.
        random_state : {None, int, np.random.Generator}, optional
            Random number generator specification.

        Returns
        -------
        rvs : array_like
            Random variates on $[0, 2\pi)$.
        """
        return super().rvs(mu, kappa, nu, size=size, random_state=random_state)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        optimizer="L-BFGS-B",
        options=None,
        nu_grid=None,
        kappa_bounds=(1e-6, _VMFT_KAPPA_UPPER),
        nu_bounds=(-0.99, 0.99),
        return_info=False,
        **minimize_kwargs,
    ):
        r"""
        Estimate $(\mu, \kappa, \nu)$ from circular data.

        The default ``method='mle'`` maximises the weighted log-likelihood

        $$
        \ell(\mu, \kappa, \nu) = \sum_i w_i
        \left[
            \kappa \cos(\phi_i + \nu \sin \phi_i) - \log Z(\kappa, \nu)
        \right],\quad
        \phi_i = (\theta_i - \mu) \bmod 2\pi,
        $$

        where $Z$ is the normalising constant reused from the cached spectral
        table. The routine initialises $(\mu, \kappa)$ from the first trigonometric
        moment and profiles a small grid for $\nu$ before bounded optimisation
        (default L-BFGS-B) with $\kappa \in$ ``kappa_bounds`` and
        $\nu \in$ ``nu_bounds``.

        Parameters
        ----------
        data : array_like
            Sample of angles.
        weights : array_like, optional
            Non-negative weights broadcastable to ``data``.
        method : {'mle', 'moments'}, default 'mle'
            Estimation method. ``'moments'`` returns the circular mean,
            ``circ_kappa``, and $\nu=0$.
        optimizer : str, optional
            SciPy optimiser to use when ``method='mle'``.
        options : dict, optional
            Optimiser options forwarded to :func:`scipy.optimize.minimize`.
        nu_grid : array_like, optional
            Candidate $\nu$ values for initial profiling. Defaults to a small grid
            spanning ``nu_bounds``.
        kappa_bounds : tuple, optional
            Lower/upper bounds for $\kappa$ during optimisation.
        nu_bounds : tuple, optional
            Lower/upper bounds for $\nu$ during optimisation.
        return_info : bool, optional
            If True, also return a dictionary with optimisation diagnostics.
        **minimize_kwargs :
            Additional keyword arguments passed to :func:`scipy.optimize.minimize`.

        Returns
        -------
        params : tuple
            Estimated parameters ``(mu, kappa, nu)``.
        info : dict, optional
            Returned when ``return_info=True`` with fields such as ``loglik``,
            ``n_effective`` and ``converged``.
        """

        minimize_kwargs = self._sanitize_fit_kwargs(minimize_kwargs)
        minimize_kwargs.pop("floc", None)
        minimize_kwargs.pop("fscale", None)

        data_arr = self._wrap_angles(np.asarray(data, dtype=float)).ravel()
        if data_arr.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(data_arr, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, data_arr.shape).astype(float, copy=False).ravel()

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = float(w_sum**2 / np.sum(w**2))

        mu_mom, r1 = circ_mean_and_r(alpha=data_arr, w=w)
        if not np.isfinite(mu_mom):
            mu_mom = 0.0
        mu_mom = float(np.mod(mu_mom, 2.0 * np.pi))
        r1 = float(np.clip(r1, 1e-12, 1.0 - 1e-12))

        n_adjust = int(max(1, round(w_sum)))
        kappa_mom = float(np.clip(circ_kappa(r=r1, n=n_adjust), kappa_bounds[0], kappa_bounds[1]))

        if nu_grid is None:
            lower_nu = float(max(nu_bounds[0], -0.9))
            upper_nu = float(min(nu_bounds[1], 0.9))
            nu_grid = np.linspace(lower_nu, upper_nu, 7)
        else:
            nu_grid = np.asarray(nu_grid, dtype=float)

        def nll(params):
            mu_param, kappa_param, nu_param = params

            if not (0.0 <= mu_param <= 2.0 * np.pi):
                return np.inf
            if not (kappa_bounds[0] <= kappa_param <= kappa_bounds[1]):
                return np.inf
            if not (nu_bounds[0] <= nu_param <= nu_bounds[1]):
                return np.inf

            mu_wrapped = float(np.mod(mu_param, 2.0 * np.pi))
            two_pi = 2.0 * np.pi
            phi = ((data_arr - mu_wrapped + np.pi) % two_pi) - np.pi

            if kappa_param <= _VMFT_KAPPA_TOL:
                log_pdf = -np.log(two_pi)
                return float(-np.sum(w * log_pdf))

            table = self._get_vmft_table(float(kappa_param), float(nu_param))

            log_kernel = kappa_param * np.cos(phi + nu_param * np.sin(phi))
            log_pdf = log_kernel + table["log_normalizer"]
            if not np.all(np.isfinite(log_pdf)):
                return np.inf
            return float(-np.sum(w * log_pdf))

        method_key = str(method).lower()

        if method_key == "moments":
            estimates = (mu_mom, kappa_mom, 0.0)
            if return_info:
                info = {
                    "method": "moments",
                    "converged": True,
                    "loglik": float(-nll(estimates)),
                    "n_effective": n_eff,
                }
                return estimates, info
            return estimates

        if method_key != "mle":
            raise ValueError("`method` must be one of {'mle', 'moments'}.")

        best_nu = 0.0
        best_score = nll((mu_mom, kappa_mom, best_nu))
        for candidate in np.unique(np.concatenate(([0.0], nu_grid))):
            score = nll((mu_mom, kappa_mom, float(candidate)))
            if score < best_score:
                best_score = score
                best_nu = float(candidate)

        init = np.array([mu_mom, kappa_mom, best_nu], dtype=float)
        bounds = [
            (0.0, 2.0 * np.pi),
            (kappa_bounds[0], kappa_bounds[1]),
            (nu_bounds[0], nu_bounds[1]),
        ]

        options = {} if options is None else dict(options)

        optimizer_used = optimizer

        result = minimize(
            nll,
            init,
            method=optimizer,
            bounds=bounds,
            options=options,
            **minimize_kwargs,
        )

        if not result.success and optimizer != "Powell":
            fallback = minimize(
                nll,
                init,
                method="Powell",
                bounds=bounds,
                options={},
                **minimize_kwargs,
            )
            if fallback.success:
                result = fallback
                optimizer_used = "Powell"

        if not result.success:
            raise RuntimeError(f"Maximum likelihood fit failed: {result.message}")

        mu_hat = self._wrap_direction(float(result.x[0]))
        kappa_hat = float(np.clip(result.x[1], kappa_bounds[0], kappa_bounds[1]))
        nu_hat = float(np.clip(result.x[2], nu_bounds[0], nu_bounds[1]))

        estimates = (mu_hat, kappa_hat, nu_hat)
        if not return_info:
            return estimates

        info = {
            "method": "mle",
            "loglik": float(-result.fun),
            "n_effective": n_eff,
            "converged": bool(result.success),
            "optimizer": optimizer_used,
            "nit": getattr(result, "nit", np.nan),
            "nfev": getattr(result, "nfev", np.nan),
            "message": result.message,
        }
        return estimates, info

    def cdf(self, x, mu, kappa, nu, *args, **kwargs):
        r"""
        Cumulative distribution function of the flat-topped von Mises distribution.

        Let $\phi = (\theta - \mu) \bmod 2\pi$ re-centred onto $[-\pi, \pi]$ and
        $g_{\kappa,\nu}(\phi) = \exp\!\bigl[\kappa \cos(\phi + \nu \sin \phi)\bigr]$.
        The normalised primitive
        $$
        H_{\kappa,\nu}(\phi) = \frac{1}{Z} \int_{-\pi}^{\phi} g_{\kappa,\nu}(t)\,dt,
        \qquad Z = \int_{-\pi}^{\pi} g_{\kappa,\nu}(t)\,dt,
        $$
        is approximated with spectral accuracy by a trapezoidal rule on an
        equispaced grid (size selected from $O(\sqrt{\kappa})$). The CDF on
        $[0, 2\pi)$ then follows from $F(\theta) = H_{\kappa,\nu}(\phi) -
        H_{\kappa,\nu}(\phi_0)$ with $\phi_0 = ((-\mu) \bmod 2\pi) - \pi$. The
        precomputed cumulative grid is cached per $(\kappa, \nu)$, so repeated
        evaluations are $O(1)$ once the table is built.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        mu : float
            Location parameter, $0 \le \mu \le 2\pi$.
        kappa : float
            Concentration parameter, $\kappa \ge 0$ (capped internally at
            :data:`_VMFT_KAPPA_UPPER` for numerical stability).
        nu : float
            Shape parameter, $-1 \le \nu \le 1$.

        Returns
        -------
        cdf_values : array_like
            Cumulative probabilities corresponding to `x`.
        """
        mu_val = _vmft_ensure_scalar(mu, "mu")
        kappa_val = _vmft_ensure_scalar(kappa, "kappa")
        nu_val = _vmft_ensure_scalar(nu, "nu")
        return super().cdf(x, mu_val, kappa_val, nu_val, *args, **kwargs)

    def _get_vmft_table(self, kappa, nu, grid_size=None):
        kappa_val = float(kappa)
        nu_val = float(nu)
        if grid_size is None:
            grid_size = _vmft_grid_size(kappa_val, nu_val)
        grid_int = int(grid_size)
        key = (kappa_val, nu_val, grid_int)
        table = self._vmft_table_cache.get(key)
        if table is None:
            table = _vmft_build_table(kappa_val, nu_val, grid_int)
            self._vmft_table_cache[key] = table
        return table

vonmises_flattopped = vonmises_flattopped_gen(name="vonmises_flattopped")
vmftlss = CircularLL(vonmises_flattopped, name="vmftlss")

##############################################
## Helper Functions: Flat-topped von Mises  ##
##############################################


def _vmft_grid_size(kappa, nu):
    sharpness = (1.0 + abs(nu)) * np.sqrt(max(kappa, 0.0) + 1.0)
    target = _VMFT_GRID_BASE + _VMFT_GRID_SHARPNESS * sharpness
    target = float(np.clip(target, _VMFT_MIN_GRID, _VMFT_MAX_GRID))
    power = int(np.ceil(np.log2(target)))
    size = 1 << power
    size = int(np.clip(size, _VMFT_MIN_GRID, _VMFT_MAX_GRID))
    if size % 2 != 0:
        size += 1
    return size


def _vmft_build_table(kappa, nu, grid_size):
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4.")
    two_pi = 2.0 * np.pi
    phi = np.linspace(-np.pi, np.pi, grid_size + 1, dtype=float)
    log_kernel = kappa * np.cos(phi + nu * np.sin(phi))
    log_max = np.max(log_kernel)
    shifted = log_kernel - log_max
    weights = np.ones_like(phi)
    weights[0] = 0.5
    weights[-1] = 0.5
    log_sum = logsumexp(shifted, b=weights)
    log_Z = np.log(two_pi / grid_size) + log_max + log_sum
    log_normalizer = -log_Z
    normalizer = float(np.exp(log_normalizer))

    log_pdf = log_kernel + log_normalizer
    pdf = np.exp(np.clip(log_pdf, -700.0, 700.0))
    pdf = np.maximum(pdf, np.finfo(float).tiny)
    pdf[-1] = pdf[0]

    avg = 0.5 * (pdf[:-1] + pdf[1:])
    cumulative = np.concatenate(([0.0], np.cumsum(avg))) * (two_pi / grid_size)
    cumulative = np.clip(cumulative, 0.0, 1.0)
    cumulative = np.maximum.accumulate(cumulative)
    cumulative[-1] = 1.0

    cdf_interp = PchipInterpolator(phi, cumulative, extrapolate=True)

    knots = _inverse_cdf_knots(phi, cumulative)
    inv_interp = (
        PchipInterpolator(knots[0], knots[1], extrapolate=True)
        if knots is not None
        else None
    )

    return {
        "phi": phi,
        "pdf": pdf,
        "cdf": cumulative,
        "normalizer": normalizer,
        "log_normalizer": float(log_normalizer),
        "cdf_interp": cdf_interp,
        "inv_cdf_interp": inv_interp,
        "grid_size": int(grid_size),
        "kappa": float(kappa),
        "nu": float(nu),
    }


def _kernel_vmft(x, mu, kappa, nu):
    return np.exp(kappa * np.cos(x - mu + nu * np.sin(x - mu)))


def _c_vmft(kappa, nu):
    if kappa <= _VMFT_KAPPA_TOL:
        return 1.0 / (2.0 * np.pi)
    table = _vmft_build_table(float(kappa), float(nu), _vmft_grid_size(float(kappa), float(nu)))
    return table["normalizer"]


def _vmft_as_scalar(value):
    """Collapse a parameter to a float (tolerating constant arrays), or return
    ``None`` if it genuinely varies — the scalar-vs-regression-path switch the
    flat-topped vM ``pdf``/``logpdf`` use (mirrors ``_jp_as_scalar``)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.reshape(-1)
    if flat.size == 1:
        return float(flat[0])
    first = flat[0]
    if np.all(flat == first):
        return float(first)
    return None


def _vmft_log_c_vec(kappa, nu):
    """Vectorized ``log c(κ,ν) = −log ∫ e^{κ cos(φ + ν sinφ)} dφ`` over
    per-observation params, once per unique (κ,ν) on one shared grid (the
    ibslss ``_invbat_log_c_array`` pattern — overflow-safe max-subtracted
    trapezoid). κ ≤ tol → uniform. Matches the cached scalar table's
    ``log_normalizer`` at the same grid size."""
    kappa, nu = np.broadcast_arrays(
        np.asarray(kappa, dtype=float), np.asarray(nu, dtype=float)
    )
    out = np.full(kappa.shape, -np.log(2.0 * np.pi))
    live = kappa > _VMFT_KAPPA_TOL
    if not np.any(live):
        return out
    kl = np.clip(kappa[live], 0.0, _VMFT_KAPPA_UPPER)
    nl = nu[live]
    pairs, inverse = np.unique(
        np.stack([kl, nl], axis=1), axis=0, return_inverse=True
    )
    k = pairs[:, 0]
    nv = pairs[:, 1]
    grid = _vmft_grid_size(float(k.max()), float(np.abs(nv).max()))
    phi = np.linspace(-np.pi, np.pi, grid + 1)
    sin_phi = np.sin(phi)
    w = np.ones_like(phi)
    w[0] = w[-1] = 0.5
    log_ker = k[:, None] * np.cos(phi[None, :] + nv[:, None] * sin_phi[None, :])
    mx = np.max(log_ker, axis=1)
    log_Z = (np.log(2.0 * np.pi / grid) + mx
             + np.log(np.exp(log_ker - mx[:, None]) @ w))
    out[live] = (-log_Z)[inverse]
    return out


def _vmft_logpdf_vec(x, mu, kappa, nu):
    """Per-observation flat-topped vM log-density (regression contract): the
    log-kernel ``κ cos(φ + ν sinφ)`` plus the vectorized log-normalizer, each
    datum its own (μ, κ, ν). κ ≤ tol → uniform."""
    x, mu, kappa, nu = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, mu, kappa, nu))
    )
    kappa = np.clip(kappa, 0.0, _VMFT_KAPPA_UPPER)
    phi = ((x - mu + np.pi) % (2.0 * np.pi)) - np.pi
    B = phi + nu * np.sin(phi)
    out = kappa * np.cos(B) + _vmft_log_c_vec(kappa, nu)
    return np.where(kappa <= _VMFT_KAPPA_TOL, -np.log(2.0 * np.pi), out)


def _vmft_logZ_moments_vec(kappa, nu):
    """First and second (κ,ν)-derivatives of ``log Z(κ,ν)`` (``Z = ∫ e^{κ cos B}
    dφ``, ``B = φ + ν sinφ``) as kernel-weighted moments under the density,
    vectorized over all unique (κ,ν) pairs in one grid pass — the normalizer
    block of the flat-topped vM score/Hessian. With ``h_κ = cos B`` and
    ``h_ν = −κ sin B sinφ``:

        ∂log Z/∂κ = E[h_κ],    ∂log Z/∂ν = E[h_ν],
        ∂²log Z/∂a∂b = E[h_{ab}] + Cov(h_a, h_b),

    using the kernel second derivatives ``h_κκ = 0``, ``h_κν = −sin B sinφ``,
    ``h_νν = −κ cos B sin²φ``. Each E[·] is a normalizer-free ratio (the
    e^{κcosB−max} factor cancels), so it is accurate at any concentration.
    Returns ``(dk, dnu, dkk, dknu, dnunu)`` broadcast to the parameter shape."""
    kappa, nu = np.broadcast_arrays(
        np.asarray(kappa, dtype=float), np.asarray(nu, dtype=float)
    )
    shape = kappa.shape
    kf = np.clip(kappa.ravel(), 0.0, _VMFT_KAPPA_UPPER)
    nf = nu.ravel()
    pairs, inverse = np.unique(
        np.stack([kf, nf], axis=1), axis=0, return_inverse=True
    )
    k = pairs[:, 0]
    nv = pairs[:, 1]
    grid = _vmft_grid_size(float(k.max()), float(np.abs(nv).max()))
    phi = np.linspace(-np.pi, np.pi, grid + 1)
    sin_phi = np.sin(phi)
    sin2 = sin_phi * sin_phi
    w = np.ones_like(phi)
    w[0] = w[-1] = 0.5
    B = phi[None, :] + nv[:, None] * sin_phi[None, :]
    cosB = np.cos(B)
    sinB = np.sin(B)
    log_ker = k[:, None] * cosB
    e = np.exp(log_ker - np.max(log_ker, axis=1, keepdims=True)) * w[None, :]
    Z = np.maximum(np.sum(e, axis=1), np.finfo(float).tiny)

    def m(v):
        return np.sum(e * v, axis=1) / Z

    sBsp = sinB * sin_phi[None, :]                 # sin B sinφ
    E_cosB = m(cosB)
    E_sBsp = m(sBsp)
    dk = E_cosB                                    # ∂logZ/∂κ
    dnu = -k * E_sBsp                              # ∂logZ/∂ν = E[h_ν]
    # ∂²logZ/∂a∂b = E[h_ab] + Cov(h_a, h_b); h_κκ = 0
    dkk = m(cosB * cosB) - E_cosB * E_cosB         # Var(cos B)
    cov_k_nu = -k * (m(cosB * sBsp) - E_cosB * E_sBsp)   # Cov(h_κ, h_ν)
    dknu = -E_sBsp + cov_k_nu                      # E[h_κν] = −E[sin B sinφ]
    var_nu = k * k * (m(sBsp * sBsp) - E_sBsp * E_sBsp)  # Var(h_ν)
    dnunu = -k * m(cosB * sin2[None, :]) + var_nu  # E[h_νν] + Var(h_ν)
    vals = np.stack([dk, dnu, dkk, dknu, dnunu], axis=1)
    out = vals[inverse].reshape(shape + (5,))
    return tuple(np.moveaxis(out, -1, 0))


def _vmft_ensure_scalar(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(()))
    unique = np.unique(arr)
    if unique.size == 1:
        return float(unique[0])
    raise ValueError(
        f"Flat-topped von Mises parameter '{name}' must be scalar; "
        "array/broadcasted parameters are not supported because tables are cached per parameter set."
    )


class jonespewsey_gen(_RegressionReady, CircularContinuous):
    """Jones-Pewsey Distribution

    ![jonespewsey](../images/circ-mod-jonespewsey.png)

    Methods
    -------
    pdf(x, mu, kappa, psi)
        Probability density function.

    logpdf(x, mu, kappa, psi)
        Logarithm of the probability density function.

    cdf(x, mu, kappa, psi)
        Cumulative distribution function.

    ppf(q, mu, kappa, psi)
        Percent-point function (inverse CDF).

    rvs(mu, kappa, psi, size=None, random_state=None)
        Random variates by inverse transform on the kernel quantile table.

    fit(data, *, weights=None, method="mle", ...)
        Estimate ``(mu, kappa, psi)`` by moments or maximum likelihood.

    Note
    ----
    Scalar parameters use cached normalisation tables; ``pdf``/``logpdf`` also
    accept per-observation parameter arrays (the regression contract),
    normalised through the shared log-space Gauss–Legendre ladder
    (``_jp_log_c_vec``). Other methods (cdf, rvs, …) remain scalar-only.
    Implementation based on Section 4.3.9 of Pewsey et al. (2013)
    """

    # --- regression overlay (read by the regression engine
    # only). Book names mu/kappa/psi are preserved; ψ ∈ ℝ indexes the family
    # shape (−1 wrapped Cauchy, 0 von Mises, +1 cardioid), so it rides an
    # identity link. l1/l2 split as kernel terms (`_jp_score_terms`, exact
    # closed forms) minus log-normalizer moments (`_jp_logZ_moments_vec`, one
    # Gauss–Legendre sweep per unique (κ, ψ) pair). Note the derivatives stay
    # exact below _JP_KAPPA_TOL while ``logpdf`` flattens to the uniform
    # value there — the value gap is O(κ) ≤ 1e-3 relative, and live scores in
    # that corner let the optimizer escape it. ---
    param_roles = {"mu": "location", "kappa": "concentration", "psi": "shape"}
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): kappa toward 0 (linear) and a firmer ridge (scale 30) on the
    # Jones-Pewsey shape psi toward 0 (= von Mises), keeping its quadrature
    # normalizer fast and non-singular.
    degen_penalty = (_degen_linear("kappa"), _degen_ridge("psi", 30.0))
    default_links = {
        "location": "tanhalf",
        "concentration": "log",
        "shape": "identity",
    }

    def _concentration_start(self, Rbar):
        """Closed-form concentration start from the mean resultant ``Rbar``
        for the regression null model: the von Mises A1-inverse ``kappa`` (the
        ``psi -> 0`` reduction member), clamped — the circlss ``initialize``
        convention (see CircularLL._null_params)."""
        return float(np.clip(A1inv(Rbar), 0.01, 500.0))

    def dlogpdf(self, x, mu, kappa, psi):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        With ``h`` the log-kernel and ``Z`` the normalizer:

        $$\frac{\partial\ell}{\partial\mu} = -h_\phi,\qquad
          \frac{\partial\ell}{\partial\kappa} = h_\kappa -
          \mathbb{E}[h_\kappa],\qquad
          \frac{\partial\ell}{\partial\psi} = h_\psi - \mathbb{E}[h_\psi],$$

        the expectations under the JP density itself (one quadrature per
        unique (κ, ψ); see ``_jp_logZ_moments``). Vectorizes over
        per-observation parameter arrays; returns a book-named dict.
        """
        x = np.asarray(x, dtype=float)
        mu_b, kappa_b, psi_b = (
            np.asarray(v, dtype=float) for v in (mu, kappa, psi)
        )
        t = _jp_score_terms(x - mu_b, kappa_b, psi_b, second=False)
        dk, dp, *_ = _jp_logZ_moments_vec(kappa_b, psi_b)
        return {"mu": -t["hphi"], "kappa": t["hk"] - dk, "psi": t["hp"] - dp}

    def d2logpdf(self, x, mu, kappa, psi):
        r"""Second derivatives of ``logpdf`` (l2) — unique unordered pairs:
        kernel terms minus the log-normalizer's covariance-form second
        moments, ``∂²log Z/∂θ_a∂θ_b = E[h_{ab} + h_a h_b] − E[h_a]E[h_b]``
        (μ never enters Z)."""
        x = np.asarray(x, dtype=float)
        mu_b, kappa_b, psi_b = (
            np.asarray(v, dtype=float) for v in (mu, kappa, psi)
        )
        t = _jp_score_terms(x - mu_b, kappa_b, psi_b, second=True)
        _, _, dkk, dkp, dpp = _jp_logZ_moments_vec(kappa_b, psi_b)
        return {
            ("mu", "mu"): t["hphiphi"],
            ("mu", "kappa"): -t["hphik"],
            ("mu", "psi"): -t["hphip"],
            ("kappa", "kappa"): t["hkk"] - dkk,
            ("kappa", "psi"): t["hkp"] - dkp,
            ("psi", "psi"): t["hpp"] - dpp,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._series_cache = {}

    def _argcheck(self, mu, kappa, psi):
        try:
            mu_arr, kappa_arr, psi_arr = np.broadcast_arrays(mu, kappa, psi)
        except ValueError:
            return False
        return (
            (mu_arr >= 0.0)
            & (mu_arr <= 2.0 * np.pi)
            & (kappa_arr >= 0.0)
            & np.isfinite(kappa_arr)
            & np.isfinite(psi_arr)
        )

    def _pdf(self, x, mu, kappa, psi):
        x = np.asarray(x, dtype=float)
        kappa_scalar = _jp_as_scalar(kappa)
        psi_scalar = _jp_as_scalar(psi)

        if kappa_scalar is None or psi_scalar is None:
            # Per-observation (κ_i, ψ_i) — the regression contract path
            # (concentration smoothing on a shape family). Assembled in log
            # space: the raw kernel peaks at e^κ and overflows for κ ≳ 709
            # Uniform/von-Mises reductions applied
            # element-wise to match the scalar branch exactly.
            mu_b, kappa_b, psi_b = np.broadcast_arrays(
                *(np.asarray(a, dtype=float) for a in (mu, kappa, psi))
            )
            phi = x - mu_b
            logc = _jp_log_c_vec(kappa_b, psi_b)
            h = _jp_score_terms(phi, kappa_b, psi_b, second=False)["h"]
            vm = np.abs(psi_b) < _JP_PSI_TOL
            h = np.where(vm, kappa_b * np.cos(phi), h)
            dens = np.exp(h + logc)
            return np.where(kappa_b < _JP_KAPPA_TOL, 1.0 / (2.0 * np.pi), dens)

        if not np.isfinite(kappa_scalar) or not np.isfinite(psi_scalar):
            return np.full_like(x, np.nan, dtype=float)

        if abs(kappa_scalar) < _JP_KAPPA_TOL:
            return np.full_like(x, 1.0 / (2.0 * np.pi), dtype=float)

        log_c = self._get_cached_normalizer(
            lambda: _jp_log_c(kappa_scalar, psi_scalar),
            kappa_scalar,
            psi_scalar,
        )
        self._c = float(np.exp(log_c))  # legacy attribute (write-only)

        if abs(psi_scalar) < _JP_PSI_TOL:
            return np.exp(kappa_scalar * np.cos(x - mu) + log_c)

        h = _jp_score_terms(x - mu, kappa_scalar, psi_scalar, second=False)["h"]
        return np.exp(h + log_c)

    def pdf(self, x, mu, kappa, psi, *args, **kwargs):
        r"""
        Probability density function of the Jones-Pewsey distribution.

        $$
        f(\theta) = c(\kappa, \psi)
        \Big(\cosh(\kappa \psi) + \sinh(\kappa \psi) \cos(\theta - \mu)\Big)^{1/\psi},
        $$

        where ``c(\kappa, \psi)`` is the normalizing constant, evaluated numerically with
        stable special-case reductions:

            - ``c = 1 / (2\pi)`` when ``\kappa`` is effectively zero (uniform limit).
            - ``c = 1 / (2\pi I_0(\kappa))`` as ``\psi \to 0`` (von Mises limit).

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        kappa : float
            Concentration parameter, kappa >= 0.
        psi : float
            Shape parameter, -∞ <= psi <= ∞.

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """
        return super().pdf(x, mu, kappa, psi, *args, **kwargs)

    def _logpdf(self, x, mu, kappa, psi):
        # the log-space assembly ``_pdf`` exponentiates, returned before the
        # exp: h from ``_jp_score_terms`` is stable at any κ and the
        # log-normalizer already exists, so the antipodal tail stays finite
        # (e.g. ≈ −1200 at κ=600, ψ=0.1) where the density underflows — and
        # finite at the deep ψ < 0 mode where the density overflows to inf.
        # Mirrors ``_pdf`` branch-for-branch, incl. the per-observation
        # regression path.
        x = np.asarray(x, dtype=float)
        kappa_scalar = _jp_as_scalar(kappa)
        psi_scalar = _jp_as_scalar(psi)

        if kappa_scalar is None or psi_scalar is None:
            mu_b, kappa_b, psi_b = np.broadcast_arrays(
                *(np.asarray(a, dtype=float) for a in (mu, kappa, psi))
            )
            phi = x - mu_b
            logc = _jp_log_c_vec(kappa_b, psi_b)
            h = _jp_score_terms(phi, kappa_b, psi_b, second=False)["h"]
            vm = np.abs(psi_b) < _JP_PSI_TOL
            h = np.where(vm, kappa_b * np.cos(phi), h)
            return np.where(
                kappa_b < _JP_KAPPA_TOL, -np.log(2.0 * np.pi), h + logc
            )

        if not np.isfinite(kappa_scalar) or not np.isfinite(psi_scalar):
            return np.full_like(x, np.nan, dtype=float)

        if abs(kappa_scalar) < _JP_KAPPA_TOL:
            return np.full_like(x, -np.log(2.0 * np.pi), dtype=float)

        log_c = self._get_cached_normalizer(
            lambda: _jp_log_c(kappa_scalar, psi_scalar),
            kappa_scalar,
            psi_scalar,
        )

        if abs(psi_scalar) < _JP_PSI_TOL:
            return kappa_scalar * np.cos(x - mu) + log_c

        h = _jp_score_terms(x - mu, kappa_scalar, psi_scalar, second=False)["h"]
        return h + log_c

    def logpdf(self, x, mu, kappa, psi, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the Jones-Pewsey
        distribution: the stable log-kernel
        $h(\theta; \kappa, \psi) = \tfrac{1}{\psi}\log(\cosh\kappa\psi +
        \sinh\kappa\psi\cos(\theta-\mu))$ plus the log normalizing constant,
        assembled entirely in log space — finite in underflowed tails and at
        deep $\psi < 0$ modes whose density exceeds the double range.

        Accepts per-observation parameter arrays like ``pdf`` (the
        regression contract).

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        kappa : float
            Concentration parameter, kappa >= 0.
        psi : float
            Shape parameter.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, mu, kappa, psi, *args, **kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """Trigonometric moment via the exact ladder cosine moments: the
        centered JP law is even, so m_p = ᾱ_p·e^{ipμ} with ᾱ_p from
        ``_jp_cos_moment`` — our own evaluation of the off-cut Legendre
        ratio P_{1/ψ}-type/(2π P_{1/ψ})(cosh κψ) through its
        Mehler–Dirichlet integral representation on the feature-scale
        ladder (scipy's ``lpmv`` is the on-cut Ferrers function and
        silently fails off the cut; see ``_jp_log_c``). Exact at any
        concentration; vM and uniform limits closed-form."""
        shape_args, non_shape_kwargs = self._separate_shape_parameters(
            args, kwargs, "trig_moment"
        )
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        mu, kappa, psi = (float(np.asarray(v, dtype=float))
                          for v in self._parse_args(*shape_args, **call_kwargs)[0])

        if not np.isscalar(p):
            raise ValueError("`p` must be an integer scalar.")
        if int(round(p)) != p:
            raise ValueError("`p` must be an integer.")
        k = int(round(p))
        if k == 0:
            return complex(1.0, 0.0)
        ak = abs(k)
        value = _jp_cos_moment(kappa, psi, ak) * np.exp(1j * ak * mu)
        return complex(np.conjugate(value)) if k < 0 else complex(value)

    def _cdf(self, x, mu, kappa, psi):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        flat = arr.reshape(-1)
        if flat.size == 0:
            return arr.astype(float)

        mu_val = _jp_ensure_scalar(mu, "mu")
        kappa_val = _jp_ensure_scalar(kappa, "kappa")
        psi_val = _jp_ensure_scalar(psi, "psi")

        two_pi = 2.0 * np.pi

        if kappa_val < _JP_KAPPA_TOL:
            result = np.mod(flat, two_pi) / two_pi
            return result.reshape(arr.shape)

        if abs(psi_val) < _JP_PSI_TOL:
            return vonmises.cdf(arr, mu=mu_val, kappa=kappa_val)

        phi_start = (-mu_val) % two_pi
        phi_end = (flat - mu_val) % two_pi

        if _jp_cdf_use_ladder(kappa_val, psi_val):
            # deep ψ < 0: exact ladder cumulative — the series
            # grid cannot resolve the spike there
            H_start = float(
                _jp_cum01(
                    np.array([phi_start]), kappa_val, psi_val, want_skew=False
                )[0][0]
            )
            H_end = _jp_cum01(phi_end, kappa_val, psi_val, want_skew=False)[0]
        else:
            try:
                n_idx, coeffs = self._jp_get_series(kappa_val, psi_val)
            except Exception:  # pragma: no cover - defensive fallback
                cdf_vals = self._cdf_from_pdf(arr, mu_val, kappa_val, psi_val)
                return np.asarray(cdf_vals, dtype=float).reshape(arr.shape)

            H_start = float(self._jp_series_cumulative(np.array([phi_start]), n_idx, coeffs)[0])
            H_end = self._jp_series_cumulative(phi_end, n_idx, coeffs)

        cdf = np.where(
            phi_end >= phi_start,
            np.clip(H_end - H_start, 0.0, 1.0),
            np.clip(1.0 - (H_start - H_end), 0.0, 1.0),
        )

        return cdf.reshape(arr.shape)

    def cdf(self, x, mu, kappa, psi, *args, **kwargs):
        r"""
        Cumulative distribution function of the Jones--Pewsey distribution.

        $$
        F(\theta)=\frac{\theta-\mu}{2\pi}+\frac{1}{\pi}
        \sum_{n\ge 1}\frac{\alpha_n(\kappa,\psi)}{n}
        \sin\bigl(n(\theta-\mu)\bigr),
        $$
        where the cosine moments $\alpha_n$ are evaluated through the
        associated Legendre expression reported by Jones & Pewsey (2005).
        Coefficients are cached per parameter set and the routine falls back to
        numerical quadrature only when the series becomes unstable,
        reproducing the von Mises limit as $\psi \to 0$ and the uniform limit
        as $\kappa \to 0$. For deep $\psi < 0$ spikes beyond the series
        resolution (roughly $\kappa\psi \lesssim -3$) the routine switches to
        an exact composite Gauss--Legendre cumulative on the same
        feature-scale ladder that serves the normalizer and the sampler, so
        CDF values stay correct at any representable spike depth.

        Parameters
        ----------
        x : array_like
            Evaluation points (radians), automatically wrapped onto [0, 2π).
        mu, kappa, psi : float
            Jones--Pewsey location, concentration, and shape parameters.

        Returns
        -------
        ndarray
            CDF values matching the shape of x.
        """
        return super().cdf(x, mu, kappa, psi, *args, **kwargs)

    def _ppf(self, q, mu, kappa, psi):
        mu_val = _jp_ensure_scalar(mu, "mu")
        kappa_val = _jp_ensure_scalar(kappa, "kappa")
        psi_val = _jp_ensure_scalar(psi, "psi")
        two_pi = 2.0 * np.pi

        q_arr = np.asarray(q, dtype=float)
        if q_arr.size == 0:
            return q_arr.astype(float)

        flat = q_arr.reshape(-1)
        result = np.full_like(flat, np.nan, dtype=float)

        valid = np.isfinite(flat) & (flat >= 0.0) & (flat <= 1.0)
        if np.any(valid):
            q_valid = flat[valid]

            boundary_lo = q_valid <= 0.0
            boundary_hi = q_valid >= 1.0
            interior = (~boundary_lo) & (~boundary_hi)
            theta_vals = np.zeros_like(q_valid)

            theta_vals[boundary_lo] = 0.0
            theta_vals[boundary_hi] = two_pi

            if np.any(interior):
                q_int = q_valid[interior]
                eps = 1e-15
                q_clipped = np.clip(q_int, eps, 1.0 - eps)
                if kappa_val < _JP_KAPPA_TOL:
                    theta_vals[interior] = two_pi * q_clipped
                elif abs(psi_val) < _JP_PSI_TOL:
                    vm = vonmises(kappa=kappa_val, mu=mu_val)
                    theta_vals[interior] = vm.ppf(q_clipped)
                elif _jp_cdf_use_ladder(kappa_val, psi_val):
                    # deep ψ < 0: table-initialized exact solve —
                    # the generic Newton scaffold cannot land inside a
                    # sub-resolution spike from a uniform start
                    theta_vals[interior] = _jp_ppf_ladder(
                        q_clipped, mu_val, kappa_val, psi_val
                    )
                else:
                    theta_curr = two_pi * q_clipped
                    L = np.zeros_like(theta_curr)
                    H = np.full_like(theta_curr, two_pi)
                    tol_cdf = 1e-12
                    tol_theta = 1e-10
                    max_iter = 8

                    for _ in range(max_iter):
                        cdf_vals = np.asarray(
                            self.cdf(theta_curr, mu_val, kappa_val, psi_val), dtype=float
                        )
                        pdf_vals = np.asarray(
                            self.pdf(theta_curr, mu_val, kappa_val, psi_val), dtype=float
                        )
                        delta = cdf_vals - q_clipped

                        L = np.where(delta <= 0.0, theta_curr, L)
                        H = np.where(delta > 0.0, theta_curr, H)

                        converged = (np.abs(delta) <= tol_cdf) & ((H - L) <= tol_theta)
                        if np.all(converged):
                            break

                        denom = np.clip(pdf_vals, 1e-15, None)
                        step = np.clip(delta / denom, -np.pi, np.pi)
                        theta_next = theta_curr - step
                        midpoint = 0.5 * (L + H)
                        theta_next = np.where(
                            (theta_next <= L) | (theta_next >= H),
                            midpoint,
                            theta_next,
                        )
                        theta_curr = np.clip(theta_next, 0.0, two_pi)

                    residual = np.asarray(
                        self.cdf(theta_curr, mu_val, kappa_val, psi_val),
                        dtype=float,
                    ) - q_clipped
                    mask = (np.abs(residual) > tol_cdf) | ((H - L) > tol_theta)
                    if np.any(mask):
                        theta_b = theta_curr.copy()
                        L_b = L.copy()
                        H_b = H.copy()
                        for _ in range(30):
                            if not np.any(mask):
                                break
                            mid = 0.5 * (L_b + H_b)
                            cdf_mid = np.asarray(
                                self.cdf(mid, mu_val, kappa_val, psi_val),
                                dtype=float,
                            )
                            delta_mid = cdf_mid - q_clipped
                            take_upper = (delta_mid > 0.0) & mask
                            take_lower = (~take_upper) & mask
                            H_b = np.where(take_upper, mid, H_b)
                            L_b = np.where(take_lower, mid, L_b)
                            theta_b = np.where(mask, mid, theta_b)
                            mask = mask & (np.abs(delta_mid) > tol_cdf)
                        theta_curr = np.where(mask, 0.5 * (L_b + H_b), theta_b)

                    theta_vals[interior] = theta_curr

            result_vals = theta_vals
            result_vals[boundary_lo] = 0.0
            result_vals[boundary_hi] = two_pi
            result[valid] = result_vals

        result = result.reshape(q_arr.shape)
        return result

    def ppf(self, q, mu, kappa, psi, *args, **kwargs):
        r"""
        Quantile function of the Jones--Pewsey law.

        The inverse CDF is obtained by a safeguarded Newton iteration that uses
        the series-based CDF as the residual and the fully normalised PDF as the
        slope.  Bracketing and bisection polishing guarantee convergence on the
        circular interval [0, 2π] while the implementation switches to the
        closed-form von Mises or uniform solutions in their respective limits.
        For deep ψ < 0 spikes the solve runs in the centered angle against the
        exact ladder CDF instead, initialized by the sampler's quantile-table
        inverse, which lands inside the spike at any representable depth.

        Parameters
        ----------
        q : array_like
            Probabilities in [0, 1].
        mu, kappa, psi : float
            Jones--Pewsey parameters.

        Returns
        -------
        ndarray
            Quantiles with the same shape as q.
        """
        return super().ppf(q, mu, kappa, psi, *args, **kwargs)

    def _rvs(self, mu, kappa, psi, size=None, random_state=None):
        rng = self._init_rng(random_state)

        mu_val = _jp_ensure_scalar(mu, "mu")
        mu_val = float(np.mod(mu_val, 2.0 * np.pi))
        kappa_val = _jp_ensure_scalar(kappa, "kappa")
        psi_val = _jp_ensure_scalar(psi, "psi")

        if size is None:
            size_tuple = ()
            total = 1
        elif np.isscalar(size):
            size_tuple = (int(size),)
            total = int(size_tuple[0])
        else:
            size_tuple = tuple(int(s) for s in np.atleast_1d(size))
            total = int(np.prod(size_tuple))

        two_pi = 2.0 * np.pi
        if kappa_val < _JP_KAPPA_TOL:
            samples = rng.uniform(0.0, two_pi, size=total)
            return samples.reshape(size_tuple)

        if abs(psi_val) < _JP_PSI_TOL:
            return vonmises.rvs(mu=mu_val, kappa=kappa_val, size=size_tuple or None, random_state=rng)

        # Inverse transform on the ladder-graded kernel table — exact at
        # any spike depth, one vectorized pass, no rejection loop. (The
        # previous von Mises rejection envelope was calibrated on a
        # 2048-point uniform grid, which cannot see the ψ < 0 spike once
        # it is narrower than ~6e-3: draws were distributionally wrong
        # from κψ ≈ −6 — KS p ~ 1e-113 at (κ=8, ψ=−1) — and the
        # acceptance rate collapsed into an effective hang by κψ ≈ −50.)
        phi_draws = _jp_sample_table(kappa_val, psi_val, total, rng)
        samples = np.mod(mu_val + phi_draws, two_pi)
        return samples.reshape(size_tuple)

    def rvs(self, mu, kappa, psi, size=None, random_state=None):
        r"""
        Draw random variates from the Jones-Pewsey distribution.

        Sampling is by inverse transform through a cached quantile table of
        the centered kernel on a feature-scale-graded grid (the same
        break-point ladder that serves the normalizer), so cost and
        correctness are independent of how narrow the ψ < 0 spike gets.

        Parameters
        ----------
        mu, kappa, psi : float
            Jones-Pewsey parameters.
        size : int or tuple of ints, optional
            Output shape.  When omitted a single draw is returned.
        random_state : numpy.random.Generator or compatible seed, optional
            Source of randomness.

        Returns
        -------
        ndarray
            Sample(s) wrapped to [0, 2π).
        """
        return super().rvs(mu, kappa, psi, size=size, random_state=random_state)

    def _jp_get_series(self, kappa, psi, max_harmonics=256, grid_size=4096):
        key = (float(kappa), float(psi))
        cached = self._series_cache.get(key)
        if cached is not None:
            return cached

        phi = np.linspace(-np.pi, np.pi, int(grid_size), endpoint=False)
        theta = np.mod(phi, 2.0 * np.pi)
        pdf_vals = self.pdf(theta, 0.0, kappa, psi)
        pdf_vals = np.asarray(pdf_vals, dtype=float)

        delta = (2.0 * np.pi) / float(grid_size)
        harmonics = np.arange(0, max_harmonics + 1, dtype=float)
        cos_matrix = np.cos(np.outer(harmonics, phi))
        cos_coeffs = delta * cos_matrix @ pdf_vals
        cos_coeffs[0] = 1.0
        cos_coeffs = np.clip(cos_coeffs, -1.0, 1.0)

        n_idx = harmonics[1:]
        coeffs = cos_coeffs[1:]
        if n_idx.size == 0:
            result = (n_idx, coeffs)
            self._series_cache[key] = result
            return result

        contributions = np.abs(coeffs / n_idx)
        tol = 5e-12
        mask = contributions > tol
        if not np.any(mask):
            n_used = n_idx[:1]
            coeffs_used = coeffs[:1]
        else:
            last = int(np.nonzero(mask)[0][-1]) + 1
            n_used = n_idx[:last]
            coeffs_used = coeffs[:last]
        result = (n_used, coeffs_used)
        self._series_cache[key] = result
        return result

    @staticmethod
    def _jp_series_cumulative(phi_values, n_idx, coeffs):
        phi_values = np.asarray(phi_values, dtype=float)
        phi_flat = phi_values.reshape(-1)
        result = phi_flat / (2.0 * np.pi)
        if n_idx.size:
            sin_terms = np.sin(np.outer(phi_flat, n_idx))
            result += (sin_terms @ (coeffs / n_idx)) / np.pi
        return result.reshape(phi_values.shape)

    @staticmethod
    def _jp_series_skew_integral(phi_values, n_idx, coeffs):
        phi_values = np.asarray(phi_values, dtype=float)
        phi_flat = phi_values.reshape(-1)
        base = (1.0 - np.cos(phi_flat)) / (2.0 * np.pi)
        if n_idx.size:
            n_arr = n_idx
            coeff_arr = coeffs
            contributions = np.zeros_like(phi_flat)

            mask_one = np.isclose(n_arr, 1.0)
            if np.any(mask_one):
                coeff_one = float(np.sum(coeff_arr[mask_one]))
                contributions += coeff_one * ((1.0 - np.cos(2.0 * phi_flat)) / 4.0)

            mask_other = ~mask_one
            if np.any(mask_other):
                n_other = n_arr[mask_other]
                coeff_other = coeff_arr[mask_other]
                phi_matrix = np.outer(phi_flat, n_other)
                term_plus = (1.0 - np.cos(phi_matrix + phi_flat[:, None])) / (n_other + 1.0)
                term_minus = (1.0 - np.cos(phi_matrix - phi_flat[:, None])) / (n_other - 1.0)
                contributions += 0.5 * (term_plus - term_minus) @ coeff_other

            base += contributions / np.pi
        return base.reshape(phi_values.shape)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        return_info=False,
        psi_bounds=(-4.0, 4.0),
        kappa_bounds=(1e-6, 1e3),
        optimizer="L-BFGS-B",
        **kwargs,
    ):
        r"""
        Estimate Jones--Pewsey parameters from data.

        A moment-based start is built from the sample circular mean
        μ̂ and resultant length r₁ with the usual von Mises
        approximation for κ.  The shape parameter ψ is seeded
        by scanning a coarse grid and the three parameters are then refined via
        constrained maximum likelihood:

        ```
        ℓ(μ, κ, ψ) = Σᵢ wᵢ log( c(κ, ψ) K_JP(θᵢ − μ; κ, ψ) ).
        ```

        The normalising constant c is evaluated using the associated
        Legendre function whenever stable, with numerical quadrature as a
        fallback.  Set method="moments" to skip the optimisation and
        return the analytic seed.

        Parameters
        ----------
        data : array_like
            Sample angles (radians), wrapped internally.
        weights : array_like, optional
            Non-negative weights broadcastable to data.
        method : {"moments", "mle"}, optional
            Whether to return the analytic seed or run the numerical MLE.
        return_info : bool, optional
            If True return a diagnostics dictionary alongside the estimates.
        psi_bounds, kappa_bounds : tuple, optional
            Parameter bounds used by the optimiser.
        optimizer : str, optional
            Name of the ``scipy.optimize.minimize`` method.

        Returns
        -------
        tuple or (tuple, dict)
            Estimated parameters (mu, kappa, psi) and, optionally,
            optimisation diagnostics when return_info is True.
        """
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        x = self._wrap_angles(np.asarray(data, dtype=float)).ravel()
        if x.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, x.shape).astype(float, copy=False).ravel()

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = w_sum**2 / np.sum(w**2)

        mu_mom, r1 = circ_mean_and_r(alpha=x, w=w)
        if not np.isfinite(mu_mom):
            mu_mom = 0.0
        mu_mom = float(np.mod(mu_mom, 2.0 * np.pi))
        r1 = float(np.clip(r1, 1e-12, 1.0 - 1e-12))
        n_adjust = int(max(1, round(w_sum)))
        kappa_mom = float(np.clip(circ_kappa(r=r1, n=n_adjust), kappa_bounds[0], kappa_bounds[1]))

        psi_low, psi_high = psi_bounds
        psi_grid = np.linspace(psi_low, psi_high, 9)

        def nll(params):
            mu_param, kappa_param, psi_param = params
            if not (kappa_bounds[0] <= kappa_param <= kappa_bounds[1]):
                return np.inf
            if not (psi_low <= psi_param <= psi_high):
                return np.inf
            mu_wrapped = float(np.mod(mu_param, 2.0 * np.pi))
            pdf_vals = self.pdf(x, mu_wrapped, kappa_param, psi_param)
            if np.any(pdf_vals <= 0.0) or not np.all(np.isfinite(pdf_vals)):
                return np.inf
            return float(-np.sum(w * np.log(pdf_vals)))

        psi_init = 0.0
        best_score = nll((mu_mom, kappa_mom, psi_init))
        for candidate in psi_grid:
            score = nll((mu_mom, kappa_mom, candidate))
            if score < best_score:
                best_score = score
                psi_init = float(candidate)

        method_key = method.lower()
        alias = {"analytical": "moments", "numerical": "mle"}
        method_key = alias.get(method_key, method_key)
        if method_key not in {"moments", "mle"}:
            raise ValueError("`method` must be either 'moments' or 'mle'.")

        if method_key == "moments":
            estimates = (self._wrap_direction(mu_mom), kappa_mom, 0.0)
            info = {
                "method": "moments",
                "loglik": float(-best_score),
                "n_effective": float(n_eff),
                "converged": True,
            }
        else:
            bounds = [(0.0, 2.0 * np.pi), kappa_bounds, psi_bounds]
            init = np.array([mu_mom, kappa_mom, psi_init], dtype=float)
            result = minimize(
                nll,
                init,
                method=optimizer,
                bounds=bounds,
                **kwargs,
            )
            if not result.success:
                raise RuntimeError(f"jonespewsey.fit(method='mle') failed: {result.message}")
            mu_hat = self._wrap_direction(float(result.x[0]))
            kappa_hat = float(np.clip(result.x[1], kappa_bounds[0], kappa_bounds[1]))
            psi_hat = float(np.clip(result.x[2], psi_bounds[0], psi_bounds[1]))
            final_nll = float(result.fun)
            estimates = (mu_hat, kappa_hat, psi_hat)
            info = {
                "method": "mle",
                "loglik": float(-final_nll),
                "n_effective": float(n_eff),
                "converged": bool(result.success),
                "nit": result.nit,
                "optimizer": optimizer,
                "initial": (mu_mom, kappa_mom, psi_init),
            }

        if return_info:
            return estimates, info
        return estimates


jonespewsey = jonespewsey_gen(name="jonespewsey")
jplss = CircularLL(jonespewsey, name="jplss")

####################################
## Helper Functions: Jones-Pewsey ##
####################################


_JP_KAPPA_TOL = 1e-3
_JP_PSI_TOL = 1e-6


def _jp_as_scalar(value):
    """The single value of an effectively-scalar parameter, else ``None``
    (meaning genuinely per-observation)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(()))
    unique = np.unique(arr)
    return float(unique[0]) if unique.size == 1 else None


def _jp_ensure_scalar(value, name):
    scalar = _jp_as_scalar(value)
    if scalar is None:
        raise ValueError(
            f"Jones-Pewsey parameter '{name}' must be scalar for this method; "
            "per-observation parameter arrays are supported only by pdf/logpdf."
        )
    return scalar


def _jp_feature_scales(kappa, psi):
    """The two length scales of the JP kernel in its own angle: the peak
    curvature width 1/√κ_eff at φ = 0 (the ψ < 0 spike) and the antipodal
    near-kink scale 2e^{−|κψ|} at φ = π, where g ≈ e^A cos²(φ/2) + e^{−A}
    crosses over (a √-type feature the kernel and the moment integrands
    inherit). Returns ``(w_peak, w_anti)``, clipped to [1e-320, 1].

    The lower clip is a pure positivity guard at the denormal floor (the
    width formula underflows to 0 beyond |κψ| ≈ 745, which would hang the
    geometric rung loop) — it must NOT be a resolution floor: under the
    fixed-panel GL ladder any floor above the true feature scale silently
    truncates the ladder and loses the ψ < 0 spike that carries all of the
    mass. An earlier 1e-13 floor — harmless for the adaptive quadrature it
    was written for — made the normalizer wrong beyond |κψ| ≈ 33 and
    garbage by |κψ| ≈ 40 (caught against the exact ψ = −1 identity
    Z ≡ 2π); a 1e-300 floor still broke |κψ| ≳ 690. Rungs below ~1e-320
    only resolve because the peak sits at φ = 0, where doubles are
    denormally dense — which is also why the asymmetric family must
    integrate in its *unwarped* angle (see ``_jp_log_c_asym``)."""
    A = kappa * psi
    if abs(A) < 1e-12:
        keff = max(kappa, 1e-12)
        w_peak = float(np.clip(1.0 / np.sqrt(keff), 1e-320, 1.0))
    else:
        with np.errstate(over="ignore"):  # expm1 → inf is fine: w_peak
            # lands on the 1e-320 denormal guard either way
            if A >= 0.0:
                log_keff = np.log(-np.expm1(-2.0 * A)) - np.log(2.0 * abs(psi))
            else:
                log_keff = np.log(np.expm1(-2.0 * A)) - np.log(2.0 * abs(psi))
            w_peak = float(np.clip(np.exp(-0.5 * log_keff), 1e-320, 1.0))
    w_anti = float(np.clip(2.0 * np.exp(-abs(A)), 1e-320, 1.0))
    return w_peak, w_anti


def _gl_panels_from_edges(edges):
    """Composite 24-point Gauss–Legendre nodes/weights over consecutive
    panel ``edges`` (sorted). Returns ``(nodes, weights)`` in panel-major
    order — built by one broadcast (no Python panel loop), bitwise the same
    as the per-panel concatenation it replaces; this runs once per unique
    parameter tuple on every JP-family normalizer sweep."""
    xi, wgl = _JP_GL_XW
    edges = np.asarray(edges, dtype=float)
    mid = 0.5 * (edges[:-1] + edges[1:])
    hw = 0.5 * (edges[1:] - edges[:-1])
    nodes = (mid[:, None] + hw[:, None] * xi[None, :]).ravel()
    wts = (hw[:, None] * wgl[None, :]).ravel()
    return nodes, wts


def _jp_ladder_edges(kappa, psi):
    """Break-point ladder over the half circle [0, π]: rungs grow
    geometrically (decade steps) from *both* ends at the
    ``_jp_feature_scales`` of the peak (φ = 0) and the antipodal near-kink
    (φ = π), so every panel spans at most one decade of its feature scale.
    Returns the sorted edge array."""
    w_peak, w_anti = _jp_feature_scales(kappa, psi)
    edge_set = {0.0, float(np.pi)}
    r = w_peak
    while r < np.pi:
        edge_set.add(r)
        r *= 10.0
    r = w_anti
    while r < np.pi:
        edge_set.add(float(np.pi) - r)
        r *= 10.0
    return np.asarray(sorted(edge_set))


def _jp_gl_panels(kappa, psi):
    """[0, π] composite Gauss–Legendre nodes/weights on the JP break-point
    ladder (the kernel is even in φ, so a half-circle sweep suffices) —
    24-point GL per ``_jp_ladder_edges`` panel is at quadrature precision.
    Returns ``(nodes, weights)``.
    """
    return _gl_panels_from_edges(_jp_ladder_edges(kappa, psi))


@lru_cache(maxsize=8192)
def _jp_grid(kappa: float, psi: float):
    """Cached [0, π] GL nodes/weights for one symmetric Jones–Pewsey tuple
    (deterministic in (κ, ψ)) — the un-warped twin of ``_jp_asym_grid``, shared
    by the value (``_jp_log_c_vec``) and moment (``_jp_logZ_moments_vec``)
    sweeps of a fit iteration. Callers copy via ``np.stack``, so the cached
    arrays are never mutated."""
    return _jp_gl_panels(kappa, psi)


def _jp_size_groups(pairs):
    """Symmetric-JP GL grids grouped by node count for batched evaluation —
    the un-warped twin of ``_jp_asym_size_groups`` (jplss / ssjplss). Each group
    stacks into one rectangular ``(g, L)`` batch with no padding: every tuple
    keeps its own adaptive ``_jp_ladder_edges`` grid, so a batched sweep is
    bit-for-bit the per-tuple scalar, and a lone deep-ψ spike forms its own
    one-row group instead of inflating every datum's node count. Returns
    ``[(idx, nodes, wts), ...]`` with ``idx`` indexing rows of ``pairs``."""
    grids = [_jp_grid(float(k), float(p)) for k, p in pairs]
    sizes = np.array([g[0].size for g in grids])
    groups = []
    for L in np.unique(sizes):
        idx = np.nonzero(sizes == L)[0]
        nodes = np.stack([grids[i][0] for i in idx])
        wts = np.stack([grids[i][1] for i in idx])
        groups.append((idx, nodes, wts))
    return groups


@lru_cache(maxsize=1024)
def _jp_quantile_table(kappa: float, psi: float):
    """Centered sampling table for the JP kernel law: density and cdf knots
    on a ladder-graded grid (32 log-spaced points per feature-scale decade,
    mirrored about 0), so the ψ < 0 spike is resolved at any representable
    depth — the failure mode of the retired von Mises rejection envelope,
    whose 2048-point uniform calibration grid never saw spikes narrower
    than ~6e-3 (wrong draws from κψ ≈ −6, hangs by −50). Within-knot cells
    the density is taken as linear, so inverse-transform draws sample
    *exactly* from the piecewise-linear density through true knot values
    (knot spacing ≤ 7% of the local scale ⇒ cdf bias ≪ KS resolution).
    Returns ``(phi, cdf, dens)`` with cdf[0] = 0, cdf[-1] = 1.
    """
    edges = _jp_ladder_edges(kappa, psi)

    half = [0.0]
    for a, b in zip(edges[:-1], edges[1:]):
        if a == 0.0:  # the flat-topped innermost cell: linear subdivision
            half.extend(np.linspace(a, b, 33)[1:])
        else:
            half.extend(np.geomspace(a, b, 33)[1:])
    half = np.asarray(half, dtype=float)
    phi = np.concatenate([-half[::-1], half[1:]])

    h = _jp_log_kernel(phi, kappa, psi)
    dens = np.exp(h - kappa)  # ∝ density; peak value exactly 1
    seg = 0.5 * (dens[1:] + dens[:-1]) * np.diff(phi)
    cdf = np.concatenate([[0.0], np.cumsum(seg)])
    total = max(float(cdf[-1]), np.finfo(float).tiny)
    return phi, cdf / total, dens / total


def _jp_table_invert(u, kappa, psi):
    """Invert the ``_jp_quantile_table`` cdf at probabilities ``u``: locate
    the cell by searchsorted, then invert the cell's quadratic cdf (linear
    density) in closed form. Returns centered angles in [−π, π] — exact for
    the table's piecewise-linear density at any representable spike depth.
    Serves the sampler (with uniform draws) and the deep-cdf ppf branch
    (with target probabilities, as the Newton initialization)."""
    phi, cdf, dens = _jp_quantile_table(float(kappa), float(psi))
    u = np.asarray(u, dtype=float)
    idx = np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, len(phi) - 2)
    du = u - cdf[idx]
    dx = phi[idx + 1] - phi[idx]
    a = 0.5 * (dens[idx + 1] - dens[idx]) * dx
    b = dens[idx] * dx
    # solve a t² + b t = du for the position fraction t ∈ [0, 1]
    lin = np.abs(a) <= 1e-12 * np.maximum(np.abs(b), np.finfo(float).tiny)
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = np.sqrt(np.maximum(b * b + 4.0 * a * du, 0.0))
        t_quad = (disc - b) / np.where(lin, 1.0, 2.0 * a)
        t_lin = du / np.maximum(b, np.finfo(float).tiny)
    t = np.clip(np.where(lin, t_lin, t_quad), 0.0, 1.0)
    return phi[idx] + t * dx


def _jp_sample_table(kappa, psi, total, rng):
    """Draw ``total`` centered angles from the JP kernel law by inverse
    transform on ``_jp_quantile_table`` (see ``_jp_table_invert``)."""
    return _jp_table_invert(rng.random(total), kappa, psi)


# --- deep-spike cdf/ppf branch ------------------------------------------------
# The series cdf path (4096-point coefficient grid, ≤ 256 harmonics) cannot
# represent ψ < 0 spikes much narrower than the harmonic cap resolves:
# probed 2026-06-11, its cdf error is ≤ 4e-11 at κψ = −3 for ψ ∈
# [−2.5, −0.3] but reaches 1.5e-4 by κψ = −4 and 1.0 (at spike-interior
# points) by −7. Below the gate the cdf is evaluated exactly instead, by
# composite Gauss–Legendre on the same feature-scale ladder that serves the
# normalizer and the sampler: cached cumulatives at the ladder edges plus a
# 24-point partial panel per query.

_JP_CDF_DEEP_A = -3.0  # κψ at/below which the series path is retired …
_JP_CDF_DEEP_W = 0.06  # … or the peak feature scale that forces it


def _jp_cdf_use_ladder(kappa, psi):
    """True where the JP-clan cdf/ppf must leave the series / uniform-grid
    path for the exact ladder branch: ψ < 0 with the spike near or below
    the series resolution (the gate sits where the series is still clean,
    so both branches agree to ~1e-10 at the boundary)."""
    if psi >= 0.0 or kappa < _JP_KAPPA_TOL or abs(psi) < _JP_PSI_TOL:
        return False
    if kappa * psi <= _JP_CDF_DEEP_A:
        return True
    w_peak, _ = _jp_feature_scales(kappa, psi)
    return w_peak < _JP_CDF_DEEP_W


def _jp_log_kernel(phi, kappa, psi):
    """The JP log-kernel ``h(φ; κ, ψ)`` alone — the ladder integrand and
    moment paths need no scores, and the full `_jp_score_terms` bundle
    costs ~15 array ops where one or four suffice. Branch-for-branch
    bitwise-identical to ``_jp_score_terms(...)["h"]``: ``ψ = 0`` is the
    von Mises member ``h = κ cos φ`` (equal to the cumulant series at
    A = 0), ``|κψ| < _JP_A_SMALL`` the cumulant series, else the
    overflow-safe ``logaddexp`` form."""
    phi = np.asarray(phi, dtype=float)
    if psi == 0.0:
        return kappa * np.cos(phi)
    A = kappa * psi
    if abs(A) < _JP_A_SMALL:
        s = np.sin(phi)
        c = np.cos(phi)
        s2 = s * s
        k3 = -2.0 * c * s2
        k4 = s2 * (4.0 - 6.0 * s2)
        return kappa * (c + s2 * A / 2.0 + k3 * A * A / 6.0 + k4 * A**3 / 24.0)
    half = 0.5 * phi
    with np.errstate(divide="ignore"):
        lp = 2.0 * np.log(np.abs(np.cos(half)))
        lq = 2.0 * np.log(np.abs(np.sin(half)))
    return np.logaddexp(A + lp, -A + lq) / psi


@lru_cache(maxsize=1024)
def _jp_cdf_ladder(kappa: float, psi: float):
    """Half-line cumulative tables for the centered JP kernel law: GL-exact
    cumulatives of e^{h−κ} (base) and e^{h−κ} sin t (the sine-skew moment)
    at the ``_jp_ladder_edges`` panel edges on [0, π]. Returns
    ``(edges, cum_base, cum_skew)`` with ``cum_*[0] = 0``."""
    edges = _jp_ladder_edges(kappa, psi)
    nodes, wts = _gl_panels_from_edges(edges)
    h = _jp_log_kernel(nodes, kappa, psi)
    e = wts * np.exp(h - kappa)
    n_gl = _JP_GL_XW[0].size
    base = e.reshape(edges.size - 1, n_gl).sum(axis=1)
    skew = (e * np.sin(nodes)).reshape(edges.size - 1, n_gl).sum(axis=1)
    cum_base = np.concatenate([[0.0], np.cumsum(base)])
    cum_skew = np.concatenate([[0.0], np.cumsum(skew)])
    return edges, cum_base, cum_skew


def _jp_half_cum(x, kappa, psi, want_skew=True):
    """R(x) = ∫₀ˣ e^{h−κ} dt and S(x) = ∫₀ˣ e^{h−κ} sin t dt for x ∈
    [0, π]: cached edge cumulatives plus a 24-point partial panel
    [edge_k, x] per query — quadrature-exact at any spike depth. With
    ``want_skew=False`` skips the sine moment and returns ``(R, None)``."""
    edges, cum_base, cum_skew = _jp_cdf_ladder(kappa, psi)
    x = np.clip(np.asarray(x, dtype=float).reshape(-1), 0.0, np.pi)
    k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, edges.size - 2)
    hw = 0.5 * (x - edges[k])
    xi_gl, w_gl = _JP_GL_XW
    nodes = (edges[k] + hw)[:, None] + hw[:, None] * xi_gl[None, :]
    h = _jp_log_kernel(nodes, kappa, psi)
    e = np.exp(h - kappa)
    base = cum_base[k] + hw * (e @ w_gl)
    if not want_skew:
        return base, None
    skew = cum_skew[k] + hw * ((e * np.sin(nodes)) @ w_gl)
    return base, skew


def _jp_cum01(phi, kappa, psi, want_skew=True):
    """Exact H(φ) = ∫₀^φ f_c dt and J(φ) = ∫₀^φ f_c sin t dt over
    φ ∈ [0, 2π) for the centered normalized JP kernel law f_c (drop-in
    replacements for the series cumulative and skew integral in the deep
    regime). The kernel is even, so both fold onto the half-line tables:
    H = R(φ)/Z for φ ≤ π and 1 − R(2π−φ)/Z above; J = S(min(φ, 2π−φ))/Z.
    Beyond κψ ≈ −740 (spike below the smallest denormal) the tiny-floor
    guard makes values best-effort, never nan — same boundary as the
    normalizer. With ``want_skew=False`` returns ``(H, None)``."""
    phi = np.asarray(phi, dtype=float).reshape(-1)
    upper = phi > np.pi
    x = np.where(upper, 2.0 * np.pi - phi, phi)
    R, S = _jp_half_cum(x, kappa, psi, want_skew=want_skew)
    _, cum_base, _ = _jp_cdf_ladder(kappa, psi)
    ztot = max(2.0 * float(cum_base[-1]), np.finfo(float).tiny)
    H = np.where(upper, 1.0 - R / ztot, R / ztot)
    return H, (None if S is None else S / ztot)


def _jp_solve_quantile(target, x0, lo, hi, cdf_fn, pdf_fn, tol=1e-13,
                       max_iter=100):
    """Vectorized safeguarded Newton on a monotone cdf: solve
    cdf_fn(x) = target from init ``x0`` with brackets [lo, hi] (bisection
    midpoint wherever the Newton step is non-finite or leaves the
    bracket). Convergence is on the cdf residual — in dead-flat tails the
    bracket midpoint is the honest answer and the iteration cap bounds the
    cost."""
    x = np.clip(np.asarray(x0, dtype=float), lo, hi)
    target = np.asarray(target, dtype=float)
    L = np.full_like(x, lo)
    H = np.full_like(x, hi)
    for _ in range(max_iter):
        d = cdf_fn(x) - target
        L = np.where(d <= 0.0, x, L)
        H = np.where(d > 0.0, x, H)
        done = np.abs(d) <= tol
        if np.all(done):
            break
        p = pdf_fn(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = x - d / p
        xn = np.where(np.isfinite(xn) & (xn > L) & (xn < H), xn,
                      0.5 * (L + H))
        x = np.where(done, x, xn)
    return x


def _jp_ppf_ladder(q, mu, kappa, psi):
    """Deep-regime JP quantiles: solve F_c(ζ) = u in the centered angle
    (spike at ζ = 0, where doubles are dense) with the exact ladder cdf,
    initialized by the sampler's quantile-table inverse — which lands
    inside the spike at any representable depth. ``q`` must be interior."""
    two_pi = 2.0 * np.pi
    H_start, _ = _jp_cum01(np.array([(-mu) % two_pi]), kappa, psi, want_skew=False)
    u_t = (np.asarray(q, dtype=float) + float(H_start[0]) + 0.5) % 1.0
    zeta0 = _jp_table_invert(u_t, kappa, psi)
    _, cum_base, _ = _jp_cdf_ladder(kappa, psi)
    ztot = max(2.0 * float(cum_base[-1]), np.finfo(float).tiny)

    def cdf_fn(z):
        R, _ = _jp_half_cum(np.abs(z), kappa, psi, want_skew=False)
        return 0.5 + np.sign(z) * R / ztot

    def pdf_fn(z):
        h = _jp_log_kernel(z, kappa, psi)
        return np.exp(h - kappa) / ztot

    zeta = _jp_solve_quantile(u_t, zeta0, -np.pi, np.pi, cdf_fn, pdf_fn)
    return np.mod(mu + zeta, two_pi)


def _jp_ppf_ladder_sineskewed(q, xi, kappa, psi, lmbd):
    """Deep-regime sine-skewed JP quantiles: same centered-angle solve as
    ``_jp_ppf_ladder`` against F_c(ζ) + λ(S(|ζ|) − S(π))/Z. The init from
    the unskewed kernel table is skew-blind, which the Newton polish
    absorbs (inside a deep spike sin ζ ≈ 0, so the skew factor is ≈ 1
    exactly where the init has to be precise)."""
    two_pi = 2.0 * np.pi
    H_s, J_s = _jp_cum01(np.array([(-xi) % two_pi]), kappa, psi)
    _, cum_base, cum_skew = _jp_cdf_ladder(kappa, psi)
    ztot = max(2.0 * float(cum_base[-1]), np.finfo(float).tiny)
    sn_pi = float(cum_skew[-1])
    g_start = float(H_s[0]) + lmbd * float(J_s[0])
    g_c0 = 0.5 - lmbd * sn_pi / ztot
    u_t = (np.asarray(q, dtype=float) + g_start + g_c0) % 1.0
    zeta0 = _jp_table_invert(u_t, kappa, psi)

    def cdf_fn(z):
        R, S = _jp_half_cum(np.abs(z), kappa, psi)
        return 0.5 + np.sign(z) * R / ztot + lmbd * (S - sn_pi) / ztot

    def pdf_fn(z):
        h = _jp_log_kernel(z, kappa, psi)
        return np.exp(h - kappa) * (1.0 + lmbd * np.sin(z)) / ztot

    zeta = _jp_solve_quantile(u_t, zeta0, -np.pi, np.pi, cdf_fn, pdf_fn)
    return np.mod(xi + zeta, two_pi)


@lru_cache(maxsize=4096)
def _jp_log_c(kappa: float, psi: float) -> float:
    """log of the Jones–Pewsey normalizing constant, log c(κ, ψ) =
    −log ∫ kernel dφ (μ-invariant).

    Same preference order as the historical linear-space normalizer
    (uniform and von Mises reductions first; the Legendre closed form
    ``1/(2π P_{1/ψ}(cosh κψ))`` stays banned — scipy's ``lpmv`` is the
    *Ferrers* function, domain |x| ≤ 1, and off the cut it silently
    returns garbage for non-integer degree: plausible near z = 1, −1e63
    by z ≈ 10, nan beyond. The off-cut route via
    ``hyp2f1(−ν, ν+1; 1; (1−z)/2)`` was probed 2026-06-11: ~1e-13 in the
    moderate band but linear-space — overflows at the e^709 wall for
    ψ > 0 and internally from κψ ≈ −30 for ψ < 0. This GL ladder *is*
    the same function — the kernel integral is its Mehler–Dirichlet-type
    representation — evaluated in log space at any depth), but evaluated
    **entirely in log space** with the kernel's peak value e^κ factored
    out: the raw kernel
    maximum is exp(κ) for every ψ, so any linear-space evaluation turns
    the whole JP clan's pdf into nan for κ ≳ 709.
    The general branch integrates e^{h−κ} ≤ 1 by composite Gauss–Legendre
    on ``_jp_gl_panels`` — the same engine as the regression moment
    machinery (``_jp_logZ_moments``), which retired the per-call adaptive
    quadrature this replaced.
    """
    if kappa < _JP_KAPPA_TOL:
        return float(-np.log(2.0 * np.pi))
    if abs(psi) < _JP_PSI_TOL:
        return float(-(np.log(2.0 * np.pi * i0e(kappa)) + kappa))
    nodes, wts = _jp_gl_panels(kappa, psi)
    h = _jp_log_kernel(nodes, kappa, psi)
    # kernel even in φ with peak h(0) = κ exactly. The tiny-floor guard
    # only engages beyond |κψ| ≈ 740 with ψ < 0, where the spike is below
    # float resolution even at φ = 0 (the distribution is numerically a
    # point mass): values there are best-effort, never nan.
    integral = 2.0 * np.sum(wts * np.exp(h - kappa))
    return float(-(kappa + np.log(max(integral, np.finfo(float).tiny))))


def _jp_log_c_vec(kappa, psi):
    """Per-observation ``_jp_log_c(κ_i, ψ_i)`` — element-wise mirror of the
    scalar preference order, evaluated once per unique ``(κ, ψ)`` pair (the
    lru cache on the scalar makes repeated regression iterations cheap)."""
    kappa, psi = np.broadcast_arrays(
        np.asarray(kappa, dtype=float), np.asarray(psi, dtype=float)
    )
    out = np.full(kappa.shape, -np.log(2.0 * np.pi))
    live = kappa >= _JP_KAPPA_TOL
    vm = live & (np.abs(psi) < _JP_PSI_TOL)
    if np.any(vm):
        out[vm] = -(np.log(2.0 * np.pi * i0e(kappa[vm])) + kappa[vm])
    gen = live & ~vm
    if not np.any(gen):
        return out
    pairs, inverse = np.unique(
        np.stack([kappa[gen], psi[gen]], axis=1), axis=0, return_inverse=True
    )
    vals = np.empty(pairs.shape[0])
    tiny = np.finfo(float).tiny
    for idx, nodes, wts in _jp_size_groups(pairs):
        k, p = pairs[idx, 0:1], pairs[idx, 1:2]
        # h via score_terms (2D-safe, bitwise-identical to _jp_log_kernel);
        # 2· for the even-kernel half-circle, peak e^κ factored out
        h = _jp_score_terms(nodes, k, p, second=False)["h"]
        integral = 2.0 * np.sum(wts * np.exp(h - k), axis=1)
        vals[idx] = -(k[:, 0] + np.log(np.maximum(integral, tiny)))
    out[gen] = vals[inverse]
    return out


@lru_cache(maxsize=8192)
def _jp_cos_moment(kappa: float, psi: float, q: int) -> float:
    """q-th cosine moment ᾱ_q = E[cos(qΦ)] of the *centered* JP law (the
    sine moments vanish — the kernel is even). One Gauss–Legendre sweep on
    the feature-scale ladder as a normalizer-free ratio, so it is exact at
    any concentration; the vM and uniform reductions are closed-form.
    Serves the sine-skewed JP trig moments (book §4.3.11), which are exact
    combinations of the base moments ᾱ_{p−1}, ᾱ_p, ᾱ_{p+1}."""
    if q == 0:
        return 1.0
    if kappa < _JP_KAPPA_TOL:
        return 0.0
    if abs(psi) < _JP_PSI_TOL:
        return float(ive(q, kappa) / ive(0, kappa))
    nodes, wts = _jp_gl_panels(kappa, psi)
    e = wts * np.exp(_jp_log_kernel(nodes, kappa, psi) - kappa)
    denom = max(float(np.sum(e)), np.finfo(float).tiny)
    return float(np.sum(np.cos(q * nodes) * e) / denom)


# --- Jones–Pewsey regression derivatives (the l1/l2 contract) ----------------
# Everything below differentiates the JP log-kernel h(φ; κ, ψ) =
# (1/ψ) log(cosh κψ + sinh κψ cos φ) and the log-normalizer log Z(κ, ψ); the
# distribution methods assemble them into per-observation score/Hessian
# entries. Derivatives are exact down to κ → 0 (no _JP_KAPPA_TOL flat spot:
# the uniform reduction in ``_pdf`` is a value-only shortcut, and keeping the
# scores smooth there preserves the gradient signal the optimizer needs to
# leave the near-uniform corner).

_JP_A_SMALL = 1e-4  # |κψ| below this switches the ψ-direction to series
_JP_LOG_CAP = 250.0  # cap on log Ṽ; see _jp_score_terms


def _jp_score_terms(phi, kappa, psi, second=True):
    r"""Stable per-observation derivatives of the JP log-kernel
    ``h(φ; κ, ψ) = (1/ψ) log(cosh κψ + sinh κψ cos φ)``.

    All quantities derive from overflow-safe primitives of the exact
    decomposition ``g = e^A cos²(φ/2) + e^{−A} sin²(φ/2)`` (A = κψ):

    - ``lg = log g`` via ``logaddexp``;
    - ``arg = A + ½(lp − lq)`` so that ``T := g_A/g = tanh(arg)`` and
      ``∂T/∂A = sech²(arg)`` — and, since ``∂arg/∂φ = −1/sin φ``,
      ``∂T/∂φ = −sech²(arg)/sin φ`` (→ 0 at φ ≡ 0, π);
    - ``Ṽ := sinh(A)/(ψ g) = κ·sinhc(A)/g ≥ 0`` in log form, so that
      ``h_φ = −Ṽ sin φ`` and ``h_φφ = −Ṽ cos φ − ψ (Ṽ sin φ)²``.

    The ψ-direction uses the cumulant view: ``log g = K(A)``, the cgf of a
    ±1 variable with ``P(+1) = cos²(φ/2)`` and cumulants κ₁ = c, κ₂ = s²,
    κ₃ = −2cs², κ₄ = s²(4−6s²), κ₅ = −8cs²(1−3s²) (c = cos φ, s = sin φ):

    $$h_ψ = κ²\frac{AT − K}{A²},\qquad
      h_{ψψ} = κ³\frac{A²\,\mathrm{sech}²(arg) − 2AT + 2K}{A³},\qquad
      h_{φψ} = κ²\frac{A\,T_φ + ψ\,Ṽ\sinφ}{A²}.$$

    These cancel catastrophically as A → 0, so ``|A| < _JP_A_SMALL``
    switches to the cumulant series (agreement ≲1e-9 at the boundary,
    checked by the dev harness). log Ṽ is capped at ``_JP_LOG_CAP`` so the
    0·∞ antipode corner (|κψ| beyond ~350) degrades to large-but-finite
    scores instead of NaN — far outside any optimizer-recoverable region.

    Returns a dict with the log-kernel ``h`` and first derivatives
    ``hphi``, ``hk``, ``hp``; with ``second=True`` adds ``hphiphi``,
    ``hphik``, ``hphip``, ``hkk``, ``hkp``, ``hpp``.
    """
    phi_b, kappa_b, psi_b = np.broadcast_arrays(
        np.asarray(phi, dtype=float),
        np.asarray(kappa, dtype=float),
        np.asarray(psi, dtype=float),
    )
    A = kappa_b * psi_b
    absA = np.abs(A)
    half = 0.5 * phi_b
    s, c = np.sin(phi_b), np.cos(phi_b)
    s2 = s * s
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = 2.0 * np.log(np.abs(np.cos(half)))
        lq = 2.0 * np.log(np.abs(np.sin(half)))
        logk = np.log(kappa_b)  # −inf at κ = 0 (uniform: Ṽ = 0)
        logabss = np.log(np.abs(s))  # −inf at φ ≡ 0, π
    lg = np.logaddexp(A + lp, -A + lq)
    arg = A + 0.5 * (lp - lq)
    T = np.tanh(arg)
    a2 = np.exp(-2.0 * np.abs(arg))
    sech2 = 4.0 * a2 / (1.0 + a2) ** 2
    small = absA < _JP_A_SMALL
    with np.errstate(divide="ignore", invalid="ignore"):
        lsinhc = np.where(
            small,
            A * A / 6.0,
            absA + np.log1p(-np.exp(-2.0 * absA)) - np.log(2.0 * absA),
        )
    lVt = np.minimum(logk + lsinhc - lg, _JP_LOG_CAP)
    Vt = np.exp(lVt)
    sVt = np.sign(s) * np.exp(logabss + lVt)  # Ṽ·sin φ, 0·∞-safe

    # cumulants of the ±1 cgf K(A) at this φ
    k3 = -2.0 * c * s2
    k4 = s2 * (4.0 - 6.0 * s2)
    k5 = -8.0 * c * s2 * (1.0 - 3.0 * s2)

    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(
            small,
            kappa_b
            * (c + s2 * A / 2.0 + k3 * A * A / 6.0 + k4 * A**3 / 24.0),
            lg / psi_b,
        )
        W = np.where(
            small,
            s2 / 2.0 + k3 * A / 3.0 + k4 * A * A / 8.0 + k5 * A**3 / 30.0,
            (A * T - lg) / (A * A),
        )
    out = {
        "h": h,
        "hphi": -sVt,
        "hk": T,
        "hp": kappa_b * kappa_b * W,
    }
    if not second:
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        Tphi = np.where(s == 0.0, 0.0, -sech2 / s)
        Wp = np.where(
            small,
            k3 / 3.0 + k4 * A / 4.0 + k5 * A * A / 10.0,
            (A * A * sech2 - 2.0 * A * T + 2.0 * lg) / A**3,
        )
        Wphi = np.where(
            small,
            s * c
            + 2.0 * s * (1.0 - 3.0 * c * c) * A / 3.0
            + s * c * (1.0 - 3.0 * s2) * A * A,
            (A * Tphi + psi_b * sVt) / (A * A),
        )
    out.update(
        {
            "hphiphi": -c * Vt - psi_b * sVt * sVt,
            "hphik": Tphi,
            "hphip": kappa_b * kappa_b * Wphi,
            "hkk": psi_b * sech2,
            "hkp": kappa_b * sech2,
            "hpp": kappa_b**3 * Wp,
        }
    )
    return out


_JP_GL_XW = roots_legendre(24)


@lru_cache(maxsize=4096)
def _jp_logZ_moments(kappa: float, psi: float):
    """First and second (κ, ψ)-derivatives of ``log Z(κ, ψ)`` with
    ``Z = ∫ kernel dφ``, as kernel-weighted moments:

        ∂log Z/∂θ_a  = E[h_a],
        ∂²log Z/∂θ_a∂θ_b = E[h_ab + h_a h_b] − E[h_a]E[h_b],

    the expectations taken under the JP density itself. Evaluated by
    composite 24-point Gauss–Legendre on the two-ended break-point ladder
    of ``_jp_gl_panels`` (for ψ > 0 the hard feature is the antipodal
    near-kink — g ≈ e^A cos²(φ/2) + e^{−A} crosses over at π − φ ≈
    2e^{−|A|}, which the kernel *and* the moment integrands (T swings
    1 → −1 there) inherit; a coarse panel straddling it is only ~1e-7
    accurate). One node sweep serves all five integrands — the "one
    numeric expectation per unique parameter tuple" cost,
    cached per (κ, ψ) so ``dlogpdf``/``d2logpdf`` within one ``ll()``
    evaluation share the work.

    Returns ``(dk, dp, dkk, dkp, dpp)``.
    """
    nodes, wts = _jp_gl_panels(kappa, psi)
    t = _jp_score_terms(nodes, kappa, psi, second=True)
    e = np.exp(t["h"] - float(np.max(t["h"]))) * wts
    Z = max(float(np.sum(e)), np.finfo(float).tiny)

    def m(v):
        return float(np.sum(e * v)) / Z

    dk, dp = m(t["hk"]), m(t["hp"])
    dkk = m(t["hkk"] + t["hk"] * t["hk"]) - dk * dk
    dkp = m(t["hkp"] + t["hk"] * t["hp"]) - dk * dp
    dpp = m(t["hpp"] + t["hp"] * t["hp"]) - dp * dp
    return dk, dp, dkk, dkp, dpp


def _jp_logZ_moments_vec(kappa, psi):
    """Batched ``_jp_logZ_moments`` over per-observation (κ_i, ψ_i): unique
    pairs grouped by GL node count and evaluated as rectangular ``(group ×
    node)`` batches — one ``_jp_score_terms`` sweep per size group rather than
    per pair, on each tuple's own adaptive grid (bit-for-bit the scalar). This
    is the jplss / ssjplss hot path. Returns five arrays broadcast to the input
    shape: ``(dk, dp, dkk, dkp, dpp)``."""
    kappa, psi = np.broadcast_arrays(
        np.asarray(kappa, dtype=float), np.asarray(psi, dtype=float)
    )
    pairs, inverse = np.unique(
        np.stack([kappa.ravel(), psi.ravel()], axis=1), axis=0,
        return_inverse=True,
    )
    vals = np.empty((pairs.shape[0], 5))
    tiny = np.finfo(float).tiny
    for idx, nodes, wts in _jp_size_groups(pairs):
        k, p = pairs[idx, 0:1], pairs[idx, 1:2]
        t = _jp_score_terms(nodes, k, p, second=True)
        e = np.exp(t["h"] - np.max(t["h"], axis=1, keepdims=True)) * wts
        Z = np.maximum(np.sum(e, axis=1, keepdims=True), tiny)

        def m(v):
            return np.sum(e * v, axis=1, keepdims=True) / Z

        hk, hp = t["hk"], t["hp"]
        dk, dp = m(hk), m(hp)
        dkk = m(t["hkk"] + hk * hk) - dk * dk
        dkp = m(t["hkp"] + hk * hp) - dk * dp
        dpp = m(t["hpp"] + hp * hp) - dp * dp
        vals[idx] = np.concatenate([dk, dp, dkk, dkp, dpp], axis=1)
    out = vals[inverse].reshape(kappa.shape + (5,))
    return tuple(np.moveaxis(out, -1, 0))


###########################
## Sine-Skewed Extention ##
###########################


class jonespewsey_sineskewed_gen(_RegressionReady, CircularContinuous):
    r"""Sine-Skewed Jones-Pewsey Distribution

    The Sine-Skewed Jones-Pewsey distribution is a circular distribution defined on $[0, 2\pi)$
    that extends the Jones-Pewsey family by incorporating a sine-based skewness adjustment.

    ![jonespewsey-sineskewed](../images/circ-mod-jonespewsey-sineskewed.png)

    Methods
    -------
    pdf(x, xi, kappa, psi, lmbd)
        Probability density function.

    logpdf(x, xi, kappa, psi, lmbd)
        Logarithm of the probability density function.

    cdf(x, xi, kappa, psi, lmbd)
        Cumulative distribution function.

    ppf(q, xi, kappa, psi, lmbd)
        Percent-point function (inverse CDF).

    rvs(xi, kappa, psi, lmbd, size=None, random_state=None)
        Random variates (base-JP draws with sine-skew rejection).

    fit(data, *, weights=None, method="mle", ...)
        Estimate ``(xi, kappa, psi, lmbd)`` by moments or maximum likelihood.

    Note
    ----
    Scalar parameters use cached normalisation tables; ``pdf``/``logpdf`` also
    accept per-observation parameter arrays (the regression contract). Other
    methods (cdf, rvs, …) remain scalar-only.
    Implementation based on Section 4.3.11 of Pewsey et al. (2013)
    """

    # --- regression overlay (read by the regression engine
    # only). Book names xi/kappa/psi/lmbd are preserved. The sine-skew factor
    # 1 + λ sin(θ−ξ) leaves the JP normalizer c(κ, ψ) untouched, so a λ
    # linear predictor costs nothing beyond its trivial derivative terms —
    # this is the recommended asymmetric response family. λ ∈ (−1, 1) rides the
    # tanh link. Caveat (book §4.3.11): once
    # λ ≠ 0, ξ is the *mode anchor*, not the mean direction — the engine's
    # fitted-direction report inherits that reading. At |λ| → 1 the density
    # touches 0 where λ sin(θ−ξ) = −1 (logpdf → −∞; the link keeps λ
    # interior). ---
    param_roles = {
        "xi": "location",
        "kappa": "concentration",
        "psi": "shape",
        "lmbd": "skewness",
    }
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): kappa toward 0 (linear), a firmer ridge (scale 30) on psi, and
    # the sine-skew lmbd off its +/-1 walls (scale 10).
    degen_penalty = (
        _degen_linear("kappa"),
        _degen_ridge("psi", 30.0),
        _degen_boundary_sym("lmbd", 1.0, 10.0),
    )
    default_links = {
        "location": "tanhalf",
        "concentration": "log",
        "shape": "identity",
        "skewness": "tanh",
    }

    def _concentration_start(self, Rbar):
        """Closed-form concentration start from the mean resultant ``Rbar``
        for the regression null model: the von Mises A1-inverse ``kappa`` (the
        ``psi -> 0``, ``lambda -> 0`` reduction member), clamped — the circlss
        ``initialize`` convention (see CircularLL._null_params)."""
        return float(np.clip(A1inv(Rbar), 0.01, 500.0))

    def dlogpdf(self, x, xi, kappa, psi, lmbd):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        The log-density is ``log Q + h − log Z`` with ``Q = 1 + λ sin φ``,
        ``φ = θ − ξ`` and ``h``/``Z`` the Jones–Pewsey kernel/normalizer:

        $$\ell_\xi = -\frac{\lambda\cos\phi}{Q} - h_\phi,\quad
          \ell_\kappa = h_\kappa - \mathbb{E}[h_\kappa],\quad
          \ell_\psi = h_\psi - \mathbb{E}[h_\psi],\quad
          \ell_\lambda = \frac{\sin\phi}{Q}.$$

        Vectorizes over per-observation parameter arrays; returns a
        book-named dict.
        """
        x = np.asarray(x, dtype=float)
        xi_b, kappa_b, psi_b, lmbd_b = (
            np.asarray(v, dtype=float) for v in (xi, kappa, psi, lmbd)
        )
        phi = x - xi_b
        s, c = np.sin(phi), np.cos(phi)
        Q = 1.0 + lmbd_b * s
        t = _jp_score_terms(phi, kappa_b, psi_b, second=False)
        dk, dp, *_ = _jp_logZ_moments_vec(kappa_b, psi_b)
        return {
            "xi": -lmbd_b * c / Q - t["hphi"],
            "kappa": t["hk"] - dk,
            "psi": t["hp"] - dp,
            "lmbd": s / Q,
        }

    def d2logpdf(self, x, xi, kappa, psi, lmbd):
        r"""Second derivatives of ``logpdf`` (l2) — unique unordered pairs.
        The skew factor contributes only to the (ξ, λ) block (its normalizer
        is λ- and ξ-free), so the κ/ψ entries are exactly the Jones–Pewsey
        ones and the (κ, λ), (ψ, λ) cross terms vanish identically."""
        x = np.asarray(x, dtype=float)
        xi_b, kappa_b, psi_b, lmbd_b = (
            np.asarray(v, dtype=float) for v in (xi, kappa, psi, lmbd)
        )
        phi = x - xi_b
        s, c = np.sin(phi), np.cos(phi)
        Q = 1.0 + lmbd_b * s
        Q2 = Q * Q
        t = _jp_score_terms(phi, kappa_b, psi_b, second=True)
        _, _, dkk, dkp, dpp = _jp_logZ_moments_vec(kappa_b, psi_b)
        zero = np.zeros(np.broadcast_shapes(phi.shape, Q.shape))
        return {
            ("xi", "xi"): -lmbd_b * (s * Q + lmbd_b * c * c) / Q2
            + t["hphiphi"],
            ("xi", "kappa"): -t["hphik"],
            ("xi", "psi"): -t["hphip"],
            ("xi", "lmbd"): -c / Q2,
            ("kappa", "kappa"): t["hkk"] - dkk,
            ("kappa", "psi"): t["hkp"] - dkp,
            ("kappa", "lmbd"): zero,
            ("psi", "psi"): t["hpp"] - dpp,
            ("psi", "lmbd"): zero,
            ("lmbd", "lmbd"): -s * s / Q2,
        }

    def _argcheck(self, xi, kappa, psi, lmbd):
        try:
            xi_arr, kappa_arr, psi_arr, lmbd_arr = np.broadcast_arrays(xi, kappa, psi, lmbd)
        except ValueError:
            return False
        return (
            (xi_arr >= 0.0)
            & (xi_arr <= 2.0 * np.pi)
            & (kappa_arr >= 0.0)
            & np.isfinite(kappa_arr)
            & np.isfinite(psi_arr)
            & (lmbd_arr >= -1.0)
            & (lmbd_arr <= 1.0)
        )

    def _pdf(self, x, xi, kappa, psi, lmbd):
        x = np.asarray(x, dtype=float)
        xi_scalar = _jp_as_scalar(xi)
        kappa_scalar = _jp_as_scalar(kappa)
        psi_scalar = _jp_as_scalar(psi)
        lmbd_scalar = _jp_as_scalar(lmbd)

        if any(v is None for v in (xi_scalar, kappa_scalar, psi_scalar, lmbd_scalar)):
            # Per-observation parameters — regression contract path,
            # assembled in log space like the base JP.
            xi_b, kappa_b, psi_b, lmbd_b = np.broadcast_arrays(
                *(np.asarray(a, dtype=float) for a in (xi, kappa, psi, lmbd))
            )
            phi = x - xi_b
            skew = 1.0 + lmbd_b * np.sin(phi)
            logc = _jp_log_c_vec(kappa_b, psi_b)
            h = _jp_score_terms(phi, kappa_b, psi_b, second=False)["h"]
            vm = np.abs(psi_b) < _JP_PSI_TOL
            h = np.where(vm, kappa_b * np.cos(phi), h)
            dens = np.exp(h + logc) * skew
            return np.where(
                kappa_b < _JP_KAPPA_TOL, skew / (2.0 * np.pi), dens
            )

        if abs(kappa_scalar) < _JP_KAPPA_TOL:
            return (1.0 / (2.0 * np.pi)) * (1.0 + lmbd_scalar * np.sin(x - xi_scalar))

        log_c = self._get_cached_normalizer(
            lambda: _jp_log_c(kappa_scalar, psi_scalar),
            kappa_scalar,
            psi_scalar,
        )
        self._c = float(np.exp(log_c))  # legacy attribute (write-only)

        phi = x - xi_scalar
        if abs(psi_scalar) < _JP_PSI_TOL:
            h = kappa_scalar * np.cos(phi)
        else:
            h = _jp_score_terms(phi, kappa_scalar, psi_scalar, second=False)["h"]
        return np.exp(h + log_c) * (1.0 + lmbd_scalar * np.sin(phi))

    def pdf(self, x, xi, kappa, psi, lmbd, *args, **kwargs):
        r"""
        Probability density function of the Sine-Skewed Jones-Pewsey distribution.

        $$
        f(\theta) = c(\kappa,\psi)\Bigl(\cosh(\kappa\psi)+
        \sinh(\kappa\psi)\cos(\theta-\xi)\Bigr)^{1/\psi}
        \bigl(1+\lambda \sin(\theta-\xi)\bigr).
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        xi : float
            Direction parameter (generally not the mean), 0 <= ξ <= 2*pi.
        kappa : float
            Concentration parameter, κ >= 0. Higher values indicate a sharper peak.
        psi : float
            Shape parameter, -∞ <= ψ <= ∞. When ψ=-1, the distribution reduces to the wrapped Cauchy,
            when ψ=0, von Mises, and when ψ=1, cardioid.
        lmbd : float
            Skewness parameter, -1 <= λ <= 1. Controls the asymmetry introduced by the sine-skewing.

        Returns
        -------
        pdf_values: float
            Values of the probability density function at the specified points.
        """

        return super().pdf(x, xi, kappa, psi, lmbd, *args, **kwargs)

    def _logpdf(self, x, xi, kappa, psi, lmbd):
        # base-JP log density + log1p(λ sin φ), mirroring ``_pdf``
        # branch-for-branch incl. the per-observation regression path;
        # −inf only at the honest sine-skew zero (|λ| = 1 at sin φ = ∓1)
        x = np.asarray(x, dtype=float)
        xi_scalar = _jp_as_scalar(xi)
        kappa_scalar = _jp_as_scalar(kappa)
        psi_scalar = _jp_as_scalar(psi)
        lmbd_scalar = _jp_as_scalar(lmbd)

        if any(v is None for v in (xi_scalar, kappa_scalar, psi_scalar, lmbd_scalar)):
            xi_b, kappa_b, psi_b, lmbd_b = np.broadcast_arrays(
                *(np.asarray(a, dtype=float) for a in (xi, kappa, psi, lmbd))
            )
            phi = x - xi_b
            with np.errstate(divide="ignore"):
                log_skew = np.log1p(lmbd_b * np.sin(phi))
            logc = _jp_log_c_vec(kappa_b, psi_b)
            h = _jp_score_terms(phi, kappa_b, psi_b, second=False)["h"]
            vm = np.abs(psi_b) < _JP_PSI_TOL
            h = np.where(vm, kappa_b * np.cos(phi), h)
            return np.where(
                kappa_b < _JP_KAPPA_TOL,
                log_skew - np.log(2.0 * np.pi),
                h + logc + log_skew,
            )

        phi = x - xi_scalar
        with np.errstate(divide="ignore"):
            log_skew = np.log1p(lmbd_scalar * np.sin(phi))

        if abs(kappa_scalar) < _JP_KAPPA_TOL:
            return log_skew - np.log(2.0 * np.pi)

        log_c = self._get_cached_normalizer(
            lambda: _jp_log_c(kappa_scalar, psi_scalar),
            kappa_scalar,
            psi_scalar,
        )

        if abs(psi_scalar) < _JP_PSI_TOL:
            h = kappa_scalar * np.cos(phi)
        else:
            h = _jp_score_terms(phi, kappa_scalar, psi_scalar, second=False)["h"]
        return h + log_c + log_skew

    def logpdf(self, x, xi, kappa, psi, lmbd, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the sine-skewed
        Jones-Pewsey distribution: the base-JP log density plus
        $\mathrm{log1p}(\lambda\sin(\theta-\xi))$, assembled in log space —
        finite in underflowed tails, $-\infty$ only at the honest sine-skew
        zero ($|\lambda| = 1$).

        Accepts per-observation parameter arrays like ``pdf`` (the
        regression contract).

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.
        xi : float
            Direction parameter, 0 <= xi <= 2*pi.
        kappa : float
            Concentration parameter, kappa >= 0.
        psi : float
            Shape parameter.
        lmbd : float
            Skewness parameter, -1 <= lmbd <= 1.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, xi, kappa, psi, lmbd, *args, **kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """Trigonometric moment via the exact sine-skewing relation (book
        §4.3.11): with ᾱ_q the centered base-JP cosine moments
        (``_jp_cos_moment``, one GL-ladder sweep each, exact),

            m_p = [ᾱ_p + iλ(ᾱ_{p−1} − ᾱ_{p+1})/2]·e^{ipξ}.
        """
        shape_args, non_shape_kwargs = self._separate_shape_parameters(
            args, kwargs, "trig_moment"
        )
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        xi, kappa, psi, lmbd = (
            float(np.asarray(v, dtype=float))
            for v in self._parse_args(*shape_args, **call_kwargs)[0]
        )

        if not np.isscalar(p):
            raise ValueError("`p` must be an integer scalar.")
        if int(round(p)) != p:
            raise ValueError("`p` must be an integer.")
        k = int(round(p))
        if k == 0:
            return complex(1.0, 0.0)
        ak = abs(k)
        a_p = _jp_cos_moment(kappa, psi, ak)
        b_p = 0.5 * lmbd * (
            _jp_cos_moment(kappa, psi, ak - 1) - _jp_cos_moment(kappa, psi, ak + 1)
        )
        value = (a_p + 1j * b_p) * np.exp(1j * ak * xi)
        return complex(np.conjugate(value)) if k < 0 else complex(value)

    def _cdf(self, x, xi, kappa, psi, lmbd):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        flat = arr.reshape(-1)
        if flat.size == 0:
            return arr.astype(float)

        xi_val = _jp_ensure_scalar(xi, "xi")
        kappa_val = _jp_ensure_scalar(kappa, "kappa")
        psi_val = _jp_ensure_scalar(psi, "psi")
        lmbd_val = _jp_ensure_scalar(lmbd, "lmbd")

        two_pi = 2.0 * np.pi

        if kappa_val < _JP_KAPPA_TOL:
            phi = (flat - xi_val) % two_pi
            base = phi / two_pi
            skew = (1.0 - np.cos(phi)) / (2.0 * np.pi)
            cdf = base + lmbd_val * skew
            return np.clip(cdf, 0.0, 1.0).reshape(arr.shape)

        if abs(psi_val) < _JP_PSI_TOL and abs(lmbd_val) < 1e-12:
            return jonespewsey.cdf(arr, mu=xi_val, kappa=kappa_val, psi=psi_val)

        phi_start = (-xi_val) % two_pi
        phi_end = (flat - xi_val) % two_pi

        if _jp_cdf_use_ladder(kappa_val, psi_val):
            # deep ψ < 0: exact ladder cumulatives for both the base and
            # the sine-skew term
            H_s, J_s = _jp_cum01(np.array([phi_start]), kappa_val, psi_val)
            H_start, J_start = float(H_s[0]), float(J_s[0])
            H_end, J_end = _jp_cum01(phi_end, kappa_val, psi_val)
        else:
            n_idx, coeffs = jonespewsey._jp_get_series(kappa_val, psi_val)

            H_start = float(jonespewsey._jp_series_cumulative(np.array([phi_start]), n_idx, coeffs)[0])
            H_end = jonespewsey._jp_series_cumulative(phi_end, n_idx, coeffs)

            if abs(lmbd_val) > 0:
                J_start = float(jonespewsey._jp_series_skew_integral(np.array([phi_start]), n_idx, coeffs)[0])
                J_end = jonespewsey._jp_series_skew_integral(phi_end, n_idx, coeffs)
            else:
                J_start = 0.0
                J_end = np.zeros_like(H_end)

        base_cdf = np.where(
            phi_end >= phi_start,
            H_end - H_start,
            1.0 - (H_start - H_end),
        )

        skew_cdf = np.where(
            phi_end >= phi_start,
            J_end - J_start,
            -(J_start - J_end),
        )

        cdf = base_cdf + lmbd_val * skew_cdf
        return np.clip(cdf, 0.0, 1.0).reshape(arr.shape)

    def cdf(self, x, xi, kappa, psi, lmbd, *args, **kwargs):
        r"""
        Cumulative distribution function of the sine-skewed Jones--Pewsey law.

        No closed form is available; the base-JP Fourier series supplies both
        the symmetric cumulative and the sine-skew integral, honouring the
        symmetric JP and uniform limits when ``lambda`` or ``kappa`` approach
        zero. For deep ψ < 0 spikes beyond the series resolution both terms
        are evaluated exactly on the kernel's feature-scale ladder instead.
        """
        return super().cdf(x, xi, kappa, psi, lmbd, *args, **kwargs)

    def _ppf(self, q, xi, kappa, psi, lmbd):
        xi_val = _jp_ensure_scalar(xi, "xi")
        xi_val = float(np.mod(xi_val, 2.0 * np.pi))
        kappa_val = _jp_ensure_scalar(kappa, "kappa")
        psi_val = _jp_ensure_scalar(psi, "psi")
        lmbd_val = _jp_ensure_scalar(lmbd, "lmbd")

        two_pi = 2.0 * np.pi
        q_arr = np.asarray(q, dtype=float)
        if q_arr.size == 0:
            return q_arr.astype(float)

        flat = q_arr.reshape(-1)
        result = np.full_like(flat, np.nan, dtype=float)

        valid = np.isfinite(flat) & (flat >= 0.0) & (flat <= 1.0)
        if np.any(valid):
            q_valid = flat[valid]
            boundary_lo = q_valid <= 0.0
            boundary_hi = q_valid >= 1.0
            interior = (~boundary_lo) & (~boundary_hi)
            theta_vals = np.zeros_like(q_valid)
            theta_vals[boundary_lo] = 0.0
            theta_vals[boundary_hi] = two_pi

            if np.any(interior):
                q_int = q_valid[interior]
                eps = 1e-15
                q_clipped = np.clip(q_int, eps, 1.0 - eps)
                if kappa_val < _JP_KAPPA_TOL:
                    theta_vals[interior] = two_pi * q_clipped
                elif abs(lmbd_val) < 1e-12:
                    theta_vals[interior] = jonespewsey.ppf(
                        q_clipped, mu=xi_val, kappa=kappa_val, psi=psi_val
                    )
                elif _jp_cdf_use_ladder(kappa_val, psi_val):
                    # deep ψ < 0: table-initialized exact solve
                    theta_vals[interior] = _jp_ppf_ladder_sineskewed(
                        q_clipped, xi_val, kappa_val, psi_val, lmbd_val
                    )
                else:
                    theta_curr = two_pi * q_clipped
                    L = np.zeros_like(theta_curr)
                    H = np.full_like(theta_curr, two_pi)
                    tol_cdf = 1e-12
                    tol_theta = 1e-10
                    max_iter = 8

                    for _ in range(max_iter):
                        cdf_vals = np.asarray(
                            self.cdf(theta_curr, xi_val, kappa_val, psi_val, lmbd_val),
                            dtype=float,
                        )
                        pdf_vals = np.asarray(
                            self.pdf(theta_curr, xi_val, kappa_val, psi_val, lmbd_val),
                            dtype=float,
                        )
                        delta = cdf_vals - q_clipped
                        L = np.where(delta <= 0.0, theta_curr, L)
                        H = np.where(delta > 0.0, theta_curr, H)

                        converged = (np.abs(delta) <= tol_cdf) & ((H - L) <= tol_theta)
                        if np.all(converged):
                            break

                        denom = np.clip(pdf_vals, 1e-15, None)
                        step = np.clip(delta / denom, -np.pi, np.pi)
                        theta_next = theta_curr - step
                        midpoint = 0.5 * (L + H)
                        theta_next = np.where(
                            (theta_next <= L) | (theta_next >= H),
                            midpoint,
                            theta_next,
                        )
                        theta_curr = np.clip(theta_next, 0.0, two_pi)

                    residual = np.asarray(
                        self.cdf(theta_curr, xi_val, kappa_val, psi_val, lmbd_val),
                        dtype=float,
                    ) - q_clipped
                    mask = (np.abs(residual) > tol_cdf) | ((H - L) > tol_theta)
                    if np.any(mask):
                        theta_b = theta_curr.copy()
                        L_b = L.copy()
                        H_b = H.copy()
                        for _ in range(30):
                            if not np.any(mask):
                                break
                            mid = 0.5 * (L_b + H_b)
                            cdf_mid = np.asarray(
                                self.cdf(mid, xi_val, kappa_val, psi_val, lmbd_val),
                                dtype=float,
                            )
                            delta_mid = cdf_mid - q_clipped
                            take_upper = (delta_mid > 0.0) & mask
                            take_lower = (~take_upper) & mask
                            H_b = np.where(take_upper, mid, H_b)
                            L_b = np.where(take_lower, mid, L_b)
                            theta_b = np.where(mask, mid, theta_b)
                            mask = mask & (np.abs(delta_mid) > tol_cdf)
                        theta_curr = np.where(mask, 0.5 * (L_b + H_b), theta_b)

                    theta_vals[interior] = theta_curr

            result_vals = theta_vals
            result_vals[boundary_lo] = 0.0
            result_vals[boundary_hi] = two_pi
            result[valid] = result_vals

        return result.reshape(q_arr.shape)

    def ppf(self, q, xi, kappa, psi, lmbd, *args, **kwargs):
        r"""
        Quantile function of the sine-skewed Jones--Pewsey distribution.

        The solver mirrors the symmetric JP inverse CDF while reusing the
        skew-aware CDF so that round-trip accuracy is preserved even for large
        skewness.  Uniform and purely symmetric edge cases are delegated to the
        corresponding closed forms.
        """
        return super().ppf(q, xi, kappa, psi, lmbd, *args, **kwargs)

    def _rvs(self, xi, kappa, psi, lmbd, size=None, random_state=None):
        rng = self._init_rng(random_state)

        xi_val = _jp_ensure_scalar(xi, "xi")
        xi_val = float(np.mod(xi_val, 2.0 * np.pi))
        kappa_val = _jp_ensure_scalar(kappa, "kappa")
        psi_val = _jp_ensure_scalar(psi, "psi")
        lmbd_val = _jp_ensure_scalar(lmbd, "lmbd")
        if abs(lmbd_val) >= 1.0:
            raise ValueError("|lmbd| must be < 1 for sine-skewed Jones-Pewsey.")

        if size is None:
            size_tuple = ()
            total = 1
        elif np.isscalar(size):
            size_tuple = (int(size),)
            total = int(size_tuple[0])
        else:
            size_tuple = tuple(int(s) for s in np.atleast_1d(size))
            total = int(np.prod(size_tuple))

        base_dist = jonespewsey(kappa=kappa_val, psi=psi_val, mu=xi_val)
        weights_max = 1.0 + abs(lmbd_val)

        samples = np.empty(total, dtype=float)
        filled = 0
        while filled < total:
            remaining = total - filled
            proposals = base_dist.rvs(size=remaining, random_state=rng)
            accept_prob = (1.0 + lmbd_val * np.sin(proposals - xi_val)) / weights_max
            u = rng.uniform(0.0, 1.0, size=remaining)
            accept = u <= accept_prob
            n_accept = int(np.sum(accept))
            if n_accept > 0:
                samples[filled:filled + n_accept] = proposals[accept][:n_accept]
                filled += n_accept

        return samples.reshape(size_tuple)

    def rvs(self, xi, kappa, psi, lmbd, size=None, random_state=None):
        r"""
        Draw random variates from the sine-skewed Jones--Pewsey distribution.

        Sampling follows the acceptance-rejection construction of Abe & Pewsey
        (2011): draw from the symmetric JP base and accept with probability
        $$\frac{1 + \lambda \sin\phi}{1 + |\lambda|}.$$  This scheme is exact,
        automatically respects the skew symmetry, and retains the base
        sampler's efficiency.
        """
        return super().rvs(xi, kappa, psi, lmbd, size=size, random_state=random_state)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="two-step",
        return_info=False,
        optimizer="L-BFGS-B",
        refine=False,
        psi_bounds=(-4.0, 4.0),
        kappa_bounds=(1e-6, 1e3),
        lmbd_bounds=(-0.99, 0.99),
        base_kwargs=None,
        **kwargs,
    ):
        r"""
        Estimate sine-skewed JP parameters via a two-step maximum likelihood fit.

        1. Fit the symmetric JP base (xi, kappa, psi) using the MLE routine.
        2. Maximise the weighted log term sum log(1 + lambda sin(theta_i - xi)).
        3. Optionally refine all four parameters jointly (set refine=True).

        The acceptance-rejection sampler used for the skewed density makes the
        likelihood well behaved across |lambda| < 1, while moment starts ensure
        stability near the uniform limit.
        """
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        x = self._wrap_angles(np.asarray(data, dtype=float)).ravel()
        if x.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, x.shape).astype(float, copy=False).ravel()

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = w_sum**2 / np.sum(w**2)

        base_kwargs = {} if base_kwargs is None else dict(base_kwargs)
        base_estimates, base_info = jonespewsey.fit(
            x,
            weights=w,
            method="mle",
            psi_bounds=psi_bounds,
            kappa_bounds=kappa_bounds,
            optimizer=optimizer,
            return_info=True,
            **base_kwargs,
        )
        xi_hat, kappa_hat, psi_hat = base_estimates

        lam_low, lam_high = lmbd_bounds

        def lambda_nll(lmbd):
            if not (lam_low < lmbd < lam_high):
                return np.inf
            vals = 1.0 + lmbd * np.sin(x - xi_hat)
            if np.any(vals <= 0.0) or not np.all(np.isfinite(vals)):
                return np.inf
            return float(-np.sum(w * np.log(vals)))

        lambda_result = minimize_scalar(
            lambda_nll,
            bounds=lmbd_bounds,
            method="bounded",
        )
        if not lambda_result.success:
            raise RuntimeError("Failed to estimate skewness parameter `lmbd`.")
        lmbd_hat = float(np.clip(lambda_result.x, lam_low, lam_high))

        method_key = method.lower()
        alias = {"twostep": "two-step", "two_step": "two-step", "mle": "mle"}
        method_key = alias.get(method_key, method_key)
        if method_key not in {"two-step", "mle"}:
            raise ValueError("`method` must be either 'two-step' or 'mle'.")

        if method_key == "mle":
            refine = True

        info = {
            "base": base_info,
            "lambda_opt": {
                "success": bool(lambda_result.success),
                "nit": getattr(lambda_result, "nit", None),
                "nfev": getattr(lambda_result, "nfev", None),
            },
            "n_effective": float(n_eff),
        }

        if refine:
            bounds = [
                (0.0, 2.0 * np.pi),
                kappa_bounds,
                psi_bounds,
                lmbd_bounds,
            ]

            def total_nll(params):
                xi_param, kappa_param, psi_param, lmbd_param = params
                if not (kappa_bounds[0] <= kappa_param <= kappa_bounds[1]):
                    return np.inf
                if not (psi_bounds[0] <= psi_param <= psi_bounds[1]):
                    return np.inf
                if not (lmbd_bounds[0] < lmbd_param < lmbd_bounds[1]):
                    return np.inf
                xi_wrapped = float(np.mod(xi_param, 2.0 * np.pi))
                pdf_vals = self.pdf(x, xi_wrapped, kappa_param, psi_param, lmbd_param)
                if np.any(pdf_vals <= 0.0) or not np.all(np.isfinite(pdf_vals)):
                    return np.inf
                return float(-np.sum(w * np.log(pdf_vals)))

            init = np.array([xi_hat, kappa_hat, psi_hat, lmbd_hat], dtype=float)
            result = minimize(
                total_nll,
                init,
                method=optimizer,
                bounds=bounds,
                **kwargs,
            )
            if not result.success:
                raise RuntimeError("Sine-skewed JP fit refinement failed: " + result.message)
            xi_hat = self._wrap_direction(float(result.x[0]))
            kappa_hat = float(np.clip(result.x[1], kappa_bounds[0], kappa_bounds[1]))
            psi_hat = float(np.clip(result.x[2], psi_bounds[0], psi_bounds[1]))
            lmbd_hat = float(np.clip(result.x[3], lmbd_bounds[0], lmbd_bounds[1]))
            info["refinement"] = {
                "success": bool(result.success),
                "nit": result.nit,
                "optimizer": optimizer,
            }

        final_pdf = self.pdf(x, xi_hat, kappa_hat, psi_hat, lmbd_hat)
        loglik = float(np.sum(w * np.log(final_pdf)))

        estimates = (xi_hat, kappa_hat, psi_hat, lmbd_hat)
        if return_info:
            info.update(
                {
                    "loglik": loglik,
                    "method": method_key,
                    "estimates": estimates,
                }
            )
            return estimates, info
        return estimates


jonespewsey_sineskewed = jonespewsey_sineskewed_gen(name="jonespewsey_sineskewed")
ssjplss = CircularLL(jonespewsey_sineskewed, name="ssjplss")

##########################
## Asymmetric Extention ##
##########################


class jonespewsey_asym_gen(_RegressionReady, CircularContinuous):
    r"""Asymmetric Extended Jones-Pewsey Distribution

    This distribution is an extension of the Jones-Pewsey family, incorporating asymmetry
    through a secondary parameter $\nu$. It is defined on the circular domain $[0, 2\pi)$.

    ![jonespewsey-asymext](../images/circ-mod-jonespewsey-asym.png)

    Methods
    -------
    pdf(x, xi, kappa, psi, nu)
        Probability density function.

    logpdf(x, xi, kappa, psi, nu)
        Logarithm of the probability density function.

    cdf(x, xi, kappa, psi, nu)
        Cumulative distribution function.

    ppf(q, xi, kappa, psi, nu)
        Percent-point function (inverse CDF).

    rvs(xi, kappa, psi, nu, size=None, random_state=None)
        Random variates (kernel-table proposals in u = g(φ) with the
        bounded warp-weight acceptance).

    fit(data, *, weights=None, method="mle" | "moments", ...)
        Estimate ``(xi, kappa, psi, nu)`` by maximum likelihood, or
        return the analytic seed with ``method="moments"``.

    Note
    ----
    ``cdf``/``ppf``/``rvs`` take scalar parameters (cached normalisation tables
    are built per parameter set); ``pdf``/``logpdf`` and the regression
    derivatives ``dlogpdf``/``d2logpdf`` additionally accept per-observation
    parameter arrays (the regression contract — this is the ``ajplss``
    asymmetric location-concentration-shape-skewness family). Implementation
    from 4.3.12 of Pewsey et al. (2013).
    """

    # --- regression overlay (read by the regression engine
    # only). Book names xi/kappa/psi/nu preserved. The asymmetry warps the JP
    # kernel argument *forward*, g = φ + ν cosφ (φ = θ − ξ), so the score
    # reuses jplss's `_jp_score_terms` chained through g (g_φ = 1 − ν sinφ,
    # g_ν = cosφ) — no implicit differentiation. Unlike ssjplss's sine-skew
    # (which leaves the normalizer untouched), the warp *moves* Z, so c depends
    # on (κ, ψ, ν) and ℓ_ν carries a grid-expectation term `−E[h_φ cosφ]`
    # (`_jp_logZ_moments_asym_vec`). ψ is unbounded (identity link, as jplss);
    # ν ∈ (−1,1) rides the tanh link. Reduction member ν=0 is plain `jplss`. ---
    param_roles = {
        "xi": "location",
        "kappa": "concentration",
        "psi": "shape",
        "nu": "skewness",
    }
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): kappa toward 0 (linear), a firmer ridge (scale 30) on psi, and
    # the asymmetry nu off its +/-1 walls (scale 10).
    degen_penalty = (
        _degen_linear("kappa"),
        _degen_ridge("psi", 30.0),
        _degen_boundary_sym("nu", 1.0, 10.0),
    )
    default_links = {
        "location": "tanhalf",
        "concentration": "log",
        "shape": "identity",
        "skewness": "tanh",
    }

    def _concentration_start(self, Rbar):
        """Closed-form concentration start from the mean resultant ``Rbar`` for
        the regression null model: the von Mises A1-inverse ``kappa`` (the
        ψ→0, ν→0 reduction member), clamped — the circlss ``initialize``
        convention (see CircularLL._null_params)."""
        return float(np.clip(A1inv(Rbar), 0.01, 500.0))

    def dlogpdf(self, x, xi, kappa, psi, nu):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        With ``φ = θ − ξ``, the forward warp ``g = φ + ν cosφ`` (``g_φ = 1 −
        ν sinφ``) and the JP log-kernel/normalizer ``h``/``Z`` evaluated at
        ``g``:

        $$\ell_\xi = -h_\phi(g)\,g_\phi,\quad
          \ell_\kappa = h_\kappa(g) - \mathbb{E}[h_\kappa],\quad
          \ell_\psi = h_\psi(g) - \mathbb{E}[h_\psi],\quad
          \ell_\nu = h_\phi(g)\cos\phi - \mathbb{E}[h_\phi(g)\cos\phi].$$

        ``Z`` is ξ-invariant (translation), so ``ℓ_ξ`` has no normalizer term;
        the κ/ψ/ν expectations come from :func:`_jp_logZ_moments_asym_vec`.
        Vectorizes over per-observation arrays; returns a book-named dict.
        """
        x, xi_b, kappa_b, psi_b, nu_b = (
            np.asarray(v, dtype=float) for v in (x, xi, kappa, psi, nu)
        )
        phi = x - xi_b
        cphi, sphi = np.cos(phi), np.sin(phi)
        g = phi + nu_b * cphi
        gphi = 1.0 - nu_b * sphi
        t = _jp_score_terms(g, kappa_b, psi_b, second=False)
        dk, dp, dnu, *_ = _jp_logZ_moments_asym_vec(kappa_b, psi_b, nu_b)
        return {
            "xi": -t["hphi"] * gphi,
            "kappa": t["hk"] - dk,
            "psi": t["hp"] - dp,
            "nu": t["hphi"] * cphi - dnu,
        }

    def d2logpdf(self, x, xi, kappa, psi, nu):
        r"""Second derivatives of ``logpdf`` (l2) — unique unordered pairs.
        The location (ξ) blocks are pure kernel chained through the warp (``Z``
        is ξ-free, ``g_ξ = −g_φ``, ``g_{ξξ} = −ν cosφ``); the κ/ψ/ν blocks
        subtract the normalizer second derivatives ``∂²log Z = E[h_{ab}] +
        Cov(h_a, h_b)`` from :func:`_jp_logZ_moments_asym_vec`."""
        x, xi_b, kappa_b, psi_b, nu_b = (
            np.asarray(v, dtype=float) for v in (x, xi, kappa, psi, nu)
        )
        phi = x - xi_b
        cphi, sphi = np.cos(phi), np.sin(phi)
        g = phi + nu_b * cphi
        gphi = 1.0 - nu_b * sphi          # ∂g/∂φ ; ∂g/∂ξ = −gphi
        t = _jp_score_terms(g, kappa_b, psi_b, second=True)
        hphi, hphiphi = t["hphi"], t["hphiphi"]
        hphik, hphip = t["hphik"], t["hphip"]
        (_, _, _, dkk, dkp, dpp, dknu, dpnu,
         dnunu) = _jp_logZ_moments_asym_vec(kappa_b, psi_b, nu_b)
        return {
            ("xi", "xi"): hphiphi * gphi * gphi - nu_b * cphi * hphi,
            ("xi", "kappa"): -hphik * gphi,
            ("xi", "psi"): -hphip * gphi,
            ("xi", "nu"): -hphiphi * cphi * gphi + hphi * sphi,
            ("kappa", "kappa"): t["hkk"] - dkk,
            ("kappa", "psi"): t["hkp"] - dkp,
            ("kappa", "nu"): hphik * cphi - dknu,
            ("psi", "psi"): t["hpp"] - dpp,
            ("psi", "nu"): hphip * cphi - dpnu,
            ("nu", "nu"): hphiphi * cphi * cphi - dnunu,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cdf_table_cache = {}

    def _argcheck(self, xi, kappa, psi, nu):
        try:
            xi_arr, kappa_arr, psi_arr, nu_arr = np.broadcast_arrays(xi, kappa, psi, nu)
        except ValueError:
            return False
        return (
            (xi_arr >= 0.0)
            & (xi_arr <= 2.0 * np.pi)
            & (kappa_arr >= 0.0)
            & np.isfinite(kappa_arr)
            & np.isfinite(psi_arr)
            & (nu_arr > -1.0)
            & (nu_arr < 1.0)
        )

    def _pdf(self, x, xi, kappa, psi, nu):
        x = np.asarray(x, dtype=float)
        xi_scalar = _jp_ensure_scalar(xi, "xi")
        kappa_scalar = _jp_ensure_scalar(kappa, "kappa")
        psi_scalar = _jp_ensure_scalar(psi, "psi")
        nu_scalar = _jp_ensure_scalar(nu, "nu")

        if abs(kappa_scalar) < _JP_KAPPA_TOL:
            return np.full_like(x, 1.0 / (2.0 * np.pi), dtype=float)

        log_c = self._get_cached_normalizer(
            lambda: _jp_log_c_asym(kappa_scalar, psi_scalar, nu_scalar),
            kappa_scalar,
            psi_scalar,
            nu_scalar,
        )
        self._c = float(np.exp(log_c))  # legacy attribute (write-only)
        phi = x - xi_scalar
        g = phi + nu_scalar * np.cos(phi)
        h = _jp_score_terms(g, kappa_scalar, psi_scalar, second=False)["h"]
        return np.exp(h + log_c)

    def pdf(self, x, xi, kappa, psi, nu, *args, **kwargs):
        r"""
        Probability density function (PDF) of the Asymmetric Extended Jones-Pewsey distribution.

        The PDF is given by:

        $$
        f(\theta) = \frac{k(\theta; \xi, \kappa, \psi, \nu)}{c}
        $$

        where $k(\theta; \xi, \kappa, \psi, \nu)$ is the kernel function defined as:

        $$
        k(\theta; \xi, \kappa, \psi, \nu) =
        \begin{cases}
        \exp\left(\kappa \cos(\theta - \xi + \nu \cos(\theta - \xi))\right) & \text{if } \psi = 0 \\
        \left[\cosh(\kappa \psi) + \sinh(\kappa \psi) \cos(\theta - \xi + \nu \cos(\theta - \xi))\right]^{1/\psi} & \text{if } \psi \neq 0
        \end{cases}
        $$

        and $c$ is the normalization constant:

        $$
        c = \int_{-\pi}^{\pi} k(\theta; \xi, \kappa, \psi, \nu) \, d\theta
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF, defined on the interval $[0, 2\pi)$.
        xi : float
            Direction parameter, $0 \leq \xi \leq 2\pi$. This typically represents the mode of the distribution.
        kappa : float
            Concentration parameter, $\kappa \geq 0$. Higher values result in a sharper peak around $\xi$.
        psi : float
            Shape parameter, $-\infty \leq \psi \leq \infty$. When $\psi = 0$, the distribution reduces to a simpler von Mises-like form.
        nu : float
            Asymmetry parameter, $-1 < \nu < 1$. Introduces skewness in the circular distribution.

        Returns
        -------
        pdf_values : array_like
            Values of the probability density function at the specified points.

        Notes
        -----
        - The normalization constant $c$ is computed numerically using integration.
        - Special cases:
            - When $\psi = 0$, the kernel simplifies to the von Mises-like asymmetric form.
            - When $\kappa = 0$, the distribution becomes uniform on $[0, 2\pi)$.
        """
        return super().pdf(x, xi, kappa, psi, nu, *args, **kwargs)

    def _logpdf(self, x, xi, kappa, psi, nu):
        # the log-space assembly ``_pdf`` exponentiates (stable kernel h at
        # the warped angle g(φ) plus the u-substituted log-normalizer),
        # returned before the exp so deep-spike tails stay finite
        if any(_jp_as_scalar(v) is None for v in (xi, kappa, psi, nu)):
            # per-observation parameters — regression contract path
            return _ajp_logpdf_vec(x, xi, kappa, psi, nu)
        x = np.asarray(x, dtype=float)
        xi_scalar = _jp_ensure_scalar(xi, "xi")
        kappa_scalar = _jp_ensure_scalar(kappa, "kappa")
        psi_scalar = _jp_ensure_scalar(psi, "psi")
        nu_scalar = _jp_ensure_scalar(nu, "nu")

        if abs(kappa_scalar) < _JP_KAPPA_TOL:
            return np.full_like(x, -np.log(2.0 * np.pi), dtype=float)

        log_c = self._get_cached_normalizer(
            lambda: _jp_log_c_asym(kappa_scalar, psi_scalar, nu_scalar),
            kappa_scalar,
            psi_scalar,
            nu_scalar,
        )
        phi = x - xi_scalar
        g = phi + nu_scalar * np.cos(phi)
        h = _jp_score_terms(g, kappa_scalar, psi_scalar, second=False)["h"]
        return h + log_c

    def logpdf(self, x, xi, kappa, psi, nu, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the asymmetric
        extended Jones-Pewsey distribution: the stable log-kernel evaluated
        at the warped angle $g(\phi) = \phi + \nu\cos\phi$ plus the
        u-substituted log normalizing constant — finite in underflowed
        tails and at deep $\psi < 0$ spikes.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.
        xi : float
            Direction parameter, 0 <= xi <= 2*pi.
        kappa : float
            Concentration parameter, kappa >= 0.
        psi : float
            Shape parameter.
        nu : float
            Asymmetry parameter, -1 < nu < 1.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, xi, kappa, psi, nu, *args, **kwargs)

    def _cdf(self, x, xi, kappa, psi, nu):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        flat = arr.reshape(-1)
        if flat.size == 0:
            return arr.astype(float)

        xi_val = _jp_ensure_scalar(xi, "xi")
        kappa_val = _jp_ensure_scalar(kappa, "kappa")
        psi_val = _jp_ensure_scalar(psi, "psi")
        nu_val = _jp_ensure_scalar(nu, "nu")

        two_pi = 2.0 * np.pi

        if kappa_val < _JP_KAPPA_TOL and abs(nu_val) < 1e-12:
            return jonespewsey.cdf(arr, mu=xi_val, kappa=kappa_val, psi=psi_val)

        phi_start = (-xi_val) % two_pi
        phi_end = (flat - xi_val) % two_pi

        if _jp_cdf_use_ladder(kappa_val, psi_val):
            # deep ψ < 0: exact u-space ladder cumulative — the
            # uniform 4096-point table cannot resolve the spike there
            H_start = float(
                _jp_cum01_asym(np.array([phi_start]), kappa_val, psi_val, nu_val)[0]
            )
            H_end = _jp_cum01_asym(phi_end, kappa_val, psi_val, nu_val)
        else:
            phi_grid, cdf_grid = self._asym_cdf_table(xi_val, kappa_val, psi_val, nu_val)

            H_start = float(np.interp(phi_start, phi_grid, cdf_grid, left=0.0, right=1.0))
            H_end = np.interp(phi_end, phi_grid, cdf_grid, left=0.0, right=1.0)

        cdf = np.where(
            phi_end >= phi_start,
            np.clip(H_end - H_start, 0.0, 1.0),
            np.clip(1.0 - (H_start - H_end), 0.0, 1.0),
        )

        return cdf.reshape(arr.shape)

    def cdf(self, x, xi, kappa, psi, nu, *args, **kwargs):
        r"""
        Cumulative distribution function of the argument-warped JP family.

        The asymmetric transformation phi -> phi + nu cos(phi) is handled by
        precomputing a high-resolution trapezoidal cumulative table for each
        parameter set.  Interpolation of this table gives fast evaluations while
        preserving the limiting cases (nu -> 0 reduces to the symmetric JP CDF).
        For deep psi < 0 spikes beyond the table resolution the cumulative is
        instead evaluated exactly in the kernel's own angle u = g(phi) on the
        same feature-scale ladder that serves the normalizer and the sampler.
        """
        return super().cdf(x, xi, kappa, psi, nu, *args, **kwargs)

    def _ppf(self, q, xi, kappa, psi, nu):
        xi_val = _jp_ensure_scalar(xi, "xi")
        xi_val = float(np.mod(xi_val, 2.0 * np.pi))
        kappa_val = _jp_ensure_scalar(kappa, "kappa")
        psi_val = _jp_ensure_scalar(psi, "psi")
        nu_val = _jp_ensure_scalar(nu, "nu")

        two_pi = 2.0 * np.pi
        q_arr = np.asarray(q, dtype=float)
        if q_arr.size == 0:
            return q_arr.astype(float)

        flat = q_arr.reshape(-1)
        result = np.full_like(flat, np.nan, dtype=float)

        valid = np.isfinite(flat) & (flat >= 0.0) & (flat <= 1.0)
        if np.any(valid):
            q_valid = flat[valid]
            boundary_lo = q_valid <= 0.0
            boundary_hi = q_valid >= 1.0
            interior = (~boundary_lo) & (~boundary_hi)
            theta_vals = np.zeros_like(q_valid)
            theta_vals[boundary_lo] = 0.0
            theta_vals[boundary_hi] = two_pi

            if np.any(interior):
                q_int = q_valid[interior]
                eps = 1e-15
                q_clipped = np.clip(q_int, eps, 1.0 - eps)
                if kappa_val < _JP_KAPPA_TOL and abs(nu_val) < 1e-12:
                    theta_vals[interior] = two_pi * q_clipped
                elif _jp_cdf_use_ladder(kappa_val, psi_val):
                    # deep ψ < 0: table-initialized exact u-space solve
                    theta_vals[interior] = _jp_ppf_ladder_asym(
                        q_clipped, xi_val, kappa_val, psi_val, nu_val
                    )
                else:
                    theta_curr = two_pi * q_clipped
                    L = np.zeros_like(theta_curr)
                    H = np.full_like(theta_curr, two_pi)
                    tol_cdf = 1e-12
                    tol_theta = 1e-10
                    max_iter = 8

                    for _ in range(max_iter):
                        cdf_vals = np.asarray(
                            self.cdf(theta_curr, xi_val, kappa_val, psi_val, nu_val),
                            dtype=float,
                        )
                        pdf_vals = np.asarray(
                            self.pdf(theta_curr, xi_val, kappa_val, psi_val, nu_val),
                            dtype=float,
                        )
                        delta = cdf_vals - q_clipped
                        L = np.where(delta <= 0.0, theta_curr, L)
                        H = np.where(delta > 0.0, theta_curr, H)

                        converged = (np.abs(delta) <= tol_cdf) & ((H - L) <= tol_theta)
                        if np.all(converged):
                            break

                        denom = np.clip(pdf_vals, 1e-15, None)
                        step = np.clip(delta / denom, -np.pi, np.pi)
                        theta_next = theta_curr - step
                        midpoint = 0.5 * (L + H)
                        theta_next = np.where(
                            (theta_next <= L) | (theta_next >= H),
                            midpoint,
                            theta_next,
                        )
                        theta_curr = np.clip(theta_next, 0.0, two_pi)

                    residual = np.asarray(
                        self.cdf(theta_curr, xi_val, kappa_val, psi_val, nu_val),
                        dtype=float,
                    ) - q_clipped
                    mask = (np.abs(residual) > tol_cdf) | ((H - L) > tol_theta)
                    if np.any(mask):
                        theta_b = theta_curr.copy()
                        L_b = L.copy()
                        H_b = H.copy()
                        for _ in range(30):
                            if not np.any(mask):
                                break
                            mid = 0.5 * (L_b + H_b)
                            cdf_mid = np.asarray(
                                self.cdf(mid, xi_val, kappa_val, psi_val, nu_val),
                                dtype=float,
                            )
                            delta_mid = cdf_mid - q_clipped
                            take_upper = (delta_mid > 0.0) & mask
                            take_lower = (~take_upper) & mask
                            H_b = np.where(take_upper, mid, H_b)
                            L_b = np.where(take_lower, mid, L_b)
                            theta_b = np.where(mask, mid, theta_b)
                            mask = mask & (np.abs(delta_mid) > tol_cdf)
                        theta_curr = np.where(mask, 0.5 * (L_b + H_b), theta_b)

                    theta_vals[interior] = theta_curr

            result_vals = theta_vals
            result_vals[boundary_lo] = 0.0
            result_vals[boundary_hi] = two_pi
            result[valid] = result_vals

        return result.reshape(q_arr.shape)

    def ppf(self, q, xi, kappa, psi, nu, *args, **kwargs):
        r"""
        Quantile function of the asymmetric Jones--Pewsey distribution.

        Quantiles are obtained by the same safeguarded Newton iteration as in
        the symmetric case, with the warp-aware CDF supplying residuals.  For
        deep psi < 0 spikes the solve runs in the kernel's own angle against
        the exact u-space cumulative, initialized from the kernel's quantile
        table, and the warp is inverted by bisection at the end.
        """
        return super().ppf(q, xi, kappa, psi, nu, *args, **kwargs)

    def _rvs(self, xi, kappa, psi, nu, size=None, random_state=None):
        rng = self._init_rng(random_state)

        xi_val = _jp_ensure_scalar(xi, "xi")
        xi_val = float(np.mod(xi_val, 2.0 * np.pi))
        kappa_val = _jp_ensure_scalar(kappa, "kappa")
        psi_val = _jp_ensure_scalar(psi, "psi")
        nu_val = _jp_ensure_scalar(nu, "nu")
        if not (-1.0 < nu_val < 1.0):
            raise ValueError("`nu` must lie in (-1, 1).")

        if size is None:
            size_tuple = ()
            total = 1
        elif np.isscalar(size):
            size_tuple = (int(size),)
            total = int(size_tuple[0])
        else:
            size_tuple = tuple(int(s) for s in np.atleast_1d(size))
            total = int(np.prod(size_tuple))

        two_pi = 2.0 * np.pi
        if kappa_val < _JP_KAPPA_TOL:
            samples = rng.uniform(0.0, two_pi, size=total)
            return samples.reshape(size_tuple)

        if abs(psi_val) < _JP_PSI_TOL and abs(nu_val) < 1e-12:
            return vonmises.rvs(mu=xi_val, kappa=kappa_val, size=size_tuple or None, random_state=rng)

        # In the kernel's own angle U = g(Φ) the target law has density
        # ∝ kernel(u)/g′(g⁻¹(u)): propose u from the pure-kernel table
        # (exact at any spike depth — see ``_jp_quantile_table``), accept
        # with the bounded weight ratio (1−|ν|)/g′ ≤ 1 (min g′ = 1 − |ν|
        # for either sign of ν), then invert the monotone warp by
        # bisection. (The previous von Mises rejection envelope shared the
        # symmetric sampler's blindness to sub-grid ψ < 0 spikes.)
        two_pi_f = 2.0 * np.pi
        samples = np.empty(total, dtype=float)
        filled = 0
        while filled < total:
            remaining = total - filled
            u_prop = _jp_sample_table(kappa_val, psi_val, remaining, rng)
            # map into g's principal range [−π−ν, π−ν] (kernel is
            # periodic; for ν > 0 the table's upper sliver folds down, for
            # ν < 0 the lower sliver folds up)
            u_prop = np.where(u_prop > np.pi - nu_val, u_prop - two_pi_f, u_prop)
            u_prop = np.where(u_prop < -np.pi - nu_val, u_prop + two_pi_f, u_prop)
            phi = _jp_warp_inv(u_prop, nu_val)
            accept = rng.uniform(0.0, 1.0, size=remaining) <= (
                (1.0 - abs(nu_val)) / (1.0 - nu_val * np.sin(phi))
            )
            n_accept = int(np.sum(accept))
            if n_accept > 0:
                samples[filled:filled + n_accept] = np.mod(
                    xi_val + phi[accept][:n_accept], two_pi_f
                )
                filled += n_accept

        return samples.reshape(size_tuple)

    def rvs(self, xi, kappa, psi, nu, size=None, random_state=None):
        r"""
        Draw random variates from the asymmetric Jones--Pewsey distribution.

        Sampling works in the kernel's own angle: proposals come from the
        symmetric kernel's quantile table (exact at any concentration), the
        warp's Jacobian enters as a bounded acceptance weight
        ``(1−|ν|)/g′ ≤ 1`` (min g′ = 1 − |ν| for either sign), and the
        monotone warp is inverted by
        bisection. Uniform and von Mises limits are handled explicitly.
        """
        return super().rvs(xi, kappa, psi, nu, size=size, random_state=random_state)

    def _asym_cdf_table(self, xi, kappa, psi, nu, grid_size=4096):
        key = (float(np.mod(xi, 2.0 * np.pi)), float(kappa), float(psi), float(nu), int(grid_size))
        cached = self._cdf_table_cache.get(key)
        if cached is not None:
            return cached

        phi_grid = np.linspace(0.0, 2.0 * np.pi, int(grid_size) + 1)
        theta = np.mod(xi + phi_grid, 2.0 * np.pi)
        pdf_vals = self.pdf(theta, xi, kappa, psi, nu)
        pdf_vals = np.asarray(pdf_vals, dtype=float)

        delta = (2.0 * np.pi) / float(grid_size)
        trap = 0.5 * (pdf_vals[:-1] + pdf_vals[1:]) * delta
        cdf_vals = np.empty_like(phi_grid)
        cdf_vals[0] = 0.0
        cdf_vals[1:] = np.cumsum(trap)
        total = cdf_vals[-1]
        if not np.isfinite(total) or total <= 0.0:
            total = 1.0
        cdf_vals /= total

        result = (phi_grid, cdf_vals)
        self._cdf_table_cache[key] = result
        return result

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        return_info=False,
        optimizer="L-BFGS-B",
        psi_bounds=(-4.0, 4.0),
        kappa_bounds=(1e-6, 1e3),
        nu_bounds=(-0.99, 0.99),
        base_kwargs=None,
        **kwargs,
    ):
        r"""
        Estimate asymmetric JP parameters ``(xi, kappa, psi, nu)``.

        The symmetric JP fit supplies starting values for (xi, kappa, psi)
        with nu initialised at zero.  With ``method="mle"`` the full
        four-parameter log-likelihood is then optimised under simple
        bounds, re-using the cached normalising constant machinery of the
        JP core; ``method="moments"`` skips all optimisation and returns
        the analytic seed (the symmetric base's moment estimates with
        ``nu = 0``).

        Parameters
        ----------
        data : array_like
            Sample angles (radians), wrapped internally.
        weights : array_like, optional
            Non-negative weights broadcastable to ``data``.
        method : {"mle", "moments"}, optional
            Full four-parameter MLE (default; aliases: "numerical") or
            the analytic seed (alias: "analytical").
        return_info : bool, optional
            If True, also return a diagnostics dictionary.
        optimizer : str, optional
            Name of the ``scipy.optimize.minimize`` method.
        psi_bounds, kappa_bounds, nu_bounds : tuple, optional
            Parameter bounds used by the optimiser.
        base_kwargs : dict, optional
            Extra keyword arguments forwarded to the symmetric
            ``jonespewsey.fit`` seeding call.
        **kwargs :
            Additional keyword arguments forwarded to the optimiser
            (ignored under ``method="moments"``).

        Returns
        -------
        tuple or (tuple, dict)
            Estimated parameters ``(xi, kappa, psi, nu)`` and, optionally,
            fit diagnostics when ``return_info`` is True.
        """
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        x = self._wrap_angles(np.asarray(data, dtype=float)).ravel()
        if x.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, x.shape).astype(float, copy=False).ravel()

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = w_sum**2 / np.sum(w**2)

        method_key = method.lower()
        alias = {"analytical": "moments", "numerical": "mle"}
        method_key = alias.get(method_key, method_key)
        if method_key not in {"moments", "mle"}:
            raise ValueError("`method` must be either 'moments' or 'mle'.")

        base_kwargs = {} if base_kwargs is None else dict(base_kwargs)

        if method_key == "moments":
            seed_estimates, base_info = jonespewsey.fit(
                x,
                weights=w,
                method="moments",
                psi_bounds=psi_bounds,
                kappa_bounds=kappa_bounds,
                optimizer=optimizer,
                return_info=True,
                **base_kwargs,
            )
            xi_hat, kappa_hat, psi_hat = seed_estimates
            nu_hat = 0.0
            seed_pdf = self.pdf(x, xi_hat, kappa_hat, psi_hat, nu_hat)
            estimates = (xi_hat, kappa_hat, psi_hat, nu_hat)
            if return_info:
                info = {
                    "method": "moments",
                    "base": base_info,
                    "loglik": float(np.sum(w * np.log(seed_pdf))),
                    "n_effective": float(n_eff),
                    "converged": True,
                }
                return estimates, info
            return estimates

        init_estimates, base_info = jonespewsey.fit(
            x,
            weights=w,
            method="mle",
            psi_bounds=psi_bounds,
            kappa_bounds=kappa_bounds,
            optimizer=optimizer,
            return_info=True,
            **base_kwargs,
        )
        xi_init, kappa_init, psi_init = init_estimates
        nu_init = 0.0

        kappa_low, kappa_high = kappa_bounds
        psi_low, psi_high = psi_bounds
        nu_low, nu_high = nu_bounds

        def nll(params):
            xi_param, kappa_param, psi_param, nu_param = params
            if not (kappa_low <= kappa_param <= kappa_high):
                return np.inf
            if not (psi_low <= psi_param <= psi_high):
                return np.inf
            if not (nu_low <= nu_param < nu_high):
                return np.inf
            xi_wrapped = float(np.mod(xi_param, 2.0 * np.pi))
            pdf_vals = self.pdf(x, xi_wrapped, kappa_param, psi_param, nu_param)
            if np.any(pdf_vals <= 0.0) or not np.all(np.isfinite(pdf_vals)):
                return np.inf
            return float(-np.sum(w * np.log(pdf_vals)))

        init = np.array([xi_init, kappa_init, psi_init, nu_init], dtype=float)
        bounds = [
            (0.0, 2.0 * np.pi),
            kappa_bounds,
            psi_bounds,
            nu_bounds,
        ]
        result = minimize(
            nll,
            init,
            method=optimizer,
            bounds=bounds,
            **kwargs,
        )
        if not result.success:
            raise RuntimeError("jonespewsey_asym.fit failed: " + result.message)

        xi_hat = self._wrap_direction(float(result.x[0]))
        kappa_hat = float(np.clip(result.x[1], kappa_low, kappa_high))
        psi_hat = float(np.clip(result.x[2], psi_low, psi_high))
        nu_hat = float(np.clip(result.x[3], nu_low, nu_high - 1e-9))

        final_pdf = self.pdf(x, xi_hat, kappa_hat, psi_hat, nu_hat)
        loglik = float(np.sum(w * np.log(final_pdf)))

        estimates = (xi_hat, kappa_hat, psi_hat, nu_hat)
        if return_info:
            info = {
                "method": "mle",
                "base": base_info,
                "loglik": loglik,
                "converged": bool(result.success),
                "nit": result.nit,
                "optimizer": optimizer,
                "n_effective": float(n_eff),
            }
            return estimates, info
        return estimates


jonespewsey_asym = jonespewsey_asym_gen(name="jonespewsey_asym")
ajplss = CircularLL(jonespewsey_asym, name="ajplss")


def _jp_warp_inv(u, nu):
    """φ = g⁻¹(u) on the principal branch of the asymmetry warp
    g(φ) = φ + ν cos φ. Monotone (g′ = 1 − ν sinφ ≥ 1 − |ν| > 0), so a short
    bisection warms a tight bracket and clamped Newton — using the closed-form
    g′ — polishes it: ≤3e-14 across |ν| ≤ 0.999, indistinguishable to the
    smooth weight 1/g′ that consumes φ(u) from the ~5e-18 of 60 pure halvings
    this replaces, at ~4× the speed. The Newton step is clamped to the live
    bracket so it cannot run away where g′ → 0 (ν → 1 near φ = π/2) — the
    failure mode of unguarded Newton from a far start. Feeds every
    asymmetric-JP value/derivative/cdf/rvs path."""
    u = np.asarray(u, dtype=float)
    a = np.full_like(u, -np.pi)
    b = np.full_like(u, np.pi)
    for _ in range(5):                       # warm a tight bracket (width ~2π/32)
        m = 0.5 * (a + b)
        too_high = m + nu * np.cos(m) > u
        b = np.where(too_high, m, b)
        a = np.where(too_high, a, m)
    phi = 0.5 * (a + b)
    for _ in range(6):                       # Newton polish, clamped to [a, b]
        phi = np.clip(
            phi - (phi + nu * np.cos(phi) - u) / (1.0 - nu * np.sin(phi)), a, b
        )
    return phi


def _jp_ladder_edges_asym(kappa, psi, nu):
    """Break-point ladder in the kernel's own angle u = g(φ) over one
    period [−π − ν, π − ν]: decade rungs at the ``_jp_feature_scales`` of
    the peak (u = 0, interior) and the antipodal near-kink (u ≡ ±π; for
    ν > 0 only −π is interior, for ν < 0 only +π — candidates outside the
    window are filtered), plus rungs for the weight 1/g′'s own bump at
    u = g(±π/2) = ±π/2 (the +π/2 bump for ν > 0, −π/2 for ν < 0) of
    u-width ~(1−|ν|)^{3/2} — unresolved it costs ~1e-5 relative at
    ν = 0.9 (the kernel ladders have no rungs mid-window)."""
    two_pi = 2.0 * np.pi
    lo, hi = -np.pi - nu, np.pi - nu
    w_peak, w_anti = _jp_feature_scales(kappa, psi)
    edge_set = {lo, hi, 0.0}
    r = w_peak
    while r < two_pi:
        for cand in (-r, r):
            if lo < cand < hi:
                edge_set.add(cand)
        r *= 10.0
    r = w_anti
    while r < two_pi:
        for cand in (-np.pi - r, -np.pi + r, np.pi - r, np.pi + r):
            if lo < cand < hi:
                edge_set.add(cand)
        r *= 10.0
    # the 1/g′ weight bump sits at u = g(±π/2) = ±π/2 (at +π/2 for ν > 0,
    # at −π/2 for ν < 0); rung both, the window filter drops the inert one
    r = max((1.0 - abs(nu)) ** 1.5, 1e-3)
    while r < two_pi:
        for cand in (0.5 * np.pi - r, 0.5 * np.pi + r,
                     -0.5 * np.pi - r, -0.5 * np.pi + r):
            if lo < cand < hi:
                edge_set.add(cand)
        r *= 10.0
    return np.asarray(sorted(edge_set))


@lru_cache(maxsize=4096)
def _jp_log_c_asym(kappa: float, psi: float, nu: float) -> float:
    """log normalizing constant of the asymmetric-extended JP kernel
    (ξ-invariant), via the substitution u = g(φ) = φ + ν cos φ:

        ∫ kernel(g(φ)) dφ  =  ∫_{−π−ν}^{π−ν} kernel(u) / g′(g⁻¹(u)) du.

    Working in the kernel's own angle u is what makes the spike resolvable:
    the warp puts the peak at a generic φ* where adjacent doubles are
    ~|φ*|·1e-16 apart, so for deep ψ < 0 the spike (width ~2e^{−|κψ|})
    falls *between representable numbers* and no φ-space ladder can see
    it — in u the peak sits at exactly 0, where doubles are denormally
    dense, and the kernel features are at the known points 0/±π (no
    root-solving). The warp survives only as the smooth bounded weight
    1/g′ ∈ [1/(1+ν), 1/(1−ν)], whose argument g⁻¹(u) a float-limited
    bisection serves perfectly well. One vectorized ``_jp_score_terms``
    call over the GL ladder (the scalar-node adaptive quadrature this
    replaces paid ~100 µs of tiny-array numpy per node).
    """
    if kappa < _JP_KAPPA_TOL:
        return float(-np.log(2.0 * np.pi))

    nodes, wts = _gl_panels_from_edges(_jp_ladder_edges_asym(kappa, psi, nu))
    weight = 1.0 / (1.0 - nu * np.sin(_jp_warp_inv(nodes, nu)))
    h = _jp_log_kernel(nodes, kappa, psi)
    integral = float(np.sum(wts * np.exp(h - kappa) * weight))
    return float(-(kappa + np.log(max(integral, np.finfo(float).tiny))))


@lru_cache(maxsize=8192)
def _jp_asym_grid(kappa: float, psi: float, nu: float):
    """Cached GL nodes/weights for one asymmetric-JP tuple — deterministic in
    (κ, ψ, ν), so the value sweep and the moment sweep of a fit iteration build
    each tuple's adaptive ladder once and share it (and a re-visited coefficient
    in a line search is free). Callers copy via ``np.stack`` before use, so the
    cached arrays are never mutated."""
    return _gl_panels_from_edges(_jp_ladder_edges_asym(kappa, psi, nu))


def _jp_asym_size_groups(pairs):
    """Per-tuple GL grids for the asymmetric-JP normalizer, grouped by node
    count so each group stacks into one rectangular ``(g, L)`` batch with **no
    padding** — every tuple keeps its own adaptive ``_jp_ladder_edges_asym``
    grid, so a batched sweep is bit-for-bit the per-tuple scalar. Grouping by
    size (rather than a shared/padded grid) is what makes this exact *and*
    blow-up-free: a lone deep-ψ tuple with a 100-panel ladder forms its own
    one-row group instead of forcing its node count onto every other datum.
    Returns ``[(idx, nodes, wts), ...]`` with ``idx`` indexing rows of
    ``pairs`` and ``nodes``/``wts`` shaped ``(g, L)``."""
    grids = [_jp_asym_grid(float(k), float(p), float(n)) for k, p, n in pairs]
    sizes = np.array([g[0].size for g in grids])
    groups = []
    for L in np.unique(sizes):
        idx = np.nonzero(sizes == L)[0]
        nodes = np.stack([grids[i][0] for i in idx])
        wts = np.stack([grids[i][1] for i in idx])
        groups.append((idx, nodes, wts))
    return groups


def _jp_log_c_asym_vec(kappa, psi, nu):
    """Per-observation ``_jp_log_c_asym(κ_i, ψ_i, ν_i)``, batched: unique
    triples are grouped by GL node count and evaluated as rectangular
    ``(group × node)`` array ops — one ``_jp_warp_inv``/``_jp_log_kernel`` call
    per size group instead of one per tuple — so a distributional κ-smooth pays
    a handful of vectorized sweeps, not a Python loop over every datum. Each
    tuple keeps its own adaptive grid, so the result matches the lru-cached
    scalar to the last bit. κ ≈ 0 → uniform."""
    kappa, psi, nu = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (kappa, psi, nu))
    )
    out = np.full(kappa.shape, -np.log(2.0 * np.pi))
    live = kappa >= _JP_KAPPA_TOL
    if not np.any(live):
        return out
    pairs, inverse = np.unique(
        np.stack([kappa[live], psi[live], nu[live]], axis=1), axis=0,
        return_inverse=True,
    )
    vals = np.empty(pairs.shape[0])
    tiny = np.finfo(float).tiny
    for idx, nodes, wts in _jp_asym_size_groups(pairs):
        k, p, n = pairs[idx, 0:1], pairs[idx, 1:2], pairs[idx, 2:3]
        weight = 1.0 / (1.0 - n * np.sin(_jp_warp_inv(nodes, n)))
        # h via score_terms (2D-safe, bitwise-identical to _jp_log_kernel,
        # which special-cases scalar ψ == 0 and so cannot take array params)
        h = _jp_score_terms(nodes, k, p, second=False)["h"]
        integral = np.sum(wts * np.exp(h - k) * weight, axis=1)
        vals[idx] = -(k[:, 0] + np.log(np.maximum(integral, tiny)))
    out[live] = vals[inverse]
    return out


def _ajp_logpdf_vec(x, xi, kappa, psi, nu):
    """Per-observation asymmetric-extended JP log-density (regression
    contract): the stable log-kernel at the warped angle ``g = φ + ν cosφ``
    plus the vectorized u-substituted log-normalizer, each datum its own
    (ξ, κ, ψ, ν). κ ≈ 0 → uniform."""
    x, xi, kappa, psi, nu = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, xi, kappa, psi, nu))
    )
    phi = x - xi
    g = phi + nu * np.cos(phi)
    h = _jp_score_terms(g, kappa, psi, second=False)["h"]
    h = np.where(np.abs(kappa) < _JP_KAPPA_TOL, 0.0, h)
    return h + _jp_log_c_asym_vec(kappa, psi, nu)


@lru_cache(maxsize=4096)
def _jp_logZ_moments_asym(kappa: float, psi: float, nu: float):
    """First and second (κ, ψ, ν)-derivatives of ``log Z(κ, ψ, ν)`` for the
    asymmetric-extended JP law, as kernel-weighted moments under the density:

        ∂log Z/∂a = E[h_a],
        ∂²log Z/∂a∂b = E[h_{ab} + h_a h_b] − E[h_a]E[h_b].

    The forward warp ``g = φ + ν cosφ`` (``g_ν = cosφ``) enters by the chain
    rule, so the parameter derivatives of the log-kernel are ``h_κ = h_k``,
    ``h_ψ = h_p``, ``h_ν = h_φ cosφ`` and the cross seconds ``h_κν = h_{φκ}
    cosφ``, ``h_ψν = h_{φψ} cosφ``, ``h_νν = h_{φφ} cos²φ`` — with ``h_·`` the
    ``_jp_score_terms`` derivatives at the warped argument ``g``. Same
    ``u = g(φ)`` substitution and adaptive GL ladder as :func:`_jp_log_c_asym`
    (so the antipodal near-kink is resolved); one node sweep serves all nine
    integrands, cached per (κ, ψ, ν). Returns
    ``(dk, dp, dnu, dkk, dkp, dpp, dknu, dpnu, dnunu)``."""
    if kappa < _JP_KAPPA_TOL:
        return (0.0,) * 9
    nodes, wts = _gl_panels_from_edges(_jp_ladder_edges_asym(kappa, psi, nu))
    phi = _jp_warp_inv(nodes, nu)
    cphi = np.cos(phi)
    weight = 1.0 / (1.0 - nu * np.sin(phi))
    t = _jp_score_terms(nodes, kappa, psi, second=True)
    e = wts * np.exp(t["h"] - kappa) * weight
    Z = max(float(np.sum(e)), np.finfo(float).tiny)

    def m(v):
        return float(np.sum(e * v)) / Z

    hk, hp, hphi = t["hk"], t["hp"], t["hphi"]
    hnu = hphi * cphi                              # h_ν
    dk, dp, dnu = m(hk), m(hp), m(hnu)
    h_knu = t["hphik"] * cphi                      # h_κν
    h_pnu = t["hphip"] * cphi                      # h_ψν
    h_nunu = t["hphiphi"] * cphi * cphi            # h_νν
    dkk = m(t["hkk"] + hk * hk) - dk * dk
    dkp = m(t["hkp"] + hk * hp) - dk * dp
    dpp = m(t["hpp"] + hp * hp) - dp * dp
    dknu = m(h_knu + hk * hnu) - dk * dnu
    dpnu = m(h_pnu + hp * hnu) - dp * dnu
    dnunu = m(h_nunu + hnu * hnu) - dnu * dnu
    return dk, dp, dnu, dkk, dkp, dpp, dknu, dpnu, dnunu


def _jp_logZ_moments_asym_vec(kappa, psi, nu):
    """Batched :func:`_jp_logZ_moments_asym` over per-observation
    (κ_i, ψ_i, ν_i): unique triples grouped by GL node count and evaluated as
    rectangular ``(group × node)`` batches — one ``_jp_warp_inv``/
    ``_jp_score_terms`` sweep per size group rather than per tuple, on each
    tuple's own adaptive grid (bit-for-bit the scalar). κ < tol rows stay 0
    (the scalar contract). Returns nine arrays broadcast to the input shape:
    ``(dk, dp, dnu, dkk, dkp, dpp, dknu, dpnu, dnunu)``."""
    kappa, psi, nu = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (kappa, psi, nu))
    )
    pairs, inverse = np.unique(
        np.stack([kappa.ravel(), psi.ravel(), nu.ravel()], axis=1), axis=0,
        return_inverse=True,
    )
    vals = np.zeros((pairs.shape[0], 9))      # κ < tol → 0 (scalar contract)
    live = pairs[:, 0] >= _JP_KAPPA_TOL
    tiny = np.finfo(float).tiny
    if np.any(live):
        live_pairs = pairs[live]
        live_vals = np.empty((live_pairs.shape[0], 9))
        for idx, nodes, wts in _jp_asym_size_groups(live_pairs):
            k, p, n = live_pairs[idx, 0:1], live_pairs[idx, 1:2], live_pairs[idx, 2:3]
            phi = _jp_warp_inv(nodes, n)
            cphi = np.cos(phi)
            weight = 1.0 / (1.0 - n * np.sin(phi))
            t = _jp_score_terms(nodes, k, p, second=True)
            e = wts * np.exp(t["h"] - k) * weight
            Z = np.maximum(np.sum(e, axis=1, keepdims=True), tiny)

            def m(v):
                return np.sum(e * v, axis=1, keepdims=True) / Z

            hk, hp, hphi = t["hk"], t["hp"], t["hphi"]
            hnu = hphi * cphi                          # h_ν
            dk, dp, dnu = m(hk), m(hp), m(hnu)
            h_knu = t["hphik"] * cphi                  # h_κν
            h_pnu = t["hphip"] * cphi                  # h_ψν
            h_nunu = t["hphiphi"] * cphi * cphi        # h_νν
            dkk = m(t["hkk"] + hk * hk) - dk * dk
            dkp = m(t["hkp"] + hk * hp) - dk * dp
            dpp = m(t["hpp"] + hp * hp) - dp * dp
            dknu = m(h_knu + hk * hnu) - dk * dnu
            dpnu = m(h_pnu + hp * hnu) - dp * dnu
            dnunu = m(h_nunu + hnu * hnu) - dnu * dnu
            live_vals[idx] = np.concatenate(
                [dk, dp, dnu, dkk, dkp, dpp, dknu, dpnu, dnunu], axis=1)
        vals[live] = live_vals
    out = vals[inverse].reshape(kappa.shape + (9,))
    return tuple(np.moveaxis(out, -1, 0))


@lru_cache(maxsize=1024)
def _jp_cdf_ladder_asym(kappa: float, psi: float, nu: float):
    """Cumulative table for the asymmetric-extended JP law in the kernel's
    own angle: GL-exact cumulatives of kernel(u)/g′(g⁻¹(u)) (peak value
    suppressed to e^{h−κ}) at the ``_jp_ladder_edges_asym`` panel edges
    over [−π − ν, π − ν]. Returns ``(edges, cum)`` with ``cum[0] = 0``."""
    edges = _jp_ladder_edges_asym(kappa, psi, nu)
    nodes, wts = _gl_panels_from_edges(edges)
    weight = 1.0 / (1.0 - nu * np.sin(_jp_warp_inv(nodes, nu)))
    h = _jp_log_kernel(nodes, kappa, psi)
    n_gl = _JP_GL_XW[0].size
    panels = (wts * np.exp(h - kappa) * weight).reshape(
        edges.size - 1, n_gl
    ).sum(axis=1)
    return edges, np.concatenate([[0.0], np.cumsum(panels)])


def _jp_weighted_cum_asym(u, kappa, psi, nu):
    """Cw(u) = ∫_{−π−ν}^{u} e^{h(t)−κ}/g′(g⁻¹(t)) dt: cached edge
    cumulatives plus a 24-point partial panel per query (the aeJP analog
    of ``_jp_half_cum``; no half-line fold — the weight is not even)."""
    edges, cum = _jp_cdf_ladder_asym(kappa, psi, nu)
    u = np.clip(np.asarray(u, dtype=float).reshape(-1), edges[0], edges[-1])
    k = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, edges.size - 2)
    hw = 0.5 * (u - edges[k])
    xi_gl, w_gl = _JP_GL_XW
    nodes = (edges[k] + hw)[:, None] + hw[:, None] * xi_gl[None, :]
    weight = 1.0 / (1.0 - nu * np.sin(_jp_warp_inv(nodes, nu)))
    h = _jp_log_kernel(nodes, kappa, psi)
    return cum[k] + hw * ((np.exp(h - kappa) * weight) @ w_gl)


def _jp_cum01_asym(s, kappa, psi, nu):
    """Exact H(s) = ∫₀ˢ f_ae(ξ + t) dt over s ∈ [0, 2π) for the
    asymmetric-extended JP law (ξ-invariant): fold s into the centered
    offset ζ ∈ (−π, π], map through the warp u = g(ζ), and difference the
    u-space cumulative against the mass below the start point g(0) = ν.
    Same denormal best-effort boundary as ``_jp_cum01``."""
    s = np.asarray(s, dtype=float).reshape(-1)
    upper = s > np.pi
    zeta = np.where(upper, s - 2.0 * np.pi, s)
    u = zeta + nu * np.cos(zeta)
    edges, cum = _jp_cdf_ladder_asym(kappa, psi, nu)
    z = max(float(cum[-1]), np.finfo(float).tiny)
    C = _jp_weighted_cum_asym(u, kappa, psi, nu) / z
    C_nu = float(_jp_weighted_cum_asym(np.array([nu]), kappa, psi, nu)[0]) / z
    return np.where(upper, 1.0 - C_nu + C, C - C_nu)


def _jp_ppf_ladder_asym(q, xi, kappa, psi, nu):
    """Deep-regime aeJP quantiles: solve C(u) = target in the kernel's own
    angle (spike at u = 0), initialized weight-blind from the symmetric
    kernel's quantile table (the weight is a bounded smooth factor the
    Newton polish absorbs), then invert the warp. ``q`` must be
    interior."""
    two_pi = 2.0 * np.pi
    H_start = float(
        _jp_cum01_asym(np.array([(-xi) % two_pi]), kappa, psi, nu)[0]
    )
    edges, cum = _jp_cdf_ladder_asym(kappa, psi, nu)
    z = max(float(cum[-1]), np.finfo(float).tiny)
    C_nu = float(_jp_weighted_cum_asym(np.array([nu]), kappa, psi, nu)[0]) / z
    u_t = (np.asarray(q, dtype=float) + H_start + C_nu) % 1.0
    u0 = _jp_table_invert(u_t, kappa, psi)
    # the kernel table spans [−π, π]; fold the out-of-window sliver onto
    # the equivalent stretch of the aeJP period [−π − ν, π − ν] (upper
    # sliver for ν > 0, lower for ν < 0 — cf. _rvs)
    u0 = np.where(u0 > np.pi - nu, u0 - two_pi, u0)
    u0 = np.where(u0 < -np.pi - nu, u0 + two_pi, u0)

    def cdf_fn(u):
        return _jp_weighted_cum_asym(u, kappa, psi, nu) / z

    def pdf_fn(u):
        weight = 1.0 / (1.0 - nu * np.sin(_jp_warp_inv(u, nu)))
        h = _jp_log_kernel(u, kappa, psi)
        return np.exp(h - kappa) * weight / z

    u_root = _jp_solve_quantile(
        u_t, u0, -np.pi - nu, np.pi - nu, cdf_fn, pdf_fn
    )
    return np.mod(xi + _jp_warp_inv(u_root, nu), two_pi)


class inverse_batschelet_gen(_RegressionReady, CircularContinuous):
    r"""Inverse Batschelet Distribution

    ![inverse-batschelet](../images/circ-mod-inverse-batschelet.png)

    The inverse Batschelet family (Pewsey, Neuhäuser & Ruxton, 2013, §4.3.13)
    extends the von Mises distribution by applying two inverse angular warps:
    a skewness transform controlled by $\nu$, and an inverse
    Batschelet peakedness transform governed by $\lambda$. The resulting density on
    $[0, 2\pi)$ takes the form

    $$
    f(\theta) = c(\kappa, \lambda)
    \exp\left[\kappa \cos\left(a\,t_\nu^{-1}(\varphi) + b\,s_\lambda^{-1}\bigl(t_\nu^{-1}(\varphi)\bigr)\right)\right],
    $$

    where $\varphi = (\theta - \xi) \bmod 2\pi - \pi$,
    $a = \tfrac{1 - \lambda}{1 + \lambda}$,
    $b = \tfrac{2\lambda}{1 + \lambda}$, and the normalising constant
    $c(\kappa, \lambda)$ depends only on $\kappa$ and $\lambda$.
    Setting $\nu = \lambda = 0$ recovers the von Mises distribution, while
    $\kappa \to 0$ yields the circular uniform law.

    This law is regression-ready: the module-level ``ibslss`` is its
    location-scale-shape family (``CircularLL(inverse_batschelet)``), with
    ξ via tanhalf, log κ, and ν (skewness) / λ (peakedness) via tanh — e.g.
    ``circ_gam(["theta ~ s(x)", "~1", "~1", "~1"], data, family=ibslss)``.
    Accordingly ``pdf``/``logpdf``/``dlogpdf``/``d2logpdf`` accept
    per-observation parameter arrays (the regression contract); ``cdf``,
    ``ppf``, ``rvs`` and the cached normalisation tables remain scalar-only.

    Methods
    -------
    pdf(x, xi, kappa, nu, lmbd)
        Probability density function.

    logpdf(x, xi, kappa, nu, lmbd)
        Logarithm of the probability density function.

    cdf(x, xi, kappa, nu, lmbd)
        Cumulative distribution function.

    ppf(q, xi, kappa, nu, lmbd)
        Percent-point function (inverse CDF).

    rvs(xi, kappa, nu, lmbd, size=None, random_state=None)
        Random variates via von Mises acceptance–rejection.

    fit(data, *, method='mle', ...)
        Moments or maximum-likelihood parameter estimation.
    """

    # --- regression overlay (read by the regression engine
    # only). Book names ξ/κ/ν/λ preserved. The density warps θ *forward* into
    # the von-Mises kernel via the two inverse maps t_ν⁻¹, s_λ⁻¹ (the vectorized
    # `_solve_monotone_increasing` solver), so the per-observation score is
    # closed-form by implicit differentiation, reusing the solved roots and
    # their slopes. ν (skewness) and λ (peakedness) both ride the tanh link;
    # ξ = tanhalf, κ = log. Reduction member ν=λ=0 is the von Mises (vmlss).
    # The normalizer c(κ,λ) = (1−λ)/[(1+λ)·2π·I0(κ) − 2λ·∫e^{κcos B}] is numeric;
    # its κ,λ gradient is finite-differenced off the tested `_c_invbatschelet`.
    # The *lss
    # alias is the module-level `ibslss`. ---
    param_roles = {
        "xi": "location",
        "kappa": "concentration",
        "nu": "skewness",
        "lmbd": "shape",
    }
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): kappa toward 0 (linear); skewness nu and peakedness lmbd off
    # their +/-1 walls (scale 10), keeping the inverse-Batschelet normalizer fast
    # and non-singular.
    degen_penalty = (
        _degen_linear("kappa"),
        _degen_boundary_sym("nu", 1.0, 10.0),
        _degen_boundary_sym("lmbd", 1.0, 10.0),
    )
    default_links = {
        "location": "tanhalf",
        "concentration": "log",
        "skewness": "tanh",     # ν ∈ (−1, 1)
        "shape": "tanh",        # λ ∈ (−1, 1)
    }

    def _concentration_start(self, Rbar):
        """Closed-form κ start from the mean resultant ``Rbar`` for the
        regression null model: the von Mises A1-inverse (the ν→0, λ→0 reduction
        member), clamped — the circlss ``initialize`` convention (see
        CircularLL._null_params)."""
        return float(np.clip(A1inv(Rbar), 0.01, 500.0))

    def dlogpdf(self, x, xi, kappa, nu, lmbd):
        r"""First derivatives of ``logpdf`` w.r.t. the parameters (l1).

        With `φ⋆ = t_ν⁻¹(φ)`, `u⋆ = s_λ⁻¹(φ⋆)`, the kernel argument
        `A = u⋆ − ½(1−λ) sin u⋆ = B(u⋆)`, `S = κ sin A`, and the solver
        slopes `Tν = 1 + ν sin φ⋆`, `Sλ = 1 − ½(1+λ) cos u⋆`,
        `B′ = 1 − ½(1−λ) cos u⋆`:

        $$\ell_\xi = S\,B'/(T_\nu S_\lambda),\quad
          \ell_\nu = -S\,B'\,(1+\cos\varphi^\star)/(T_\nu S_\lambda),\quad
          \ell_\kappa = \cos A + \partial_\kappa\log c,\quad
          \ell_\lambda = -S\,\tfrac12\sin u^\star\,(1+B'/S_\lambda)
                          + \partial_\lambda\log c.$$

        The kernel terms are fully analytic (implicit differentiation of the
        two warps); the normalizer gradient `∂log c/∂{κ,λ}` is finite-differenced
        off `_c_invbatschelet`, once per unique (κ,λ). Vectorizes over
        per-observation parameter arrays; returns a book-named dict.
        """
        kappa = np.asarray(kappa, dtype=float)
        lmbd = np.asarray(lmbd, dtype=float)
        g = _invbat_dlogkernel(x, xi, kappa, nu, lmbd)   # (..., 4): ξ, κ, ν, λ
        dlogc_dk, dlogc_dl = _invbat_logc_grad_vec(kappa, lmbd)
        return {
            "xi": g[..., 0],
            "kappa": g[..., 1] + dlogc_dk,
            "nu": g[..., 2],
            "lmbd": g[..., 3] + dlogc_dl,
        }

    def d2logpdf(self, x, xi, kappa, nu, lmbd):
        r"""Second derivatives of ``logpdf`` (l2) — the 10 unique unordered
        pairs of the 4-LP family. Two blocks:

        - **kernel** (every pair): central finite-difference of the *analytic*
          kernel gradient :func:`_invbat_dlogkernel` (FD of a closed-form
          gradient, not FD-of-FD), symmetrized;
        - **normalizer** (only the κ,κ / κ,λ / λ,λ pairs, since c⊥ξ,ν):
          :func:`_invbat_logc_hess_vec`, direct second differences of log c.

        EFS-grade Hessian; the end-to-end gate is intercept-only parity vs
        ``inverse_batschelet.fit``. Returns a book-named dict
        keyed by unordered parameter pairs.
        """
        x, xi, kappa, nu, lmbd = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (x, xi, kappa, nu, lmbd))
        )
        base = [xi, kappa, nu, lmbd]
        steps = [1e-6, 1e-5 * np.maximum(1.0, np.abs(kappa)), 1e-6, 1e-6]
        # kernel Hessian: H[..., a, b] = ∂(kernel grad)_a / ∂param_b
        H = np.empty(x.shape + (4, 4), dtype=float)
        for b in range(4):
            hb = steps[b]
            pp = list(base)
            pm = list(base)
            pp[b] = base[b] + hb
            pm[b] = base[b] - hb
            gp = _invbat_dlogkernel(x, *pp)
            gm = _invbat_dlogkernel(x, *pm)
            H[..., :, b] = (gp - gm) / (2.0 * np.asarray(hb)[..., None])
        H = 0.5 * (H + np.swapaxes(H, -1, -2))           # symmetrize

        hkk, hkl, hll = _invbat_logc_hess_vec(kappa, lmbd)
        i = {"xi": 0, "kappa": 1, "nu": 2, "lmbd": 3}
        norm = {("kappa", "kappa"): hkk, ("kappa", "lmbd"): hkl,
                ("lmbd", "lmbd"): hll}
        out = {}
        for a, b in combinations_with_replacement(("xi", "kappa", "nu", "lmbd"), 2):
            val = H[..., i[a], i[b]]
            if (a, b) in norm:
                val = val + norm[(a, b)]
            out[(a, b)] = val
        return out

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._invbat_table_cache = {}
        self._invbat_sampler_cache = {}

    def _clear_normalization_cache(self):
        super()._clear_normalization_cache()
        self._invbat_table_cache = {}
        self._invbat_sampler_cache = {}

    def _argcheck(self, xi, kappa, nu, lmbd):
        try:
            xi_arr, kappa_arr, nu_arr, lmbd_arr = np.broadcast_arrays(xi, kappa, nu, lmbd)
        except ValueError:
            return False
        return (
            (xi_arr >= 0.0)
            & (xi_arr <= 2.0 * np.pi)
            & (kappa_arr >= 0.0)
            & np.isfinite(kappa_arr)
            & (nu_arr >= -1.0)
            & (nu_arr <= 1.0)
            & (lmbd_arr >= -1.0)
            & (lmbd_arr <= 1.0)
        )

    def _pdf(self, x, xi, kappa, nu, lmbd):
        scalar_input = np.isscalar(x)
        x_arr = np.asarray([x], dtype=float) if scalar_input else np.asarray(x, dtype=float)
        if x_arr.size == 0:
            return x_arr.astype(float)

        xi_val = _invbat_ensure_scalar(xi, "xi")
        kappa_val = float(np.clip(_invbat_ensure_scalar(kappa, "kappa"), 0.0, _INVBAT_KAPPA_UPPER))
        nu_val = _invbat_ensure_scalar(nu, "nu")
        lmbd_val = _invbat_ensure_scalar(lmbd, "lmbd")

        if not (
            np.isfinite(xi_val)
            and np.isfinite(kappa_val)
            and np.isfinite(nu_val)
            and np.isfinite(lmbd_val)
        ):
            result = np.full_like(x_arr, np.nan, dtype=float)
            return float(result[0]) if scalar_input else result

        if kappa_val <= _INVBAT_KAPPA_TOL:
            self._c = 1.0 / (2.0 * np.pi)
            result = np.full_like(x_arr, self._c, dtype=float)
            return float(result[0]) if scalar_input else result

        normalizer = self._get_cached_normalizer(
            lambda: _c_invbatschelet(kappa_val, lmbd_val),
            kappa_val,
            lmbd_val,
        )
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            normalizer = _c_invbatschelet_numeric(kappa_val, lmbd_val, grid_size=_INVBAT_NUMERIC_GRID)
        self._c = normalizer

        phi = _tnu(x_arr, nu_val, xi_val)
        skew = _slmbdinv(phi, lmbd_val)

        if np.isclose(lmbd_val, -1.0):
            log_kernel = kappa_val * np.cos(phi - np.sin(phi))
        else:
            con1 = (1.0 - lmbd_val) / (1.0 + lmbd_val)
            con2 = (2.0 * lmbd_val) / (1.0 + lmbd_val)
            log_kernel = kappa_val * np.cos(con1 * phi + con2 * skew)

        pdf_vals = normalizer * np.exp(log_kernel)
        pdf_vals = np.clip(pdf_vals, 0.0, None).astype(float, copy=False)

        if scalar_input:
            return float(pdf_vals.reshape(-1)[0])
        return pdf_vals

    def pdf(self, x, xi, kappa, nu, lmbd, *args, **kwargs):
        r"""
        Probability density function (PDF) of the inverse Batschelet distribution.

        Let

        - $\varphi = ((\theta - \xi + \pi) \bmod 2\pi) - \pi$,
        - $t_\nu^{-1}(\varphi)$ solve $y - \nu (1 + \cos y) = \varphi$,
        - $s_\lambda^{-1}(\cdot)$ solve $u - \tfrac{1 + \lambda}{2} \sin u = \cdot$,

        and set
        $\phi^\star = t_\nu^{-1}(\varphi)$,
        $u^\star = s_\lambda^{-1}(\phi^\star)$,
        $a = \tfrac{1 - \lambda}{1 + \lambda}$,
        $b = \tfrac{2 \lambda}{1 + \lambda}$.
        The inverse Batschelet density is

        $$
        f(\theta) = c(\kappa, \lambda)
        \exp\bigl[\kappa \cos\bigl(a\,\phi^\star + b\,u^\star\bigr)\bigr],
        $$

        where $c(\kappa,\lambda)$ is the normalising constant (independent of
        $\xi$ and $\nu$). For $\kappa \rightarrow 0$ the distribution reduces to
        the circular uniform density $1/(2\pi)$.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF, defined on the interval $[0, 2\pi)$.
        xi : float
            Direction parameter, $0 \leq \xi \leq 2\pi$.
        kappa : float
            Concentration parameter, $\kappa \geq 0$. Higher values result in sharper peaks around $\xi$.
        nu : float
            Skewness parameter, $-1 \leq \nu \leq 1$. Controls asymmetry through the angular transformation.
        lmbd : float
            Peakedness parameter, $-1 \leq \lambda \leq 1$. Controls the peak shape, from flat-topped to sharply peaked.

        Returns
        -------
        pdf_values : array_like
            Values of the probability density function at the specified points.
        """
        return super().pdf(x, xi, kappa, nu, lmbd, *args, **kwargs)

    def _logpdf(self, x, xi, kappa, nu, lmbd):
        # log(normalizer) + log-kernel — the assembly ``_pdf`` exponentiates
        # the warped vM kernel underflows from κ ≈ 360 at the
        # antipodal flank while the log form stays finite across the
        # κ ≤ 700 range
        scalar_input = np.isscalar(x)
        x_arr = np.asarray([x], dtype=float) if scalar_input else np.asarray(x, dtype=float)
        if x_arr.size == 0:
            return x_arr.astype(float)

        # Regression path: per-observation parameter arrays (each datum its own
        # κ_i, λ_i → a per-observation normalizer). The scalar descriptive path
        # below can't take array warp/normalizer params; this is what
        # CircularLL.ll()/fit drive. Mirrors jonespewsey._logpdf's branch.
        if any(_invbat_as_scalar(v) is None for v in (xi, kappa, nu, lmbd)):
            xb, xib, kb, nb, lb = np.broadcast_arrays(
                *(np.asarray(v, dtype=float) for v in (x, xi, kappa, nu, lmbd))
            )
            kb = np.clip(kb, 0.0, _INVBAT_KAPPA_UPPER)
            phi_star, u_star = _invbat_warp_vec(xb, xib, nb, lb)
            A = u_star - 0.5 * (1.0 - lb) * np.sin(u_star)
            log_kernel = kb * np.cos(A)
            return np.where(kb <= _INVBAT_KAPPA_TOL, -np.log(2.0 * np.pi),
                            log_kernel + _invbat_log_c_vec(kb, lb))

        xi_val = _invbat_ensure_scalar(xi, "xi")
        kappa_val = float(np.clip(_invbat_ensure_scalar(kappa, "kappa"), 0.0, _INVBAT_KAPPA_UPPER))
        nu_val = _invbat_ensure_scalar(nu, "nu")
        lmbd_val = _invbat_ensure_scalar(lmbd, "lmbd")

        if not (
            np.isfinite(xi_val)
            and np.isfinite(kappa_val)
            and np.isfinite(nu_val)
            and np.isfinite(lmbd_val)
        ):
            result = np.full_like(x_arr, np.nan, dtype=float)
            return float(result[0]) if scalar_input else result

        if kappa_val <= _INVBAT_KAPPA_TOL:
            result = np.full_like(x_arr, -np.log(2.0 * np.pi), dtype=float)
            return float(result[0]) if scalar_input else result

        normalizer = self._get_cached_normalizer(
            lambda: _c_invbatschelet(kappa_val, lmbd_val),
            kappa_val,
            lmbd_val,
        )
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            normalizer = _c_invbatschelet_numeric(kappa_val, lmbd_val, grid_size=_INVBAT_NUMERIC_GRID)

        phi = _tnu(x_arr, nu_val, xi_val)
        skew = _slmbdinv(phi, lmbd_val)

        if np.isclose(lmbd_val, -1.0):
            log_kernel = kappa_val * np.cos(phi - np.sin(phi))
        else:
            con1 = (1.0 - lmbd_val) / (1.0 + lmbd_val)
            con2 = (2.0 * lmbd_val) / (1.0 + lmbd_val)
            log_kernel = kappa_val * np.cos(con1 * phi + con2 * skew)

        logpdf_vals = np.log(normalizer) + log_kernel
        if scalar_input:
            return float(logpdf_vals.reshape(-1)[0])
        return logpdf_vals

    def logpdf(self, x, xi, kappa, nu, lmbd, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the inverse
        Batschelet distribution: the log normalizing constant plus the
        warped von Mises log-kernel — finite across the advertised
        parameter range, including tails where the density underflows.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.
        xi : float
            Direction parameter, 0 <= xi <= 2*pi.
        kappa : float
            Concentration parameter, 0 <= kappa <= 700.
        nu : float
            Skewness parameter, -1 <= nu <= 1.
        lmbd : float
            Peakedness parameter, -1 <= lmbd <= 1.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, xi, kappa, nu, lmbd, *args, **kwargs)

    def _cdf(self, x, xi, kappa, nu, lmbd):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        flat = arr.reshape(-1)

        if flat.size == 0:
            return arr.astype(float)

        xi_val = _invbat_ensure_scalar(xi, "xi")
        kappa_val = float(np.clip(_invbat_ensure_scalar(kappa, "kappa"), 0.0, _INVBAT_KAPPA_UPPER))
        nu_val = _invbat_ensure_scalar(nu, "nu")
        lmbd_val = _invbat_ensure_scalar(lmbd, "lmbd")

        if not (
            np.isfinite(xi_val)
            and np.isfinite(kappa_val)
            and np.isfinite(nu_val)
            and np.isfinite(lmbd_val)
        ):
            return np.full_like(arr, np.nan, dtype=float)

        two_pi = 2.0 * np.pi

        if kappa_val <= _INVBAT_KAPPA_TOL:
            cdf_flat = flat / two_pi
        else:
            table = self._get_invbat_table(kappa_val, nu_val, lmbd_val)
            phi = ((flat - xi_val + np.pi) % two_pi) - np.pi
            phi_start = ((-xi_val + np.pi) % two_pi) - np.pi
            H = table["cdf_interp"](phi)
            H_start = float(table["cdf_interp"](phi_start))
            diff = H - H_start
            cdf_flat = np.where(diff < 0.0, diff + 1.0, diff)
            cdf_flat = np.clip(cdf_flat, 0.0, 1.0)

        if arr.ndim == 0:
            value = float(cdf_flat[0])
            if np.isclose(float(wrapped), two_pi, rtol=0.0, atol=1e-12):
                return 1.0
        else:
            value = cdf_flat.reshape(arr.shape)
            mask_upper = np.isclose(arr, two_pi, rtol=0.0, atol=1e-12)
            if np.any(mask_upper):
                value = value.copy()
                value[mask_upper] = 1.0
        return value

    def cdf(self, x, xi, kappa, nu, lmbd, *args, **kwargs):
        r"""
        Cumulative distribution function of the inverse Batschelet distribution.

        The implementation precomputes the normalised primitive on a periodic grid
        in the centred angle $\varphi = (\theta - \xi) \bmod 2\pi - \pi$. For each
        grid node, the inverse skewness transform $t_\nu^{-1}$ and inverse
        Batschelet peakedness $s_\lambda^{-1}$ are evaluated, and the resulting kernel is
        accumulated via a trapezoidal rule. The cumulative table is cached per
        parameter triple $(\kappa, \nu, \lambda)$, enabling $O(1)$ queries after the
        initial $O(N)$ precomputation. The limit $\kappa \to 0$ reduces to the
        circular uniform CDF $\theta / (2\pi)$.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        xi : float
            Direction parameter, $0 \leq \xi \leq 2\pi$.
        kappa : float
            Concentration parameter, $\kappa \geq 0$.
        nu : float
            Skewness parameter, $-1 \leq \nu \leq 1$.
        lmbd : float
            Peakedness parameter, $-1 \leq \lambda \leq 1$.

        Returns
        -------
        cdf_values : array_like
            Cumulative probabilities corresponding to `x`.
        """
        xi_val = _invbat_ensure_scalar(xi, "xi")
        kappa_val = float(np.clip(_invbat_ensure_scalar(kappa, "kappa"), 0.0, _INVBAT_KAPPA_UPPER))
        nu_val = _invbat_ensure_scalar(nu, "nu")
        lmbd_val = _invbat_ensure_scalar(lmbd, "lmbd")
        return super().cdf(x, xi_val, kappa_val, nu_val, lmbd_val, *args, **kwargs)

    def _ppf(self, q, xi, kappa, nu, lmbd):
        xi_val = _invbat_ensure_scalar(xi, "xi")
        kappa_val = float(np.clip(_invbat_ensure_scalar(kappa, "kappa"), 0.0, _INVBAT_KAPPA_UPPER))
        nu_val = _invbat_ensure_scalar(nu, "nu")
        lmbd_val = _invbat_ensure_scalar(lmbd, "lmbd")

        q_arr = np.asarray(q, dtype=float)
        flat = q_arr.reshape(-1)
        if flat.size == 0:
            return q_arr.astype(float)

        two_pi = 2.0 * np.pi
        result = np.full_like(flat, np.nan, dtype=float)

        valid = np.isfinite(flat) & (flat >= 0.0) & (flat <= 1.0)
        if not np.any(valid):
            shaped = result.reshape(q_arr.shape)
            return float(shaped) if q_arr.ndim == 0 else shaped

        q_valid = flat[valid]
        close_zero = np.isclose(q_valid, 0.0, rtol=0.0, atol=1e-12)
        close_one = np.isclose(q_valid, 1.0, rtol=0.0, atol=1e-12)

        if kappa_val <= _INVBAT_KAPPA_TOL:
            theta = (two_pi * q_valid) % two_pi
            if np.any(close_zero):
                theta[close_zero] = 0.0
            if np.any(close_one):
                theta[close_one] = two_pi
            result[valid] = theta
        else:
            table = self._get_invbat_table(kappa_val, nu_val, lmbd_val)
            phi_grid = table["phi"]
            cdf_grid = table["cdf"]
            cdf_interp = table["cdf_interp"]
            inv_interp = table["inv_cdf_interp"]
            pdf_interp = table["pdf_interp"]

            phi_start = ((-xi_val + np.pi) % two_pi) - np.pi
            H_start = float(cdf_interp(phi_start))
            targets = (H_start + q_valid) % 1.0

            phi_candidates = (
                inv_interp(targets)
                if inv_interp is not None
                else np.interp(targets, cdf_grid, phi_grid, left=phi_grid[0], right=phi_grid[-1])
            )

            # Vectorized safeguarded Newton on the monotone cdf: all targets
            # at once, each point's [phi_lo, phi_hi] bracket from the grid,
            # converged points frozen. Replaces the per-point loop that called
            # the Pchip interpolators ~30k times scalar (one call/iteration);
            # now ≤ _INVBAT_NEWTON_MAXITER array evaluations total.
            i_hi = np.clip(np.searchsorted(cdf_grid, targets, side="right"),
                           1, len(phi_grid) - 1)
            phi_lo = phi_grid[i_hi - 1].astype(float, copy=True)
            phi_hi = phi_grid[i_hi].astype(float, copy=True)
            phi = np.clip(phi_candidates, phi_lo, phi_hi)

            done = np.zeros(q_valid.shape, dtype=bool)
            tiny = np.finfo(float).tiny
            for _ in range(_INVBAT_NEWTON_MAXITER):
                H_phi = np.asarray(cdf_interp(phi), dtype=float)
                residual = H_phi - targets
                pdf_val = np.maximum(np.asarray(pdf_interp(phi), dtype=float), tiny)

                done |= (np.abs(residual) <= _INVBAT_NEWTON_TOL) & (
                    (phi_hi - phi_lo) <= _INVBAT_NEWTON_WIDTH_TOL)
                if np.all(done):
                    break

                hi_upd = residual > 0.0
                phi_hi = np.where(hi_upd, np.minimum(phi_hi, phi), phi_hi)
                phi_lo = np.where(~hi_upd, np.maximum(phi_lo, phi), phi_lo)

                cand = phi - residual / pdf_val
                fallback = ~np.isfinite(cand) | (cand <= phi_lo) | (cand >= phi_hi)
                cand = np.where(fallback, 0.5 * (phi_lo + phi_hi), cand)
                phi = np.where(done, phi, np.clip(cand, phi_lo, phi_hi))

            theta_vals = (xi_val + phi) % two_pi
            theta_vals[close_zero] = 0.0
            theta_vals[close_one] = two_pi
            result[valid] = theta_vals

        shaped = result.reshape(q_arr.shape)
        if q_arr.ndim == 0:
            return float(shaped)
        return shaped

    def ppf(self, q, xi, kappa, nu, lmbd, *args, **kwargs):
        r"""
        Percent-point function (quantile) of the inverse Batschelet distribution.

        Quantiles are obtained by inverting the cached cumulative table described in
        `cdf`. A monotone initial guess supplied by the table inverse is refined
        with safeguarded Newton steps that leverage the tabulated density, while
        preserving a bracketing interval. For $\kappa \rightarrow 0$, the quantile
        reduces to the linear uniform mapping $2\pi q$.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate (0 <= q <= 1).
        xi : float
            Direction parameter, $0 \leq \xi \leq 2\pi$.
        kappa : float
            Concentration parameter, $\kappa \geq 0$.
        nu : float
            Skewness parameter, $-1 \leq \nu \leq 1$.
        lmbd : float
            Peakedness parameter, $-1 \leq \lambda \leq 1$.

        Returns
        -------
        ppf_values : array_like
            Angles corresponding to the probabilities in `q`.
        """
        xi_val = _invbat_ensure_scalar(xi, "xi")
        kappa_val = float(np.clip(_invbat_ensure_scalar(kappa, "kappa"), 0.0, _INVBAT_KAPPA_UPPER))
        nu_val = _invbat_ensure_scalar(nu, "nu")
        lmbd_val = _invbat_ensure_scalar(lmbd, "lmbd")
        return super().ppf(q, xi_val, kappa_val, nu_val, lmbd_val, *args, **kwargs)

    def _get_invbat_sampler_params(self, kappa, nu, lmbd):
        key = (float(kappa), float(nu), float(lmbd))
        params = self._invbat_sampler_cache.get(key)
        if params is not None:
            return params

        table = self._get_invbat_table(kappa, nu, lmbd)
        phi = table["phi"]
        pdf = table["pdf"]
        log_pdf = np.log(np.clip(pdf, np.finfo(float).tiny, None))

        idx0 = int(np.argmin(np.abs(phi)))
        if idx0 == 0:
            idx0 = 1
        elif idx0 == phi.size - 1:
            idx0 = phi.size - 2

        h1 = phi[idx0] - phi[idx0 - 1]
        h2 = phi[idx0 + 1] - phi[idx0]
        if not np.isfinite(h1) or not np.isfinite(h2) or h1 == 0.0 or h2 == 0.0:
            curvature = max(kappa, 1.0)
        else:
            d2 = (
                log_pdf[idx0 + 1]
                - 2.0 * log_pdf[idx0]
                + log_pdf[idx0 - 1]
            ) / ((0.5 * (h1 + h2)) ** 2)
            curvature = max(-d2, 1e-3)

        kappa_env = float(np.clip(curvature, _INVBAT_ENV_MIN_KAPPA, _INVBAT_KAPPA_UPPER))
        log_vm_norm = np.log(2.0 * np.pi) + np.log(i0e(kappa_env)) + kappa_env
        log_ratio = log_pdf + log_vm_norm - kappa_env * np.cos(phi)
        log_multiplier = float(np.max(log_ratio))
        multiplier = float(np.exp(log_multiplier) * 1.02)

        params = {
            "kappa_env": kappa_env,
            "log_vm_norm": log_vm_norm,
            "log_multiplier": np.log(multiplier),
            "multiplier": multiplier,
        }
        self._invbat_sampler_cache[key] = params
        return params

    def _rvs(self, xi, kappa, nu, lmbd, size=None, random_state=None):
        rng = self._init_rng(random_state)

        xi_val = float(np.mod(_invbat_ensure_scalar(xi, "xi"), 2.0 * np.pi))
        kappa_val = float(np.clip(_invbat_ensure_scalar(kappa, "kappa"), 0.0, _INVBAT_KAPPA_UPPER))
        nu_val = _invbat_ensure_scalar(nu, "nu")
        lmbd_val = _invbat_ensure_scalar(lmbd, "lmbd")

        if not (
            np.isfinite(xi_val)
            and np.isfinite(kappa_val)
            and np.isfinite(nu_val)
            and np.isfinite(lmbd_val)
        ):
            raise ValueError("`xi`, `kappa`, `nu`, and `lmbd` must be finite scalars.")

        if size is None:
            shape = ()
            total = 1
        else:
            if np.isscalar(size):
                shape = (int(size),)
            else:
                shape = tuple(int(dim) for dim in np.atleast_1d(size))
            total = int(np.prod(shape, dtype=int))
            if total < 0:
                raise ValueError("`size` must describe a non-negative number of samples.")

        two_pi = 2.0 * np.pi

        if total == 0:
            empty = np.empty(shape, dtype=float)
            return float(empty) if empty.ndim == 0 else empty

        if kappa_val <= _INVBAT_KAPPA_TOL:
            samples = rng.uniform(0.0, two_pi, size=shape)
            return float(samples) if samples.ndim == 0 else samples

        table = self._get_invbat_table(kappa_val, nu_val, lmbd_val)
        sampler = self._get_invbat_sampler_params(kappa_val, nu_val, lmbd_val)
        kappa_env = sampler["kappa_env"]
        log_vm_norm = sampler["log_vm_norm"]
        log_multiplier = sampler["log_multiplier"]
        pdf_interp = table["pdf_interp"]

        samples = np.empty(total, dtype=float)
        filled = 0
        batch_base = max(8, min(4 * total, 4096))

        while filled < total:
            batch = min(batch_base, total - filled) if filled > 0 else batch_base
            proposals = rng.vonmises(xi_val, kappa_env, size=batch)
            phi = ((proposals - xi_val + np.pi) % two_pi) - np.pi

            pdf_vals = np.clip(pdf_interp(phi), np.finfo(float).tiny, None)
            log_target = np.log(pdf_vals)
            log_env = kappa_env * np.cos(phi) - log_vm_norm
            log_accept = log_target - log_env - log_multiplier

            accept_mask = np.log(rng.random(size=batch)) <= log_accept
            if not np.any(accept_mask):
                continue

            accepted = proposals[accept_mask]
            take = min(accepted.size, total - filled)
            samples[filled : filled + take] = accepted[:take]
            filled += take

        samples = np.mod(samples, two_pi)
        samples = samples.reshape(shape)
        if samples.ndim == 0:
            return float(samples)
        return samples

    def rvs(self, xi=None, kappa=None, nu=None, lmbd=None, size=None, random_state=None):
        r"""
        Draw random variates from the inverse Batschelet distribution.

        Sampling proceeds by acceptance--rejection with a von Mises envelope whose
        concentration is matched to the curvature of the inverse Batschelet kernel at
        the mode. Envelope constants are calibrated on the cached spectral grid used
        for `cdf`, so repeated sampling calls with the same parameters are fast
        and stable across the entire parameter range.

        Parameters
        ----------
        xi : float
            Direction parameter, $0 \leq \xi \leq 2\pi$.
        kappa : float
            Concentration parameter, $\kappa \geq 0$.
        nu : float
            Skewness parameter, $-1 \leq \nu \leq 1$.
        lmbd : float
            Peakedness parameter, $-1 \leq \lambda \leq 1$.
        size : int or tuple of ints, optional
            Desired output shape.
        random_state : {None, int, np.random.Generator}, optional
            Random number generator specification.

        Returns
        -------
        rvs : array_like
            Random variates on $[0, 2\pi)$ sampled from the inverse Batschelet
            distribution.
        """

        xi_val = _invbat_ensure_scalar(xi, "xi")
        kappa_val = _invbat_ensure_scalar(kappa, "kappa")
        nu_val = _invbat_ensure_scalar(nu, "nu")
        lmbd_val = _invbat_ensure_scalar(lmbd, "lmbd")
        return super().rvs(xi_val, kappa_val, nu_val, lmbd_val, size=size, random_state=random_state)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        optimizer="L-BFGS-B",
        options=None,
        nu_grid=None,
        lmbd_grid=None,
        kappa_bounds=(1e-6, _INVBAT_KAPPA_UPPER),
        nu_bounds=(-0.99, 0.99),
        lmbd_bounds=(-0.99, 0.99),
        return_info=False,
        **minimize_kwargs,
    ):
        r"""
        Estimate $(\xi, \kappa, \nu, \lambda)$ from circular data.

        ``method='mle'`` maximises the weighted log-likelihood using the cached
        spectral tables for the pdf and normalising constant. ``method='moments'``
        returns the circular mean, ``circ_kappa`` estimate, and sets $(\nu, \lambda)
        = (0, 0)$.

        Parameters
        ----------
        data : array_like
            Sample of angles.
        weights : array_like, optional
            Non-negative weights broadcastable to ``data``.
        method : {'mle', 'moments'}, default 'mle'
            Estimation method.
        optimizer : str, optional
            SciPy optimiser for maximum likelihood.
        options : dict, optional
            Optimiser options forwarded to :func:`scipy.optimize.minimize`.
        nu_grid : array_like, optional
            Candidate $
            u$ values for profiling the starting point.
        lmbd_grid : array_like, optional
            Candidate $
            u$ values for $
            lambda$ profiling.
        kappa_bounds, nu_bounds, lmbd_bounds : tuple, optional
            Parameter bounds enforced during optimisation.
        return_info : bool, optional
            If True, also return a dictionary with optimisation diagnostics.
        **minimize_kwargs :
            Additional keyword arguments forwarded to
            :func:`scipy.optimize.minimize`.

        Returns
        -------
        params : tuple
            Estimated parameters ``(xi, kappa, nu, lmbd)``.
        info : dict, optional
            Returned when ``return_info=True`` with optimisation diagnostics.
        """

        minimize_kwargs = self._sanitize_fit_kwargs(minimize_kwargs)
        minimize_kwargs.pop("floc", None)
        minimize_kwargs.pop("fscale", None)

        data_arr = self._wrap_angles(np.asarray(data, dtype=float)).ravel()
        if data_arr.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(data_arr, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, data_arr.shape).astype(float, copy=False).ravel()

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = float(w_sum**2 / np.sum(w**2))

        xi_mom, r1 = circ_mean_and_r(alpha=data_arr, w=w)
        if not np.isfinite(xi_mom):
            xi_mom = 0.0
        xi_mom = float(np.mod(xi_mom, 2.0 * np.pi))
        r1 = float(np.clip(r1, 1e-12, 1.0 - 1e-12))
        n_adjust = int(max(1, round(w_sum)))
        kappa_mom = float(np.clip(circ_kappa(r=r1, n=n_adjust), kappa_bounds[0], kappa_bounds[1]))

        if method == "moments":
            estimates = (xi_mom, kappa_mom, 0.0, 0.0)
            if return_info:
                info = {
                    "method": "moments",
                    "converged": True,
                    "loglik": float(-np.sum(w) * np.log(2.0 * np.pi)) if kappa_mom <= _INVBAT_KAPPA_TOL else float("nan"),
                    "n_effective": n_eff,
                }
                return estimates, info
            return estimates

        method_key = str(method).lower()
        if method_key != "mle":
            raise ValueError("`method` must be one of {'mle', 'moments' }.")

        two_pi = 2.0 * np.pi

        if nu_grid is None:
            nu_grid = np.linspace(nu_bounds[0], nu_bounds[1], 5)
        else:
            nu_grid = np.asarray(nu_grid, dtype=float)

        if lmbd_grid is None:
            lmbd_grid = np.linspace(lmbd_bounds[0], lmbd_bounds[1], 5)
        else:
            lmbd_grid = np.asarray(lmbd_grid, dtype=float)

        def nll(params):
            xi_param, kappa_param, nu_param, lmbd_param = params
            if not (0.0 <= xi_param <= two_pi):
                return np.inf
            if not (kappa_bounds[0] <= kappa_param <= kappa_bounds[1]):
                return np.inf
            if not (nu_bounds[0] <= nu_param <= nu_bounds[1]):
                return np.inf
            if not (lmbd_bounds[0] <= lmbd_param <= lmbd_bounds[1]):
                return np.inf

            xi_wrapped = float(np.mod(xi_param, two_pi))
            if kappa_param <= _INVBAT_KAPPA_TOL:
                log_pdf = -np.log(two_pi)
                return float(-np.sum(w * log_pdf))

            table = self._get_invbat_table(float(kappa_param), float(nu_param), float(lmbd_param))
            phi = ((data_arr - xi_wrapped + np.pi) % two_pi) - np.pi
            pdf_vals = table["pdf_interp"](phi)
            if np.any(pdf_vals <= 0.0) or not np.all(np.isfinite(pdf_vals)):
                return np.inf
            return float(-np.sum(w * np.log(pdf_vals)))

        best_nu = 0.0
        best_lmbd = 0.0
        best_score = nll((xi_mom, kappa_mom, best_nu, best_lmbd))
        for nu_candidate in np.unique(np.concatenate(([0.0], nu_grid))):
            for lmbd_candidate in np.unique(np.concatenate(([0.0], lmbd_grid))):
                score = nll((xi_mom, kappa_mom, float(nu_candidate), float(lmbd_candidate)))
                if score < best_score:
                    best_score = score
                    best_nu = float(nu_candidate)
                    best_lmbd = float(lmbd_candidate)

        init = np.array([xi_mom, kappa_mom, best_nu, best_lmbd], dtype=float)
        bounds = [
            (0.0, two_pi),
            (kappa_bounds[0], kappa_bounds[1]),
            (nu_bounds[0], nu_bounds[1]),
            (lmbd_bounds[0], lmbd_bounds[1]),
        ]

        options = {} if options is None else dict(options)

        result = minimize(
            nll,
            init,
            method=optimizer,
            bounds=bounds,
            options=options,
            **minimize_kwargs,
        )

        optimizer_used = optimizer
        if not result.success and optimizer != "Powell":
            fallback = minimize(
                nll,
                init,
                method="Powell",
                bounds=bounds,
                options={},
                **minimize_kwargs,
            )
            if fallback.success:
                result = fallback
                optimizer_used = "Powell"

        if not result.success:
            raise RuntimeError(f"Maximum likelihood fit failed: {result.message}")

        xi_hat = self._wrap_direction(float(result.x[0]))
        kappa_hat = float(np.clip(result.x[1], kappa_bounds[0], kappa_bounds[1]))
        nu_hat = float(np.clip(result.x[2], nu_bounds[0], nu_bounds[1]))
        lmbd_hat = float(np.clip(result.x[3], lmbd_bounds[0], lmbd_bounds[1]))

        estimates = (xi_hat, kappa_hat, nu_hat, lmbd_hat)
        if not return_info:
            return estimates

        info = {
            "method": "mle",
            "loglik": float(-result.fun),
            "n_effective": n_eff,
            "converged": bool(result.success),
            "optimizer": optimizer_used,
            "nit": getattr(result, "nit", np.nan),
            "nfev": getattr(result, "nfev", np.nan),
            "message": result.message,
        }
        return estimates, info

    def _get_invbat_table(self, kappa, nu, lmbd, grid_size=None):
        kappa_val = float(np.clip(kappa, 0.0, _INVBAT_KAPPA_UPPER))
        nu_val = float(nu)
        lmbd_val = float(lmbd)
        if kappa_val <= _INVBAT_KAPPA_TOL:
            phi = np.array([-np.pi, np.pi], dtype=float)
            pdf_vals = np.full(2, 1.0 / (2.0 * np.pi), dtype=float)
            cdf_interp = PchipInterpolator(phi, [0.0, 1.0], extrapolate=True)
            pdf_interp = PchipInterpolator(phi, pdf_vals, extrapolate=True)
            return {
                "phi": phi,
                "pdf": pdf_vals,
                "cdf": np.array([0.0, 1.0], dtype=float),
                "cdf_interp": cdf_interp,
                "pdf_interp": pdf_interp,
                "inv_cdf_interp": PchipInterpolator([0.0, 1.0], phi, extrapolate=True),
                "log_normalizer": -np.log(2.0 * np.pi),
            }

        if grid_size is None:
            grid_size = _invbat_grid_size(kappa_val, nu_val, lmbd_val)
        grid_int = int(grid_size)
        key = (kappa_val, nu_val, lmbd_val, grid_int)
        table = self._invbat_table_cache.get(key)
        if table is None:
            table = self._build_invbat_table(kappa_val, nu_val, lmbd_val, grid_int)
            self._invbat_table_cache[key] = table
        return table

    def _build_invbat_table(self, kappa, nu, lmbd, grid_size):
        phi = np.linspace(-np.pi, np.pi, grid_size + 1, dtype=float)
        phi_star = _tnu(phi, nu, 0.0)
        skew = _slmbdinv(phi_star, lmbd)

        if np.isclose(lmbd, -1.0, atol=_INVBAT_LMBDA_TOL):
            log_kernel = kappa * np.cos(phi_star - np.sin(phi_star))
        else:
            con1 = (1.0 - lmbd) / (1.0 + lmbd)
            con2 = (2.0 * lmbd) / (1.0 + lmbd)
            log_kernel = kappa * np.cos(con1 * phi_star + con2 * skew)

        normalizer = self._get_cached_normalizer(
            lambda: _c_invbatschelet(kappa, lmbd),
            kappa,
            lmbd,
        )
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            normalizer = _c_invbatschelet_numeric(kappa, lmbd, grid_size=_INVBAT_NUMERIC_GRID)
            cache = self._get_normalization_cache()
            cache[(kappa, lmbd)] = normalizer

        log_norm = np.log(normalizer)
        log_pdf = log_norm + log_kernel
        log_pdf = np.clip(log_pdf, -745.0, 700.0)
        pdf = np.exp(log_pdf)

        step = (2.0 * np.pi) / grid_size
        avg = 0.5 * (pdf[:-1] + pdf[1:])
        mass = float(np.sum(avg) * step)
        if not np.isfinite(mass) or mass <= 0.0:
            pdf = np.full_like(pdf, 1.0 / (2.0 * np.pi), dtype=float)
            log_norm = -np.log(2.0 * np.pi)
            mass = 1.0
        elif abs(mass - 1.0) > 5e-10:
            scale = 1.0 / mass
            pdf *= scale
            log_norm += np.log(scale)
            mass = 1.0
            cache = self._get_normalization_cache()
            cache[(kappa, lmbd)] = np.exp(log_norm)

        avg = 0.5 * (pdf[:-1] + pdf[1:])
        cumulative = np.concatenate(([0.0], np.cumsum(avg))) * step
        cumulative = np.maximum.accumulate(np.clip(cumulative, 0.0, 1.0))
        cumulative[-1] = 1.0

        cdf_interp = PchipInterpolator(phi, cumulative, extrapolate=True)

        knots = _inverse_cdf_knots(phi, cumulative)
        inv_cdf_interp = (
            PchipInterpolator(knots[0], knots[1], extrapolate=True)
            if knots is not None
            else None
        )

        pdf_interp = PchipInterpolator(phi, pdf, extrapolate=True)

        return {
            "phi": phi,
            "pdf": pdf,
            "cdf": cumulative,
            "cdf_interp": cdf_interp,
            "pdf_interp": pdf_interp,
            "inv_cdf_interp": inv_cdf_interp,
            "log_normalizer": log_norm,
        }


inverse_batschelet = inverse_batschelet_gen(name="inverse_batschelet")
ibslss = CircularLL(inverse_batschelet, name="ibslss")


##########################################
## Helper Functions: inverse_batschelet ##
##########################################


def _solve_monotone_increasing(rhs, g, gprime, *, lo=-np.pi, hi=np.pi,
                               x0=None, tol=1e-14, max_iter=20):
    """Vectorized root of a smooth, monotone-increasing ``g(y) = rhs`` on
    ``[lo, hi]``, with ``g(lo) <= rhs <= g(hi)`` (the caller's bracket).

    Newton from ``x0`` (default ``rhs``), each step clamped into the bracket,
    then a vectorized bisection mop-up for any point Newton leaves with a
    residual above ``1e-12`` — the near-boundary cases where ``gprime`` → 0
    (ν, λ → ±1). Replaces the per-point ``brentq`` loop the inverse-Batschelet
    warps used to run (≈10⁵ scalar root-finds per ``fit``): one warp call is
    now a handful of array ops. Returns ``(y, gprime(y))`` — the root and its
    local slope, the latter being the Jacobian factor the ``ibslss`` score
    reuses."""
    rhs = np.asarray(rhs, dtype=float)
    y = (np.clip(rhs, lo, hi).copy() if x0 is None
         else np.clip(np.broadcast_to(x0, rhs.shape).astype(float), lo, hi))
    if y.size:
        for _ in range(max_iter):
            gp = gprime(y)
            step = np.divide(g(y) - rhs, gp, out=np.zeros_like(y),
                             where=np.abs(gp) > 1e-15)
            y_new = np.clip(y - step, lo, hi)
            if np.max(np.abs(y_new - y)) < tol:
                y = y_new
                break
            y = y_new
        bad = np.abs(g(y) - rhs) > 1e-12
        if np.any(bad):
            a = np.full(y.shape, lo, dtype=float)
            b = np.full(y.shape, hi, dtype=float)
            for _ in range(60):
                m = 0.5 * (a + b)
                left = g(m) <= rhs          # g increasing → root at/above m
                a = np.where(left, m, a)
                b = np.where(left, b, m)
            y = np.where(bad, 0.5 * (a + b), y)
    return y, gprime(y)


def _tnu(x, nu, xi):
    x_arr = np.asarray(x, dtype=float)
    scalar_input = x_arr.ndim == 0
    phi = np.mod(x_arr - xi + np.pi, 2.0 * np.pi) - np.pi

    if abs(nu) <= _INVBAT_NU_TOL:
        results = phi
    else:
        root, _ = _solve_monotone_increasing(
            phi,
            lambda y: y - nu * (1.0 + np.cos(y)),
            lambda y: 1.0 + nu * np.sin(y),
            x0=phi,
        )
        results = (root + np.pi) % (2.0 * np.pi) - np.pi

    if scalar_input:
        return float(np.asarray(results).reshape(-1)[0])
    return np.asarray(results).reshape(phi.shape)


def _slmbdinv(x, lmbd):
    x_arr = np.asarray(x, dtype=float)
    scalar_input = x_arr.ndim == 0
    x_flat = np.atleast_1d(x_arr).astype(float, copy=False)

    if np.isclose(lmbd, -1.0, atol=_INVBAT_LMBDA_TOL):
        result = x_flat.copy()
    else:
        c = 0.5 * (1.0 + lmbd)
        root, _ = _solve_monotone_increasing(
            x_flat,
            lambda u: u - c * np.sin(u),
            lambda u: 1.0 - c * np.cos(u),
            x0=x_flat,
        )
        result = (root + np.pi) % (2.0 * np.pi) - np.pi

    if scalar_input:
        return float(result[0])
    return result.reshape(x_arr.shape)


def _invbat_warp_vec(x, xi, nu, lmbd):
    """Vectorized per-observation inverse warps for the regression path:
    `φ⋆ = t_ν⁻¹(φ)` then `u⋆ = s_λ⁻¹(φ⋆)`, with `φ = (θ−ξ)` wrapped and every
    parameter an array (each datum its own ν_i, λ_i). The scalar `_tnu`/
    `_slmbdinv` can't take array warp parameters; this calls the shared
    monotone solver with array-valued closures. Returns `(φ⋆, u⋆)`, both
    wrapped to [−π, π)."""
    x, xi, nu, lmbd = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, xi, nu, lmbd))
    )
    phi = np.mod(x - xi + np.pi, 2.0 * np.pi) - np.pi
    phi_star, _ = _solve_monotone_increasing(
        phi,
        lambda y: y - nu * (1.0 + np.cos(y)),
        lambda y: 1.0 + nu * np.sin(y),
        x0=phi,
    )
    phi_star = (phi_star + np.pi) % (2.0 * np.pi) - np.pi
    c = 0.5 * (1.0 + lmbd)
    u_star, _ = _solve_monotone_increasing(
        phi_star,
        lambda u: u - c * np.sin(u),
        lambda u: 1.0 - c * np.cos(u),
        x0=phi_star,
    )
    u_star = (u_star + np.pi) % (2.0 * np.pi) - np.pi
    return phi_star, u_star


def _invbat_unique_pairs(kappa, lmbd):
    """Shared ``np.unique`` over (κ,λ) for the FD'd normalizer derivatives:
    returns ``(k, l, inverse, shape)``. Differencing only the *unique* pairs
    collapses an intercept-only or repeated-covariate fit to a handful of grid
    passes; the caller picks its own central step."""
    kappa, lmbd = np.broadcast_arrays(
        np.asarray(kappa, dtype=float), np.asarray(lmbd, dtype=float)
    )
    pairs, inverse = np.unique(
        np.stack([kappa.ravel(), lmbd.ravel()], axis=1), axis=0,
        return_inverse=True,
    )
    return pairs[:, 0], pairs[:, 1], inverse, kappa.shape


def _invbat_logc_grad_vec(kappa, lmbd):
    """``(∂log c/∂κ, ∂log c/∂λ)`` for the inverse-Batschelet normalizer
    ``c(κ,λ) = (1−λ)/[(1+λ)·2π·I0(κ) − 2λ·∫e^{κ cos B}]``, by central
    finite-difference of :func:`_invbat_log_c_array` — evaluated for *all*
    unique (κ,λ) pairs at once (no Python per-pair loop) on the coarser
    ``_INVBAT_DERIV_GRID``. The analytic gradient is closed-form but its
    overflow-safe assembly duplicates ``_c_invbatschelet_numeric``; FD reuses
    that exact normalizer, so it stays consistent with the logpdf the
    derivative tests difference."""
    k, lm, inverse, shape = _invbat_unique_pairs(kappa, lmbd)
    hk = 1e-6 * np.maximum(1.0, np.abs(k))
    hl = np.minimum(1e-6, 0.25 * (1.0 - np.abs(lm)))

    def lc(kk, ll):
        return _invbat_log_c_array(kk, ll, grid_size=_INVBAT_DERIV_GRID)

    gk = (lc(k + hk, lm) - lc(k - hk, lm)) / (2.0 * hk)
    gl = (lc(k, lm + hl) - lc(k, lm - hl)) / (2.0 * hl)
    grad = np.stack([gk, gl], axis=1)
    out = grad[inverse].reshape(shape + (2,))
    return out[..., 0], out[..., 1]


def _invbat_logc_hess_vec(kappa, lmbd):
    """``(∂²log c/∂κ², ∂²log c/∂κ∂λ, ∂²log c/∂λ²)`` by direct second central
    differences of :func:`_invbat_log_c_array`, vectorized over all unique
    (κ,λ) pairs on ``_INVBAT_DERIV_GRID`` — the normalizer block of `d2logpdf`.
    Direct second differences (one FD level on the smooth log c) avoid the
    FD-of-FD noise of differencing the already-FD'd gradient."""
    k, lm, inverse, shape = _invbat_unique_pairs(kappa, lmbd)
    # second differences want a larger step than the first-difference default
    hk = 1e-4 * np.maximum(1.0, np.abs(k))
    hl = np.minimum(1e-4, 0.25 * (1.0 - np.abs(lm)))

    def lc(kk, ll):
        return _invbat_log_c_array(kk, ll, grid_size=_INVBAT_DERIV_GRID)

    f0 = lc(k, lm)
    hkk = (lc(k + hk, lm) - 2.0 * f0 + lc(k - hk, lm)) / (hk * hk)
    hll = (lc(k, lm + hl) - 2.0 * f0 + lc(k, lm - hl)) / (hl * hl)
    hkl = (
        lc(k + hk, lm + hl) - lc(k + hk, lm - hl)
        - lc(k - hk, lm + hl) + lc(k - hk, lm - hl)
    ) / (4.0 * hk * hl)
    hess = np.stack([hkk, hkl, hll], axis=1)
    out = hess[inverse].reshape(shape + (3,))
    return out[..., 0], out[..., 1], out[..., 2]


def _invbat_log_c_vec(kappa, lmbd):
    """Vectorized ``log c(κ,λ)`` over per-observation params, once per unique
    (κ,λ) — the normalizer for the regression-path ``_logpdf`` (each datum its
    own κ_i, λ_i). Evaluates :func:`_invbat_log_c_array` (machine-identical to
    the scalar ``_c_invbatschelet``, edge pairs deferred to it) on the unique
    pairs at the full value grid — one vectorized pass replaces a per-pair
    Python loop over the scalar normalizer (the logpdf-array hotspot)."""
    kappa, lmbd = np.broadcast_arrays(
        np.asarray(kappa, dtype=float), np.asarray(lmbd, dtype=float)
    )
    pairs, inverse = np.unique(
        np.stack([kappa.ravel(), lmbd.ravel()], axis=1), axis=0,
        return_inverse=True,
    )
    vals = _invbat_log_c_array(
        pairs[:, 0], pairs[:, 1], grid_size=_INVBAT_NUMERIC_GRID
    )
    return vals[inverse].reshape(kappa.shape)


def _invbat_as_scalar(value):
    """Collapse a parameter to a float (tolerating constant arrays), or return
    ``None`` if it genuinely varies — the scalar-vs-regression-path switch the
    ``_logpdf`` uses (mirrors ``_jp_as_scalar``)."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.reshape(-1)
    if flat.size == 1:
        return float(flat[0])
    first = flat[0]
    if np.all(flat == first):
        return float(first)
    return None


def _invbat_dlogkernel(x, xi, kappa, nu, lmbd):
    """The four analytic log-*kernel* gradient components (no normalizer),
    stacked as ``(..., 4)`` in book order [ξ, κ, ν, λ]. The per-observation
    score reuses the two warps' solved roots φ⋆, u⋆ and their slopes; see
    ``inverse_batschelet_gen.dlogpdf``. Isolated as a module function so
    ``d2logpdf`` can central-difference it for the kernel Hessian block
    (FD of an analytic gradient — not FD-of-FD)."""
    x, xi, kappa, nu, lmbd = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, xi, kappa, nu, lmbd))
    )
    phi_star, u_star = _invbat_warp_vec(x, xi, nu, lmbd)
    half = 0.5 * (1.0 - lmbd)
    A = u_star - half * np.sin(u_star)
    Bp = 1.0 - half * np.cos(u_star)
    Tnu = 1.0 + nu * np.sin(phi_star)
    Slam = 1.0 - 0.5 * (1.0 + lmbd) * np.cos(u_star)
    S = kappa * np.sin(A)
    chain = Bp / (Tnu * Slam)
    d_xi = S * chain
    d_nu = -S * chain * (1.0 + np.cos(phi_star))
    d_lmbd = -S * 0.5 * np.sin(u_star) * (1.0 + Bp / Slam)
    d_kappa = np.cos(A)
    return np.stack([d_xi, d_kappa, d_nu, d_lmbd], axis=-1)


def _A1(kappa):
    return i1e(kappa) / i0e(kappa)  # scaled: i1/i0 is nan for κ ≥ 713


def _c_invbatschelet(kappa, lmbd):
    kappa_val = float(np.clip(kappa, 0.0, _INVBAT_KAPPA_UPPER))
    lmbd_val = float(lmbd)

    if kappa_val <= _INVBAT_KAPPA_TOL:
        return 1.0 / (2.0 * np.pi)

    if np.isclose(lmbd_val, 1.0, atol=_INVBAT_LMBDA_TOL):
        log_mult = np.log(2.0 * np.pi) + np.log(i0e(kappa_val)) + kappa_val
        K = 1.0 - _A1(kappa_val)
        if not np.isfinite(K) or K <= 0.0:
            return _c_invbatschelet_numeric(kappa_val, lmbd_val, grid_size=_INVBAT_NUMERIC_GRID * 2)
        log_c = -log_mult - np.log(K)
        return float(np.exp(log_c))

    c_val = _c_invbatschelet_numeric(kappa_val, lmbd_val, grid_size=_INVBAT_NUMERIC_GRID)
    if not np.isfinite(c_val) or c_val <= 0.0:
        c_val = _c_invbatschelet_numeric(kappa_val, lmbd_val, grid_size=_INVBAT_NUMERIC_GRID * 2)
    return c_val


def _log_invbatschelet_kernel_integral(kappa, lmbd, grid_size):
    phi = np.linspace(-np.pi, np.pi, grid_size + 1, dtype=float)
    log_kernel = kappa * np.cos(phi - 0.5 * (1.0 - lmbd) * np.sin(phi))
    max_log = np.max(log_kernel)
    weights = np.ones_like(phi)
    weights[0] = weights[-1] = 0.5
    log_sum = logsumexp(log_kernel - max_log, b=weights)
    return np.log(2.0 * np.pi / grid_size) + max_log + log_sum


def _c_invbatschelet_numeric(kappa, lmbd, *, grid_size):
    log_mult = np.log(2.0 * np.pi) + np.log(i0e(kappa)) + kappa
    log_int = _log_invbatschelet_kernel_integral(kappa, lmbd, grid_size)

    if np.isclose(lmbd, -1.0, atol=_INVBAT_LMBDA_TOL):
        return float(np.exp(-log_int))

    log_term1 = np.log1p(lmbd) + log_mult
    if abs(lmbd) <= _INVBAT_LMBDA_TOL:
        log_term2 = -np.inf
    else:
        log_term2 = np.log(2.0 * abs(lmbd)) + log_int

    max_log = max(log_term1, log_term2)
    term1 = np.exp(log_term1 - max_log)
    term2 = np.exp(log_term2 - max_log) if log_term2 > -np.inf else 0.0

    if lmbd >= 0.0:
        denom_scaled = term1 - term2
    else:
        denom_scaled = term1 + term2

    if denom_scaled <= 0.0 or not np.isfinite(denom_scaled):
        return float("nan")

    log_denom = max_log + np.log(denom_scaled)
    log_num = np.log1p(-lmbd)
    return float(np.exp(log_num - log_denom))


def _invbat_log_c_array(kappa, lmbd, *, grid_size):
    """Vectorized ``log c(κ,λ)`` for arrays of (κ,λ) on one shared grid — a
    faithful, overflow-safe vectorization of :func:`_c_invbatschelet_numeric`
    over the interior region (κ>0, |λ|<1) that the log/tanh links guarantee.
    Edge pairs (κ≈0, |λ|≈1, or any non-finite result) fall back to the scalar
    :func:`_c_invbatschelet` for exact-limit parity. Lets the normalizer
    gradient/Hessian be finite-differenced for *all* unique pairs at once
    instead of looping the scalar normalizer per pair (the κ(x)/λ(x) hotspot).
    """
    shape = np.broadcast(kappa, lmbd).shape
    k_b, l_b = np.broadcast_arrays(
        np.asarray(kappa, dtype=float), np.asarray(lmbd, dtype=float)
    )
    kf = np.clip(k_b.ravel(), 0.0, _INVBAT_KAPPA_UPPER)
    lf = l_b.ravel().astype(float, copy=True)

    # log J = log ∫ e^{κ cos B(u)} du,  B(u) = u − ½(1−λ) sin u  (max-subtracted)
    phi = np.linspace(-np.pi, np.pi, grid_size + 1)
    sin_phi = np.sin(phi)
    weights = np.ones_like(phi)
    weights[0] = weights[-1] = 0.5
    B = phi[None, :] - 0.5 * (1.0 - lf[:, None]) * sin_phi[None, :]
    log_kernel = kf[:, None] * np.cos(B)
    max_log = np.max(log_kernel, axis=1)
    # log ∫ = log(2π/G) + max + log Σ_j w_j·exp(logkernel_j − max). The max
    # subtraction already bounds the exp (≤1, and the max row contributes ≥0.5),
    # so a plain exp → trapezoid (gemv with the weights) → log is overflow-safe
    # and skips scipy.logsumexp's array-API dispatch overhead — the hot path.
    weighted_sum = np.exp(log_kernel - max_log[:, None]) @ weights
    log_int = np.log(2.0 * np.pi / grid_size) + max_log + np.log(weighted_sum)

    log_mult = np.log(2.0 * np.pi) + np.log(i0e(kf)) + kf  # = log(2π I0(κ))

    # D = (1+λ)·2πI0 − 2λ·J, assembled overflow-safe (mirrors the scalar split)
    log_t1 = np.log1p(lf) + log_mult
    with np.errstate(divide="ignore"):
        log_t2 = np.log(2.0 * np.abs(lf)) + log_int  # −inf at λ=0 → term 0
    m = np.maximum(log_t1, log_t2)
    t1 = np.exp(log_t1 - m)
    t2 = np.where(np.isfinite(log_t2), np.exp(log_t2 - m), 0.0)
    denom_scaled = np.where(lf >= 0.0, t1 - t2, t1 + t2)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_c = np.log1p(-lf) - (m + np.log(denom_scaled))

    # Exact-limit edges (rare under log/tanh links): defer to the scalar form.
    edge = (
        (kf <= _INVBAT_KAPPA_TOL)
        | (np.abs(np.abs(lf) - 1.0) <= _INVBAT_LMBDA_TOL)
        | ~np.isfinite(log_c)
        | (denom_scaled <= 0.0)
    )
    if np.any(edge):
        for idx in np.nonzero(edge)[0]:
            log_c[idx] = np.log(_c_invbatschelet(float(kf[idx]), float(lf[idx])))

    return log_c.reshape(shape)


def _invbat_ensure_scalar(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(()))
    unique = np.unique(arr)
    if unique.size == 1:
        return float(unique[0])
    raise ValueError(
        f"Inverse Batschelet parameter '{name}' must be scalar; "
        "vectorised parameters are not supported because numeric grids are cached per parameter."
    )


def _invbat_grid_size(kappa, nu, lmbd):
    sharpness = (1.0 + 0.75 * abs(lmbd)) * (1.0 + abs(nu)) * np.sqrt(kappa + 1.0)
    target = 64.0 + 12.0 * sharpness
    target = float(np.clip(target, _INVBAT_MIN_GRID, _INVBAT_MAX_GRID))
    power = int(np.ceil(np.log2(target)))
    size = 1 << power
    size = int(np.clip(size, _INVBAT_MIN_GRID, _INVBAT_MAX_GRID))
    if size % 2 != 0:
        size += 1
    return size


class wrapstable_gen(CircularContinuous):
    r"""Wrapped Stable Distribution

    ![wrapstable](../images/circ-mod-wrapstable.png)

    The wrapped stable family results from wrapping a linear stable law onto
    ``[0, 2π)``. Its trigonometric moments satisfy

    $$
    \mathbb{E}\big[e^{ip\Theta}\big] = \rho_p e^{i\mu_p}, \qquad
    \rho_p = \exp\left[-(\gamma p)^\alpha\right],
    $$

    with

    $$
    \mu_p =
    \begin{cases}
        \delta p + \beta \tan\left(\tfrac{\pi\alpha}{2}\right)\bigl((\gamma p)^\alpha - \gamma p\bigr), & \alpha \ne 1, \\[6pt]
        \delta p - \tfrac{2}{\pi}\beta\gamma p \log(\gamma p), & \alpha = 1,
    \end{cases}
    $$

    the S0 (Nolan) parameterization of Pewsey (2008), jointly continuous in
    all four parameters.

    Special cases include the wrapped normal (``α=2, β=0``), wrapped Cauchy
    (``α=1, β=0``), and wrapped Lévy (``α=1/2, β=1``).

    Methods
    -------
    pdf(x, delta, alpha, beta, gamma)
        Probability density function via adaptive Fourier series.

    logpdf(x, delta, alpha, beta, gamma)
        Logarithm of the probability density function (series, floored at
        float-tiny — the series is its own accuracy floor).

    cdf(x, delta, alpha, beta, gamma)
        Analytic cumulative distribution function using integrated series.

    ppf(q, delta, alpha, beta, gamma)
        Quantile function obtained by safeguarded Newton refinement.

    rvs(delta, alpha, beta, gamma, size=None, random_state=None)
        Random variates by Chambers–Mallows–Stuck sampling and wrapping.

    fit(data, *, method='mle' | 'moments', ...)
        Estimate parameters via moment starts with optional MLE refinement.

    References
    ----------
    - Pewsey (2008). *Computational Statistics & Data Analysis* 52(3), 1516-1523.

    Parameters must be scalar; Fourier series coefficients are cached per
    parameter set.

    **Small-α boundary (performance over diagnostics, by design).** As
    α → 0 the law approaches an atom-plus-uniform mixture
    (ρ_p = e^{−(γp)^α} → e^{−1} for *every* harmonic) which has no
    density, and the number of terms a faithful density series would
    need grows like (−ln ε)^{1/α}/γ. The series build is therefore
    capped at 20 000 terms (`_WRAPSTABLE_MAX_TERMS`) and the cap engages
    *silently*: below the cap-onset α (≈ 0.38 at γ = 0.7; earlier for
    smaller γ) `pdf` degrades to a truncated-kernel artifact —
    oscillatory, with negative side-lobes floored by `logpdf`'s tiny
    clamp — while `cdf`/`ppf` inherit only the milder 1/p-damped
    truncation error and stay monotone, and `trig_moment` is exact at
    any α (closed characteristic-function form, no truncation). No
    warning is raised and α is not floored: the cap is the performance
    guard, and callers needing densities in the deep-α regime should
    treat α ≲ 0.4 as out of the series' faithful range.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._series_cache = {}

    def _clear_normalization_cache(self):
        super()._clear_normalization_cache()
        self._series_cache = {}

    def _argcheck(self, delta, alpha, beta, gamma):
        try:
            delta_arr, alpha_arr, beta_arr, gamma_arr = np.broadcast_arrays(delta, alpha, beta, gamma)
        except ValueError:
            return False
        return (
            (delta_arr >= 0.0)
            & (delta_arr <= 2.0 * np.pi)
            & (alpha_arr > 0.0)
            & (alpha_arr <= 2.0)
            & (beta_arr >= -1.0)
            & (beta_arr <= 1.0)
            & (gamma_arr > 0.0)
        )

    def _pdf(self, x, delta, alpha, beta, gamma):
        x_arr = np.asarray(x, dtype=float)
        rho_vals, mu_vals, p = self._get_series_terms(delta, alpha, beta, gamma)
        cos_args = p[:, np.newaxis] * x_arr[np.newaxis, ...] - mu_vals[:, np.newaxis]
        series_sum = np.sum(rho_vals[:, np.newaxis] * np.cos(cos_args), axis=0)
        pdf_values = 1 / (2 * np.pi) * (1 + 2 * series_sum)
        if np.isscalar(x):
            return np.asarray(pdf_values, dtype=float).reshape(-1)[0]
        return pdf_values

    def pdf(self, x, delta, alpha, beta, gamma, *args, **kwargs):
        r"""
        Probability density function of the Wrapped Stable distribution.

        $$
        f(\theta) = \frac{1}{2\pi} \left[1 + 2 \sum_{p=1}^{\infty} \rho_p \cos\left(p(\theta - \mu_p)\right)\right]
        $$

        , where $\rho_p$ is the $p$th mean resultant length and $\mu_p$ is the $p$th mean direction:

        $$
        \rho_p = \exp\left(-(\gamma p)^\alpha\right)
        $$

        $$
        \mu_p =
        \begin{cases}
            \delta p + \beta \tan\left(\frac{\pi \alpha}{2}\right) \left((\gamma p)^\alpha - \gamma p\right), & \alpha \neq 1 \\
            \delta p - \frac{2}{\pi} \beta \gamma p \log(\gamma p), & \text{if } \alpha = 1
        \end{cases}
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF, defined on the interval $[0, 2\pi)$.
        delta : float
            Location parameter, $0 \leq \delta \leq 2\pi$. This is the mean direction of the distribution.
        alpha : float
            Stability parameter, $0 < \alpha \leq 2$. Higher values indicate heavier tails.
        beta : float
            Skewness parameter, $-1 < \beta < 1$. Controls the asymmetry of the distribution.
        gamma : float
            Scale parameter, $\gamma > 0$. Scales the distribution.

        Returns
        -------
        pdf_values : array_like
            Values of the probability density function at the specified points.
        """
        return super().pdf(x, delta, alpha, beta, gamma, *args, **kwargs)

    def _logpdf(self, x, delta, alpha, beta, gamma):
        # log of the Fourier-series density, floored at float-tiny
        # the series is a truncated trig
        # polynomial whose ringing noise is the density's own accuracy
        # floor, so values below ~2.2e-308 are not meaningful — the floor
        # keeps the log finite (≈ −708) instead of nan on a noise-negative
        # cell. Tails above that floor are exact.
        pdf_vals = self._pdf(x, delta, alpha, beta, gamma)
        return np.log(np.clip(pdf_vals, np.finfo(float).tiny, None))

    def logpdf(self, x, delta, alpha, beta, gamma, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the Wrapped Stable
        distribution: the log of the Fourier-series density, floored at the
        smallest normal double. The series is the density's own accuracy
        floor (a truncated trigonometric polynomial), so log-density values
        below ≈ −708 are reported as the floor rather than as noise.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-density.
        delta : float
            Location parameter, 0 <= delta <= 2*pi.
        alpha : float
            Stability parameter, 0 < alpha <= 2.
        beta : float
            Skewness parameter, -1 < beta < 1.
        gamma : float
            Scale parameter, gamma > 0.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, delta, alpha, beta, gamma, *args, **kwargs)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        """Closed-form trigonometric moment (Pewsey 2008 eqs. 3–4, the S0
        characteristic-function terms the series itself is built from):
        m_p = ρ_p·e^{iμ_p} with ρ_p = e^{−(γp)^α} and the α = 1 phase
        carrying the minus sign (validation bug #3). Exact for every p —
        unlike the density series, no truncation cap is involved."""
        shape_args, non_shape_kwargs = self._separate_shape_parameters(
            args, kwargs, "trig_moment"
        )
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        delta, alpha, beta, gamma = (
            float(np.asarray(v, dtype=float))
            for v in self._parse_args(*shape_args, **call_kwargs)[0]
        )

        if not np.isscalar(p):
            raise ValueError("`p` must be an integer scalar.")
        if int(round(p)) != p:
            raise ValueError("`p` must be an integer.")
        k = int(round(p))
        if k == 0:
            return complex(1.0, 0.0)
        ak = float(abs(k))
        rho_p = np.exp(-((gamma * ak) ** alpha))
        if abs(alpha - 1.0) <= _WRAPSTABLE_ALPHA_TOL:
            mu_p = delta * ak - (2.0 / np.pi) * beta * gamma * ak * np.log(gamma * ak)
        else:
            mu_p = delta * ak + beta * np.tan(0.5 * np.pi * alpha) * (
                (gamma * ak) ** alpha - gamma * ak
            )
        value = rho_p * np.exp(1j * mu_p)
        return complex(np.conjugate(value)) if k < 0 else complex(value)

    def _cdf(self, x, delta, alpha, beta, gamma):
        x_arr = np.asarray(x, dtype=float)
        scalar_input = x_arr.ndim == 0
        theta = np.atleast_1d(x_arr)

        rho_vals, mu_vals, p = self._get_series_terms(delta, alpha, beta, gamma)
        theta_flat = theta.reshape(1, -1)
        p_vals = p.astype(float)

        sin_args = p_vals[:, np.newaxis] * theta_flat - mu_vals[:, np.newaxis]
        coeffs = (rho_vals / p_vals)[:, np.newaxis]
        series_sum = np.sum(coeffs * np.sin(sin_args), axis=0)
        cdf_vals = (theta_flat[0] / (2.0 * np.pi)) + (1.0 / np.pi) * series_sum

        anchor = (1.0 / np.pi) * np.sum((rho_vals / p_vals) * np.sin(-mu_vals))
        # The anchored difference raw(θ) − raw(0) of one increasing function
        # is ≥ 0 for every θ ∈ [0, 2π]; only float/truncation dust can go
        # negative, so clip — never wrap by +1 (that would turn −1e-17 into
        # ≈ 1).
        cdf_vals = np.clip(cdf_vals - anchor, 0.0, 1.0)

        # Exact-endpoint pins only: an isclose() here (whose default rtol
        # survives an atol override) would swallow honest tail values up to
        # ~6e-5 away from 2π.
        two_pi = 2.0 * np.pi
        cdf_vals[theta_flat[0] == 0.0] = 0.0
        cdf_vals[theta_flat[0] == two_pi] = 1.0

        if scalar_input:
            return float(cdf_vals.reshape(-1)[0])
        return cdf_vals.reshape(x_arr.shape)

    def cdf(self, x, delta, alpha, beta, gamma, *args, **kwargs):
        r"""
        Cumulative distribution function of the Wrapped Stable distribution.

        The characteristic-function series integrates term by term, giving
        the analytic form

        $$
        F(\theta) = \frac{\theta}{2\pi} + \frac{1}{\pi}\sum_{p\ge 1}
        \frac{\rho_p}{p}\,\Bigl[\sin(p\theta - \mu_p) - \sin(-\mu_p)\Bigr],
        \qquad \rho_p = e^{-(\gamma p)^\alpha},
        $$

        truncated once the exact tail bound drops below tolerance — no
        quadrature is involved.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the CDF.
        delta : float
            Location (mean-direction) parameter, ``0 <= delta <= 2*pi``.
        alpha : float
            Stability index, ``0 < alpha <= 2``.
        beta : float
            Skewness parameter, ``-1 < beta < 1``.
        gamma : float
            Scale parameter, ``gamma > 0``.

        Returns
        -------
        cdf_values : array_like
            CDF evaluated at `x`.
        """
        return super().cdf(x, delta, alpha, beta, gamma, *args, **kwargs)

    def _ppf(self, q, delta, alpha, beta, gamma):
        q_arr = np.asarray(q, dtype=float)
        flat = q_arr.reshape(-1)
        if flat.size == 0:
            return q_arr.astype(float)

        delta_val = self._scalar_param(delta)
        alpha_val = self._scalar_param(alpha)
        beta_val = self._scalar_param(beta)
        gamma_val = self._scalar_param(gamma)

        result = np.full_like(flat, np.nan, dtype=float)
        valid = np.isfinite(flat) & (flat >= 0.0) & (flat <= 1.0)
        if not np.any(valid):
            shaped = result.reshape(q_arr.shape)
            return float(shaped) if q_arr.ndim == 0 else shaped

        two_pi = 2.0 * np.pi
        close_zero = valid & np.isclose(flat, 0.0, atol=1e-12, rtol=0.0)
        close_one = valid & np.isclose(flat, 1.0, atol=1e-12, rtol=0.0)
        result[close_zero] = 0.0
        result[close_one] = two_pi

        interior = valid & ~(close_zero | close_one)
        if not np.any(interior):
            shaped = result.reshape(q_arr.shape)
            return float(shaped) if q_arr.ndim == 0 else shaped

        q_sub = flat[interior]

        def cdf_fn(t):
            return np.asarray(
                self._cdf(t, delta_val, alpha_val, beta_val, gamma_val),
                dtype=float,
            )

        def pdf_fn(t):
            return np.asarray(
                self._pdf(t, delta_val, alpha_val, beta_val, gamma_val),
                dtype=float,
            )

        # Bracket every quantile on a coarse grid of the series cdf, then
        # polish with one shared bracket-safeguarded Newton, iterating only
        # the unconverged subset (the former per-quantile scalar loop paid
        # the full series matrix per q per iteration).
        grid = np.linspace(0.0, two_pi, 33)
        f_grid = cdf_fn(grid)
        idx = np.clip(np.sum(f_grid[:, None] <= q_sub[None, :], axis=0), 1, 32)
        lower = grid[idx - 1]
        upper = grid[idx]
        f_lo = f_grid[idx - 1]
        f_hi = f_grid[idx]
        span = np.clip(f_hi - f_lo, 1e-300, None)
        theta = lower + (upper - lower) * np.clip((q_sub - f_lo) / span, 0.0, 1.0)

        residual = cdf_fn(theta) - q_sub
        lower = np.where(residual <= 0.0, theta, lower)
        upper = np.where(residual > 0.0, theta, upper)
        act = np.flatnonzero(np.abs(residual) > _WRAPSTABLE_NEWTON_TOL)
        for _ in range(_WRAPSTABLE_NEWTON_MAXITER):
            if not act.size:
                break
            th_a = theta[act]
            lo_a = lower[act]
            hi_a = upper[act]
            p_a = pdf_fn(th_a)
            step = residual[act] / np.clip(p_a, np.finfo(float).tiny, None)
            th_n = th_a - step
            bad = (
                ~np.isfinite(th_n)
                | (th_n <= lo_a)
                | (th_n >= hi_a)
                | (p_a <= 0.0)
                | ~np.isfinite(p_a)
            )
            th_n = np.where(bad, 0.5 * (lo_a + hi_a), th_n)
            r_n = cdf_fn(th_n) - q_sub[act]
            theta[act] = th_n
            residual[act] = r_n
            lower[act] = np.where(r_n <= 0.0, th_n, lo_a)
            upper[act] = np.where(r_n > 0.0, th_n, hi_a)
            act = act[np.abs(r_n) > _WRAPSTABLE_NEWTON_TOL]

        # Residual-converged cells still owe the width certificate
        # (|F − q| ≤ tol alone cannot pin θ in a flat stretch). One probe
        # pass settles all sharp cells with two cdf evals — a ±w bracket is
        # confirmed wherever the cdf visibly crosses q inside it — instead
        # of the ~30 blanket bisections the per-quantile loop paid.
        need = np.flatnonzero(upper - lower > _WRAPSTABLE_NEWTON_WIDTH_TOL)
        if need.size:
            w = 0.4 * _WRAPSTABLE_NEWTON_WIDTH_TOL
            lo_p = np.maximum(theta[need] - w, lower[need])
            hi_p = np.minimum(theta[need] + w, upper[need])
            below = cdf_fn(lo_p) - q_sub[need] <= 0.0
            above = cdf_fn(hi_p) - q_sub[need] > 0.0
            lower[need] = np.where(below, lo_p, lower[need])
            upper[need] = np.where(above, hi_p, upper[need])

        # flat-zone stragglers: bisect the bracket to the width tolerance
        rem = np.flatnonzero(upper - lower > _WRAPSTABLE_NEWTON_WIDTH_TOL)
        if rem.size:
            lo_u = lower[rem]
            hi_u = upper[rem]
            q_u = q_sub[rem]
            for _ in range(40):
                if np.all(hi_u - lo_u <= _WRAPSTABLE_NEWTON_WIDTH_TOL):
                    break
                mid = 0.5 * (lo_u + hi_u)
                go_up = cdf_fn(mid) <= q_u
                lo_u = np.where(go_up, mid, lo_u)
                hi_u = np.where(go_up, hi_u, mid)
            theta[rem] = 0.5 * (lo_u + hi_u)
            lower[rem] = lo_u
            upper[rem] = hi_u

        result[interior] = (theta + two_pi) % two_pi
        shaped = result.reshape(q_arr.shape)
        if q_arr.ndim == 0:
            return float(shaped)
        return shaped

    def ppf(self, q, delta, alpha, beta, gamma, *args, **kwargs):
        """
        Percent-point function (inverse CDF) of the Wrapped Stable
        distribution.

        Quantiles invert the analytic series CDF with a vectorized
        bracket-safeguarded Newton iteration (series PDF as the slope) and
        a bracket-width certificate, so ``ppf`` stays in exact sync with
        ``cdf``.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate (values in ``[0, 1]``).
        delta : float
            Location (mean-direction) parameter, ``0 <= delta <= 2*pi``.
        alpha : float
            Stability index, ``0 < alpha <= 2``.
        beta : float
            Skewness parameter, ``-1 < beta < 1``.
        gamma : float
            Scale parameter, ``gamma > 0``.

        Returns
        -------
        ppf_values : array_like
            Angles in ``[0, 2π)`` such that ``cdf(angle) = q``.
        """
        return super().ppf(q, delta, alpha, beta, gamma, *args, **kwargs)

    def _rvs(self, delta, alpha, beta, gamma, size=None, random_state=None):
        rng = self._init_rng(random_state)

        delta_val = self._scalar_param(delta)
        alpha_val = self._scalar_param(alpha)
        beta_val = self._scalar_param(beta)
        gamma_val = self._scalar_param(gamma)

        if not (0.0 < alpha_val <= 2.0):
            raise ValueError("`alpha` must lie in (0, 2].")
        if not (-1.0 <= beta_val <= 1.0):
            raise ValueError("`beta` must lie in [-1, 1].")
        if not (gamma_val > 0.0):
            raise ValueError("`gamma` must be positive.")

        if size is None:
            shape = ()
            total = 1
        else:
            if np.isscalar(size):
                shape = (int(size),)
            else:
                shape = tuple(int(dim) for dim in np.atleast_1d(size))
            total = int(np.prod(shape, dtype=int))
            if total < 0:
                raise ValueError("`size` must describe a non-negative number of samples.")

        if total == 0:
            empty = np.empty(shape, dtype=float)
            return float(empty) if empty.ndim == 0 else empty

        linear_samples = _wrapstable_sample_linear(
            alpha=alpha_val,
            beta=beta_val,
            gamma=gamma_val,
            delta=delta_val,
            size=total,
            rng=rng,
        )

        samples = np.mod(linear_samples, 2.0 * np.pi).reshape(shape)
        if samples.ndim == 0:
            return float(samples)
        return samples

    def rvs(self, delta=None, alpha=None, beta=None, gamma=None, size=None, random_state=None):
        r"""Draw random variates from the wrapped stable distribution."""

        delta_val = self._scalar_param(delta)
        alpha_val = self._scalar_param(alpha)
        beta_val = self._scalar_param(beta)
        gamma_val = self._scalar_param(gamma)
        return super().rvs(delta_val, alpha_val, beta_val, gamma_val, size=size, random_state=random_state)

    def fit(
        self,
        data,
        *,
        weights=None,
        method="mle",
        optimizer="L-BFGS-B",
        options=None,
        alpha_bounds=(1e-3, 2.0),
        beta_bounds=(-0.99, 0.99),
        gamma_bounds=(1e-6, 10.0),
        return_info=False,
        **minimize_kwargs,
    ):
        r"""Estimate ``(delta, alpha, beta, gamma)`` from circular data."""

        minimize_kwargs = self._sanitize_fit_kwargs(minimize_kwargs)
        minimize_kwargs.pop("floc", None)
        minimize_kwargs.pop("fscale", None)

        data_arr = self._wrap_angles(np.asarray(data, dtype=float)).ravel()
        if data_arr.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(data_arr, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")
            w = np.broadcast_to(w, data_arr.shape).astype(float, copy=False).ravel()

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = float(w_sum**2 / np.sum(w**2))

        def weighted_moment(p):
            return np.sum(w * np.exp(1j * p * data_arr)) / w_sum

        m1 = weighted_moment(1)
        m2 = weighted_moment(2)

        r1 = float(np.clip(abs(m1), 1e-9, 1 - 1e-9))
        r2 = float(np.clip(abs(m2), 1e-9, 1 - 1e-9))

        if r1 >= 1 - 1e-6 or r2 >= 1 - 1e-6:
            alpha_mom = 1.0
            gamma_mom = 1e-3
        else:
            y1 = float(np.log(-np.log(r1)))
            y2 = float(np.log(-np.log(r2)))
            slope = (y2 - y1) / np.log(2.0)
            alpha_mom = float(np.clip(slope, alpha_bounds[0], alpha_bounds[1]))
            gamma_mom = float(np.exp(y1 / alpha_mom))
            gamma_mom = float(np.clip(gamma_mom, gamma_bounds[0], gamma_bounds[1]))

        phi1 = float(np.angle(m1))
        phi2_raw = float(np.angle(m2))
        phi2 = phi2_raw + 2.0 * np.pi * round((2.0 * phi1 - phi2_raw) / (2.0 * np.pi))

        if abs(alpha_mom - 1.0) <= _WRAPSTABLE_ALPHA_TOL:
            # mu_p = delta*p - beta*B_p with B_p = (2/pi)*gamma*p*log(gamma*p)
            # (S0, alpha = 1), so phi2 - 2*phi1 = -beta*(B2 - 2*B1).
            B1 = (2.0 / np.pi) * gamma_mom * np.log(gamma_mom)
            B2 = (2.0 / np.pi) * (gamma_mom * 2.0) * np.log(gamma_mom * 2.0)
            denom = B2 - 2.0 * B1
            if abs(denom) < 1e-8:
                beta_mom = 0.0
                delta_mom = phi1
            else:
                beta_mom = (2.0 * phi1 - phi2) / denom
                beta_mom = float(np.clip(beta_mom, beta_bounds[0], beta_bounds[1]))
                delta_mom = phi1 + beta_mom * B1
        else:
            A = np.tan(0.5 * np.pi * alpha_mom)
            B1 = (gamma_mom) ** alpha_mom - gamma_mom
            B2 = (gamma_mom * 2.0) ** alpha_mom - gamma_mom * 2.0
            denom = A * (B2 - 2.0 * B1)
            if abs(denom) < 1e-8:
                beta_mom = 0.0
                delta_mom = phi1
            else:
                beta_mom = (phi2 - 2.0 * phi1) / denom
                beta_mom = float(np.clip(beta_mom, beta_bounds[0], beta_bounds[1]))
                delta_mom = phi1 - beta_mom * A * B1

        delta_mom = float(np.mod(delta_mom, 2.0 * np.pi))

        if method == "moments":
            estimates = (delta_mom, alpha_mom, beta_mom, gamma_mom)
            if return_info:
                info = {
                    "method": "moments",
                    "converged": True,
                    "n_effective": n_eff,
                }
                return estimates, info
            return estimates

        method_key = str(method).lower()
        if method_key != "mle":
            raise ValueError("`method` must be one of {'mle', 'moments' }.")

        def nll(params):
            delta_param, alpha_param, beta_param, gamma_param = params
            if not (0.0 <= delta_param <= 2.0 * np.pi):
                return np.inf
            if not (alpha_bounds[0] <= alpha_param <= alpha_bounds[1]):
                return np.inf
            if not (beta_bounds[0] <= beta_param <= beta_bounds[1]):
                return np.inf
            if not (gamma_bounds[0] <= gamma_param <= gamma_bounds[1]):
                return np.inf

            pdf_vals = self._pdf(data_arr, delta_param, alpha_param, beta_param, gamma_param)
            if np.any(pdf_vals <= 0.0) or not np.all(np.isfinite(pdf_vals)):
                return np.inf
            return float(-np.sum(w * np.log(pdf_vals)))

        delta_candidates = np.mod(
            np.array([delta_mom, phi1, phi2 / 2.0]), 2.0 * np.pi
        )
        alpha_candidates = np.clip(
            np.array([alpha_mom, 1.0, min(1.9, alpha_mom * 1.2)]), alpha_bounds[0], alpha_bounds[1]
        )
        beta_candidates = np.clip(
            np.array([beta_mom, 0.0, np.sign(beta_mom) * 0.5]), beta_bounds[0], beta_bounds[1]
        )
        gamma_candidates = np.clip(
            np.array([gamma_mom, max(gamma_bounds[0], gamma_mom * 0.8), min(gamma_bounds[1], gamma_mom * 1.2)]),
            gamma_bounds[0],
            gamma_bounds[1],
        )

        best_params = (delta_mom, alpha_mom, beta_mom, gamma_mom)
        best_score = nll(best_params)
        for d0 in delta_candidates:
            for a0 in alpha_candidates:
                for b0 in beta_candidates:
                    for g0 in gamma_candidates:
                        cand = (float(d0), float(a0), float(b0), float(g0))
                        score = nll(cand)
                        if score < best_score:
                            best_score = score
                            best_params = cand

        bounds = [
            (0.0, 2.0 * np.pi),
            tuple(alpha_bounds),
            tuple(beta_bounds),
            tuple(gamma_bounds),
        ]

        init = np.array(best_params, dtype=float)
        options = {} if options is None else dict(options)

        optimizer_used = optimizer
        result = minimize(
            nll,
            init,
            method=optimizer,
            bounds=bounds,
            options=options,
            **minimize_kwargs,
        )

        if not result.success and optimizer != "Powell":
            fallback = minimize(
                nll,
                init,
                method="Powell",
                bounds=bounds,
                options={},
                **minimize_kwargs,
            )
            if fallback.success:
                result = fallback
                optimizer_used = "Powell"

        if not result.success:
            raise RuntimeError(f"Maximum likelihood fit failed: {result.message}")

        delta_hat = float(np.mod(result.x[0], 2.0 * np.pi))
        alpha_hat = float(np.clip(result.x[1], alpha_bounds[0], alpha_bounds[1]))
        beta_hat = float(np.clip(result.x[2], beta_bounds[0], beta_bounds[1]))
        gamma_hat = float(np.clip(result.x[3], gamma_bounds[0], gamma_bounds[1]))

        estimates = (delta_hat, alpha_hat, beta_hat, gamma_hat)
        if not return_info:
            return estimates

        info = {
            "method": "mle",
            "loglik": float(-result.fun),
            "n_effective": n_eff,
            "converged": bool(result.success),
            "optimizer": optimizer_used,
            "nit": getattr(result, "nit", np.nan),
            "nfev": getattr(result, "nfev", np.nan),
            "message": result.message,
        }
        return estimates, info

    def _get_series_terms(self, delta, alpha, beta, gamma):
        delta_s = self._scalar_param(delta)
        alpha_s = self._scalar_param(alpha)
        beta_s = self._scalar_param(beta)
        gamma_s = self._scalar_param(gamma)
        key = self._normalization_cache_key(delta_s, alpha_s, beta_s, gamma_s)
        if key is None:
            return self._compute_series_terms(delta_s, alpha_s, beta_s, gamma_s)
        cache = self._series_cache
        if key not in cache:
            cache[key] = self._compute_series_terms(delta_s, alpha_s, beta_s, gamma_s)
        return cache[key]

    def _compute_series_terms(self, delta, alpha, beta, gamma):
        if gamma <= 0.0:
            raise ValueError("`gamma` must be positive for wrapstable.")

        def _initial_order(tol):
            if tol <= 0.0:
                return 1
            log_term = -np.log(tol)
            if log_term <= 0.0:
                return 1
            if not np.isfinite(alpha) or alpha <= 0.0:
                return 1

            exponent = (np.log(log_term) / alpha) - np.log(gamma)
            if not np.isfinite(exponent):
                return _WRAPSTABLE_MAX_TERMS
            if exponent > np.log(_WRAPSTABLE_MAX_TERMS):
                return _WRAPSTABLE_MAX_TERMS

            value = np.exp(exponent)
            if not np.isfinite(value):
                return _WRAPSTABLE_MAX_TERMS
            value = max(1.0, value)
            return int(min(_WRAPSTABLE_MAX_TERMS, np.ceil(value)))

        p_pdf = _initial_order(_WRAPSTABLE_PDF_TOL)
        p_cdf = _initial_order(_WRAPSTABLE_CDF_TOL)
        P = max(1, p_pdf, p_cdf)

        for _ in range(_WRAPSTABLE_MAX_TERMS):
            rho_P = np.exp(-((gamma * P) ** alpha))
            if rho_P <= _WRAPSTABLE_PDF_TOL and rho_P / P <= _WRAPSTABLE_CDF_TOL:
                break
            P += 1
            if P >= _WRAPSTABLE_MAX_TERMS:
                break

        p = np.arange(1, P + 1, dtype=float)
        rho_vals = np.exp(-((gamma * p) ** alpha))

        if abs(alpha - 1.0) <= _WRAPSTABLE_ALPHA_TOL:
            # S0 convention (Pewsey 2008, eq. 4): the alpha = 1 phase carries a
            # minus sign — also the alpha -> 1 limit of the branch below.
            mu_vals = delta * p - (2.0 / np.pi) * beta * gamma * p * np.log(gamma * p)
        else:
            mu_vals = delta * p + beta * np.tan(0.5 * np.pi * alpha) * (
                (gamma * p) ** alpha - gamma * p
            )

        return rho_vals, mu_vals, p

    @staticmethod
    def _scalar_param(value):
        arr = np.asarray(value, dtype=float)
        if arr.size == 1:
            return float(np.asarray(arr, dtype=float).reshape(-1)[0])
        first = float(arr.flat[0])
        if not np.allclose(arr, first):
            raise ValueError(
                "wrapstable parameters must be scalar; vectorised parameters are not supported "
                "because Fourier series coefficients are cached per parameter set."
            )
        return first


wrapstable = wrapstable_gen(name="wrapstable")


def _wrapstable_sample_linear(alpha, beta, gamma, delta, *, size, rng):
    size = int(size)
    if size <= 0:
        return np.empty(0, dtype=float)

    alpha = float(alpha)
    beta = float(beta)
    gamma = float(gamma)
    delta = float(delta)

    V = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=size)
    W = rng.exponential(1.0, size=size)

    # CMS draws are standard S1 variates; the series/pdf use the S0(delta)
    # convention (Pewsey 2008, eq. 2), so the location shift below converts
    # the scaled draw to S0 location ``delta`` (Nolan's delta0/delta1 relation).
    if abs(alpha - 1.0) > _WRAPSTABLE_ALPHA_TOL:
        tan_term = np.tan(0.5 * np.pi * alpha)
        theta0 = np.arctan(beta * tan_term) / alpha
        factor = (1.0 + (beta * tan_term) ** 2) ** (1.0 / (2.0 * alpha))

        part1 = np.sin(alpha * (V + theta0)) / (np.cos(V) ** (1.0 / alpha))
        part2 = (np.cos(V - alpha * (V + theta0)) / W) ** ((1.0 - alpha) / alpha)
        # gamma*Z + c is S1 with location c; S0 location delta needs
        # c = delta - beta*gamma*tan(pi*alpha/2).
        x = gamma * factor * part1 * part2 + delta - gamma * beta * tan_term
    else:
        factor = 2.0 / np.pi
        term = (
            (0.5 * np.pi + beta * V) * np.tan(V)
            - beta * np.log((0.5 * np.pi * W * np.cos(V)) / (0.5 * np.pi + beta * V))
        )
        # At alpha = 1 scaling a standard S1 draw already shifts the S1
        # location by -(2/pi)*beta*gamma*log(gamma), which exactly cancels
        # the S0<->S1 conversion: gamma*Z + delta has S0 location delta.
        x = gamma * factor * term + delta

    return x

def _kj_cart_scores(x, mu, gamma, a, b, second=True):
    r"""Derivatives of the Kato–Jones log-density in Cartesian shape
    coordinates ``(a, b) = (ρ cos λ, ρ sin λ)``.

    In these coordinates the log-density is elementary:

    $$\ell = \log N - \log 2\pi,\qquad
      N = 1 + \frac{2\gamma(c - a)}{D},\qquad
      D = 1 + a^2 + b^2 - 2ac - 2bs,$$

    with ``c = cos(θ−μ)``, ``s = sin(θ−μ)``. All partials follow from the
    quotient rule on ``F = (c−a)/D`` (``N = 1 + 2γF``); the only nonzero
    second partials of the numerator/denominator are ``E_{μμ} = −c`` and
    ``D_{aa} = D_{bb} = 2``, ``D_{μa} = −2s``, ``D_{μb} = 2c``,
    ``D_{μμ} = 2(bs + ac)``. ``N`` is floored at a tiny positive value for
    the same reason ``_pdf`` clips: strictly inside the Theorem-1 disc the
    density is positive, but boundary-hugging parameters can round it to 0.

    Returns ``(l1, l2)`` dicts keyed by ``mu``/``gamma``/``a``/``b`` and
    their unordered pairs (``l2 = None`` when ``second=False``).
    """
    x_b, mu_b, g_b, a_b, b_b = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, mu, gamma, a, b))
    )
    delta = x_b - mu_b
    c, s = np.cos(delta), np.sin(delta)
    D = 1.0 + a_b * a_b + b_b * b_b - 2.0 * a_b * c - 2.0 * b_b * s
    D = np.maximum(D, 1e-300)
    E = c - a_b
    F = E / D
    N = np.maximum(1.0 + 2.0 * g_b * F, 1e-300)

    E1 = {"mu": s, "a": -np.ones_like(s), "b": np.zeros_like(s)}
    D1 = {
        "mu": 2.0 * (b_b * c - a_b * s),
        "a": 2.0 * (a_b - c),
        "b": 2.0 * (b_b - s),
    }
    names = ("mu", "a", "b")
    F1 = {k: (E1[k] * D - E * D1[k]) / (D * D) for k in names}

    l1 = {
        "mu": 2.0 * g_b * F1["mu"] / N,
        "gamma": 2.0 * F / N,
        "a": 2.0 * g_b * F1["a"] / N,
        "b": 2.0 * g_b * F1["b"] / N,
    }
    if not second:
        return l1, None

    zero = np.zeros_like(s)
    E2 = {("mu", "mu"): -c}
    D2 = {
        ("mu", "mu"): 2.0 * (b_b * s + a_b * c),
        ("mu", "a"): -2.0 * s,
        ("mu", "b"): 2.0 * c,
        ("a", "a"): 2.0 * np.ones_like(s),
        ("a", "b"): zero,
        ("b", "b"): 2.0 * np.ones_like(s),
    }
    l2 = {}
    for i, xn in enumerate(names):
        for yn in names[i:]:
            Exy = E2.get((xn, yn), zero)
            Dxy = D2[(xn, yn)]
            Fxy = (
                Exy / D
                - (E1[xn] * D1[yn] + E1[yn] * D1[xn] + E * Dxy) / (D * D)
                + 2.0 * E * D1[xn] * D1[yn] / (D * D * D)
            )
            l2[(xn, yn)] = (
                2.0 * g_b * Fxy / N
                - 4.0 * g_b * g_b * F1[xn] * F1[yn] / (N * N)
            )
    l2[("gamma", "gamma")] = -4.0 * F * F / (N * N)
    for xn in names:
        l2[("gamma", xn)] = (
            2.0 * F1[xn] / N - 4.0 * g_b * F * F1[xn] / (N * N)
        )
    return l1, l2


class katojones_gen(_RegressionReady, CircularContinuous):
    """
    Kato--Jones (2015) Distribution

    ![katojones](../images/circ-mod-katojones.png)

    Methods
    -------
    pdf(x, mu, gamma, rho, lam)
        Probability density function.
    logpdf(x, mu, gamma, rho, lam)
        Logarithm of the probability density function.
    cdf(x, mu, gamma, rho, lam)
        Cumulative distribution function via adaptive Fourier series.
    ppf(q, mu, gamma, rho, lam)
        Percent-point function (vectorized bracket-safeguarded Newton on
        the series CDF).

    rvs(mu, gamma, rho, lam, size=None, random_state=None)
        Random variates obtained by inverting the CDF.
    fit(data, method=\"moments\" | \"mle\", ...)
        Method-of-moments or maximum-likelihood parameter estimation.
    Notes
    -----
    Implements the tractable four-parameter unimodal family proposed by Kato and
    Jones (2015). Parameters control the first two trigonometric moments:
    ``mu`` sets the mean direction, ``gamma`` the mean resultant length, and
    ``rho``/``lam`` encode the magnitude/phase of the second-order moment.
    Feasible parameter tuples satisfy ``0 <= mu < 2*pi``, ``0 <= gamma < 1``,
    ``0 <= rho < 1``, ``0 <= lam < 2*pi`` together with the constraint enforced
    in `_argcheck`.

    Special cases include the uniform distribution (``gamma = 0``), the cardioid
    (``rho = 0``) and the wrapped Cauchy (``lambda = 0`` with ``gamma = rho``).

    References
    ----------
    - Kato, S., & Jones, M. C. (2015). *A tractable and interpretable
      four-parameter family of unimodal distributions on the circle*. Biometrika,
      102(1), 181-190.
    """

    # --- regression overlay (read by the regression engine
    # only — descriptive use keeps the book parameterization (mu, gamma, rho,
    # lam)). The regression coordinates are the **disc chart**:
    # the Theorem-1 feasible set for the Cartesian shape pair
    # (a, b) = (ρ cos λ, ρ sin λ) is the closed disc of center (γ, 0) and
    # radius 1−γ, and the chart
    #
    #     (a, b) = (γ, 0) + (1−γ)·u/√(1+‖u‖²),   u ∈ ℝ²,
    #
    # maps an unconstrained u to its open interior — every (μ, γ, u₁, u₂)
    # with γ ∈ (0, 1) is feasible, so univariate links suffice and no
    # constraint ever reaches the optimizer. u = 0 is exactly the wrapped
    # Cauchy WC(μ, γ) (the nesting test point). ``dlogpdf``/``d2logpdf``
    # therefore take (mu, gamma, u1, u2); the ``KatoJonesLL`` family in
    # ``regression.py`` owns the (γ, u) → (ρ, λ) translation for ``logpdf``
    # and ``fit``. ---
    param_roles = {
        "mu": "location",
        "gamma": "concentration",
        "u1": "shape",
        "u2": "shape",
    }
    # size-aware MAP degeneracy guard (reweighted circ_mix M-step; inert
    # otherwise): a ridge on the disc-chart coordinates (u1, u2) toward 0 =
    # wrapped Cauchy, keeping the shape off the feasibility-disc boundary where
    # the kernel denominator -> 0 and the Hessian (~ 1/(1-rho)^6) blows up.
    degen_penalty = (_degen_ridge("u1"), _degen_ridge("u2"))
    default_links = {
        "location": "tanhalf",
        "concentration": "logit",
        "shape": "identity",
    }

    @staticmethod
    def disc_chart(gamma, u1, u2):
        """Map unconstrained chart coordinates ``u`` to the Cartesian shape
        pair ``(a, b)`` strictly inside the Theorem-1 disc."""
        gamma, u1, u2 = (np.asarray(v, dtype=float) for v in (gamma, u1, u2))
        r = np.sqrt(1.0 + u1 * u1 + u2 * u2)
        om = 1.0 - gamma
        return gamma + om * u1 / r, om * u2 / r

    @staticmethod
    def disc_chart_inverse(gamma, rho, lam, *, vmax=1.0 - 1e-9):
        """Closed-form chart inverse: book shape parameters ``(ρ, λ)`` →
        chart coordinates ``(u₁, u₂)``. Boundary or infeasible (ρ, λ) — a
        moments fit can land exactly on the Theorem-1 circle — are first
        pulled radially to ``vmax`` times the disc radius so the inverse
        stays finite."""
        gamma, rho, lam = (np.asarray(v, dtype=float) for v in (gamma, rho, lam))
        om = np.maximum(1.0 - gamma, 1e-12)
        v1 = (rho * np.cos(lam) - gamma) / om
        v2 = rho * np.sin(lam) / om
        n = np.hypot(v1, v2)
        scale = np.where(n > vmax, vmax / np.maximum(n, 1e-300), 1.0)
        v1, v2 = v1 * scale, v2 * scale
        den = np.sqrt(np.maximum(1.0 - (v1 * v1 + v2 * v2), 1e-18))
        return v1 / den, v2 / den

    @staticmethod
    def _chart_pieces(gamma, u1, u2, second):
        """Chart value plus the Jacobian/Hessian pieces the chain rule
        needs: ``(a, b)``, ``∂(a,b)/∂γ``, ``∂(a,b)/∂u_j`` and, for l2, the
        ``∂²(a,b)`` blocks (the chart is linear in γ, so only the mixed
        γ–u and u–u second derivatives survive)."""
        r2 = 1.0 + u1 * u1 + u2 * u2
        r = np.sqrt(r2)
        r3 = r2 * r
        v1, v2 = u1 / r, u2 / r
        om = 1.0 - gamma
        a = gamma + om * v1
        b = om * v2
        J11 = 1.0 / r - u1 * u1 / r3
        J12 = -u1 * u2 / r3
        J22 = 1.0 / r - u2 * u2 / r3
        first = {
            "a": a, "b": b,
            "a_g": 1.0 - v1, "b_g": -v2,
            "a_u": (om * J11, om * J12),
            "b_u": (om * J12, om * J22),
        }
        if not second:
            return first, None
        r5 = r3 * r2
        H1 = (
            -3.0 * u1 / r3 + 3.0 * u1**3 / r5,        # ∂²v₁/∂u₁²
            -u2 / r3 + 3.0 * u1 * u1 * u2 / r5,       # ∂²v₁/∂u₁∂u₂
            -u1 / r3 + 3.0 * u1 * u2 * u2 / r5,       # ∂²v₁/∂u₂²
        )
        H2 = (
            -u2 / r3 + 3.0 * u1 * u1 * u2 / r5,       # ∂²v₂/∂u₁²
            -u1 / r3 + 3.0 * u1 * u2 * u2 / r5,       # ∂²v₂/∂u₁∂u₂
            -3.0 * u2 / r3 + 3.0 * u2**3 / r5,        # ∂²v₂/∂u₂²
        )
        return first, {"J": (J11, J12, J22), "H1": H1, "H2": H2}

    def dlogpdf(self, x, mu, gamma, u1, u2):
        r"""First derivatives of ``logpdf`` w.r.t. the **chart parameters**
        ``(μ, γ, u₁, u₂)`` (l1) — the Cartesian scores of
        ``_kj_cart_scores`` pushed through the disc chart; the γ column
        picks up the chart's own γ-dependence:

        $$\frac{\partial\ell}{\partial\gamma}\Big|_{\text{chart}} =
          \ell_\gamma + \ell_a(1 - v_1) - \ell_b v_2 .$$

        Vectorizes over per-observation parameter arrays; returns a dict
        keyed by the declared parameter names.
        """
        mu_b, g_b, u1_b, u2_b = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (mu, gamma, u1, u2))
        )
        ch, _ = self._chart_pieces(g_b, u1_b, u2_b, second=False)
        l1, _ = _kj_cart_scores(x, mu_b, g_b, ch["a"], ch["b"], second=False)
        la, lb = l1["a"], l1["b"]
        return {
            "mu": l1["mu"],
            "gamma": l1["gamma"] + la * ch["a_g"] + lb * ch["b_g"],
            "u1": la * ch["a_u"][0] + lb * ch["b_u"][0],
            "u2": la * ch["a_u"][1] + lb * ch["b_u"][1],
        }

    def d2logpdf(self, x, mu, gamma, u1, u2):
        r"""Second derivatives of ``logpdf`` w.r.t. the chart parameters
        (l2) — the full second-order chain rule through the disc chart:
        quadratic forms of the Cartesian Hessian in the chart Jacobian,
        plus first-order Cartesian scores times the chart's curvature
        (``∂²(a,b)/∂γ∂u`` and ``∂²(a,b)/∂u∂u``; the chart is linear in γ).
        """
        mu_b, g_b, u1_b, u2_b = np.broadcast_arrays(
            *(np.asarray(v, dtype=float) for v in (mu, gamma, u1, u2))
        )
        ch, ch2 = self._chart_pieces(g_b, u1_b, u2_b, second=True)
        l1, l2 = _kj_cart_scores(x, mu_b, g_b, ch["a"], ch["b"], second=True)
        la, lb = l1["a"], l1["b"]
        ag, bg = ch["a_g"], ch["b_g"]
        au, bu = ch["a_u"], ch["b_u"]
        J11, J12, J22 = ch2["J"]
        Jrow = ((J11, J12), (J12, J22))  # J[k][j] = ∂v_k/∂u_j
        H1, H2 = ch2["H1"], ch2["H2"]
        Hidx = {(0, 0): 0, (0, 1): 1, (1, 1): 2}
        om = 1.0 - g_b

        lmm = l2[("mu", "mu")]
        lma, lmb = l2[("mu", "a")], l2[("mu", "b")]
        lgg = l2[("gamma", "gamma")]
        lgm = l2[("gamma", "mu")]
        lga, lgb = l2[("gamma", "a")], l2[("gamma", "b")]
        laa, lab, lbb = l2[("a", "a")], l2[("a", "b")], l2[("b", "b")]

        out = {
            ("mu", "mu"): lmm,
            ("mu", "gamma"): lgm + lma * ag + lmb * bg,
            ("mu", "u1"): lma * au[0] + lmb * bu[0],
            ("mu", "u2"): lma * au[1] + lmb * bu[1],
            ("gamma", "gamma"): lgg
            + 2.0 * (lga * ag + lgb * bg)
            + laa * ag * ag
            + 2.0 * lab * ag * bg
            + lbb * bg * bg,
        }
        for j, name in enumerate(("u1", "u2")):
            out[("gamma", name)] = (
                lga * au[j]
                + lgb * bu[j]
                + (laa * ag + lab * bg) * au[j]
                + (lab * ag + lbb * bg) * bu[j]
                - la * Jrow[0][j]
                - lb * Jrow[1][j]
            )
        for key, i, j in ((("u1", "u1"), 0, 0), (("u1", "u2"), 0, 1),
                          (("u2", "u2"), 1, 1)):
            k = Hidx[(i, j)]
            out[key] = (
                laa * au[i] * au[j]
                + lab * (au[i] * bu[j] + bu[i] * au[j])
                + lbb * bu[i] * bu[j]
                + la * om * H1[k]
                + lb * om * H2[k]
            )
        return out

    _moment_tolerance = 1e-12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._series_cache = {}

    def _clear_normalization_cache(self):
        super()._clear_normalization_cache()
        self._series_cache = {}

    @staticmethod
    def _scalar_param(value):
        arr = np.asarray(value, dtype=float)
        if arr.size == 1:
            return float(np.asarray(arr, dtype=float).reshape(-1)[0])
        first = float(arr.flat[0])
        if not np.allclose(arr, first):
            raise ValueError(
                "katojones parameters must be scalar; vectorised parameters are not supported "
                "because series expansions are cached per parameter set."
            )
        return first

    def _argcheck(self, mu, gamma, rho, lam):
        try:
            mu_arr, gamma_arr, rho_arr, lam_arr = np.broadcast_arrays(mu, gamma, rho, lam)
        except ValueError:
            return False

        base = (
            (mu_arr >= 0.0)
            & (mu_arr < 2.0 * np.pi)
            & (gamma_arr >= 0.0)
            & (gamma_arr < 1.0)
            & (rho_arr >= 0.0)
            & (rho_arr < 1.0)
            & (lam_arr >= 0.0)
            & (lam_arr < 2.0 * np.pi)
        )

        cos_lam = np.cos(lam_arr)
        sin_lam = np.sin(lam_arr)
        constraint_val = (rho_arr * cos_lam - gamma_arr) ** 2 + (rho_arr * sin_lam) ** 2
        constraint_limit = (1.0 - gamma_arr) ** 2 + 1e-12
        admissible = constraint_val <= constraint_limit
        return base & admissible

    def _pdf(self, x, mu, gamma, rho, lam):
        x_arr = np.asarray(x, dtype=float)
        delta = x_arr - mu
        denom = 1.0 + rho**2 - 2.0 * rho * np.cos(delta - lam)
        denom = np.clip(denom, 1e-15, None)
        numerator = 1.0 + (2.0 * gamma * (np.cos(delta) - rho * np.cos(lam))) / denom
        pdf = numerator / (2.0 * np.pi)
        pdf = np.clip(pdf, 0.0, None)
        if np.isscalar(x):
            return np.asarray(pdf, dtype=float).reshape(-1)[0]
        return pdf

    def pdf(self, x, mu, gamma, rho, lam, *args, **kwargs):
        r"""
        Probability density function of the Kato--Jones (2015) distribution.

        $$
        g(\theta) = \frac{1}{2\pi}\left[1 + \frac{2\gamma\,(\cos(\theta-\mu) - \rho\cos\lambda)}
        {1 + \rho^2 - 2\rho\cos(\theta-\mu-\lambda)}\right]
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        mu : float
            Mean direction, $0 \leq \mu < 2\pi$.
        gamma : float
            Mean resultant length, $0 \leq \gamma < 1$.
        rho : float
            Second-order magnitude, $0 \leq \rho < 1$.
        lam : float
            Second-order phase, $0 \leq \lambda < 2\pi$.

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """
        return super().pdf(x, mu, gamma, rho, lam, *args, **kwargs)

    def _cdf(self, x, mu, gamma, rho, lam):
        x_arr = np.asarray(x, dtype=float)
        scalar_input = x_arr.ndim == 0
        flat = x_arr.reshape(-1)

        mu_val = float(np.mod(self._scalar_param(mu), 2.0 * np.pi))
        gamma_val = float(np.clip(self._scalar_param(gamma), 0.0, 1.0 - 1e-12))
        rho_val = float(np.clip(self._scalar_param(rho), 0.0, 1.0 - 1e-12))
        lam_val = float(np.mod(self._scalar_param(lam), 2.0 * np.pi))

        if gamma_val <= _KJ_GAMMA_TOL:
            cdf_flat = flat / (2.0 * np.pi)
        else:
            series = self._get_series_terms(mu_val, gamma_val, rho_val, lam_val)
            cdf_raw = self._evaluate_cdf_series(flat, mu_val, gamma_val, rho_val, lam_val, series=series)
            # The anchored series G(θ) − G(0) is ≥ 0 and ≤ 1 for every
            # θ ∈ [0, 2π]; only dust strays outside, so clip — a mod(·, 1)
            # here would wrap −1e-17 to ≈ 1 (and 1 + 1e-17 to ≈ 0).
            cdf_flat = cdf_raw

        # exact-endpoint pins only (isclose's default rtol survives an
        # atol override and would swallow honest tail values near 2π)
        cdf_flat = np.clip(cdf_flat, 0.0, 1.0)
        cdf_flat[flat == 0.0] = 0.0
        cdf_flat[flat == 2.0 * np.pi] = 1.0

        if scalar_input:
            return float(cdf_flat[0])
        return cdf_flat.reshape(x_arr.shape)

    def cdf(self, x, mu, gamma, rho, lam, *args, **kwargs):
        r"""
        Cumulative distribution function of the Kato--Jones (2015) distribution.

        The CDF has the closed-form Fourier expansion

        $$
        G(\theta) = \frac{\theta}{2\pi}
        + \frac{1}{\pi}\sum_{p=1}^{\infty} \frac{\gamma \rho^{p-1}}{p}
        \sin\!\bigl(p\theta - [p\mu + (p-1)\lambda]\bigr),
        $$

        which is evaluated adaptively by truncating the series once the tail
        contribution drops below a specified tolerance. No numerical quadrature
        is required.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        mu : float
            Mean direction, $0 \leq \mu < 2\pi$.
        gamma : float
            Mean resultant length, $0 \leq \gamma < 1$.
        rho : float
            Second-order magnitude, $0 \leq \rho < 1$.
        lam : float
            Second-order phase, $0 \leq \lambda < 2\pi$.

        Returns
        -------
        cdf_values : array_like
            Cumulative distribution function evaluated at `x`.
        """
        return super().cdf(x, mu, gamma, rho, lam, *args, **kwargs)

    def _logpdf(self, x, mu, gamma, rho, lam):
        pdf_vals = self._pdf(x, mu, gamma, rho, lam)
        return np.log(np.clip(pdf_vals, np.finfo(float).tiny, None))

    def logpdf(self, x, mu, gamma, rho, lam, *args, **kwargs):
        r"""
        Logarithm of the probability density function of the Kato--Jones (2015)
        distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-PDF.
        mu : float
            Mean direction, $0 \leq \mu < 2\pi$.
        gamma : float
            Mean resultant length, $0 \leq \gamma < 1$.
        rho : float
            Second-order magnitude, $0 \leq \rho < 1$.
        lam : float
            Second-order phase, $0 \leq \lambda < 2\pi$.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, mu, gamma, rho, lam, *args, **kwargs)

    def _ppf(self, q, mu, gamma, rho, lam):
        mu_val = float(np.mod(self._scalar_param(mu), 2.0 * np.pi))
        gamma_val = float(np.clip(self._scalar_param(gamma), 0.0, 1.0 - 1e-12))
        rho_val = float(np.clip(self._scalar_param(rho), 0.0, 1.0 - 1e-12))
        lam_val = float(np.mod(self._scalar_param(lam), 2.0 * np.pi))

        q_arr = np.asarray(q, dtype=float)
        if q_arr.size == 0:
            return q_arr.astype(float)

        if gamma_val <= _KJ_GAMMA_TOL:
            return (2.0 * np.pi * q_arr).astype(float)

        scalar_input = q_arr.ndim == 0
        flat = q_arr.reshape(-1)
        result = np.full_like(flat, np.nan, dtype=float)

        valid = np.isfinite(flat) & (flat >= 0.0) & (flat <= 1.0)
        if not np.any(valid):
            return float(result) if scalar_input else result.reshape(q_arr.shape)

        series = self._get_series_terms(mu_val, gamma_val, rho_val, lam_val)
        two_pi = 2.0 * np.pi

        # rtol=0.0 matters: isclose's default rtol against 1.0 would remap
        # every q ≥ 1 − 1e-5 to the upper endpoint
        close_zero = valid & np.isclose(flat, 0.0, atol=1e-12, rtol=0.0)
        close_one = valid & np.isclose(flat, 1.0, atol=1e-12, rtol=0.0)
        result[close_zero] = 0.0
        result[close_one] = two_pi

        interior = valid & ~(close_zero | close_one)
        if not np.any(interior):
            return float(result[0]) if scalar_input else result.reshape(q_arr.shape)

        q_sub = flat[interior]

        def cdf_fn(t):
            value = self._evaluate_cdf_series(
                t, mu_val, gamma_val, rho_val, lam_val, series=series
            )
            return np.clip(np.asarray(value, dtype=float), 0.0, 1.0)

        def pdf_fn(t):
            return np.asarray(
                self._pdf(t, mu_val, gamma_val, rho_val, lam_val), dtype=float
            )

        # Bracket on a coarse grid of the series cdf, then one shared
        # bracket-safeguarded Newton over the unconverged subset (the former
        # per-quantile scalar loop re-evaluated the series per q per step).
        grid = np.linspace(0.0, two_pi, 33)
        f_grid = cdf_fn(grid)
        idx = np.clip(np.sum(f_grid[:, None] <= q_sub[None, :], axis=0), 1, 32)
        lower = grid[idx - 1]
        upper = grid[idx]
        f_lo = f_grid[idx - 1]
        f_hi = f_grid[idx]
        span = np.clip(f_hi - f_lo, 1e-300, None)
        theta = lower + (upper - lower) * np.clip((q_sub - f_lo) / span, 0.0, 1.0)

        residual = cdf_fn(theta) - q_sub
        lower = np.where(residual <= 0.0, theta, lower)
        upper = np.where(residual > 0.0, theta, upper)
        act = np.flatnonzero(np.abs(residual) > _KJ_NEWTON_TOL)
        for _ in range(_KJ_NEWTON_MAXITER):
            if not act.size:
                break
            th_a = theta[act]
            lo_a = lower[act]
            hi_a = upper[act]
            p_a = pdf_fn(th_a)
            step = residual[act] / np.clip(p_a, np.finfo(float).tiny, None)
            th_n = th_a - step
            bad = (
                ~np.isfinite(th_n)
                | (th_n <= lo_a)
                | (th_n >= hi_a)
                | (p_a <= 0.0)
                | ~np.isfinite(p_a)
            )
            th_n = np.where(bad, 0.5 * (lo_a + hi_a), th_n)
            r_n = cdf_fn(th_n) - q_sub[act]
            theta[act] = th_n
            residual[act] = r_n
            lower[act] = np.where(r_n <= 0.0, th_n, lo_a)
            upper[act] = np.where(r_n > 0.0, th_n, hi_a)
            act = act[np.abs(r_n) > _KJ_NEWTON_TOL]

        # width certificate: one probe pass settles sharp cells with two
        # cdf evals; only flat-zone stragglers fall through to bisection
        need = np.flatnonzero(upper - lower > _KJ_NEWTON_WIDTH_TOL)
        if need.size:
            w = 0.4 * _KJ_NEWTON_WIDTH_TOL
            lo_p = np.maximum(theta[need] - w, lower[need])
            hi_p = np.minimum(theta[need] + w, upper[need])
            below = cdf_fn(lo_p) - q_sub[need] <= 0.0
            above = cdf_fn(hi_p) - q_sub[need] > 0.0
            lower[need] = np.where(below, lo_p, lower[need])
            upper[need] = np.where(above, hi_p, upper[need])

        rem = np.flatnonzero(upper - lower > _KJ_NEWTON_WIDTH_TOL)
        if rem.size:
            lo_u = lower[rem]
            hi_u = upper[rem]
            q_u = q_sub[rem]
            for _ in range(40):
                if np.all(hi_u - lo_u <= _KJ_NEWTON_WIDTH_TOL):
                    break
                mid = 0.5 * (lo_u + hi_u)
                go_up = cdf_fn(mid) <= q_u
                lo_u = np.where(go_up, mid, lo_u)
                hi_u = np.where(go_up, hi_u, mid)
            theta[rem] = 0.5 * (lo_u + hi_u)

        result[interior] = theta % two_pi

        if scalar_input:
            return float(result[0])
        return result.reshape(q_arr.shape)

    def ppf(self, q, mu, gamma, rho, lam, *args, **kwargs):
        """
        Percent-point function (inverse CDF) of the Kato--Jones (2015)
        distribution.

        Quantiles invert the analytic Fourier-series CDF with a vectorized
        bracket-safeguarded Newton iteration (closed-form PDF as the
        slope) and a bracket-width certificate, so ``ppf`` stays in exact
        sync with ``cdf``.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate (values in ``[0, 1]``).
        mu : float
            Mean direction, ``0 <= mu < 2*pi``.
        gamma : float
            Mean resultant length, ``0 <= gamma < 1``.
        rho : float
            Second-order magnitude, ``0 <= rho < 1``.
        lam : float
            Second-order phase, ``0 <= lam < 2*pi``.

        Returns
        -------
        ppf_values : array_like
            Angles in ``[0, 2π)`` such that ``cdf(angle) = q``.
        """
        return super().ppf(q, mu, gamma, rho, lam, *args, **kwargs)

    def _rvs(self, mu, gamma, rho, lam, size=None, random_state=None):
        rng = self._init_rng(random_state)

        if size is None:
            u = rng.random()
            return float(self._ppf(u, mu, gamma, rho, lam))

        if np.isscalar(size):
            shape = (int(size),)
        else:
            shape = tuple(int(dim) for dim in np.atleast_1d(size))

        total = int(np.prod(shape, dtype=int))
        if total < 0:
            raise ValueError("`size` must describe a non-negative number of samples.")
        if total == 0:
            return np.empty(shape, dtype=float)

        u = rng.random(size=shape)
        return self._ppf(u, mu, gamma, rho, lam)

    def rvs(self, mu=None, gamma=None, rho=None, lam=None, size=None, random_state=None):
        mu_val = self._scalar_param(mu)
        gamma_val = self._scalar_param(gamma)
        rho_val = self._scalar_param(rho)
        lam_val = self._scalar_param(lam)
        return super().rvs(mu_val, gamma_val, rho_val, lam_val, size=size, random_state=random_state)

    def trig_moment(self, p: int = 1, *args, **kwargs) -> complex:
        shape_args, non_shape_kwargs = self._separate_shape_parameters(
            args, kwargs, "trig_moment"
        )
        call_kwargs = self._prepare_call_kwargs(non_shape_kwargs, "trig_moment")
        params = self._parse_args(*shape_args, **call_kwargs)[0]
        if len(params) != 4:
            raise ValueError("Expected parameters (mu, gamma, rho, lam).")
        mu, gamma, rho, lam = [float(np.asarray(val, dtype=float)) for val in params]

        if not np.isscalar(p):
            raise ValueError("`p` must be an integer scalar.")
        if int(round(p)) != p:
            raise ValueError("`p` must be an integer.")

        k = int(round(p))
        if k == 0:
            return complex(1.0, 0.0)

        abs_k = abs(k)
        mag = float(gamma) if abs_k == 1 else float(gamma * (rho ** (abs_k - 1)))
        angle = abs_k * mu + (abs_k - 1) * lam
        value = mag * np.exp(1j * angle)

        if k < 0:
            return np.conjugate(value)
        return complex(value)

    def _prepare_data_weights(self, data, weights=None):
        data_arr = self._wrap_angles(np.asarray(data, dtype=float)).ravel()
        if data_arr.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if weights is None:
            w = np.ones_like(data_arr, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            try:
                w = np.broadcast_to(w, data_arr.shape).astype(float, copy=False).ravel()
            except ValueError as exc:
                raise ValueError("`weights` must be broadcastable to the data shape.") from exc
            if np.any(w < 0):
                raise ValueError("`weights` must be non-negative.")

        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("Sum of weights must be positive.")
        n_eff = float(w_sum**2 / np.sum(w**2))
        return data_arr, w, w_sum, n_eff

    def _fit_moments(self, data, *, weights=None, return_info=False):
        data_arr, w, w_sum, n_eff = self._prepare_data_weights(data, weights=weights)

        mu_hat, r1 = circ_mean_and_r(alpha=data_arr, w=w)
        centered = angmod(data_arr - mu_hat)
        cos2 = np.cos(2.0 * centered)
        sin2 = np.sin(2.0 * centered)
        alpha2 = float(np.sum(w * cos2) / w_sum)
        beta2 = float(np.sum(w * sin2) / w_sum)
        mu_hat = self._wrap_direction(float(mu_hat))
        gamma_hat = float(np.clip(r1, 0.0, 1.0 - 1e-9))

        alpha2_proj, beta2_proj = self._project_second_order(gamma_hat, alpha2, beta2)

        if gamma_hat < self._moment_tolerance:
            rho_hat = 0.0
            lam_hat = 0.0
        else:
            r2 = np.hypot(alpha2_proj, beta2_proj)
            rho_hat = float(np.clip(r2 / max(gamma_hat, 1e-12), 0.0, 1.0 - 1e-9))
            lam_hat = float(np.mod(np.arctan2(beta2_proj, alpha2_proj), 2.0 * np.pi))
            if rho_hat < self._moment_tolerance:
                lam_hat = 0.0

        estimates = (mu_hat, gamma_hat, rho_hat, lam_hat)
        if return_info:
            info = {
                "method": "moments",
                "converged": True,
                "n_effective": n_eff,
            }
            return estimates, info
        return estimates

    @staticmethod
    def _project_second_order(gamma, alpha2, beta2):
        gamma = float(gamma)
        radius = gamma * (1.0 - gamma)
        center_alpha = gamma * gamma
        vec_alpha = alpha2 - center_alpha
        vec_beta = beta2
        distance = np.hypot(vec_alpha, vec_beta)
        if radius <= 0.0:
            return center_alpha, 0.0
        if distance <= radius:
            return alpha2, beta2
        if distance == 0.0:
            return center_alpha + radius, 0.0
        scale = radius / distance
        alpha_proj = center_alpha + vec_alpha * scale
        beta_proj = vec_beta * scale
        return alpha_proj, beta_proj

    @staticmethod
    def convert_alpha2_beta2(gamma, alpha2, beta2, *, verify=True):
        """
        Convert second-order moment parameters to (rho, lambda).

        Parameters
        ----------
        gamma : float
            Mean resultant length, 0 <= gamma < 1.
        alpha2 : float
            Second-order cosine moment around mu.
        beta2 : float
            Second-order sine moment around mu.
        verify : bool, optional
            If True (default), check that (alpha2, beta2) lies within the feasible
            disk for the supplied gamma and raise a ValueError if not.

        Returns
        -------
        rho : float
            Second-order magnitude parameter.
        lam : float
            Second-order phase parameter in [0, 2 pi).
        """
        gamma = float(gamma)
        alpha2 = float(alpha2)
        beta2 = float(beta2)

        if not (0.0 <= gamma < 1.0):
            raise ValueError("`gamma` must lie in [0, 1).")

        radius_sq = (gamma * (1.0 - gamma)) ** 2
        center_alpha = gamma * gamma
        dist_sq = (alpha2 - center_alpha) ** 2 + beta2**2

        tol = 1e-12
        if verify and dist_sq > radius_sq + tol:
            raise ValueError(
                f"(alpha2, beta2) = ({alpha2}, {beta2}) is outside the feasible disk "
                f"for gamma={gamma}."
            )

        r2 = np.hypot(alpha2, beta2)
        if gamma <= katojones_gen._moment_tolerance:
            if verify and r2 > tol:
                raise ValueError(
                    "When gamma is approximately zero, alpha2 and beta2 must also be near zero."
                )
            return 0.0, 0.0

        rho = float(np.clip(r2 / gamma, 0.0, 1.0 - 1e-12))
        if r2 <= tol:
            lam = 0.0
        else:
            lam = float(np.mod(np.arctan2(beta2, alpha2), 2.0 * np.pi))
        return rho, lam

    @staticmethod
    def convert_rho_lambda(gamma, rho, lam, *, verify=True):
        """
        Convert (rho, lambda) parameters to second-order moments (alpha2, beta2).

        Parameters
        ----------
        gamma : float
            Mean resultant length, 0 <= gamma < 1.
        rho : float
            Second-order magnitude, 0 <= rho < 1.
        lam : float
            Second-order phase, 0 <= lam < 2*pi.
        verify : bool, optional
            If True (default), ensure (gamma, rho, lam) satisfies the feasibility
            constraint and raise a ValueError otherwise.

        Returns
        -------
        alpha2 : float
            Second-order cosine moment around mu.
        beta2 : float
            Second-order sine moment around mu.
        """
        gamma = float(gamma)
        rho = float(rho)
        lam = float(lam)

        if not (0.0 <= gamma < 1.0):
            raise ValueError("`gamma` must lie in [0, 1).")
        if not (0.0 <= rho < 1.0):
            raise ValueError("`rho` must lie in [0, 1).")

        if verify:
            constraint = (rho * np.cos(lam) - gamma) ** 2 + (rho * np.sin(lam)) ** 2
            if constraint > (1.0 - gamma) ** 2 + 1e-12:
                raise ValueError(
                    f"(gamma, rho, lam)=({gamma}, {rho}, {lam}) violates the feasibility constraint."
                )

        alpha2 = float(gamma * rho * np.cos(lam))
        beta2 = float(gamma * rho * np.sin(lam))
        return alpha2, beta2

    @staticmethod
    def _aux_from_rho_lam(gamma, rho, lam):
        gamma = float(gamma)
        rho = float(rho)
        lam = float(lam)
        gamma = np.clip(gamma, 0.0, 1.0 - 1e-12)
        rho = np.clip(rho, 0.0, 1.0 - 1e-12)
        lam = float(np.mod(lam, 2.0 * np.pi))

        if gamma >= 1.0 - 1e-12:
            return 0.0, 0.0

        denom = max(1e-12, 1.0 - gamma)
        delta_cos = rho * np.cos(lam) - gamma
        delta_sin = rho * np.sin(lam)
        s = float(np.clip(np.hypot(delta_cos, delta_sin) / denom, 0.0, 1.0 - 1e-9))
        phi = float(np.mod(np.arctan2(delta_sin, delta_cos), 2.0 * np.pi))
        if s < katojones_gen._moment_tolerance:
            phi = 0.0
        return s, phi

    @staticmethod
    def _rho_lam_from_aux(gamma, s, phi):
        gamma = float(np.clip(gamma, 0.0, 1.0 - 1e-9))
        s = float(np.clip(s, 0.0, 1.0 - 1e-9))
        phi = float(np.mod(phi, 2.0 * np.pi))

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        delta_cos = (1.0 - gamma) * s * cos_phi
        delta_sin = (1.0 - gamma) * s * sin_phi
        rho_cos = gamma + delta_cos
        rho_sin = delta_sin
        rho = float(np.clip(np.hypot(rho_cos, rho_sin), 0.0, 1.0 - 1e-9))
        lam = float(np.mod(np.arctan2(rho_sin, rho_cos), 2.0 * np.pi))
        return rho, lam

    def _get_series_terms(self, mu, gamma, rho, lam):
        mu_val = float(np.mod(self._scalar_param(mu), 2.0 * np.pi))
        gamma_val = float(np.clip(self._scalar_param(gamma), 0.0, 1.0 - 1e-12))
        rho_val = float(np.clip(self._scalar_param(rho), 0.0, 1.0 - 1e-12))
        lam_val = float(np.mod(self._scalar_param(lam), 2.0 * np.pi))

        key = self._normalization_cache_key(mu_val, gamma_val, rho_val, lam_val)
        if key is None:
            return self._compute_series_terms(mu_val, gamma_val, rho_val, lam_val)

        cache = self._series_cache
        if key not in cache:
            cache[key] = self._compute_series_terms(mu_val, gamma_val, rho_val, lam_val)
        return cache[key]

    def _compute_series_terms(self, mu, gamma, rho, lam):
        # Only gamma ~ 0 collapses the CDF to the uniform theta/(2*pi); at
        # rho = 0 (the cardioid case) the series keeps its single p = 1 term
        # with coefficient gamma, handled by the P = 1 branch below.
        if gamma <= _KJ_GAMMA_TOL:
            return {
                "coeffs": np.empty(0, dtype=float),
                "phases": np.empty(0, dtype=float),
                "p": np.empty(0, dtype=float),
                "anchor": 0.0,
            }

        rho_val = float(np.clip(rho, 0.0, 1.0 - 1e-12))
        gamma_val = float(np.clip(gamma, 0.0, 1.0 - 1e-12))

        if rho_val == 0.0:
            P = 1
        else:
            P = 1
            for _ in range(_KJ_MAX_TERMS):
                tail = (gamma_val / max(P, 1)) * (rho_val ** max(P - 1, 0)) / max(1e-12, 1.0 - rho_val)
                if tail <= _KJ_CDF_TOL:
                    break
                P += 1
            P = min(P, _KJ_MAX_TERMS)

        p = np.arange(1, P + 1, dtype=float)
        rho_pows = rho_val ** (p - 1.0)
        coeffs = gamma_val * rho_pows / p
        phases = np.mod(p * mu + (p - 1.0) * lam, 2.0 * np.pi)
        anchor = -(1.0 / np.pi) * np.sum(coeffs * np.sin(phases))

        return {
            "coeffs": coeffs,
            "phases": phases,
            "p": p,
            "anchor": float(anchor),
        }

    def _evaluate_cdf_series(self, theta, mu, gamma, rho, lam, *, series=None):
        theta_arr = np.asarray(theta, dtype=float)
        flat = theta_arr.reshape(-1)

        if series is None:
            series = self._get_series_terms(mu, gamma, rho, lam)

        coeffs = series["coeffs"]
        if coeffs.size == 0:
            return (flat / (2.0 * np.pi)).reshape(theta_arr.shape)

        phases = series["phases"]
        p = series["p"][:, np.newaxis]
        theta_col = flat[np.newaxis, :]
        sin_terms = np.sin(p * theta_col - phases[:, np.newaxis])
        series_sum = np.sum(coeffs[:, np.newaxis] * sin_terms, axis=0)
        base = flat / (2.0 * np.pi)
        values = base + (1.0 / np.pi) * series_sum - series["anchor"]
        return values.reshape(theta_arr.shape)

    def _fit_mle(
        self,
        data,
        *,
        weights=None,
        initial,
        optimizer,
        options,
        return_info=False,
        **minimize_kwargs,
    ):
        data_arr, w, w_sum, n_eff = self._prepare_data_weights(data, weights=weights)

        if initial is None:
            initial = self._fit_moments(data_arr, weights=w)

        mu0, gamma0, rho0, lam0 = initial
        mu0 = self._wrap_direction(float(mu0))
        gamma0 = float(np.clip(gamma0, 1e-6, 1.0 - 1e-6))
        lam0 = float(np.mod(lam0, 2.0 * np.pi))
        rho0 = float(np.clip(rho0, 0.0, 1.0 - 1e-6))
        if rho0 < self._moment_tolerance:
            rho0 = 0.0
            lam0 = 0.0
        s0, phi0 = self._aux_from_rho_lam(gamma0, rho0, lam0)
        x0 = np.array([mu0, gamma0, s0, phi0], dtype=float)

        def objective(params):
            mu, gamma, s, phi = params
            mu = self._wrap_direction(float(mu))
            gamma = float(np.clip(gamma, 1e-6, 1.0 - 1e-9))
            s = float(np.clip(s, 0.0, 1.0 - 1e-9))
            phi = float(np.mod(phi, 2.0 * np.pi))
            rho, lam = self._rho_lam_from_aux(gamma, s, phi)
            if not self._argcheck(mu, gamma, rho, lam):
                return 1e12
            pdf_vals = self._pdf(data_arr, mu, gamma, rho, lam)
            if np.any(pdf_vals <= 0.0) or not np.all(np.isfinite(pdf_vals)):
                return 1e12
            return -np.sum(w * np.log(pdf_vals))

        bounds = [
            (0.0, 2.0 * np.pi),
            (1e-6, 1.0 - 1e-6),
            (0.0, 1.0 - 1e-6),
            (0.0, 2.0 * np.pi),
        ]

        result = minimize(
            objective,
            x0,
            method=optimizer,
            bounds=bounds,
            options=options,
            **minimize_kwargs,
        )

        if not result.success:
            fallback_method = "Powell" if optimizer != "Powell" else None
            if fallback_method is not None:
                fallback_result = minimize(
                    objective,
                    x0,
                    method=fallback_method,
                    bounds=bounds,
                    options={},
                    **minimize_kwargs,
                )
                if fallback_result.success:
                    result = fallback_result
            if not result.success:
                raise RuntimeError(f"Maximum likelihood fit failed: {result.message}")

        mu_hat, gamma_hat, s_hat, phi_hat = result.x
        mu_hat = self._wrap_direction(float(mu_hat))
        gamma_hat = float(np.clip(gamma_hat, 0.0, 1.0 - 1e-9))
        s_hat = float(np.clip(s_hat, 0.0, 1.0 - 1e-9))
        phi_hat = float(np.mod(phi_hat, 2.0 * np.pi))
        rho_hat, lam_hat = self._rho_lam_from_aux(gamma_hat, s_hat, phi_hat)

        estimates = (mu_hat, gamma_hat, rho_hat, lam_hat)
        if return_info:
            final_nll = objective(result.x)
            info = {
                "method": "mle",
                "converged": bool(result.success),
                "loglik": float(-final_nll),
                "n_effective": n_eff,
                "nit": getattr(result, "nit", None),
                "optimizer": optimizer,
                "initial": initial,
            }
            return estimates, info
        return estimates

    def fit(
        self,
        data,
        method="moments",
        *,
        weights=None,
        initial=None,
        optimizer="L-BFGS-B",
        options=None,
        return_info=False,
        **kwargs,
    ):
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        kwargs.pop("floc", None)
        kwargs.pop("fscale", None)

        if method == "moments":
            if kwargs:
                raise TypeError("Unexpected optimizer arguments for method='moments'.")
            estimates, info = self._fit_moments(
                data,
                weights=weights,
                return_info=True,
            )
            return (estimates, info) if return_info else estimates

        if method != "mle":
            raise ValueError("method must be either 'moments' or 'mle'.")

        options = {} if options is None else dict(options)
        estimates, info = self._fit_mle(
            data,
            weights=weights,
            initial=initial,
            optimizer=optimizer,
            options=options,
            return_info=True,
            **kwargs,
        )
        return (estimates, info) if return_info else estimates


katojones = katojones_gen(name="katojones")
kjlss = KatoJonesLL(name="kjlss")
