"""Circular regression.

- :func:`circ_lm` — classical parametric circular regression: circular response
  on linear covariates (von Mises), circular on circular, and linear on circular
  (harmonic), selected by ``type``.
- :func:`circ_gam` — penalized-smooth distributional circular regression via
  ``hea.models.gam``.
"""

import re
import warnings
from typing import List, Tuple

import numpy as np
import polars as pl
from hea.formula import prepare_design
from hea.models import gam, lm
from scipy.special import i0e
from scipy.stats import chi2

# Circular families for circ_gam's family resolution.
from .distributions import (
    CircularLL,
    _circular_family,
    ajplss,
    cardlss,
    cartlss,
    ibslss,
    jplss,
    kjlss,
    pnlss,
    ssjplss,
    vmftlss,
    vmlss,
    wclss,
    wnlss,
)
from .utils import A1, A1inv, A1prime

__all__ = ["circ_lm", "circ_gam"]


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
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
    """Flatten an ``hea`` output (polars frame or ndarray) to a 1-D float array."""
    if isinstance(v, pl.DataFrame):
        return v.to_numpy().ravel()
    return np.asarray(v, dtype=float).ravel()


# A smooth term — s()/te()/ti()/t2(), mgcv's smooth constructors — routes a fit
# to circ_gam (penalized REML/GCV smooths); circ_lm is parametric-only and
# rejects it. The ``\b`` keeps the trailing ``s(`` of ``cos(``/``sin(`` from
# matching the ``s`` smooth constructor (no word boundary between ``o`` and ``s``).
_SMOOTH_RE = re.compile(r"\b(?:s|te|ti|t2)\s*\(")


def _has_smooth(formula: str) -> bool:
    """True if the formula RHS contains a smooth term (→ circ_gam)."""
    rhs = formula.split("~", 1)[1] if "~" in formula else formula
    return bool(_SMOOTH_RE.search(rhs))


# Cyclic smooth bases (mgcv): bs='cc' (cyclic cubic) / bs='cp' (cyclic p-spline).
# Their boundary knots set the *period*; for circular predictors that is 2π, so
# circ_gam supplies it by default (mgcv otherwise defaults cyclic knots to the
# data range, collapsing f(0)=f(period) for angles).
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


# --------------------------------------------------------------------------- #
# circ_lm — classical (parametric) circular regression
# --------------------------------------------------------------------------- #
def circ_lm(formula, data, type="cl", order=1, init=None, tol=1e-8,
            maxit=100, verbose=False):
    """Classical (parametric) circular regression.

    ``type`` selects the fit (hyphenated spellings such as ``"c-l"`` accepted):

    - ``"cl"`` circular response ~ linear covariate(s): the Fisher & Lee (1992)
      von Mises regression by Green's (1984) IRLS, with the mean / kappa / mixed
      concentration extensions of Fisher (1993) §6.4. The mean direction is
      ``μ_i = μ₀ + 2·atan(x_iᵀβ)``. A one- or two-formula list selects the
      sub-model: ``"θ ~ x"`` (≡ ``["θ ~ x", "~ 1"]``) models μ with constant κ;
      ``["θ ~ 1", "~ z"]`` models log κ with constant μ; ``["θ ~ x", "~ x"]`` is
      the mixed model (μ and κ share one design).
    - ``"cc"`` circular ~ circular: the Sarma & Jammalamadaka (1993) harmonic
      least-squares fit of cos θ, sin θ on a degree-``order`` trigonometric
      polynomial of the angular covariate, with the higher-order significance
      test.
    - ``"lc"`` linear ~ circular: an ordinary-least-squares fit of a linear
      response on the angular harmonics. Write the terms in the formula
      (``y ~ cos(θ) + sin(θ) + sin(2*θ)``); ``order`` is not used here.

    A smooth term (``s()``/``te()``/…) raises, pointing to :func:`circ_gam`,
    which also covers penalized smooths and other response families.

    Parameters
    ----------
    formula : str or list of str
        A formula ``"y ~ x"``, or (``type="cl"``) a one/two-formula list
        ``[μ-formula, logκ-formula]``. The first formula names the response.
    data : polars.DataFrame
        A polars (or pandas) DataFrame holding the response and covariates.
    type : {"cl", "cc", "lc"}, optional
        Which classical fit (default ``"cl"``). Hyphenated ``"c-l"``/``"c-c"``/
        ``"l-c"`` are accepted.
    order : int, optional
        Trigonometric-polynomial order for ``"cc"`` (number of harmonics of the
        angular predictor). Ignored for ``"cl"`` and ``"lc"``.
    init : array-like, optional
        Starting values for the mean-direction coefficients (``"cl"`` only);
        defaults to zero.
    tol, maxit, verbose : optional
        IRLS convergence tolerance, iteration cap, and per-iteration logging
        (``"cl"`` only).

    Returns
    -------
    object
        ``"lc"`` returns the fitted ``hea.models.lm`` (with its own
        ``summary``/``plot``/``predict``/``coefficients`` interface). ``"cl"``
        and ``"cc"`` return a ``dict`` of the circular estimates; the ``"cc"``
        dict also carries the underlying ``cos_lm``/``sin_lm`` fits.

    References
    ----------
    - Fisher, N. I. & Lee, A. J. (1992). Regression models for an angular
      response. Biometrics 48, 665-677.
    - Fisher, N. I. (1993). Statistical Analysis of Circular Data. Cambridge
      University Press.
    - Sarma, Y. & Jammalamadaka, S. R. (1993). Circular regression. In
      Statistical Sciences and Data Analysis, 109-128. VSP, Utrecht.
    - Pewsey, A., Neuhäuser, M. & Ruxton, G. D. (2013). Circular Statistics in
      R. Oxford University Press.
    """
    key = _circ_lm_type(type)
    formulas = list(formula) if isinstance(formula, (list, tuple)) else [formula]
    if not formulas or not all(isinstance(f, str) for f in formulas):
        raise ValueError("`formula` must be a formula string or a list of them.")
    _circ_lm_no_smooth(formulas)
    df = _to_polars(data)

    if key == "cl":
        return _circ_lm_cl(formulas, df, init, tol, int(maxit), verbose)
    if key == "cc":
        return _circ_lm_cc(formulas, df, int(order))
    return _circ_lm_lc(formulas, df)


# --- circ_lm internals ------------------------------------------------------ #
def _circ_lm_type(type) -> str:
    """Normalize the leg selector: ``'cl'``/``'c-l'``/``'C-L'`` → ``'cl'`` …"""
    key = str(type).strip().lower().replace("-", "")
    if key not in ("cl", "cc", "lc"):
        raise ValueError(
            "type must be one of 'cl'/'c-l' (circular ~ linear), 'cc'/'c-c' "
            "(circular ~ circular), or 'lc'/'l-c' (linear ~ circular); got "
            f"{type!r}."
        )
    return key


def _circ_lm_no_smooth(formulas) -> None:
    """Reject mgcv smooth constructors on any RHS — circ_lm is parametric only."""
    bad = sorted({
        m.group(0).split("(")[0].strip()
        for f in formulas
        for m in _SMOOTH_RE.finditer(f.split("~", 1)[1] if "~" in f else f)
    })
    if bad:
        terms = ", ".join(f"{b}()" for b in bad)
        raise ValueError(
            f"circ_lm fits parametric models only; smooth term(s) {terms} are "
            "not allowed. Use circ_gam for penalized smooths."
        )


def _circ_lm_design(formula, data, response) -> Tuple[np.ndarray, List[str]]:
    """Intercept-free design matrix for the RHS of ``formula``, parsed by hea
    (so ``cos()``/``sin()``/``harmonic()`` expressions and bare columns both
    work). ``~ 1`` / ``~`` gives a 0-column design. ``response`` names the LHS
    column for the parser (a dummy is added when it is not present, e.g. at
    predict time). Returns ``(design (n × p), column names)``.
    """
    rhs = (formula.split("~", 1)[1] if "~" in formula else formula).strip()
    if rhs in ("", "1"):
        return np.empty((data.height, 0), dtype=float), []
    if response not in data.columns:
        data = data.with_columns(pl.lit(0.0).alias(response))
    design = prepare_design(f"{response} ~ 0 + {rhs}", data, na_action="pass")
    x = design.X
    return x.to_numpy().astype(float), list(x.columns)


def _circ_lm_one_predictor(formula, data):
    """Resolve the single angular predictor of a ``"y ~ x"`` formula (cc).

    Takes the RHS identifiers that are data columns and requires exactly one.
    Rows with a non-finite response or predictor are dropped. Returns
    ``(response, var, y, x, n)``.
    """
    if "~" not in formula:
        raise ValueError("the formula must name the response, e.g. 'theta ~ phi'.")
    resp, rhs = (s.strip() for s in formula.split("~", 1))
    if not resp:
        raise ValueError("the formula must name the response, e.g. 'theta ~ phi'.")
    cols = set(data.columns)
    if resp not in cols:
        raise ValueError(f"unknown response {resp!r}; columns are {sorted(cols)}.")
    seen, vars_ = set(), []
    for tok in re.findall(r"\w+", rhs):
        if tok in cols and tok not in seen:
            seen.add(tok)
            vars_.append(tok)
    if len(vars_) != 1:
        found = ", ".join(repr(v) for v in vars_) if vars_ else "none"
        raise ValueError(
            "circ_lm(type='cc') takes exactly one angular predictor; got "
            f"{found}."
        )
    var = vars_[0]
    y = np.asarray(data[resp].to_numpy(), dtype=float)
    x = np.asarray(data[var].to_numpy(), dtype=float)
    keep = np.isfinite(y) & np.isfinite(x)
    return resp, var, y[keep], x[keep], int(keep.sum())


# === cl — Fisher-Lee von Mises regression (mean / kappa / mixed) ============ #
def _circ_lm_cl(formulas, data, init, tol, maxit, verbose) -> dict:
    if len(formulas) > 2:
        raise ValueError(
            "circ_lm(type='cl') takes at most two formulas: "
            "[mu-formula, logkappa-formula]."
        )
    mu_f = formulas[0]
    if "~" not in mu_f or not mu_f.split("~", 1)[0].strip():
        raise ValueError("the first formula must name the response, e.g. 'theta ~ x'.")
    kappa_f = formulas[1] if len(formulas) == 2 else "~ 1"
    response = mu_f.split("~", 1)[0].strip()
    if response not in set(data.columns):
        raise ValueError(
            f"unknown response {response!r}; columns are {sorted(data.columns)}."
        )
    theta = np.mod(np.asarray(data[response].to_numpy(), dtype=float), 2.0 * np.pi)
    x_mu, mu_terms = _circ_lm_design(mu_f, data, response)
    x_ka, kappa_terms = _circ_lm_design(kappa_f, data, response)
    has_mu, has_ka = len(mu_terms) > 0, len(kappa_terms) > 0
    if has_mu and not has_ka:
        model = "mean"
    elif has_ka and not has_mu:
        model = "kappa"
    elif has_mu and has_ka:
        model = "mixed"
    else:
        raise ValueError("no predictors in either formula; nothing to regress.")
    if model == "mixed" and mu_terms != kappa_terms:
        raise ValueError(
            "the fisher-lee fitter ties mu and kappa to one shared design, but "
            f"mu uses {mu_terms} and kappa uses {kappa_terms}. Use circ_gam for "
            "different covariates per predictor."
        )
    x = x_ka if model == "kappa" else x_mu
    fit = _circ_lm_cl_fit(theta, x, model, init, tol, maxit, verbose)

    n = theta.size
    p = x.shape[1]
    npar = {"mean": p, "kappa": 1 + p, "mixed": 1 + 2 * p}[model]
    if model == "kappa":
        fitted = np.full(n, np.mod(fit["mu"], 2.0 * np.pi))
    else:
        fitted = np.mod(fit["mu"] + 2.0 * np.arctan(x @ fit["beta"]), 2.0 * np.pi)
    fit.update({
        "model": model, "n": n, "npar": npar,
        "aic": -2.0 * fit["loglik"] + 2.0 * npar,
        "bic": -2.0 * fit["loglik"] + np.log(n) * npar,
        "response": response, "mu_formula": mu_f, "kappa_formula": kappa_f,
        "mu_terms": mu_terms, "kappa_terms": kappa_terms,
        "fitted": fitted,
        "residuals": np.angle(np.exp(1j * (theta - fitted))),
    })
    return fit


def _circ_lm_cl_fit(theta, x, model, init, tol, maxit, verbose) -> dict:
    """Green (1984) IRLS for the von Mises MLE. Mean: ``mu_i = mu0 +
    2*atan(X beta)``, constant kappa. Kappa: ``log kappa_i = alpha + X gamma``,
    constant mu. Mixed: both. Scores and expected-information weights follow
    Fisher (1993) §6.4. The tan-half link is inlined (``linkinv = 2*atan``,
    ``mu_eta = 2/(1+eta^2)``).
    """
    n = theta.size
    p = x.shape[1]
    x1 = np.column_stack((np.ones(n), x))
    ridge_x = 1e-8 * np.eye(p)
    ridge_1 = 1e-8 * np.eye(p + 1)

    def log_i0(k):
        # log I0(k) via the exponentially scaled Bessel (raw i0 overflows ~710).
        return k + np.log(i0e(k))

    if init is not None and model in ("mean", "mixed"):
        beta = np.resize(np.asarray(init, dtype=float).ravel(), p).astype(float)
    else:
        beta = np.zeros(p)
    gamma = np.zeros(p)
    alpha = 0.0
    mu, kappa = 0.0, 1.0
    ll = ll_old = -np.inf
    diff = tol + 1.0
    it = 0

    for it in range(1, maxit + 1):
        if model == "mean":
            eta = x @ beta
            rdev = theta - 2.0 * np.arctan(eta)
            s, c = np.mean(np.sin(rdev)), np.mean(np.cos(rdev))
            mu = np.arctan2(s, c)
            kappa = float(A1inv(np.hypot(s, c)))
            g = (2.0 / (1.0 + eta**2))[:, None] * x
            w = kappa * A1(kappa)
            u = kappa * np.sin(rdev - mu)
            gtg = g.T @ g
            beta = _safe_solve(w * gtg + ridge_x, g.T @ u + w * gtg @ beta)
            ll = -n * log_i0(kappa) + kappa * np.sum(np.cos(rdev - mu))

        elif model == "kappa":
            kappa = np.exp(np.clip(alpha + x @ gamma, -50.0, 50.0))
            mu = np.arctan2(np.sum(kappa * np.sin(theta)),
                            np.sum(kappa * np.cos(theta)))
            a1p = np.maximum(A1prime(kappa), 1e-12)
            y = (np.cos(theta - mu) - A1(kappa)) / (a1p * kappa)
            w = kappa**2 * a1p
            upd = _safe_solve(x1.T @ (w[:, None] * x1) + ridge_1, x1.T @ (w * y))
            alpha = alpha + upd[0]
            gamma = gamma + upd[1:]
            ll = -np.sum(log_i0(kappa)) + np.sum(kappa * np.cos(theta - mu))

        else:  # mixed
            kappa = np.exp(np.clip(alpha + x @ gamma, -50.0, 50.0))
            eta = x @ beta
            rdev = theta - 2.0 * np.arctan(eta)
            mu = np.arctan2(np.sum(kappa * np.sin(rdev)),
                            np.sum(kappa * np.cos(rdev)))
            g = (2.0 / (1.0 + eta**2))[:, None] * x
            wb = kappa * A1(kappa)
            gtwg = g.T @ (wb[:, None] * g)
            beta = _safe_solve(gtwg + ridge_x,
                               g.T @ (kappa * np.sin(rdev - mu)) + gtwg @ beta)
            a1p = np.maximum(A1prime(kappa), 1e-12)
            y = (np.cos(rdev - mu) - A1(kappa)) / (a1p * kappa)
            wg = kappa**2 * a1p
            upd = _safe_solve(x1.T @ (wg[:, None] * x1) + ridge_1, x1.T @ (wg * y))
            alpha = alpha + upd[0]
            gamma = gamma + upd[1:]
            ll = -np.sum(log_i0(kappa)) + np.sum(kappa * np.cos(rdev - mu))

        diff = abs(ll - ll_old)
        if verbose:
            print(f"iter {it}: logLik = {ll:.6f}, diff = {diff:.2e}")
        if diff < tol:
            break
        ll_old = ll

    converged = diff < tol
    if not converged:
        warnings.warn(
            f"circ_lm(type='cl') did not converge in {maxit} iterations "
            f"(last diff={diff:.2e}, tol={tol:.2e}).",
            RuntimeWarning, stacklevel=2,
        )

    # mu/kappa/loglik evaluated at the converged coefficients (the loop carries
    # them one IRLS step behind the final beta/alpha/gamma update).
    if model == "mean":
        rdev = theta - 2.0 * np.arctan(x @ beta)
        s, c = np.mean(np.sin(rdev)), np.mean(np.cos(rdev))
        mu = np.arctan2(s, c)
        kappa = float(A1inv(np.hypot(s, c)))
        ll = -n * log_i0(kappa) + kappa * np.sum(np.cos(rdev - mu))
    else:
        kappa = np.exp(np.clip(alpha + x @ gamma, -50.0, 50.0))
        rdev = theta if model == "kappa" else theta - 2.0 * np.arctan(x @ beta)
        mu = np.arctan2(np.sum(kappa * np.sin(rdev)), np.sum(kappa * np.cos(rdev)))
        ll = -np.sum(log_i0(kappa)) + np.sum(kappa * np.cos(rdev - mu))

    se = _circ_lm_cl_se(theta, x, model, beta, alpha, gamma, kappa, mu)
    return {
        "mu": float(mu),
        "kappa": float(kappa) if np.ndim(kappa) == 0 else kappa,
        "beta": beta, "alpha": float(alpha), "gamma": gamma,
        "loglik": float(ll), "iter": it, "converged": bool(converged),
        **se,
    }


def _circ_lm_cl_se(theta, x, model, beta, alpha, gamma, kappa, mu) -> dict:
    """Large-sample SEs from the expected information (Fisher 1993, eq. 6.62-
    6.64, 6.82). Per-observation kappa_i SE (kappa/mixed) by the delta method
    on ``(alpha, gamma)``."""
    n = theta.size
    p = x.shape[1]
    x1 = np.column_stack((np.ones(n), x))
    out: dict = {"se_beta": None, "se_alpha": None, "se_gamma": None}
    if model == "mean":
        eta = x @ beta
        g = (2.0 / (1.0 + eta**2))[:, None] * x
        w = kappa * A1(kappa)
        cov_b = _safe_inverse(w * (g.T @ g))
        out["se_beta"] = np.sqrt(np.clip(np.diag(cov_b), 0.0, None))
        out["se_mu"] = 1.0 / np.sqrt(max((n - p) * w, 1e-12))
        out["se_kappa"] = float(np.sqrt(
            1.0 / max(n * (1 - A1(kappa) ** 2 - A1(kappa) / kappa), 1e-12)))
    else:
        if model == "mixed":
            eta = x @ beta
            g = (2.0 / (1.0 + eta**2))[:, None] * x
            cov_b = _safe_inverse(g.T @ ((kappa * A1(kappa))[:, None] * g))
            out["se_beta"] = np.sqrt(np.clip(np.diag(cov_b), 0.0, None))
        w = kappa**2 * A1prime(kappa)
        cov_ag = _safe_inverse(x1.T @ (w[:, None] * x1))
        out["se_alpha"] = float(np.sqrt(max(cov_ag[0, 0], 0.0)))
        out["se_gamma"] = np.sqrt(np.clip(np.diag(cov_ag)[1:], 0.0, None))
        out["se_mu"] = 1.0 / np.sqrt(
            max(float(np.sum(kappa * A1(kappa))) - 0.5, 1e-12))
        quad = np.einsum("ij,ij->i", x1 @ cov_ag, x1)
        out["se_kappa"] = kappa * np.sqrt(np.clip(quad, 0.0, None))
    return out


# === cc — Sarma & Jammalamadaka harmonic circular-circular regression ======= #
def _circ_lm_cc(formulas, data, order) -> dict:
    if len(formulas) != 1:
        raise ValueError(
            "circ_lm(type='cc') takes a single formula, e.g. 'theta ~ phi'."
        )
    response, var, y, x, n = _circ_lm_one_predictor(formulas[0], data)
    x = np.mod(x, 2.0 * np.pi)
    y = np.mod(y, 2.0 * np.pi)
    period = 2.0 * np.pi
    term = f"harmonic({var}, k={order}, period={period})"
    d = pl.DataFrame({var: x, "_cos_y": np.cos(y), "_sin_y": np.sin(y)})
    cos_lm = lm(f"_cos_y ~ {term}", d)
    sin_lm = lm(f"_sin_y ~ {term}", d)

    cos_fit, sin_fit = _ravel(cos_lm.yhat), _ravel(sin_lm.yhat)
    fitted = np.mod(np.arctan2(sin_fit, cos_fit), 2.0 * np.pi)
    residuals = np.mod(y - fitted, 2.0 * np.pi)
    rho = float(np.sqrt((cos_fit @ cos_fit + sin_fit @ sin_fit) / n))
    a_k = float(np.mean(np.cos(residuals)))
    if a_k < 0:
        warnings.warn(
            "mean residual cosine is negative; residuals anti-align with the fit "
            "(kappa clamped to 0). Check for misspecification.",
            UserWarning, stacklevel=3,
        )
    kappa = float(A1inv(a_k))

    # Higher-order test (Sarma & Jammalamadaka 1993): do the (order + 1)
    # harmonics add signal beyond the fitted design? One χ² statistic each for
    # cos(y), sin(y), using the fitted design's residual projection.
    xm = np.asarray(cos_lm.X.to_numpy(), dtype=float)
    w = np.column_stack((np.cos((order + 1) * x), np.sin((order + 1) * x)))
    im = np.eye(n) - xm @ _safe_inverse(xm.T @ xm) @ xm.T
    nmat = w @ _safe_inverse(w.T @ im @ w) @ w.T
    res_c, res_s = _ravel(cos_lm.residuals), _ravel(sin_lm.residuals)
    adj = max(n - (2 * order + 1), 1)
    t1 = adj * float(res_c @ nmat @ res_c) / max(float(res_c @ res_c), 1e-12)
    t2 = adj * float(res_s @ nmat @ res_s) / max(float(res_s @ res_s), 1e-12)
    p_values = np.array([1.0 - chi2.cdf(t1, 2), 1.0 - chi2.cdf(t2, 2)], dtype=float)

    return {
        "var": var, "response": response, "order": int(order), "n": n,
        "coefficients": {"cos": np.asarray(cos_lm.bhat.row(0), dtype=float),
                         "sin": np.asarray(sin_lm.bhat.row(0), dtype=float)},
        "rho": rho, "A_k": a_k, "kappa": kappa,
        "fitted": fitted, "residuals": residuals, "p_values": p_values,
        "cos_lm": cos_lm, "sin_lm": sin_lm,
    }


# === lc — harmonic linear-circular regression =============================== #
def _circ_lm_lc(formulas, data):
    """Linear response on a circular predictor: an OLS fit of the harmonic terms
    in the formula (``y ~ cos(phi) + sin(phi) + sin(2*phi)``), returned as the
    fitted ``hea.models.lm`` object."""
    if len(formulas) != 1:
        raise ValueError(
            "circ_lm(type='lc') takes a single formula, e.g. "
            "'y ~ cos(phi) + sin(phi)'."
        )
    return lm(formulas[0], data)


# --------------------------------------------------------------------------- #
# circ_gam — penalized-smooth distributional circular regression
# --------------------------------------------------------------------------- #
def _lss_catalog() -> dict:
    """Lowercased name → ``*lss`` family instance, under both the alias
    (``"vmlss"`` — shared with the circlss R package) and the distribution
    name (``"vonmises"``)."""
    cat: dict = {}
    for fam in (vmlss, wclss, pnlss, cardlss, cartlss, wnlss,
                jplss, ssjplss, kjlss, vmftlss, ajplss, ibslss):
        cat[fam.name.lower()] = fam
        cat[fam.dist.name.lower()] = fam
    return cat


def _resolve_gam_family(family):
    """``circ_gam``'s family resolution — permissive by design: circular
    names/objects get the CircularLL treatment, anything else passes
    through for hea to validate (gaussian and every other hea family ride
    untouched)."""
    if family is None:
        return vmlss
    if isinstance(family, str):
        key = family.strip().lower()
        cat = _lss_catalog()
        if key in cat:
            return cat[key]
        import hea.family as _hea_family

        obj = getattr(_hea_family, key, None)
        if obj is None:
            obj = getattr(_hea_family, key.capitalize(), None)
        if obj is not None:
            return obj
        lss = ", ".join(sorted({f.name for f in cat.values()}))
        raise ValueError(
            f"unknown family {family!r}: not a circular family ({lss}, or "
            "their distribution names) and not an hea family name. Pass an "
            "hea family object for non-circular responses."
        )
    if isinstance(family, CircularLL):
        return family
    if getattr(family, "param_roles", None):
        # a regression-ready circular distribution: wrap in its family
        # class (katojones auto-routes to KatoJonesLL)
        return _circular_family(family)
    return family


def _resolve_cyclic_knots_data(formulas, data, user_knots):
    """``circ_gam``'s cyclic-knot defaulting.

    For every cyclic smooth (``bs='cc'``/``'cp'``) whose knots the caller did
    not pin, set the boundary knots to the full circular period ``[0, 2π]`` —
    pycircstat2's angle convention (the branch ``Circular``/``angmod`` wrap
    to) — so the basis wraps at the true period, not the observed data range.
    A cyclic covariate must already be on ``[0, 2π]``: this raises (pointing at
    the wrapping helpers) when one falls outside it, rather than silently
    fitting a basis whose period is misaligned with the data. Explicit
    ``user_knots`` win per variable; returns ``None`` when nothing is set.

    (circlss's R twin instead brackets signed covariates with ``[-π, π]`` — the
    R/``atan2`` convention; the call shape is shared, the branch differs by
    ecosystem.)
    """
    cyclic = []
    for f in formulas:
        rhs = f.split("~", 1)[1] if "~" in f else f
        cyclic.extend(_cyclic_smooth_vars(rhs))
    cyclic = list(dict.fromkeys(cyclic))  # unique, formula order
    knots = dict(user_knots) if user_knots else {}
    if not cyclic:
        return knots or None
    cols = set(data.columns)
    period = 2 * np.pi
    for v in cyclic:
        if v in knots or v not in cols:
            continue  # user knots win; an absent var is left for hea to report
        x = np.asarray(data[v].to_numpy(), dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        lo, hi = float(x.min()), float(x.max())
        if lo < -1e-6 or hi > period + 1e-6:
            raise ValueError(
                f"cyclic covariate {v!r} lies outside [0, 2π] (range "
                f"[{lo:.3f}, {hi:.3f}]); pycircstat2 expects angles on that "
                "branch. Wrap it (pycircstat2.utils.angmod / data2rad, or the "
                f"Circular class) or pass knots={{{v!r}: [...]}} explicitly."
            )
        knots[v] = [0.0, period]
    return knots or None


def circ_gam(formula, data, family=None, knots=None, method="REML",
             **gam_kwargs):
    """Circular GAM — ``hea.models.gam`` with circular defaults.

    A deliberately thin front door: everything forwards to
    ``hea.models.gam`` verbatim; what this function adds is defaults and
    family resolution, nothing else.

    - ``family=vmlss`` (distributional von Mises) and ``method="REML"``
      unless overridden.
    - Cyclic smooths (``bs='cc'``/``'cp'``) default their boundary knots to
      the full circular period ``[0, 2π]`` — pycircstat2's angle convention
      (the branch ``Circular``/``angmod`` wrap to) — so the basis wraps at the
      true period, not the observed data range. Explicit ``knots=`` wins per
      variable (mgcv's ``knots=list(...)`` semantics). A cyclic covariate must
      already be on ``[0, 2π]``; otherwise ``circ_gam`` raises, pointing you to
      :func:`pycircstat2.utils.angmod` / :func:`~pycircstat2.utils.data2rad`
      (or the :class:`~pycircstat2.base.Circular` class) to wrap it.
    - ``family`` may be a ``*lss`` instance, a regression-ready circular
      distribution (auto-wrapped; ``katojones`` → :class:`KatoJonesLL`), a
      string (``"vmlss"``/``"vonmises"``, … or any hea family name such as
      ``"gaussian"``), or any hea family object — non-circular responses
      pass through untouched, so a linear response on a circular smooth is
      simply ``circ_gam("y ~ s(phi, bs='cc')", df, family="gaussian")``.
    - Fewer formulas than the family has parameters: the remaining linear
      predictors are filled with ``~ 1`` (held constant), so
      ``circ_gam("theta ~ s(x)", df, family="jplss")`` smooths μ and pins
      κ, ψ. The first formula must name the response.

    The circlss/mgcv twin call — cyclic knots auto-pinned to the period::

        # R:  b2 <- gam(list(theta ~ s(phi, bs="cc"), ~ s(phi, bs="cc")),
        #               family = vmlss(), data = dat, method = "REML")
        b2 = circ_gam(["theta ~ s(phi, bs='cc')", "~ s(phi, bs='cc')"],
                      data=dat)   # phi on [0, 2π] → knots default to [0, 2π]

    Returns the fitted ``hea`` gam object (``summary()``, ``predict()``,
    ``AIC``, ``gam_check()`` are hea's own).

    .. note:: **Reproducibility / parity.** ``hea`` inherits mgcv's loose
       default ``efs_tol=0.1`` for the EFS optimizer, which can leave ~1e-3
       run-to-run gaps in the coefficients and smoothing parameters. For
       reproducible, cross-engine-parity, or benchmarking fits, tighten the
       control knobs (they forward straight through ``**gam_kwargs``)::

           circ_gam(..., control={"efs_tol": 1e-8, "epsilon": 1e-10})

       which collapses that disagreement to machine precision.

    .. note:: **Multi-LP convergence.** The EFS optimizer (used for the
       Tier-2 families, ``available_derivs == 0``) caps its outer loop at
       ``efs_maxit=200`` to match mgcv. A 3-/4-LP shape family with two flat
       shape directions (``jplss``, ``ssjplss``) can need more than that to
       satisfy ``efs_tol`` and otherwise stops at "iteration limit reached";
       raise the cap for such hea-native fits with
       ``circ_gam(..., control={"efs_maxit": 500})``. Keep it at 200 for mgcv
       cross-engine parity.

    .. warning:: tanhalf-linked families (``vmlss``, ``wclss``, and the
       shape families) place μ in an open 2π-window: a mean that must sweep
       *through the antipode* — common when the covariate is itself
       circular — is unrepresentable. Use ``pnlss`` (projected normal, two
       identity-linked location LPs) for full-circle mean sweeps. Same
       convention as circlss documents on the R side.
    """
    fam = _resolve_gam_family(family)
    formulas = list(formula) if isinstance(formula, (list, tuple)) else [formula]
    if isinstance(fam, CircularLL):
        if "~" not in formulas[0] or not formulas[0].split("~", 1)[0].strip():
            raise ValueError(
                'the first formula must name the response, e.g. "theta ~ s(x)".'
            )
        if len(formulas) > fam.n_lp:
            raise ValueError(
                f"{fam.name} has {fam.n_lp} linear predictors; got "
                f"{len(formulas)} formulas."
            )
        # fewer formulas than parameters: hold the rest constant (~ 1), e.g.
        # theta ~ s(x) with jplss smooths mu and pins kappa, psi.
        formulas += ["~ 1"] * (fam.n_lp - len(formulas))
    df = _to_polars(data)
    merged = _resolve_cyclic_knots_data(formulas, df, knots)
    payload = formulas if len(formulas) > 1 else formulas[0]
    return gam(payload, df, family=fam,
               knots=merged, method=method, **gam_kwargs)
