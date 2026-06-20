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
from scipy.special import i0e, ive
from scipy.stats import chi2, norm

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
from .utils import A1, A1inv, A1prime, significance_code

__all__ = ["circ_lm", "circ_gam", "CircLM", "CircGAM"]


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
    CircLM
        A circular-regression result object. Every estimate is reachable both as
        an attribute and as a mapping key (``m.mu`` or ``m["mu"]``, ``m.kappa``
        / ``m["kappa"]`` …); the ``"cc"`` result carries the underlying
        ``cos_lm`` / ``sin_lm`` fits, and the ``"lc"`` result wraps the
        ``hea.models.lm`` and delegates its full interface (``bhat``, ``plot``,
        ``predict``, ``ci_bhat`` …). Adds ``summary``, ``predict``, ``coef``,
        ``logLik`` and the circular ``circ_check`` / ``circ_resid``.

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
        fields = _circ_lm_cl(formulas, df, init, tol, int(maxit), verbose)
    elif key == "cc":
        fields = _circ_lm_cc(formulas, df, int(order))
    else:
        fields = _circ_lm_lc(formulas, df)
    return CircLM(key, fields)


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


def _circ_lm_rhs_vars(formulas, data) -> List[str]:
    """The data-column identifiers on the RHS of the given formulas, in
    first-seen order (the raw covariates, distinct from hea's ``cos(x)`` design
    terms). One value ⇒ the fit's single covariate, used by the plot/check
    geometry views and the residual-vs-covariate panel."""
    cols = set(data.columns)
    seen, out = set(), []
    for f in formulas:
        rhs = f.split("~", 1)[1] if "~" in f else f
        for tok in re.findall(r"\w+", rhs):
            if tok in cols and tok not in seen:
                seen.add(tok)
                out.append(tok)
    return out


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
    # plotting/diagnostic frame: the raw covariate column(s) and the (mod 2π)
    # response, so circ_plot can overlay the data and bound the covariate grid,
    # and circ_check can place the residual-vs-covariate panel.
    rhs_vars = _circ_lm_rhs_vars([mu_f, kappa_f], data)
    frame = pl.DataFrame(
        {**{v: np.asarray(data[v].to_numpy(), dtype=float) for v in rhs_vars},
         response: theta}
    )
    fit.update({
        "model": model, "n": n, "npar": npar,
        "aic": -2.0 * fit["loglik"] + 2.0 * npar,
        "bic": -2.0 * fit["loglik"] + np.log(n) * npar,
        "response": response, "mu_formula": mu_f, "kappa_formula": kappa_f,
        "mu_terms": mu_terms, "kappa_terms": kappa_terms,
        "fitted": fitted,
        "residuals": np.angle(np.exp(1j * (theta - fitted))),
        "frame": frame,
        "covariate": rhs_vars[0] if len(rhs_vars) == 1 else None,
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
    # Vbeta / Vag are the coefficient covariances the flat-panel delta-method
    # bands need (G Vbeta Gᵀ for the mean direction; Z Vag Zᵀ on the log-κ scale).
    out: dict = {"se_beta": None, "se_alpha": None, "se_gamma": None,
                 "Vbeta": None, "Vag": None}
    if model == "mean":
        eta = x @ beta
        g = (2.0 / (1.0 + eta**2))[:, None] * x
        w = kappa * A1(kappa)
        cov_b = _safe_inverse(w * (g.T @ g))
        out["Vbeta"] = cov_b
        out["se_beta"] = np.sqrt(np.clip(np.diag(cov_b), 0.0, None))
        out["se_mu"] = 1.0 / np.sqrt(max((n - p) * w, 1e-12))
        out["se_kappa"] = float(np.sqrt(
            1.0 / max(n * (1 - A1(kappa) ** 2 - A1(kappa) / kappa), 1e-12)))
    else:
        if model == "mixed":
            eta = x @ beta
            g = (2.0 / (1.0 + eta**2))[:, None] * x
            cov_b = _safe_inverse(g.T @ ((kappa * A1(kappa))[:, None] * g))
            out["Vbeta"] = cov_b
            out["se_beta"] = np.sqrt(np.clip(np.diag(cov_b), 0.0, None))
        w = kappa**2 * A1prime(kappa)
        cov_ag = _safe_inverse(x1.T @ (w[:, None] * x1))
        out["Vag"] = cov_ag
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
        "frame": pl.DataFrame({var: x, response: y}),
        "covariate": var,
    }


# === lc — harmonic linear-circular regression =============================== #
_COSSIN_RE = re.compile(r"^(cos|sin)\((.*)\)$")


def _clean_harmonic_label(name) -> str:
    """hea's verbose harmonic design column name → the circlss-style coefficient
    label (``(Intercept)`` / ``cos1`` / ``sin1`` / …), so the cc coefficient
    table reads the same in both sibling packages."""
    name = str(name)
    if name == "(Intercept)":
        return name
    m = re.search(r"(cos|sin)\s*(\d+)$", name)
    return f"{m.group(1)}{m.group(2)}" if m else name


def _circ_lm_harmonics_table(fit) -> list:
    """Per-harmonic amplitude/phase (with delta-method SEs) from a linear fit's
    ``cos(expr)``/``sin(expr)`` coefficient pairs — the linear-circular summary
    of how strongly and at what phase each angular harmonic drives the response.
    Best-effort over hea's design column names; an unpaired term takes the
    coefficient it has. Returns ``[]`` when no harmonic columns are present."""
    names = list(fit.column_names)
    coef = dict(zip(names, np.asarray(fit.bhat.row(0), dtype=float)))
    try:
        x = np.asarray(fit.X.to_numpy(), dtype=float)
        cov = float(fit.sigma) ** 2 * _safe_inverse(x.T @ x)
        idx = {nm: i for i, nm in enumerate(names)}
    except Exception:
        cov = idx = None
    groups: dict = {}
    for nm in names:
        m = _COSSIN_RE.match(nm.replace(" ", ""))
        if m:
            groups.setdefault(m.group(2), {})[m.group(1)] = nm
    rows = []
    for expr, parts in groups.items():
        cn, sn = parts.get("cos"), parts.get("sin")
        a = coef.get(cn, 0.0) if cn else 0.0
        b = coef.get(sn, 0.0) if sn else 0.0
        amp = float(np.hypot(a, b))
        se_amp = se_phase = float("nan")
        if cov is not None and amp > 0 and cn in idx and sn in idx:
            ic, isn = idx[cn], idx[sn]
            vc, vs, cs = cov[ic, ic], cov[isn, isn], cov[ic, isn]
            se_amp = float(np.sqrt(max((a * a * vc + 2 * a * b * cs
                                        + b * b * vs) / amp**2, 0.0)))
            se_phase = float(np.sqrt(max((b * b * vc - 2 * a * b * cs
                                          + a * a * vs) / amp**4, 0.0)))
        rows.append({"term": expr, "cos": float(a), "sin": float(b),
                     "amplitude": amp, "phase": float(np.arctan2(b, a)),
                     "se_amplitude": se_amp, "se_phase": se_phase})
    return rows


def _circ_lm_lc(formulas, data) -> dict:
    """Linear response on a circular predictor: an OLS fit of the harmonic terms
    written in the formula (``y ~ cos(phi) + sin(phi) + sin(2*phi)``). Returns a
    fields dict carrying the underlying ``hea.models.lm`` (whose full interface
    CircLM delegates to) plus the harmonic amplitude/phase table and the
    least-squares fit metrics."""
    if len(formulas) != 1:
        raise ValueError(
            "circ_lm(type='lc') takes a single formula, e.g. "
            "'y ~ cos(phi) + sin(phi)'."
        )
    formula = formulas[0]
    response = formula.split("~", 1)[0].strip()
    if not response:
        raise ValueError("the formula must name the response, e.g. 'y ~ phi'.")
    fit = lm(formula, data)
    yhat = _ravel(fit.yhat)
    resid = _ravel(fit.residuals)
    rhs_vars = _circ_lm_rhs_vars([formula], data)
    var = rhs_vars[0] if len(rhs_vars) == 1 else None
    frame = None
    if var is not None:
        x = np.mod(np.asarray(data[var].to_numpy(), dtype=float), 2.0 * np.pi)
        frame = pl.DataFrame({var: x, response: yhat + resid})
    return {
        "lm": fit, "var": var, "response": response,
        "coefficients": dict(zip(fit.column_names,
                                 np.asarray(fit.bhat.row(0), dtype=float))),
        "harmonics": _circ_lm_harmonics_table(fit),
        "sigma": float(fit.sigma), "r_squared": float(fit.r_squared),
        "aic": float(fit.AIC), "bic": float(fit.BIC),
        "fitted": yhat, "residuals": resid, "frame": frame, "covariate": var,
    }


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

    Returns a :class:`CircGAM`: the fitted ``hea`` gam — every attribute and
    method (``summary()``, ``predict()``, ``AIC``, ``check()``, ``fitted``,
    ``Vp``, ``edf`` …) directly reachable, reclassed in place — plus the
    circular ``circ_check`` / ``circ_resid`` (and ``circ_plot``).

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
    fit = gam(payload, df, family=fam,
              knots=merged, method=method, **gam_kwargs)
    # reclass in place into the circular result object (the Python analog of R's
    # `class(fit) <- c("circ_gam", class(fit))`): every hea gam attribute/method
    # stays directly reachable, plus the circular circ_plot/circ_check/circ_resid.
    fit.__class__ = CircGAM
    return fit


# --------------------------------------------------------------------------- #
# Circular regression diagnostics & result objects
# --------------------------------------------------------------------------- #
def _wrap(a):
    """Wrap angle(s) to (−π, π]."""
    return np.angle(np.exp(1j * np.asarray(a, dtype=float)))


def _pvonmises(theta, mu=0.0, kappa=1.0, jmax=60, tol=1e-12):
    """von Mises CDF in the residual frame, origin at the antipode −π::

        F(d) = (d + π)/(2π) + (1/π) Σ_{j≥1} [I_j(κ)/I_0(κ)] sin(j d)/j,
        d = wrap(θ − μ)

    The analytic probability-integral transform of the von Mises residual —
    F(−π)=0, F(μ)=½, F(π)=1, so F(θ)~U(0,1) under the model. Bessel ratios use
    the exponentially scaled ``ive`` so they stay finite at large κ. Vectorised
    over θ, μ, κ; matches ``circular::pvonmises(from = μ − π)`` to machine
    precision (the cross-language quantile-residual reference)."""
    d = _wrap(np.asarray(theta, dtype=float) - np.asarray(mu, dtype=float))
    kappa = np.asarray(kappa, dtype=float)
    out = (d + np.pi) / (2.0 * np.pi)
    i0 = ive(0, kappa)
    for j in range(1, int(jmax) + 1):
        rj = ive(j, kappa) / i0
        out = out + rj * np.sin(j * d) / (j * np.pi)
        if np.max(np.abs(rj)) < tol:
            break
    return np.clip(out, 0.0, 1.0)


def _watson_u2(u):
    """Watson's U² test of uniformity for PIT values ``u`` ∈ [0, 1) — the
    rotation-invariant goodness-of-fit statistic for the circle (Kolmogorov–
    Smirnov is origin-dependent and wrong here). Returns ``{"stat", "p"}`` with
    the asymptotic upper-tail p-value; deterministic given ``u``."""
    u = np.sort(np.asarray(u, dtype=float))
    n = u.size
    i = np.arange(1, n + 1)
    ubar = float(u.mean())
    stat = float(np.sum((u - (2 * i - 1) / (2 * n)) ** 2)
                 - n * (ubar - 0.5) ** 2 + 1.0 / (12 * n))
    p = 0.0
    for m in range(1, 51):
        term = (-1) ** (m - 1) * np.exp(-2.0 * m * m * np.pi ** 2 * stat)
        p += term
        if abs(term) < 1e-12:
            break
    return {"stat": stat, "p": float(min(max(2.0 * p, 0.0), 1.0))}


class _ResidArray(np.ndarray):
    """Residual ndarray tagged with the residual ``type`` (and ``scale`` for the
    quantile residual) — the circlss ``attr(, "type")`` / ``attr(, "scale")``
    parity, surviving slicing/ufuncs via ``__array_finalize__``."""

    def __new__(cls, data, type=None, scale=None):
        obj = np.asarray(data, dtype=float).view(cls)
        obj.type = type
        obj.scale = scale
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.type = getattr(obj, "type", None)
        self.scale = getattr(obj, "scale", None)


class _CircRegressionMixin:
    """Shared circular diagnostics for :class:`CircGAM` / :class:`CircLM` — the
    object-oriented counterpart of circlss's ``circ_resid`` / ``circ_check`` S3
    generics. The public methods drive off two per-class hooks each subclass
    supplies — :meth:`_resid_parts` (the residual primitive bundle) and
    :meth:`_check_backend` (the leg-specific goodness-of-fit rows) — so there is
    no standalone dispatch function and no ``isinstance`` ladder."""

    def circ_resid(self, type="quantile", nsim=1000, scale="uniform"):
        """Circular regression residuals — the quantity every circ_check panel
        is a function of, since "observed − fitted" is undefined for two angles.

        ``type``: ``"quantile"`` (the probability-integral-transform residual,
        calibrated even when the concentration varies; default), ``"deviance"``
        (signed root of the per-observation deviance, ≈ N(0,1) under a good
        fit), ``"angular"`` (the wrapped ``y − μ̂`` ∈ (−π, π], the raw response
        residual), or ``"pearson"`` (score-standardized). ``scale`` applies to
        the quantile residual: ``"uniform"`` returns the PIT on (0, 1) (pairs
        with the Watson U² Q-Q), ``"normal"`` the Dunn–Smyth N(0, 1) residual.
        ``nsim`` is accepted for circlss API parity; pycircstat2's PIT is
        analytic for every family, so it is unused. The result is an ndarray
        carrying ``.type`` (and ``.scale`` for the quantile residual)."""
        parts = self._resid_parts()
        if type == "angular":
            r = parts["resid_response"]()
        elif type == "deviance":
            r = parts["resid_deviance"]()
        elif type == "pearson":
            fn = parts.get("resid_pearson")
            if fn is None:
                raise ValueError(
                    "Pearson residuals are not available for this fit.")
            r = fn()
        elif type == "quantile":
            r = self._quantile_resid(parts, scale)
        else:
            raise ValueError(
                "type must be 'quantile', 'deviance', 'angular' or 'pearson'; "
                f"got {type!r}.")
        return _ResidArray(np.asarray(r, dtype=float), type=type,
                           scale=scale if type == "quantile" else None)

    def _quantile_resid(self, parts, scale):
        cdf = parts.get("cdf")
        if cdf is None:
            raise ValueError(
                f"no distribution function for {parts['family_label']}; cannot "
                "form a quantile residual.")
        u = np.clip(np.asarray(cdf(), dtype=float), 1e-6, 1.0 - 1e-6)
        return norm.ppf(u) if scale == "normal" else u

    def circ_check(self, which=None, nsim=1000, rug=True, seed=None,
                   figsize=None):
        """Diagnostic-panel display (the circular ``gam.check`` / ``plot.lm``
        analogue, and the companion to ``circ_plot``): lays out the residual
        panel grid, prints the goodness-of-fit table, and returns the matplotlib
        Figure. ``which`` picks the panels (``None`` → the response-appropriate
        default ``rose``/``obsfit``/``residcov``/``qq.unif``; ``"all"`` adds the
        deviance-residual panels ``qq.norm``/``scaleloc``/``hist`` and the
        influence panel ``cook``); ``rug`` adds a covariate rug. The
        statistics are printed (Watson U² + residual location + the leg backend);
        as with ``summary`` nothing is returned but the Figure. ``nsim`` /
        ``seed`` are accepted for circlss API parity (the PIT is analytic)."""
        import matplotlib.pyplot as plt

        parts = self._resid_parts()
        keys, warn_rose = _check_keys(which, parts["response_circular"])
        if warn_rose:
            print("circ_check: panel 'rose' needs a circular response; "
                  "dropping it.")
        lev = self._leverage() if "cook" in keys else None
        if "cook" in keys and lev is None:
            print("circ_check: per-observation leverage is unavailable for this "
                  "fit; dropping the 'cook' panel.")
            keys = [k for k in keys if k != "cook"]
        ang = parts["resid_response"]()
        u = self._quantile_resid(parts, "uniform")
        watson = _watson_u2(u)
        dev = (parts["resid_deviance"]()
               if any(k in keys for k in ("qq.norm", "scaleloc", "hist", "cook"))
               else None)
        cinfo = self._check_cov() or {
            "name": "fitted direction", "values": parts["fitted_dir"],
            "circular": parts["response_circular"]}
        self._print_check_table(parts, ang, watson)

        ncol = int(np.ceil(np.sqrt(len(keys))))
        nrow = int(np.ceil(len(keys) / ncol))
        fig = plt.figure(figsize=figsize or (4.8 * ncol, 4.2 * nrow))
        for i, key in enumerate(keys):
            ax = fig.add_subplot(nrow, ncol, i + 1,
                                 projection="polar" if key == "rose" else None)
            if key == "rose":
                _panel_rose(ax, ang)
            elif key == "obsfit":
                _panel_obsfit(ax, parts)
            elif key == "residcov":
                _panel_residcov(ax, ang, cinfo, parts["response_circular"], rug)
            elif key == "qq.unif":
                _panel_qqunif(ax, u, watson)
            elif key == "qq.norm":
                _panel_qqnorm(ax, dev)
            elif key == "scaleloc":
                _panel_scaleloc(ax, dev, parts["fitted_dir"],
                                parts["response_circular"])
            elif key == "hist":
                _panel_hist(ax, dev)
            elif key == "cook":
                _panel_cook(ax, dev, lev)
        fig.tight_layout()
        return fig

    def _print_check_table(self, parts, ang, watson):
        """Print the R-style goodness-of-fit table (the ``circ_check`` header):
        the residual location, the Watson U² PIT-uniformity test, and the
        leg-specific backend rows."""
        print(f"\ncirc_check: {parts['family_label']}   n = {parts['n']}")
        if parts["response_circular"]:
            sb, cb = float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang)))
            print(f"  residual mean direction = {np.arctan2(sb, cb):+.4f} rad"
                  f"   resultant length = {np.hypot(sb, cb):.4f}")
        else:
            print(f"  residual mean = {np.mean(ang):+.4f}   "
                  f"sd = {np.std(ang, ddof=1):.4f}")
        print(f"  Watson U2 (PIT uniformity) = {watson['stat']:.4f}   "
              f"p = {watson['p']:.4f}")
        backend = self._check_backend()
        if "edf" in backend:
            print(f"  effective degrees of freedom = {backend['edf']:.3f}")
        elif "p_values" in backend and backend["p_values"] is not None:
            pv = backend["p_values"]
            print(f"  higher-order harmonic test: p(cos) = {pv[0]:.4f}   "
                  f"p(sin) = {pv[1]:.4f}")
        elif "r_squared" in backend:
            print(f"  R-squared = {backend['r_squared']:.4f}   "
                  f"residual sigma = {backend['sigma']:.4f}")
        elif "converged" in backend:
            kap = backend["kappa"]
            kstr = (f"= {kap:.3f}" if np.isscalar(kap)
                    else f"in [{kap[0]:.3f}, {kap[1]:.3f}]")
            print(f"  converged = {backend['converged']}   kappa {kstr}")

    def circ_plot(self, view="flat", n=200, se=True, rug=True, figsize=None):
        """Circular effect display — the geometry-aware counterpart to the
        per-term plots. ``view``: ``"flat"`` (default) draws one response-scale
        panel per modelled parameter against the covariate (a circular location
        broken at the ±π jump, a 2-SE band, the observed responses overlaid);
        ``"geometry"`` draws the fitted location curve on its natural surface — a
        cylinder (circular~linear), torus (circular~circular) or upright can
        (linear~circular); ``"both"`` places the surface and the location panel
        side by side. Returns the matplotlib Figure. A fit with no single
        covariate axis defers (to hea's per-term ``plot`` for a CircGAM, else a
        message pointing at ``coef``/``predict``/``summary``)."""
        import matplotlib.pyplot as plt

        kind, resp_circular, cov_circular, cov = self._geometry()
        if cov is None:
            if isinstance(self, CircGAM):
                return self.plot()
            print("circ_plot: the geometry/flat views need exactly one "
                  "covariate; this is a multi-covariate fit. Use "
                  "coef()/predict()/summary().")
            return None
        xv = np.asarray(self._model_frame()[cov].to_numpy(), dtype=float)
        if cov_circular:
            rng = (-np.pi, np.pi) if np.nanmin(xv) < 0 else (0.0, 2.0 * np.pi)
        else:
            rng = (float(np.nanmin(xv)), float(np.nanmax(xv)))
        grid = np.linspace(rng[0], rng[1], n)
        nd = pl.DataFrame({cov: grid})
        panels = self._flat_panels(grid, nd)
        if panels is None:
            print("circ_plot: this is a multi-covariate cl fit; use "
                  "coef()/predict()/summary().")
            return None
        rv = np.asarray(self._response_values(), dtype=float)
        yobs = _wrap(rv) if resp_circular else rv
        surface = {"cl": "cylinder", "cc": "torus", "lc": "can"}.get(kind)
        surf_idx = (next((i for i, p in enumerate(panels) if p.get("circular")), 0)
                    if resp_circular else 0)
        if view != "flat" and surface is None:
            print("circ_plot: no surface for this fit; drawing the flat view.")
            view = "flat"

        if view == "flat":
            npan = len(panels)
            ncol = int(np.ceil(np.sqrt(npan)))
            nrow = int(np.ceil(npan / ncol))
            fig, axes = plt.subplots(
                nrow, ncol, figsize=figsize or (5.0 * ncol, 4.0 * nrow),
                squeeze=False)
            axes = axes.ravel()
            for i, p in enumerate(panels):
                _flat_panel(axes[i], grid, p, cov, se,
                            xv if i == surf_idx else None,
                            yobs if i == surf_idx else None, rug)
            for j in range(npan, len(axes)):
                axes[j].axis("off")
            fig.tight_layout()
            return fig

        sp = panels[surf_idx]
        lo, hi = (sp.get("lo"), sp.get("hi")) if se else (None, None)
        if view == "both":
            fig = plt.figure(figsize=figsize or (12.0, 5.0))
            ax3 = fig.add_subplot(1, 2, 1, projection="3d")
            _geometry_panel(ax3, grid, sp["mid"], xv, yobs, surface, lo, hi)
            ax2 = fig.add_subplot(1, 2, 2)
            _flat_panel(ax2, grid, sp, cov, se, xv, yobs, rug)
        else:
            fig = plt.figure(figsize=figsize or (6.5, 6.0))
            ax3 = fig.add_subplot(1, 1, 1, projection="3d")
            _geometry_panel(ax3, grid, sp["mid"], xv, yobs, surface, lo, hi)
        fig.tight_layout()
        return fig


class CircGAM(_CircRegressionMixin, gam):
    """Circular-response GAM fit: the hea ``gam`` — every attribute and method
    (``summary``, ``predict``, ``fitted``, ``AIC``, ``logLik``, ``Vp``, ``edf``,
    ``influence``, ``cooks_distance``, ``check``, ``plot``, ``vis``) directly
    reachable — plus the circular-aware ``circ_check`` / ``circ_resid``.

    Returned by :func:`circ_gam`, constructed by reclassing the fitted gam in
    place (the Python analog of R's ``class(fit) <- c("circ_gam", class(fit))``),
    so it carries no ``__init__`` of its own."""

    def _response_name(self):
        fl = self.formula
        first = fl[0] if isinstance(fl, (list, tuple)) else fl
        return first.split("~", 1)[0].strip()

    def _resid_parts(self):
        fam = self.family
        fm = np.asarray(self.fitted, dtype=float)
        if fm.ndim == 1:
            fm = fm[:, None]
        y = np.asarray(self.data[self._response_name()].to_numpy(), dtype=float)
        n = y.size
        if isinstance(fam, CircularLL):
            direction = np.asarray(fam._fitted_direction(fm), dtype=float)
            if getattr(fam.dist, "name", "") == "vonmises":
                def cdf():
                    return _pvonmises(y, direction, fm[:, 1])
            else:
                params = {
                    nm: (np.mod(fm[:, j], 2.0 * np.pi)
                         if fam.links[j].name == "tanhalf" else fm[:, j])
                    for j, nm in enumerate(fam.params)
                }

                def cdf():
                    return np.clip(
                        np.asarray(fam.dist.cdf(y, **params), dtype=float),
                        0.0, 1.0)

            return {
                "y": _wrap(y), "response_circular": True,
                "fitted_dir": _wrap(direction), "n": n, "family_label": fam.name,
                "resid_response": lambda: fam.residuals(y, fm, type="response"),
                "resid_deviance": lambda: fam.residuals(y, fm, type="deviance"),
                "resid_pearson": lambda: fam.residuals(y, fm, type="pearson"),
                "cdf": cdf,
            }
        # linear-response gam (e.g. gaussian on a cyclic covariate): hea owns
        # the residuals; the PIT is the gaussian probability-integral transform.
        mean = fm[:, 0]
        sigma = np.sqrt(float(getattr(self, "scale", 1.0) or 1.0))
        return {
            "y": y, "response_circular": False, "fitted_dir": mean, "n": n,
            "family_label": getattr(fam, "name", type(fam).__name__),
            "resid_response": lambda: _ravel(self.residuals_of("response")),
            "resid_deviance": lambda: _ravel(self.residuals_of("deviance")),
            "resid_pearson": lambda: _ravel(self.residuals_of("pearson")),
            "cdf": lambda: norm.cdf((y - mean) / sigma),
        }

    def _check_backend(self):
        try:
            return {"edf": float(np.sum(np.asarray(self.edf, dtype=float)))}
        except Exception:
            return {}

    # ---- plotting hooks (the mixin's circ_plot drives off these) --------- #
    def _model_frame(self):
        return self.data

    def _response_values(self):
        return np.asarray(self.data[self._response_name()].to_numpy(),
                          dtype=float)

    def _covariate(self):
        fl = self.formula if isinstance(self.formula, (list, tuple)) else \
            [self.formula]
        resp = self._response_name()
        cols = set(self.data.columns)
        seen, out = set(), []
        for f in fl:
            rhs = f.split("~", 1)[1] if "~" in f else f
            for tok in re.findall(r"[A-Za-z_]\w*", rhs):
                if tok in cols and tok != resp and tok not in seen:
                    seen.add(tok)
                    out.append(tok)
        return out[0] if len(out) == 1 else None

    def _geometry(self):
        fam = self.family
        resp_circular = isinstance(fam, CircularLL)
        cov = self._covariate()
        fl = self.formula if isinstance(self.formula, (list, tuple)) else \
            [self.formula]
        cyclic = set()
        for f in fl:
            cyclic |= set(_cyclic_smooth_vars(
                f.split("~", 1)[1] if "~" in f else f))
        cov_circular = cov is not None and cov in cyclic
        if cov is None:
            kind = None
        elif resp_circular and not cov_circular:
            kind = "cl"
        elif resp_circular and cov_circular:
            kind = "cc"
        elif not resp_circular and cov_circular:
            kind = "lc"
        else:
            kind = "ll"
        return kind, resp_circular, cov_circular, cov

    def _flat_panels(self, grid, nd):
        fam = self.family
        if not isinstance(fam, CircularLL):
            return [{"name": self._response_name(), "circular": False,
                     "mid": _ravel(self.predict(nd, type="response")),
                     "lo": None, "hi": None}]
        pr = self.predict(nd, type="link", se_fit=True).to_numpy()
        nlp = fam.n_lp
        eta, seta = pr[:, :nlp], pr[:, nlp:2 * nlp]
        resp_fit = np.empty((nd.height, nlp))
        panels = []
        for j in range(nlp):
            li = fam.links[j].linkinv
            mid = np.asarray(li(eta[:, j]), dtype=float)
            resp_fit[:, j] = mid
            panels.append({
                "name": fam.params[j],
                "circular": fam.links[j].name == "tanhalf",
                "mid": mid,
                "lo": np.asarray(li(eta[:, j] - 2 * seta[:, j]), dtype=float),
                "hi": np.asarray(li(eta[:, j] + 2 * seta[:, j]), dtype=float),
            })
        # derived mean direction (e.g. pnlss's atan2(mu2, mu1)): a function of
        # two LPs at once, so its band is the joint delta-method interval.
        loc = fam.dist.params_by_role().get("location", [])
        if len(loc) == 2:
            i1, i2 = (fam.params.index(loc[0]), fam.params.index(loc[1]))
            mid = np.arctan2(resp_fit[:, i2], resp_fit[:, i1])
            sed = self._direction_band(nd, resp_fit[:, i1], resp_fit[:, i2])
            panels.append({"name": "direction", "circular": True, "mid": mid,
                           "lo": mid - 2 * sed, "hi": mid + 2 * sed})
        return panels

    def _direction_band(self, nd, m1, m2):
        """Joint delta-method SE for the projected-normal mean direction
        atan2(μ₂, μ₁): both components are identity-linked, so the band combines
        the lpmatrix blocks through the joint coefficient covariance Vp."""
        xp = np.asarray(self.predict(nd, type="lpmatrix"), dtype=float)
        i1, i2 = self.lpi[0], self.lpi[1]
        vp = np.asarray(self.Vp, dtype=float)
        x1, x2 = xp[:, i1], xp[:, i2]
        v11 = np.sum((x1 @ vp[np.ix_(i1, i1)]) * x1, axis=1)
        v22 = np.sum((x2 @ vp[np.ix_(i2, i2)]) * x2, axis=1)
        v12 = np.sum((x1 @ vp[np.ix_(i1, i2)]) * x2, axis=1)
        r2 = np.maximum(m1 ** 2 + m2 ** 2, 1e-8)
        return np.sqrt(np.clip(m2**2 * v11 + m1**2 * v22 - 2 * m1 * m2 * v12,
                               0.0, None)) / r2

    def _check_cov(self):
        cov = self._covariate()
        if cov is None:
            return None
        _, _, cov_circular, _ = self._geometry()
        return {"name": cov,
                "values": np.asarray(self.data[cov].to_numpy(), dtype=float),
                "circular": cov_circular}

    def _leverage(self):
        # a general-family GAM exposes no per-observation hat (mgcv/hea); the
        # principled influence summary is the k.check edf table instead, so the
        # cook panel is dropped (R-faithful).
        h = getattr(self, "hat", None)
        if h is not None and np.asarray(h).size:
            return {"h": np.asarray(h, dtype=float),
                    "p": float(np.sum(np.asarray(self.edf, dtype=float)))}
        return None


class CircLM(_CircRegressionMixin):
    """Classical circular regression result — the cl/cc/lc legs of
    :func:`circ_lm`.

    Fields are reachable both as attributes and as mapping keys (``m.mu`` or
    ``m["mu"]``, ``m.kappa``/``m["kappa"]`` …, the dict-style back-compat), and
    for the lc leg every unknown attribute delegates to the wrapped hea ``lm``
    (so ``m.bhat``, ``m.plot``, ``m.ci_bhat``, ``m.AIC`` keep working). Adds
    ``summary`` / ``predict`` / ``coef`` / ``logLik`` and the circular
    ``circ_check`` / ``circ_resid``."""

    def __init__(self, type, fields):
        self.type = type
        self._fields = dict(fields)
        for k, v in self._fields.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return self._fields[key]

    def __contains__(self, key):
        return key in self._fields

    def keys(self):
        return self._fields.keys()

    def __getattr__(self, name):
        # reached only when normal lookup fails; the lc leg delegates to its lm
        lm_fit = self.__dict__.get("lm")
        if lm_fit is not None and hasattr(lm_fit, name):
            return getattr(lm_fit, name)
        raise AttributeError(name)

    # ---- parts adapter (the mixin's residual hook) ----------------------- #
    def _resid_parts(self):
        return self._parts_vm() if self.type in ("cl", "cc") else self._parts_lc()

    def _obs_response(self):
        return np.asarray(self.frame[self.response].to_numpy(), dtype=float)

    def _parts_vm(self):
        y = self._obs_response()
        direction = np.asarray(self.fitted, dtype=float)
        kappa = np.broadcast_to(np.asarray(self.kappa, dtype=float),
                                y.shape).astype(float)
        yw, dw = _wrap(y), _wrap(direction)

        def ang():
            return _wrap(yw - dw)

        def dev():
            d = ang()
            return np.sign(np.sin(d)) * np.sqrt(
                np.clip(2.0 * kappa * (1.0 - np.cos(d)), 0.0, None))

        def pear():
            d = ang()
            safe = np.where(kappa == 0, 1.0, kappa)
            v = np.where(kappa == 0, 0.5, A1(kappa) / safe)
            return np.sin(d) / np.sqrt(v)

        return {
            "y": yw, "response_circular": True, "fitted_dir": dw, "n": y.size,
            "family_label": f"circ_lm:{self.type}",
            "resid_response": ang, "resid_deviance": dev, "resid_pearson": pear,
            "cdf": lambda: _pvonmises(y, direction, kappa),
        }

    def _parts_lc(self):
        fit = np.asarray(self.fitted, dtype=float)
        resid = np.asarray(self.residuals, dtype=float)
        sigma = float(self.sigma)
        return {
            "y": fit + resid, "response_circular": False, "fitted_dir": fit,
            "n": fit.size, "family_label": "circ_lm:lc",
            "resid_response": lambda: resid,
            "resid_deviance": lambda: resid / sigma,
            "resid_pearson": lambda: resid / sigma,
            "cdf": lambda: norm.cdf(resid / sigma),
        }

    def _check_backend(self):
        if self.type == "cc":
            return {"p_values": self._fields.get("p_values")}
        if self.type == "lc":
            return {"r_squared": float(self.r_squared), "sigma": float(self.sigma)}
        kappa = np.asarray(self.kappa, dtype=float)
        krange = (float(kappa) if kappa.size == 1
                  else (float(np.min(kappa)), float(np.max(kappa))))
        return {"converged": bool(self._fields.get("converged", True)),
                "kappa": krange}

    # ---- plotting hooks (the mixin's circ_plot drives off these) --------- #
    def _model_frame(self):
        return self.frame

    def _response_values(self):
        return np.asarray(self.frame[self.response].to_numpy(), dtype=float)

    def _geometry(self):
        return (self.type, self.type in ("cl", "cc"),
                self.type in ("cc", "lc"), self.covariate)

    def _flat_panels(self, grid, nd):
        if self.covariate is None:
            return None
        if self.type == "lc":
            return [{"name": self.response, "circular": False,
                     **self._lc_loc(grid)}]
        if self.type == "cc":
            return [{"name": self.response, "circular": True,
                     **self._cc_loc(grid)}]
        return [{"name": self.response, "circular": True, **self._cl_mu(nd)},
                {"name": "kappa", "circular": False, **self._cl_kappa(nd)}]

    def _cl_mu(self, nd):
        """Fisher–Lee mean direction μ = μ0 + 2·atan(Xβ) with its delta-method
        band: the intercept level (se_μ²) plus G Vβ Gᵀ, G = 2/(1+η²)·X."""
        xn, _ = _circ_lm_design(self.mu_formula, nd, self.response)
        eta = xn @ self.beta if xn.shape[1] else np.zeros(nd.height)
        mid = _wrap(self.mu + 2.0 * np.arctan(eta))
        v_level = self.se_mu ** 2 if self.se_mu is not None else 0.0
        vbeta = self._fields.get("Vbeta")
        if xn.shape[1] and vbeta is not None:
            g = (2.0 / (1.0 + eta ** 2))[:, None] * xn
            v_beta = np.sum((g @ vbeta) * g, axis=1)
        else:
            v_beta = 0.0
        sd = np.sqrt(np.clip(v_level + v_beta, 0.0, None))
        return {"mid": mid, "lo": mid - 2.0 * sd, "hi": mid + 2.0 * sd}

    def _cl_kappa(self, nd):
        """Concentration log κ = α + Zγ with its band on the log scale (through
        Vαγ), then exp()'d back to the response scale."""
        zn, _ = _circ_lm_design(self.kappa_formula, nd, self.response)
        if not zn.shape[1]:
            mid = np.full(nd.height, float(np.atleast_1d(self.kappa)[0]))
            if self.se_kappa is None:
                return {"mid": mid, "lo": None, "hi": None}
            sk = float(np.atleast_1d(self.se_kappa)[0])
            return {"mid": mid, "lo": mid - 2.0 * sk, "hi": mid + 2.0 * sk}
        eta = self.alpha + zn @ self.gamma
        vag = self._fields.get("Vag")
        if vag is None:
            return {"mid": np.exp(eta), "lo": None, "hi": None}
        z1 = np.column_stack([np.ones(nd.height), zn])
        se_eta = np.sqrt(np.clip(np.sum((z1 @ vag) * z1, axis=1), 0.0, None))
        return {"mid": np.exp(eta), "lo": np.exp(eta - 2.0 * se_eta),
                "hi": np.exp(eta + 2.0 * se_eta)}

    def _cc_loc(self, grid):
        """Harmonic circular–circular location atan2(ŝ, ĉ) with the delta-method
        band on the two OLS predictions, including their residual cross-
        covariance (homoskedastic SUR)."""
        nd = pl.DataFrame({self.var: np.mod(grid, 2.0 * np.pi)})
        pc = self.cos_lm.predict(nd, se_fit=True)
        ps = self.sin_lm.predict(nd, se_fit=True)
        cf, sf = pc["fit"].to_numpy(), ps["fit"].to_numpy()
        vc, vs = pc["se.fit"].to_numpy() ** 2, ps["se.fit"].to_numpy() ** 2
        sc = float(self.cos_lm.sigma)
        rc, rs = _ravel(self.cos_lm.residuals), _ravel(self.sin_lm.residuals)
        shp = np.asarray(self.cos_lm.X.to_numpy()).shape
        scs = float(rc @ rs) / max(shp[0] - shp[1], 1)
        cv = (vc / sc ** 2) * scs                       # leverage h = vc/σ_cos²
        r2 = np.maximum(cf ** 2 + sf ** 2, 1e-8)
        sd = np.sqrt(np.clip(
            (sf**2 * vc + cf**2 * vs - 2 * sf * cf * cv) / r2 ** 2, 0.0, None))
        mid = np.arctan2(sf, cf)
        return {"mid": mid, "lo": mid - 2.0 * sd, "hi": mid + 2.0 * sd}

    def _lc_loc(self, grid):
        """Harmonic linear–circular mean over the cyclic covariate: the OLS
        prediction band directly."""
        nd = pl.DataFrame({self.var: np.mod(grid, 2.0 * np.pi)})
        pr = self.lm.predict(nd, se_fit=True)
        fit, se = pr["fit"].to_numpy(), pr["se.fit"].to_numpy()
        return {"mid": fit, "lo": fit - 2.0 * se, "hi": fit + 2.0 * se}

    def _check_cov(self):
        if self.covariate is None:
            return None
        return {"name": self.covariate,
                "values": np.asarray(self.frame[self.covariate].to_numpy(),
                                     dtype=float),
                "circular": self.type in ("cc", "lc")}

    def _leverage(self):
        """Per-observation leverage for the influence panel: lc/cc are ordinary
        least squares (the hat diagonal; the cc cos/sin share one design), cl
        reconstructs the converged IRLS hat from the stored design and
        coefficient covariance."""
        if self.type == "cl":
            return self._cl_leverage()
        x = np.asarray((self.lm if self.type == "lc" else self.cos_lm).X.to_numpy(),
                       dtype=float)
        hat = np.sum((x @ _safe_inverse(x.T @ x)) * x, axis=1)
        return {"h": np.clip(hat, 0.0, 1.0), "p": float(x.shape[1])}

    def _cl_leverage(self):
        fr = self.frame
        if self.model in ("mean", "mixed") and self._fields.get("Vbeta") is not None:
            x, _ = _circ_lm_design(self.mu_formula, fr, self.response)
            if not x.shape[1]:
                return None
            eta = x @ self.beta
            g = (2.0 / (1.0 + eta ** 2))[:, None] * x
            kap = np.broadcast_to(np.asarray(self.kappa, dtype=float), eta.shape)
            h = (kap * A1(kap)) * np.sum((g @ self._fields["Vbeta"]) * g, axis=1)
            return {"h": np.asarray(h, dtype=float), "p": float(g.shape[1])}
        if self.model == "kappa" and self._fields.get("Vag") is not None:
            z, _ = _circ_lm_design(self.kappa_formula, fr, self.response)
            z1 = np.column_stack([np.ones(z.shape[0]), z])
            kap = np.broadcast_to(np.asarray(self.kappa, dtype=float),
                                  (z1.shape[0],))
            h = (kap ** 2 * A1prime(kap)) * np.sum(
                (z1 @ self._fields["Vag"]) * z1, axis=1)
            return {"h": np.asarray(h, dtype=float), "p": float(z1.shape[1])}
        return None

    # ---- result methods -------------------------------------------------- #
    def coef(self):
        if "coefficients" in self._fields:
            return self._fields["coefficients"]
        return {"beta": self.beta, "alpha": self.alpha, "gamma": self.gamma}

    def predict(self, newdata=None, type="direction"):
        """Predicted values. ``newdata=None`` returns the fitted values. cc/lc
        rebuild on the angular covariate; cl returns the mean ``"direction"`` or
        the ``"kappa"`` concentration."""
        if newdata is None:
            return self.fitted
        nd = _to_polars(newdata)
        if self.type == "lc":
            return self.lm.predict(nd)
        if self.type == "cc":
            cf = _ravel(self.cos_lm.predict(nd))
            sf = _ravel(self.sin_lm.predict(nd))
            return np.mod(np.arctan2(sf, cf), 2.0 * np.pi)
        if type == "kappa":
            if self.model == "mean":
                return np.full(nd.height, float(self.kappa))
            z, _ = _circ_lm_design(self.kappa_formula, nd, self.response)
            return np.exp(self.alpha + z @ self.gamma)
        if self.model == "kappa":
            return np.full(nd.height, np.mod(self.mu, 2.0 * np.pi))
        xn, _ = _circ_lm_design(self.mu_formula, nd, self.response)
        return np.mod(self.mu + 2.0 * np.arctan(xn @ self.beta), 2.0 * np.pi)

    def logLik(self):
        if self.type == "cc":
            raise ValueError(
                "type='cc' is two separate least-squares fits and has no single "
                "log-likelihood.")
        return float(self.loglik)

    @staticmethod
    def _print_coefmat(est, se, names):
        """R ``printCoefmat``-style coefficient table — Estimate / Std. Error /
        z value / Pr(>|z|) (normal approximation) with significance codes."""
        est = np.atleast_1d(np.asarray(est, dtype=float))
        se = np.atleast_1d(np.asarray(se, dtype=float))
        names = [str(nm) for nm in names]
        w = max([11, *(len(nm) for nm in names)])
        print(f"{'':<{w}}  {'Estimate':>11}  {'Std. Error':>11}  "
              f"{'z value':>9}  {'Pr(>|z|)':>10}")
        for nm, e, s in zip(names, est, se):
            z = e / s if s and np.isfinite(s) and s > 0 else np.nan
            p = 2.0 * float(norm.cdf(-abs(z))) if np.isfinite(z) else np.nan
            stars = significance_code(p) if np.isfinite(p) else ""
            print(f"{nm:<{w}}  {e:>11.5f}  {s:>11.5f}  {z:>9.3f}  "
                  f"{p:>10.3g} {stars}")

    def _print_harmonics(self):
        print("\nHarmonic amplitude / phase:")
        hdr = (f"{'term':<16}  {'amplitude':>10}  {'se':>9}  "
               f"{'phase':>10}  {'se':>9}")
        print(hdr)
        print("-" * len(hdr))
        for h in self.harmonics:
            print(f"{h['term']:<16}  {h['amplitude']:>10.4f}  "
                  f"{h['se_amplitude']:>9.4f}  {h['phase']:>10.4f}  "
                  f"{h['se_phase']:>9.4f}")
        print("Phase in radians; SEs from the delta method on the (cos, sin) "
              "coefficients.")

    def summary(self):
        """Print the regression summary in the R ``print.circ_lm`` style shared
        with the circlss sibling package — the Fisher–Lee mean & log-κ
        coefficient tables (cl), the cos/sin coefficient matrix with ρ, residual
        κ and the higher-order test (cc), or hea's R-style least-squares block
        with the harmonic amplitude/phase table (lc). Prints only; the values
        are reachable as attributes (``m.mu``, ``m.kappa``, ``m.aic`` …)."""
        if self.type == "cl":
            head = {
                "mean": "Circular-linear regression (Fisher–Lee), mean direction",
                "kappa": "Circular-linear regression (Fisher–Lee), concentration",
                "mixed": "Circular-linear regression (Fisher–Lee), mean and "
                         "concentration",
            }[self.model]
            print(f"\n{head}\n")
            if self.model in ("mean", "mixed"):
                print("Mean direction   mu = mu0 + 2*atan(X beta):")
                self._print_coefmat(self.beta, self.se_beta, self.mu_terms)
                print()
            if self.model in ("kappa", "mixed"):
                print("Concentration   log(kappa) = alpha + Z gamma:")
                self._print_coefmat(
                    np.concatenate([[self.alpha], np.atleast_1d(self.gamma)]),
                    np.concatenate([[self.se_alpha],
                                    np.atleast_1d(self.se_gamma)]),
                    ["(Intercept)", *self.kappa_terms])
                print()
            kap = np.asarray(self.kappa, dtype=float)
            if kap.size == 1:
                sk = (float(np.atleast_1d(self.se_kappa)[0])
                      if self.se_kappa is not None else float("nan"))
                kline = f"kappa: {float(kap):.4f} ({sk:.4f})"
            else:
                kline = (f"kappa: {kap.min():.4f} to {kap.max():.4f} "
                         "(per observation)")
            print(f"mu0: {self.mu:.4f} ({self.se_mu:.4f})    {kline}")
            print(f"logLik: {self.loglik:.4f}   AIC: {self.aic:.4f}   "
                  f"BIC: {self.bic:.4f}   n: {self.n}")
            if not self._fields.get("converged", True):
                print("** did not converge **")
            print("\nSignif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 "
                  "' ' 1")
            print("p-values use the normal approximation.")
        elif self.type == "cc":
            print(f"\nCircular-circular regression (harmonic, order {self.order})")
            print(f"  {self.response} ~ {self.var}   n = {self.n}\n")
            print("Coefficients:")
            labels = [_clean_harmonic_label(nm) for nm in self.cos_lm.column_names]
            cos = np.atleast_1d(np.asarray(self.coefficients["cos"], dtype=float))
            sin = np.atleast_1d(np.asarray(self.coefficients["sin"], dtype=float))
            w = max([11, *(len(nm) for nm in labels)])
            print(f"{'':<{w}}  {'cos':>11}  {'sin':>11}")
            for nm, c, s in zip(labels, cos, sin):
                print(f"{nm:<{w}}  {c:>11.5f}  {s:>11.5f}")
            print(f"\nrho: {self.rho:.4f}    residual kappa: {self.kappa:.4f} "
                  f"(A_k = {self.A_k:.4f})")
            print(f"Higher-order test p-values:  cos = {self.p_values[0]:.4f}, "
                  f"sin = {self.p_values[1]:.4f}")
            print("Higher-order terms not significant at the 0.05 level."
                  if np.all(np.asarray(self.p_values) > 0.05)
                  else "Higher-order terms significant at the 0.05 level.")
        else:  # lc — hea's R-style least-squares block + the harmonic table
            print("\nLinear-circular regression (harmonic)")
            print(f"  {self.response} ~ {self.var}   n = {self.n}\n")
            # hea's SummaryLm is the full R summary.lm block (residual
            # quantiles, coefficient table, residual SE, R², F, AIC/BIC); the
            # harmonic amplitude/phase table is the linear-circular addendum.
            print(self.lm.summary())
            if self.harmonics:
                self._print_harmonics()

    def __repr__(self):
        if self.type == "cl":
            return (f"CircLM(cl, model={self.model!r}, mu={self.mu:.4f}, "
                    f"n={self.n})")
        if self.type == "cc":
            return (f"CircLM(cc, {self.response} ~ {self.var}, "
                    f"order={self.order}, rho={self.rho:.4f}, n={self.n})")
        return (f"CircLM(lc, {self.response} ~ {self.var}, "
                f"R²={self.r_squared:.4f})")


# --------------------------------------------------------------------------- #
# Plotting primitives (matplotlib). Module-private and grouped so a later pass
# can lift them wholesale into visualization.py; circ_plot drives them off the
# per-class _geometry()/_flat_panels() hooks.
# --------------------------------------------------------------------------- #
def _break_wrap(v):
    """Insert NaN at each ±π branch jump of a circular curve (|Δ| > π), so a
    line/band plotted against the covariate breaks cleanly instead of drawing a
    vertical streak across the wrap."""
    if v is None:
        return None
    v = np.array(v, dtype=float)
    v[1:][np.abs(np.diff(v)) > np.pi] = np.nan
    return v


def _band_fill(ax, grid, lo, hi):
    """A translucent SE band that breaks at the NaN gaps (the ±π wraps)."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ok = np.isfinite(lo) & np.isfinite(hi)
    ax.fill_between(grid, lo, hi, where=ok, color="steelblue", alpha=0.25,
                    linewidth=0.0, interpolate=False)


def _flat_panel(ax, grid, panel, xlab, se, xobs, yobs, rug):
    """One response-scale panel against the covariate: a circular location is
    broken at the ±π jump with ``ylim=(-π, π)``; the 2-SE band is a broken
    shadow; the observed responses (and an optional rug) overlay it."""
    mid = np.asarray(panel["mid"], dtype=float)
    lo, hi = panel.get("lo"), panel.get("hi")
    if panel.get("circular"):
        mid, lo, hi = _break_wrap(mid), _break_wrap(lo), _break_wrap(hi)
        ylim = [-np.pi, np.pi]
    else:
        stack = [mid] + [np.asarray(v, float) for v in (lo, hi) if v is not None]
        cat = np.concatenate(stack)
        ylim = [float(np.nanmin(cat)), float(np.nanmax(cat))]
    if yobs is not None and len(yobs):
        ylim = [min(ylim[0], float(np.min(yobs))),
                max(ylim[1], float(np.max(yobs)))]
    if xobs is not None and yobs is not None:
        ax.scatter(xobs, yobs, s=6, c="black", alpha=0.25, edgecolors="none")
    if se and lo is not None and hi is not None:
        _band_fill(ax, grid, lo, hi)
    ax.plot(grid, mid, color="steelblue", lw=2)
    if rug and xobs is not None:
        ax.plot(xobs, np.full(len(xobs), ylim[0]), "|", color="black",
                alpha=0.3, markersize=6)
    ax.set_xlabel(xlab)
    ax.set_ylabel(panel["name"])
    ax.set_title(panel["name"])
    ax.set_ylim(*ylim)


def _coord(c):
    """Identity covariate/response coordinate map (the surface uses the value
    as-is, e.g. an angle around a ring or tube)."""
    return np.asarray(c, dtype=float)


def _surface_maps(kind, grid, xobs, yspan):
    """``(to_u, to_v, xyz, mesh)`` for a regression geometry surface — the
    covariate→u and response→v coordinate maps, the ``xyz(u, v)`` embedding, and
    the wireframe mesh. cylinder (c~l): response angle wraps the tube, covariate
    runs the axis; torus (c~c): covariate around the ring, response around the
    tube; can (l~c): cyclic covariate wraps the ring, linear response is the
    height."""
    if kind == "torus":
        big_r, r = 2.0, 0.82

        def xyz(u, v):
            u, v = np.asarray(u, float), np.asarray(v, float)
            return ((big_r + r * np.cos(v)) * np.cos(u),
                    (big_r + r * np.cos(v)) * np.sin(u), r * np.sin(v))

        to_u = to_v = _coord
        uu, vv = np.linspace(-np.pi, np.pi, 49), np.linspace(-np.pi, np.pi, 37)
    elif kind == "cylinder":
        rho, axlen = 0.95, 3.2
        ref = np.concatenate([np.asarray(grid, float), np.asarray(xobs, float)])
        lo, hi = float(np.nanmin(ref)), float(np.nanmax(ref))
        span = (hi - lo) or 1.0

        def to_u(c):
            return (np.asarray(c, float) - lo) / span * 2 * axlen - axlen

        def xyz(u, v):
            u, v = np.asarray(u, float), np.asarray(v, float)
            return (u, rho * np.cos(v), rho * np.sin(v))

        to_v = _coord
        uu, vv = np.linspace(-axlen, axlen, 25), np.linspace(-np.pi, np.pi, 37)
    else:  # can
        r, height = 1.0, 1.5
        ys = (np.asarray(yspan, float) if yspan is not None and len(yspan)
              else np.array([-1.0, 1.0]))
        ylo, yhi = float(np.nanmin(ys)), float(np.nanmax(ys))
        yspw = (yhi - ylo) or 1.0

        def to_v(c):
            return (np.asarray(c, float) - ylo) / yspw * 2 * height - height

        def xyz(u, v):
            u, v = np.asarray(u, float), np.asarray(v, float)
            return (r * np.cos(u), r * np.sin(u), v)

        to_u = _coord
        uu, vv = np.linspace(-np.pi, np.pi, 49), np.linspace(-height, height, 19)
    grid_u, grid_v = np.meshgrid(uu, vv)
    return to_u, to_v, xyz, xyz(grid_u, grid_v)


def _geometry_panel(ax, grid, zv, xobs, yobs, surface, lo=None, hi=None,
                    main=None):
    """Draw the fitted location curve on its natural surface (the 3-D
    counterpart of the flat location panel): wireframe canvas, optional 2-SE
    ribbon, the fitted curve, and the observed points."""
    yspan = None
    if surface == "can":
        parts = [np.asarray(zv, float)]
        for v in (yobs, lo, hi):
            if v is not None:
                parts.append(np.asarray(v, float))
        yspan = np.concatenate(parts)
    ref_x = xobs if xobs is not None else grid
    to_u, to_v, xyz, (mx, my, mz) = _surface_maps(surface, grid, ref_x, yspan)
    ax.plot_wireframe(mx, my, mz, color="0.85", linewidth=0.4)
    if lo is not None and hi is not None:
        band_u = np.vstack([to_u(grid), to_u(grid)])
        band_v = np.vstack([to_v(lo), to_v(hi)])
        bx, by, bz = xyz(band_u, band_v)
        ax.plot_surface(bx, by, bz, color="#c0392b", alpha=0.15, linewidth=0,
                        shade=False)
    cx, cy, cz = xyz(to_u(grid), to_v(zv))
    ax.plot(cx, cy, cz, color="#c0392b", lw=3)
    if xobs is not None and yobs is not None:
        ox, oy, oz = xyz(to_u(xobs), to_v(yobs))
        ax.scatter(ox, oy, oz, c="#1f4e79", s=8, alpha=0.6)
    ax.set_axis_off()
    ax.set_title(main or {
        "torus": "torus · circular–circular",
        "cylinder": "cylinder · circular–linear",
        "can": "can · linear–circular"}[surface])
    ax.view_init(elev=22, azim=-60)


# ---- circ_check diagnostic-panel drawers ----------------------------------- #
def _check_keys(which, response_circular):
    """Resolve the circ_check panel keys: ``None`` → the response-appropriate
    default, ``"all"`` → every panel, else the given key(s) (validated). ``rose``
    is dropped for a linear response (it needs an angular residual)."""
    known = ["rose", "obsfit", "residcov", "qq.unif", "qq.norm", "scaleloc",
             "hist", "cook"]
    default = ["rose", "obsfit", "residcov", "qq.unif"]
    if which is None:
        keys = list(default)
    elif which == "all":
        keys = list(known)
    elif isinstance(which, str):
        keys = [which]
    else:
        keys = list(which)
    bad = [k for k in keys if k not in known]
    if bad:
        raise ValueError(
            f"unknown panel key(s): {bad}. Available: {known} (or 'all').")
    warn_rose = (not response_circular and which is not None
                 and which != "all" and "rose" in keys)
    if not response_circular:
        keys = [k for k in keys if k != "rose"]
    if not keys:
        raise ValueError("no panels to draw.")
    return keys, warn_rose


def _panel_rose(ax, ang, nbins=24):
    """Rose diagram of the angular residuals (polar axis): equal-width sectors
    with radius ∝ √count (so sector area encodes frequency); the firebrick arrow
    is the residual mean resultant. One tight wedge at 0 is a good fit."""
    ang = _wrap(ang)
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    counts, _ = np.histogram(ang, bins=edges)
    rad = np.sqrt(counts / counts.max()) if counts.max() > 0 else counts.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centers, rad, width=2 * np.pi / nbins, color="steelblue", alpha=0.5,
           edgecolor="steelblue", linewidth=0.6)
    sb, cb = float(np.mean(np.sin(ang))), float(np.mean(np.cos(ang)))
    rbar = np.hypot(sb, cb)
    if rbar > 1e-8:
        ax.annotate("", xy=(np.arctan2(sb, cb), rbar), xytext=(0, 0),
                    arrowprops=dict(color="firebrick", lw=2, arrowstyle="-|>"))
    ax.set_yticklabels([])
    ax.set_title("angular residuals")


def _panel_obsfit(ax, parts):
    """Observed vs fitted. Circular: the wrapped diagonal and its ±2π copies are
    perfect calibration, so off-diagonal mass shows where on the circle the fit
    fails. Linear: the ordinary scatter with y = x."""
    y, f = parts["y"], parts["fitted_dir"]
    ax.scatter(f, y, s=6, c="black", alpha=0.4, edgecolors="none")
    if parts["response_circular"]:
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect("equal")
        for off in (-2 * np.pi, 0.0, 2 * np.pi):
            ax.plot([-np.pi, np.pi], [-np.pi + off, np.pi + off],
                    color="steelblue", lw=1, ls="-" if off == 0 else "--")
        ax.set_xlabel("fitted direction")
    else:
        lim = [float(min(np.min(f), np.min(y))), float(max(np.max(f), np.max(y)))]
        ax.plot(lim, lim, color="steelblue", lw=1)
        ax.set_xlabel("fitted")
    ax.set_ylabel("observed")
    ax.set_title("observed vs fitted")


def _panel_residcov(ax, resid, cinfo, resid_circular, rug):
    """Residual vs covariate — leftover trend means missed structure (add a
    harmonic, or raise the basis dimension). A cyclic covariate gets a circular
    x-range."""
    x = np.asarray(cinfo["values"], dtype=float)
    ax.scatter(x, resid, s=6, c="black", alpha=0.4, edgecolors="none")
    ax.axhline(0.0, color="steelblue", lw=1)
    if cinfo["circular"]:
        ax.set_xlim((-np.pi, np.pi) if np.nanmin(x) < 0 else (0.0, 2 * np.pi))
    if rug:
        ax.plot(x, np.full(x.size, ax.get_ylim()[0]), "|", color="black",
                alpha=0.3, markersize=5)
    ax.set_xlabel(cinfo["name"])
    ax.set_ylabel("angular residual" if resid_circular else "residual")
    ax.set_title("residual vs covariate")


def _panel_qqunif(ax, u, watson):
    """Quantile-residual uniform Q-Q: ordered PIT residuals against (i−½)/n on
    the unit square. Points on the line ⇒ calibrated; the corner carries the
    Watson U² statistic and its uniformity p-value."""
    us = np.sort(u)
    n = us.size
    pp = (np.arange(1, n + 1) - 0.5) / n
    ax.scatter(pp, us, s=6, c="black", alpha=0.5, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="steelblue", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("theoretical U(0, 1) quantile")
    ax.set_ylabel("ordered PIT residual")
    ax.set_title("quantile-residual Q-Q")
    ax.text(0.04, 0.92, f"Watson U2 = {watson['stat']:.3f}\np = {watson['p']:.3f}",
            transform=ax.transAxes, fontsize=8, va="top")


def _panel_qqnorm(ax, dev):
    """Normal Q-Q of the deviance residuals — holds on the circle because the
    deviance residual is constructed ≈ N(0, 1)."""
    from scipy.stats import probplot
    probplot(np.asarray(dev, dtype=float), dist="norm", plot=ax)
    ax.set_title("deviance-residual normal Q-Q")


def _panel_scaleloc(ax, dev, fitted, resp_circular):
    """Scale-location: √|deviance residual| vs the fitted location — a trend is
    the circular failure mode (the dispersion/concentration model is wrong)."""
    rs = np.sqrt(np.abs(dev))
    ax.scatter(fitted, rs, s=6, c="black", alpha=0.4, edgecolors="none")
    if resp_circular:
        ax.set_xlim(-np.pi, np.pi)
    ax.set_xlabel("fitted direction" if resp_circular else "fitted")
    ax.set_ylabel(r"$\sqrt{|\mathrm{deviance\ residual}|}$")
    ax.set_title("scale-location")


def _panel_hist(ax, dev):
    """Histogram of the deviance residuals with the standard-normal reference."""
    dev = np.asarray(dev, dtype=float)
    ax.hist(dev, bins="fd", density=True, color="steelblue", alpha=0.4,
            edgecolor="white")
    xs = np.linspace(float(dev.min()), float(dev.max()), 200)
    ax.plot(xs, norm.pdf(xs), color="firebrick", lw=1.5)
    ax.set_xlabel("deviance residual")
    ax.set_title("deviance-residual histogram")


def _panel_cook(ax, dev, lev):
    """Residuals vs leverage with Cook's-distance contours — a high-leverage
    point with a large residual swings the fit. Standardized deviance residual
    dev/√(1−h); points past 4/n are labelled by index."""
    h = np.asarray(lev["h"], dtype=float)
    p = float(lev["p"])
    rstd = dev / np.sqrt(np.clip(1.0 - h, 1e-8, None))
    cooks = rstd ** 2 * h / (p * np.clip(1.0 - h, 1e-8, None))
    ax.scatter(h, rstd, s=6, c="black", alpha=0.45, edgecolors="none")
    ax.axhline(0.0, color="steelblue", ls="--", lw=1)
    flagged = np.where(cooks > 4.0 / h.size)[0]
    for i in flagged:
        ax.annotate(str(i), (h[i], rstd[i]), fontsize=6, color="firebrick")
    ax.set_xlabel("leverage")
    ax.set_ylabel("std. deviance residual")
    ax.set_title("residuals vs leverage")
