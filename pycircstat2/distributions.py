import types
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, minimize_scalar, brentq, root_scalar
from scipy.special import beta as beta_fn
from scipy.special import (
    gamma,
    i0,
    i0e,
    i1,
    ndtr,
    ndtri,
    iv,
    betainc,
    betaincinv,
    gammaln,
    digamma,
    lpmv,
    logsumexp,
)
from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import rv_continuous_frozen

from .descriptive import circ_kappa, circ_mean_and_r
from .utils import angmod

__all__ = [
    "circularuniform",
    "triangular",
    "cardioid",
    "cartwright",
    "wrapnorm",
    "wrapcauchy",
    "vonmises",
    "vonmises_flattopped",
    "jonespewsey",
    "jonespewsey_sineskewed",
    "jonespewsey_asym",
    "inverse_batschelet",
    "wrapstable",
    "katojones",
]

INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

_VMFT_MIN_GRID = 512
_VMFT_MAX_GRID = 8192
_VMFT_GRID_BASE = 64.0
_VMFT_GRID_SHARPNESS = 12.0
_VMFT_KAPPA_TOL = 1e-9
_VMFT_KAPPA_UPPER = 1e3
_VMFT_ENV_MIN_KAPPA = 1e-6
_VMFT_ACCEPT_EPS = 1e-12
_VMFT_NEWTON_MAXITER = 50
_VMFT_NEWTON_TOL = 1e-12
_VMFT_NEWTON_WIDTH_TOL = 1e-10

_INVBAT_KAPPA_TOL = 1e-9
_INVBAT_KAPPA_UPPER = 700.0
_INVBAT_NUMERIC_GRID = 4096
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


class CircularContinuous(rv_continuous):
    """Base class for circular distributions with fixed loc=0 and scale=1."""

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
        """Circular mean direction μ = arg(m₁)."""
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

    cdf(x, rho)
        Cumulative distribution function.

    ppf(q, rho)
        Closed-form quantile (inverse CDF).
        
    rvs(rho, size=None, random_state=None)
        Random variates via inverse-transform using the closed-form quantile.

    fit(data)
        Fit the distribution to the data and return the parameter (rho).

    Notes
    -----
    Implementation based on Section 2.2.3 of Jammalamadaka & SenGupta (2001)
    """

    def _argcheck(self, rho):
        return 0 <= rho <= 4 / np.pi**2

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

    def fit(self, data, *, weights=None, method="mle", return_info=False):
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
        x = np.asarray(data, dtype=float)
        x = np.mod(x, 2*np.pi)

        if weights is None:
            w = np.ones_like(x, dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("weights must be nonnegative")
            # broadcast
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
            return rho_hat, {"loglik": ll, "se": se, "n_effective": n_eff, "method": "mle", "converged": converged}
        return rho_hat


triangular = triangular_gen(name="triangular")


class cardioid_gen(CircularContinuous):
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
    Implementation based on Section 4.3.4 of Pewsey et al. (2014).
    """

    def _argcheck(self, mu, rho):
        return 0 <= mu <= np.pi * 2 and 0 <= rho <= 0.5

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


class cartwright_gen(CircularContinuous):
    """Cartwright's Power-of-Cosine Distribution

    ![cartwright](../images/circ-mod-cartwright.png)


    Methods
    -------
    pdf(x, mu, zeta)
        Probability density function.

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
    Implementation based on Section 4.3.5 of Pewsey et al. (2014)
    """

    def _argcheck(self, mu, zeta):
        return 0 <= mu <= 2 * np.pi and zeta > 0

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
        return (
            (2 ** (-1 + 1 / zeta) * (gamma(1 + 1 / zeta)) ** 2)
            * (1 + np.cos(x - mu)) ** (1 / zeta)
            / (np.pi * gamma(1 + 2 / zeta))
        )

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

        mu_arr = np.asarray(mu, dtype=float)
        zeta_arr = np.asarray(zeta, dtype=float)
        if mu_arr.size != 1 or zeta_arr.size != 1:
            raise ValueError("cartwright parameters must be scalar-valued.")

        mu_val = float(mu_arr.reshape(-1)[0])
        zeta_val = float(zeta_arr.reshape(-1)[0])
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
            return 1.0 if np.isclose(float(wrapped), 2.0 * np.pi) else value

        result = cdf.reshape(arr.shape)
        result[np.isclose(arr, 2.0 * np.pi)] = 1.0
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


class wrapnorm_gen(CircularContinuous):
    """Wrapped Normal Distribution

    ![wrapnorm](../images/circ-mod-wrapnorm.png)

    Methods
    -------
    pdf(x, mu, rho)
        Probability density function.

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
    Implementation based on Section 4.3.7 of Pewsey et al. (2014)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._series_window_cache = {}

    def _argcheck(self, mu, rho):
        return 0 <= mu <= np.pi * 2 and 0 < rho < 1

    def _pdf(self, x, mu, rho):
        return (
            1
            + 2
            * np.sum([rho ** (p**2) * np.cos(p * (x - mu)) for p in range(1, 30)], 0)
        ) / (2 * np.pi)

    def pdf(self, x, mu, rho, *args, **kwargs):
        r"""
        Probability density function of the Wrapped Normal distribution.

        $$
        f(\theta) = \frac{1}{2\pi} \left(1 + 2\sum_{p=1}^{\infty} \rho^{p^2} \cos(p(\theta - \mu))\right)
        $$

        , here we approximate the infinite sum by summing the first 30 terms.

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

        mu_arr = np.asarray(mu, dtype=float)
        rho_arr = np.asarray(rho, dtype=float)
        if mu_arr.size != 1 or rho_arr.size != 1:
            raise ValueError("wrapnorm parameters must be scalar-valued.")

        mu_val = float(mu_arr.reshape(-1)[0])
        rho_val = float(rho_arr.reshape(-1)[0])
        two_pi = 2.0 * np.pi

        if rho_val <= 1e-12:
            uniform = flat / two_pi
            if arr.ndim == 0:
                value = float(uniform[0])
                return 1.0 if np.isclose(float(wrapped), two_pi) else value
            result = uniform.reshape(arr.shape)
            result[np.isclose(arr, two_pi)] = 1.0
            return result

        rho_clipped = np.clip(rho_val, np.finfo(float).tiny, 1.0 - 1e-15)
        sigma = float(np.sqrt(-2.0 * np.log(rho_clipped)))

        cdf_flat, _ = self._wrapnorm_cdf_pdf(flat, mu_val, sigma)
        if arr.ndim == 0:
            value = float(cdf_flat.reshape(-1)[0])
            return 1.0 if np.isclose(float(wrapped), two_pi) else value

        result = cdf_flat.reshape(arr.shape)
        result[np.isclose(arr, two_pi)] = 1.0
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


wrapnorm = wrapnorm_gen(name="wrapped_normal")


class wrapcauchy_gen(CircularContinuous):
    """Wrapped Cauchy Distribution.

    ![wrapcauchy](../images/circ-mod-wrapcauchy.png)

    Methods
    -------
    pdf(x, mu, rho)
        Probability density function.

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
    Implementation based on Section 4.3.6 of Pewsey et al. (2014).
    """

    def _argcheck(self, mu, rho):
        return 0 <= mu <= np.pi * 2 and 0 <= rho < 1

    def _pdf(self, x, mu, rho):
        return (1 - rho**2) / (2 * np.pi * (1 + rho**2 - 2 * rho * np.cos(x - mu)))

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
        return np.log(np.clip(self._pdf(x, mu, rho), 1e-16, None))

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

    def _cdf(self, x, mu, rho):
        wrapped = self._wrap_angles(x)
        arr = np.asarray(wrapped, dtype=float)
        flat = arr.reshape(-1)

        mu_arr = np.asarray(mu, dtype=float)
        if mu_arr.size != 1:
            raise ValueError("wrapcauchy parameters must be scalar-valued.")
        mu_val = float(mu_arr.reshape(-1)[0])
        rho_arr = np.asarray(rho, dtype=float)
        if rho_arr.size != 1:
            raise ValueError("wrapcauchy parameters must be scalar-valued.")
        rho_val = float(rho_arr.reshape(-1)[0])
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
            return 1.0 if np.isclose(float(wrapped), 2.0 * np.pi) else value
        reshaped = cdf.reshape(arr.shape)
        reshaped[np.isclose(arr, 2.0 * np.pi)] = 1.0
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
            denom = np.clip(1.0 + rho_param**2 - 2.0 * rho_param * np.cos(x - mu_param), 1e-15, None)
            log_pdf = np.log1p(-rho_param**2) - np.log(2.0 * np.pi) - np.log(denom)
            value = -np.sum(w * log_pdf)
            return float(value)

        def grad(params):
            mu_param, rho_param = params
            denom = np.clip(1.0 + rho_param**2 - 2.0 * rho_param * np.cos(x - mu_param), 1e-15, None)
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


class vonmises_gen(CircularContinuous):
    """Von Mises Distribution

    ![vonmises](../images/circ-mod-vonmises.png)

    Methods
    -------
    pdf(x, mu, kappa)
        Probability density function.

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
    - Section 4.3.8 of Pewsey et al. (2014)

    """

    _freeze_doc = """
    Freeze the distribution with specific parameters.

    Parameters
    ----------
    mu : float
        The mean direction of the distribution (0 <= mu <= 2*pi).
    kappa : float
        The concentration parameter of the distribution (kappa > 0).

    Returns
    -------
    rv_frozen : rv_frozen instance
        The frozen distribution instance with fixed parameters.
    """

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)

    __call__.__doc__ = _freeze_doc

    def _argcheck(self, mu, kappa):
        return 0 <= mu <= np.pi * 2 and kappa > 0

    def _pdf(self, x, mu, kappa):
        return np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))

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
            The concentration parameter of the distribution (kappa > 0).

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """
        return super().pdf(x, mu, kappa, *args, **kwargs)

    def _logpdf(self, x, mu, kappa):
        return kappa * np.cos(x - mu) - np.log(2 * np.pi * i0(kappa))

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
            The concentration parameter of the distribution (kappa > 0).

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, mu, kappa, *args, **kwargs)

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
                return 1.0 if np.isclose(float(wrapped), two_pi) else value
            result = uniform.reshape(arr.shape)
            result[np.isclose(arr, two_pi)] = 1.0
            return result

        denom = i0(kappa_val)
        if not np.isfinite(denom) or denom == 0.0:
            return self._cdf_from_pdf(x, mu, kappa)

        phi = (flat - mu_val + np.pi) % two_pi - np.pi
        base_phi = (-mu_val + np.pi) % two_pi - np.pi

        term_sum = np.zeros_like(phi)
        term_base = 0.0
        tol = 1e-12
        max_terms = 500
        converged = False

        for n in range(1, max_terms + 1):
            coeff = iv(n, kappa_val) / (denom * n)
            if not np.isfinite(coeff):
                continue

            term = coeff * np.sin(n * phi)
            term_sum += term
            term_base += coeff * np.sin(n * base_phi)

            max_term = np.max(np.abs(term))
            if max_term < tol and abs(coeff) < tol:
                converged = True
                break

        if not converged:
            return self._cdf_from_pdf(x, mu, kappa)

        cdf_raw = 0.5 + phi / two_pi + (1.0 / np.pi) * term_sum
        base_val = 0.5 + base_phi / two_pi + (1.0 / np.pi) * term_base

        forward = np.clip(cdf_raw - base_val, 0.0, 1.0)
        backward = np.clip(base_val - cdf_raw, 0.0, 1.0)
        cdf = np.where(phi >= base_phi, forward, 1.0 - backward)
        cdf = np.clip(cdf, 0.0, 1.0)

        if arr.ndim == 0:
            value = float(cdf[0])
            return 1.0 if np.isclose(float(wrapped), two_pi) else value

        result = cdf.reshape(arr.shape)
        result[np.isclose(arr, two_pi)] = 1.0
        return result

    def cdf(self, x, mu, kappa, *args, **kwargs):
        r"""
        Cumulative distribution function of the Von Mises distribution.

        $$
        F(\theta) = \frac{1}{2 \pi I_0(\kappa)}\int_{0}^{\theta} e^{\kappa \cos(\theta - \mu)} dx
        $$

        The CDF is evaluated via its Fourier-Bessel series expansion,
        $$
        F(\theta) = \frac{1}{2} + \frac{\theta - \mu}{2\pi}
        + \frac{1}{\pi}\sum_{n=1}^{\infty} \frac{I_n(\kappa)}{I_0(\kappa)\,n}
        \sin\bigl(n(\theta - \mu)\bigr),
        $$
        truncated adaptively for numerical stability and re-normalised to the
        $[0, 2\pi)$ support.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        mu : float
            The mean direction of the distribution (0 <= mu <= 2*pi).
        kappa : float
            The concentration parameter of the distribution (kappa > 0).

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

        eps = 1e-15
        q_clipped = np.clip(q_int, eps, 1.0 - eps)

        theta = (mu_val + two_pi * (q_clipped - 0.5)) % two_pi
        if kappa_val < 0.3:
            theta = (two_pi * q_clipped) % two_pi
        elif kappa_val > 5.0:
            normal_guess = mu_val + ndtri(q_clipped) / np.sqrt(kappa_val)
            normal_guess = np.mod(normal_guess, two_pi)
            blend = 0.5 if kappa_val < 20.0 else 0.8
            theta = np.mod(blend * normal_guess + (1.0 - blend) * (two_pi * q_clipped), two_pi)

        L = np.zeros_like(theta)
        H = np.full_like(theta, two_pi)

        tol_cdf = 1e-12
        tol_theta = 1e-10
        max_iter = 6

        theta_curr = theta.copy()
        for _ in range(max_iter):
            cdf_vals = np.asarray(self.cdf(theta_curr, mu_val, kappa_val), dtype=float)
            pdf_vals = np.exp(kappa_val * np.cos(theta_curr - mu_val)) / (2.0 * np.pi * i0(kappa_val))
            delta = cdf_vals - q_clipped

            L = np.where(delta <= 0.0, theta_curr, L)
            H = np.where(delta > 0.0, theta_curr, H)

            converged = (np.abs(delta) <= tol_cdf) & ((H - L) <= tol_theta)
            if np.all(converged):
                break

            denom = np.where(pdf_vals > 1e-15, pdf_vals, 1e-15)
            step = np.clip(delta / denom, -np.pi, np.pi)
            theta_next = theta_curr - step
            midpoint = 0.5 * (L + H)
            theta_next = np.where((theta_next <= L) | (theta_next >= H), midpoint, theta_next)
            theta_next = np.mod(theta_next, two_pi)
            theta_curr = theta_next

        delta = np.asarray(self.cdf(theta_curr, mu_val, kappa_val), dtype=float) - q_clipped
        mask = (np.abs(delta) > tol_cdf) | ((H - L) > tol_theta)
        if np.any(mask):
            theta_b = theta_curr.copy()
            L_b = L.copy()
            H_b = H.copy()
            for _ in range(30):
                if not np.any(mask):
                    break
                mid = 0.5 * (L_b + H_b)
                mid_vals = np.asarray(self.cdf(mid, mu_val, kappa_val), dtype=float)
                delta_mid = mid_vals - q_clipped
                take_upper = (delta_mid > 0.0) & mask
                take_lower = (~take_upper) & mask
                H_b = np.where(take_upper, mid, H_b)
                L_b = np.where(take_lower, mid, L_b)
                theta_b = np.where(mask, mid, theta_b)
                mask = mask & (np.abs(delta_mid) > tol_cdf)
            theta_curr = np.where(mask, 0.5 * (L_b + H_b), theta_b)

        result[interior] = np.mod(theta_curr, two_pi)
        return result.reshape(q_arr.shape)


    def ppf(self, q, mu, kappa, *args, **kwargs):
        """
        Percent-point function (inverse of the CDF) of the Von Mises distribution.

        The quantile is obtained by inverting the analytic Fourier–Bessel series
        using a safeguarded Newton iteration with the exact von Mises PDF as the
        slope, followed by a bisection polish.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate.
        mu : float
            The mean direction of the distribution (0 <= mu <= 2*pi).
        kappa : float
            The concentration parameter of the distribution (kappa > 0).

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
        # Check if instance-level parameters are set
        mu = getattr(self, "mu", None)
        kappa = getattr(self, "kappa", None)

        # Override instance parameters if provided in args/kwargs
        mu = kwargs.pop("mu", mu)
        kappa = kwargs.pop("kappa", kappa)

        # Ensure required parameters are provided
        if mu is None or kappa is None:
            raise ValueError("Both 'mu' and 'kappa' must be provided.")

        # Call the private _rvs method
        return self._rvs(mu, kappa, size=size, random_state=random_state)

    def support(self, *args, **kwargs):
        return (0, 2 * np.pi)

    def mean(self, *args, **kwargs):
        """
        Circular mean of the Von Mises distribution.

        Returns
        -------
        mean : float
            The circular mean direction (in radians), equal to `mu`.
        """
        (mu, _) = self._parse_args(*args, **kwargs)[0]
        return mu

    def median(self, *args, **kwargs):
        """
        Circular median of the Von Mises distribution.

        Returns
        -------
        median : float
            The circular median direction (in radians), equal to `mu`.
        """
        return self.mean(*args, **kwargs)

    def var(self, *args, **kwargs):
        """
        Circular variance of the Von Mises distribution.

        Returns
        -------
        variance : float
            The circular variance, derived from `kappa`.
        """
        (_, kappa) = self._parse_args(*args, **kwargs)[0]
        return 1 - i1(kappa) / i0(kappa)

    def std(self, *args, **kwargs):
        """
        Circular standard deviation of the Von Mises distribution.

        Returns
        -------
        std : float
            The circular standard deviation, derived from `kappa`.
        """
        (_, kappa) = self._parse_args(*args, **kwargs)[0]
        r = i1(kappa) / i0(kappa)

        return np.sqrt(-2 * np.log(r))

    def entropy(self, *args, **kwargs):
        """
        Entropy of the Von Mises distribution.

        Returns
        -------
        entropy : float
            The entropy of the distribution.
        """
        (_, kappa) = self._parse_args(*args, **kwargs)[0]
        return -np.log(i0(kappa)) + (kappa * i1(kappa)) / i0(kappa)

    def _nnlf(self, theta, data):
        """
        Custom negative log-likelihood function for the Von Mises distribution.
        """
        mu, kappa = theta

        if not self._argcheck(mu, kappa):  # Validate parameter range
            return np.inf

        # Compute log-likelihood robustly
        log_likelihood = self._logpdf(data, mu, kappa)

        # Negative log-likelihood
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
            log_i0_val = np.log(i0(kappa_param))
            return float(
                -kappa_param * sum_cos + w_sum * (np.log(2.0 * np.pi) + log_i0_val)
            )

        def grad(params):
            mu_param, kappa_param = params
            cos_term = np.cos(x - mu_param)
            sin_term = np.sin(x - mu_param)
            sum_sin = np.sum(w * sin_term)
            sum_cos = np.sum(w * cos_term)
            ratio = i1(kappa_param) / i0(kappa_param)
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


class vonmises_flattopped_gen(CircularContinuous):
    r"""Flat-topped von Mises Distribution

    The Flat-topped von Mises distribution is a modification of the von Mises distribution
    that allows for more flexible peak shapes, including flattened or sharper tops, depending
    on the value of the shape parameter $\nu$.

    ![vonmises-ext](../images/circ-mod-vonmises-flat-topped.png)

    Methods
    -------
    pdf(x, mu, kappa, nu)
        Probability density function.

    cdf(x, mu, kappa, nu)
        Cumulative distribution function.

    Note
    ----
    Implementation based on Section 4.3.10 of Pewsey et al. (2014)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vmft_table_cache = {}
        self._vmft_sampler_cache = {}

    def _validate_params(self, mu, kappa, nu):
        return (0 <= mu <= np.pi * 2) and (0 <= kappa <= _VMFT_KAPPA_UPPER) and (-1 <= nu <= 1)

    def _argcheck(self, mu, kappa, nu):
        return bool(self._validate_params(mu, kappa, nu))

    def _clear_normalization_cache(self):
        super()._clear_normalization_cache()
        self._vmft_table_cache = {}
        self._vmft_sampler_cache = {}

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
        self._c = table["normalizer"]  # retain attribute for existing code paths
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

            theta = np.empty_like(q_valid)
            for idx, (q_val, target, phi0) in enumerate(zip(q_valid, targets, phi_guess)):
                if close_zero[idx]:
                    theta[idx] = 0.0
                    continue
                if close_one[idx]:
                    theta[idx] = two_pi
                    continue

                i_hi = int(np.clip(np.searchsorted(cdf_grid, target, side="right"), 1, len(phi_grid) - 1))
                phi_lo = float(phi_grid[i_hi - 1])
                phi_hi = float(phi_grid[i_hi])
                phi = float(np.clip(phi0, phi_lo, phi_hi))

                for _ in range(_VMFT_NEWTON_MAXITER):
                    H_phi = float(cdf_interp(phi))
                    residual = H_phi - target
                    derivative = np.exp(
                        kappa_val * np.cos(phi + nu_val * np.sin(phi)) + table["log_normalizer"]
                    )
                    derivative = max(derivative, np.finfo(float).tiny)

                    if abs(residual) <= _VMFT_NEWTON_TOL and (phi_hi - phi_lo) <= _VMFT_NEWTON_WIDTH_TOL:
                        break

                    if residual > 0.0:
                        phi_hi = min(phi_hi, phi)
                    else:
                        phi_lo = max(phi_lo, phi)

                    step = residual / derivative
                    phi_candidate = phi - step
                    if not np.isfinite(phi_candidate) or phi_candidate <= phi_lo or phi_candidate >= phi_hi:
                        phi_candidate = 0.5 * (phi_lo + phi_hi)
                    phi = float(np.clip(phi_candidate, phi_lo, phi_hi))

                theta[idx] = (mu_val + phi) % two_pi

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

        table = self._get_vmft_table(kappa_val, nu_val)
        sampler_params = self._get_vmft_sampler_params(kappa_val, nu_val)
        kappa_env = sampler_params["kappa_env"]
        log_env_norm = sampler_params["log_env_norm"]
        log_multiplier = sampler_params["log_multiplier"]

        samples = np.empty(total, dtype=float)
        filled = 0
        batch_base = max(8, min(4 * total, 4096))

        while filled < total:
            batch = min(batch_base, total - filled) if filled > 0 else batch_base
            proposals = rng.vonmises(mu_val, kappa_env, size=batch)
            phi = ((proposals - mu_val + np.pi) % two_pi) - np.pi

            log_target = kappa_val * np.cos(phi + nu_val * np.sin(phi)) + table["log_normalizer"]
            log_env = kappa_env * np.cos(phi) - log_env_norm
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

    def rvs(self, mu=None, kappa=None, nu=None, size=None, random_state=None):
        r"""
        Draw random variates from the flat-topped von Mises distribution.

        Sampling uses an acceptance–rejection scheme with a curvature-matched
        von Mises envelope. Writing $\phi = \theta - \mu$ and matching the
        curvature at the mode yields a proposal concentration
        $\kappa_e = \kappa(1+\nu)^2$ (clipped to a small positive value). The
        envelope constant $M \ge \sup_\phi f(\phi)/g(\phi)$ is precomputed on
        the same spectral grid used for `cdf`, so once calibrated the
        sampler draws each variate with a single von Mises proposal followed by
        a scalar acceptance test.

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

    def _get_vmft_sampler_params(self, kappa, nu):
        key = (float(kappa), float(nu))
        params = self._vmft_sampler_cache.get(key)
        if params is not None:
            return params

        table = self._get_vmft_table(kappa, nu)
        kappa_env = float(np.clip(kappa * (1.0 + nu) ** 2, _VMFT_ENV_MIN_KAPPA, _VMFT_KAPPA_UPPER))

        log_env_norm = (
            np.log(2.0 * np.pi)
            + np.log(i0e(kappa_env))
            + kappa_env
        )
        log_env_pdf = kappa_env * np.cos(table["phi"]) - log_env_norm
        log_ratio = np.log(table["pdf"]) - log_env_pdf
        log_multiplier = float(np.max(log_ratio))
        multiplier = float(np.exp(log_multiplier) * (1.0 + 5e-12))

        params = {
            "kappa_env": kappa_env,
            "log_env_norm": float(log_env_norm),
            "log_multiplier": float(np.log(multiplier)),
            "multiplier": multiplier,
        }
        self._vmft_sampler_cache[key] = params
        return params


vonmises_flattopped = vonmises_flattopped_gen(name="vonmises_flattopped")

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

    unique_vals, unique_idx = np.unique(cumulative, return_index=True)
    if unique_vals.size >= 2:
        inv_interp = PchipInterpolator(unique_vals, phi[unique_idx], extrapolate=True)
    else:
        inv_interp = None

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


def _vmft_ensure_scalar(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(()))
    unique = np.unique(arr)
    if unique.size == 1:
        return float(unique[0])
    raise ValueError(f"Flat-topped von Mises parameter '{name}' must be a scalar.")


class jonespewsey_gen(CircularContinuous):
    """Jones-Pewsey Distribution

    ![jonespewsey](../images/circ-mod-jonespewsey.png)

    Methods
    -------
    pdf(x, mu, kappa, psi)
        Probability density function.

    cdf(x, mu, kappa, psi)
        Cumulative distribution function.


    Note
    ----
    Implementation based on Section 4.3.9 of Pewsey et al. (2014)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampler_cache = {}
        self._series_cache = {}

    def _validate_params(self, mu, kappa, psi):
        return (0 <= mu <= np.pi * 2) and (kappa >= 0) and (-np.inf <= psi <= np.inf)

    def _argcheck(self, mu, kappa, psi):
        return bool(self._validate_params(mu, kappa, psi))

    def _pdf(self, x, mu, kappa, psi):
        x = np.asarray(x, dtype=float)
        kappa_scalar = _jp_ensure_scalar(kappa, "kappa")
        psi_scalar = _jp_ensure_scalar(psi, "psi")

        if not np.isfinite(kappa_scalar) or not np.isfinite(psi_scalar):
            return np.full_like(x, np.nan, dtype=float)

        if abs(kappa_scalar) < _JP_KAPPA_TOL:
            return np.full_like(x, 1.0 / (2.0 * np.pi), dtype=float)

        normalizer = self._get_cached_normalizer(
            lambda: _c_jonespewsey(mu, kappa_scalar, psi_scalar),
            mu,
            kappa_scalar,
            psi_scalar,
        )
        self._c = normalizer

        if abs(psi_scalar) < _JP_PSI_TOL:
            return normalizer * np.exp(kappa_scalar * np.cos(x - mu))

        return normalizer * _kernel_jonespewsey(x, mu, kappa_scalar, psi_scalar)

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

        try:
            n_idx, coeffs = self._jp_get_series(kappa_val, psi_val)
        except Exception:  # pragma: no cover - defensive fallback
            cdf_vals = self._cdf_from_pdf(arr, mu_val, kappa_val, psi_val)
            return np.asarray(cdf_vals, dtype=float).reshape(arr.shape)

        phi_start = (-mu_val) % two_pi
        phi_end = (flat - mu_val) % two_pi

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
        as $\kappa \to 0$.

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

        kappa_env, envelope_const = self._jp_sampler_envelope(mu_val, kappa_val, psi_val)
        samples = np.empty(total, dtype=float)
        filled = 0

        while filled < total:
            remaining = total - filled
            proposals = vonmises.rvs(
                mu=mu_val,
                kappa=kappa_env,
                size=remaining,
                random_state=rng,
            )
            target_vals = self.pdf(proposals, mu_val, kappa_val, psi_val)
            proposal_vals = vonmises.pdf(proposals, mu=mu_val, kappa=kappa_env)
            ratio = np.where(proposal_vals > 0.0, target_vals / (envelope_const * proposal_vals), 0.0)
            u = rng.uniform(0.0, 1.0, size=remaining)
            accept = ratio >= u
            n_accept = int(np.sum(accept))
            if n_accept > 0:
                samples[filled:filled + n_accept] = proposals[accept][:n_accept]
                filled += n_accept

        return samples.reshape(size_tuple)

    def rvs(self, mu, kappa, psi, size=None, random_state=None):
        r"""
        Draw random variates from the Jones-Pewsey distribution.

        A von Mises envelope is tuned to the target density via local curvature
        matching and a grid-based optimisation, yielding an acceptance-rejection
        sampler that is both exact and efficient across the parameter space.

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

    def _jp_sampler_envelope(self, mu, kappa, psi):
        key = (float(np.mod(mu, 2.0 * np.pi)), float(kappa), float(psi))
        cached = self._sampler_cache.get(key)
        if cached is not None:
            return cached

        kappa_env = _jp_effective_kappa(kappa, psi)
        phi_grid = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
        theta_grid = np.mod(mu + phi_grid, 2.0 * np.pi)

        target_vals = self.pdf(theta_grid, mu, kappa, psi)
        log_target = np.log(np.clip(target_vals, np.finfo(float).tiny, None))

        kappa_env, envelope_const = _optimize_vonmises_envelope(
            theta_grid,
            log_target,
            mu,
            max(kappa_env, 1e-6),
        )

        self._sampler_cache[key] = (kappa_env, envelope_const)
        return kappa_env, envelope_const

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

####################################
## Helper Functions: Jones-Pewsey ##
####################################


_JP_KAPPA_TOL = 1e-3
_JP_PSI_TOL = 1e-6
_JP_MIN_BASE = np.finfo(float).tiny
_JP_MAX_EXP_ARGUMENT = 350.0  # guard for exp overflow


def _jp_ensure_scalar(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(()))
    unique = np.unique(arr)
    if unique.size == 1:
        return float(unique[0])
    raise ValueError(f"Jones-Pewsey parameter '{name}' must be a scalar.")


def _jp_kernel_base(phi, kappa, psi):
    phi = np.asarray(phi, dtype=float)
    if abs(psi) < _JP_PSI_TOL:
        return np.exp(kappa * np.cos(phi))

    A = kappa * psi
    cos_phi = np.cos(phi)

    cosh_A = np.cosh(A)
    sinh_A = np.sinh(A)
    if not np.isfinite(cosh_A) or not np.isfinite(sinh_A):
        # Fallback to stable exponential representation
        if A >= 0:
            exp_A = np.exp(np.clip(A, None, _JP_MAX_EXP_ARGUMENT))
            exp_negA = np.exp(np.clip(-A, -_JP_MAX_EXP_ARGUMENT, None))
        else:
            exp_A = np.exp(np.clip(A, -_JP_MAX_EXP_ARGUMENT, None))
            exp_negA = np.exp(np.clip(-A, None, _JP_MAX_EXP_ARGUMENT))
        cosh_A = 0.5 * (exp_A + exp_negA)
        sinh_A = 0.5 * (exp_A - exp_negA)

    base = cosh_A + sinh_A * cos_phi
    base = np.clip(base, _JP_MIN_BASE, None)
    return np.power(base, 1.0 / psi)


def _jp_effective_kappa(kappa, psi):
    if abs(psi) < _JP_PSI_TOL:
        return max(kappa, 1e-6)
    A = kappa * psi
    with np.errstate(over="ignore"):
        factor = 1.0 - np.exp(-2.0 * A)
    kappa_eff = factor / (2.0 * psi)
    if not np.isfinite(kappa_eff) or kappa_eff <= 0.0:
        return max(kappa, 1e-6)
    return float(kappa_eff)


def _log_vonmises_pdf(theta, mu, kappa):
    theta = np.asarray(theta, dtype=float)
    if kappa < 1e-8:
        return np.full_like(theta, -np.log(2.0 * np.pi), dtype=float)
    diff = theta - mu
    log_i0 = kappa + np.log(i0e(kappa))
    return kappa * np.cos(diff) - (np.log(2.0 * np.pi) + log_i0)


def _optimize_vonmises_envelope(theta, log_target, mu, initial_guess, *, max_iter=3):
    min_kappa = 1e-6
    max_kappa = 1e4
    best_kappa = max(initial_guess, min_kappa)
    log_M_best = np.inf

    def evaluate(candidates):
        nonlocal best_kappa, log_M_best
        for kappa_env in candidates:
            kappa_env = float(np.clip(kappa_env, min_kappa, max_kappa))
            log_proposal = _log_vonmises_pdf(theta, mu, kappa_env)
            log_ratio = log_target - log_proposal
            log_ratio_max = float(np.max(log_ratio))
            if not np.isfinite(log_ratio_max):
                continue
            if log_ratio_max < log_M_best:
                log_M_best = log_ratio_max
                best_kappa = kappa_env

    candidate_pool = np.array(
        [
            initial_guess,
            max(initial_guess * 0.5, min_kappa),
            initial_guess * 2.0,
            max(initial_guess * 0.25, min_kappa),
            initial_guess * 4.0,
            0.5,
            1.0,
            max(initial_guess, 1.5),
            max(initial_guess, 3.0),
        ],
        dtype=float,
    )
    candidate_pool = np.unique(np.clip(candidate_pool, min_kappa, max_kappa))
    evaluate(candidate_pool)

    for _ in range(max_iter):
        span = np.linspace(best_kappa * 0.5, best_kappa * 1.5, num=7)
        span = np.clip(span, min_kappa, max_kappa)
        evaluate(span)

    log_M_best = float(log_M_best)
    K = float(best_kappa)
    M = float(np.exp(log_M_best + np.log1p(0.02)))
    return K, max(M, 1.01)


def _kernel_jonespewsey(x, mu, kappa, psi):
    phi = np.asarray(x, dtype=float) - mu
    return _jp_kernel_base(phi, kappa, psi)


def _c_jonespewsey(mu, kappa, psi):
    if kappa < _JP_KAPPA_TOL:
        return 1.0 / (2.0 * np.pi)

    if abs(psi) < _JP_PSI_TOL:
        return 1.0 / (2.0 * np.pi * i0(kappa))

    constant = _jp_legendre_normalizer(kappa, psi)
    if np.isfinite(constant) and 1e-12 <= constant <= 1e6:
        return constant

    integral = quad_vec(
        _kernel_jonespewsey,
        a=-np.pi,
        b=np.pi,
        args=(mu, kappa, psi),
        epsabs=1e-10,
        epsrel=1e-10,
    )[0]
    return 1.0 / integral


def _jp_legendre_normalizer(kappa, psi):
    try:
        nu = 1.0 / psi
    except ZeroDivisionError:
        return np.nan

    A = kappa * psi
    z = np.cosh(A)
    try:
        legendre = lpmv(0, nu, z)
    except ValueError:
        return np.nan

    if not np.isfinite(legendre) or legendre <= 0.0:
        return np.nan

    return 1.0 / (2.0 * np.pi * legendre)


###########################
## Sine-Skewed Extention ##
###########################


class jonespewsey_sineskewed_gen(CircularContinuous):
    r"""Sine-Skewed Jones-Pewsey Distribution

    The Sine-Skewed Jones-Pewsey distribution is a circular distribution defined on $[0, 2\pi)$
    that extends the Jones-Pewsey family by incorporating a sine-based skewness adjustment.

    ![jonespewsey-sineskewed](../images/circ-mod-jonespewsey-sineskewed.png)

    Methods
    -------
    pdf(x, xi, kappa, psi, lmbd)
        Probability density function.

    cdf(x, xi, kappa, psi, lmbd)
        Cumulative distribution function.


    Note
    ----
    Implementation based on Section 4.3.11 of Pewsey et al. (2014)
    """

    def _validate_params(self, xi, kappa, psi, lmbd):
        return (
            (0 <= xi <= np.pi * 2)
            and (kappa >= 0)
            and (-np.inf <= psi <= np.inf)
            and (-1 <= lmbd <= 1)
        )

    def _argcheck(self, xi, kappa, psi, lmbd):
        return bool(self._validate_params(xi, kappa, psi, lmbd))

    def _pdf(self, x, xi, kappa, psi, lmbd):
        x = np.asarray(x, dtype=float)
        xi_scalar = _jp_ensure_scalar(xi, "xi")
        kappa_scalar = _jp_ensure_scalar(kappa, "kappa")
        psi_scalar = _jp_ensure_scalar(psi, "psi")
        lmbd_scalar = _jp_ensure_scalar(lmbd, "lmbd")

        if abs(kappa_scalar) < _JP_KAPPA_TOL:
            return (1.0 / (2.0 * np.pi)) * (1.0 + lmbd_scalar * np.sin(x - xi_scalar))

        normalizer = self._get_cached_normalizer(
            lambda: _c_jonespewsey(xi_scalar, kappa_scalar, psi_scalar),
            xi_scalar,
            kappa_scalar,
            psi_scalar,
        )
        self._c = normalizer

        base = _kernel_jonespewsey(x, xi_scalar, kappa_scalar, psi_scalar)
        return normalizer * base * (1.0 + lmbd_scalar * np.sin(x - xi_scalar))

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
            Skewness parameter, -1 < λ < 1. Controls the asymmetry introduced by the sine-skewing.

        Returns
        -------
        pdf_values: float
            Values of the probability density function at the specified points.
        """

        return super().pdf(x, xi, kappa, psi, lmbd, *args, **kwargs)

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

        n_idx, coeffs = jonespewsey._jp_get_series(kappa_val, psi_val)

        phi_start = (-xi_val) % two_pi
        phi_end = (flat - xi_val) % two_pi

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

        No closed form is available; the implementation integrates the PDF on
        [0, 2π) using adaptive quadrature, honouring the symmetric JP and
        uniform limits when ``lambda`` or ``kappa`` approach zero.
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

##########################
## Asymmetric Extention ##
##########################


class jonespewsey_asym_gen(CircularContinuous):
    r"""Asymmetric Extended Jones-Pewsey Distribution

    This distribution is an extension of the Jones-Pewsey family, incorporating asymmetry
    through a secondary parameter $\nu$. It is defined on the circular domain $[0, 2\pi)$.

    ![jonespewsey-asymext](../images/circ-mod-jonespewsey-asym.png)

    Methods
    -------
    pdf(x, xi, kappa, psi, nu)
        Probability density function.

    cdf(x, xi, kappa, psi, nu)
        Cumulative distribution function.


    Note
    ----
    Implementation from 4.3.12 of Pewsey et al. (2014)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampler_cache = {}
        self._cdf_table_cache = {}

    def _validate_params(self, xi, kappa, psi, nu):
        return (
            (0 <= xi <= np.pi * 2)
            and (kappa >= 0)
            and (-np.inf <= psi <= np.inf)
            and (0 <= nu < 1)
        )

    def _argcheck(self, xi, kappa, psi, nu):
        return bool(self._validate_params(xi, kappa, psi, nu))

    def _pdf(self, x, xi, kappa, psi, nu):
        x = np.asarray(x, dtype=float)
        xi_scalar = _jp_ensure_scalar(xi, "xi")
        kappa_scalar = _jp_ensure_scalar(kappa, "kappa")
        psi_scalar = _jp_ensure_scalar(psi, "psi")
        nu_scalar = _jp_ensure_scalar(nu, "nu")

        if abs(kappa_scalar) < _JP_KAPPA_TOL:
            return np.full_like(x, 1.0 / (2.0 * np.pi), dtype=float)

        norm = self._get_cached_normalizer(
            lambda: _c_jonespewsey_asym(xi_scalar, kappa_scalar, psi_scalar, nu_scalar),
            xi_scalar,
            kappa_scalar,
            psi_scalar,
            nu_scalar,
        )
        self._c = norm
        base = _kernel_jonespewsey_asym(x, xi_scalar, kappa_scalar, psi_scalar, nu_scalar)
        return norm * base

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
            Asymmetry parameter, $0 \leq \nu < 1$. Introduces skewness in the circular distribution.

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
                if kappa_val < _JP_KAPPA_TOL and nu_val < 1e-12:
                    theta_vals[interior] = two_pi * q_clipped
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
        the symmetric case, with the warp-aware CDF supplying residuals.  When
        nu is effectively zero the method delegates to the symmetric JP solver.
        """
        return super().ppf(q, xi, kappa, psi, nu, *args, **kwargs)

    def _rvs(self, xi, kappa, psi, nu, size=None, random_state=None):
        rng = self._init_rng(random_state)

        xi_val = _jp_ensure_scalar(xi, "xi")
        xi_val = float(np.mod(xi_val, 2.0 * np.pi))
        kappa_val = _jp_ensure_scalar(kappa, "kappa")
        psi_val = _jp_ensure_scalar(psi, "psi")
        nu_val = _jp_ensure_scalar(nu, "nu")
        if not (0.0 <= nu_val < 1.0):
            raise ValueError("`nu` must lie in [0, 1).")

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

        if abs(psi_val) < _JP_PSI_TOL and nu_val < 1e-12:
            return vonmises.rvs(mu=xi_val, kappa=kappa_val, size=size_tuple or None, random_state=rng)

        kappa_env, envelope_const = self._asym_sampler_envelope(xi_val, kappa_val, psi_val, nu_val)
        samples = np.empty(total, dtype=float)
        filled = 0

        while filled < total:
            remaining = total - filled
            proposals = vonmises.rvs(
                mu=xi_val,
                kappa=kappa_env,
                size=remaining,
                random_state=rng,
            )
            target_vals = self.pdf(proposals, xi_val, kappa_val, psi_val, nu_val)
            proposal_vals = vonmises.pdf(proposals, mu=xi_val, kappa=kappa_env)
            ratio = np.where(proposal_vals > 0.0, target_vals / (envelope_const * proposal_vals), 0.0)
            u = rng.uniform(0.0, 1.0, size=remaining)
            accept = ratio >= u
            n_accept = int(np.sum(accept))
            if n_accept > 0:
                samples[filled:filled + n_accept] = proposals[accept][:n_accept]
                filled += n_accept

        return samples.reshape(size_tuple)

    def rvs(self, xi, kappa, psi, nu, size=None, random_state=None):
        r"""
        Draw random variates from the asymmetric Jones--Pewsey distribution.

        Sampling uses a curvature-matched von Mises envelope tuned via the
        optimisation helper, providing an exact acceptance-rejection scheme that
        works well across nu in [0, 1).  Uniform and symmetric limits are
        handled explicitly.
        """
        return super().rvs(xi, kappa, psi, nu, size=size, random_state=random_state)

    def _asym_sampler_envelope(self, xi, kappa, psi, nu):
        key = (float(np.mod(xi, 2.0 * np.pi)), float(kappa), float(psi), float(nu))
        cached = self._sampler_cache.get(key)
        if cached is not None:
            return cached

        kappa_env = _jp_effective_kappa(kappa, psi)
        phi_grid = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
        theta_grid = np.mod(xi + phi_grid, 2.0 * np.pi)

        target_vals = self.pdf(theta_grid, xi, kappa, psi, nu)
        log_target = np.log(np.clip(target_vals, np.finfo(float).tiny, None))

        kappa_env, envelope_const = _optimize_vonmises_envelope(
            theta_grid,
            log_target,
            xi,
            max(kappa_env, 1e-6),
        )

        self._sampler_cache[key] = (kappa_env, envelope_const)
        return kappa_env, envelope_const

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
        return_info=False,
        optimizer="L-BFGS-B",
        psi_bounds=(-4.0, 4.0),
        kappa_bounds=(1e-6, 1e3),
        nu_bounds=(0.0, 0.99),
        base_kwargs=None,
        **kwargs,
    ):
        r"""
        Estimate asymmetric JP parameters by maximum likelihood.

        The symmetric JP fit supplies starting values for (xi, kappa, psi) with
        nu initialised at zero.  The full four-parameter log-likelihood is then
        optimised under simple bounds, re-using the cached normalising constant
        and envelope machinery developed for the JP core.
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


def _kernel_jonespewsey_asym(x, xi, kappa, psi, nu):
    x = np.asarray(x, dtype=float)
    phi = x - xi
    phi = phi + nu * np.cos(phi)
    return _jp_kernel_base(phi, kappa, psi)


def _c_jonespewsey_asym(xi, kappa, psi, nu):
    if kappa < _JP_KAPPA_TOL:
        return 1.0 / (2.0 * np.pi)

    integral = quad_vec(
        _kernel_jonespewsey_asym,
        a=-np.pi,
        b=np.pi,
        args=(xi, kappa, psi, nu),
        epsabs=1e-10,
        epsrel=1e-10,
    )[0]
    return 1.0 / integral


class inverse_batschelet_gen(CircularContinuous):
    r"""Inverse Batschelet Distribution

    ![inverse-batschelet](../images/circ-mod-inverse-batschelet.png)

    The inverse Batschelet family (Pewsey, Neuhaüser & Ruxton, 2014, §4.3.13)
    extends the von Mises distribution by applying two inverse angular warps:
    a "peakedness" transform controlled by $\nu$, and an inverse
    Batschelet skew transform governed by $\lambda$. The resulting density on
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

    Methods
    -------
    pdf(x, xi, kappa, nu, lmbd)
        Probability density function.

    cdf(x, xi, kappa, nu, lmbd)
        Cumulative distribution function.

    ppf(q, xi, kappa, nu, lmbd)
        Percent-point function (inverse CDF).

    rvs(xi, kappa, nu, lmbd, size=None, random_state=None)
        Random variates via von Mises acceptance–rejection.

    fit(data, *, method='mle', ...)
        Moments or maximum-likelihood parameter estimation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._invbat_table_cache = {}
        self._invbat_sampler_cache = {}

    def _clear_normalization_cache(self):
        super()._clear_normalization_cache()
        self._invbat_table_cache = {}
        self._invbat_sampler_cache = {}

    def _validate_params(self, xi, kappa, nu, lmbd):
        return (
            (0 <= xi <= np.pi * 2)
            and (kappa >= 0)
            and (-1 <= nu <= 1)
            and (-1 <= lmbd <= 1)
        )

    def _argcheck(self, xi, kappa, nu, lmbd):
        return bool(self._validate_params(xi, kappa, nu, lmbd))

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
            Shape parameter, $-1 \leq \nu \leq 1$. Controls asymmetry through angular transformation.
        lmbd : float
            Skewness parameter, $-1 \leq \lambda \leq 1$. Controls the degree of skewness in the distribution.

        Returns
        -------
        pdf_values : array_like
            Values of the probability density function at the specified points.
        """
        return super().pdf(x, xi, kappa, nu, lmbd, *args, **kwargs)

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
        grid node, the inverse peakedness transform $t_\nu^{-1}$ and inverse
        Batschelet skew $s_\lambda^{-1}$ are evaluated, and the resulting kernel is
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
            Shape parameter, $-1 \leq \nu \leq 1$.
        lmbd : float
            Skewness parameter, $-1 \leq \lambda \leq 1$.

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

            theta_vals = np.empty_like(q_valid)
            for idx, (target, phi0) in enumerate(zip(targets, phi_candidates)):
                if close_zero[idx]:
                    theta_vals[idx] = 0.0
                    continue
                if close_one[idx]:
                    theta_vals[idx] = two_pi
                    continue

                i_hi = int(np.clip(np.searchsorted(cdf_grid, target, side="right"), 1, len(phi_grid) - 1))
                phi_lo = float(phi_grid[i_hi - 1])
                phi_hi = float(phi_grid[i_hi])
                phi = float(np.clip(phi0, phi_lo, phi_hi))

                for _ in range(_INVBAT_NEWTON_MAXITER):
                    H_phi = float(cdf_interp(phi))
                    residual = H_phi - target
                    pdf_val = float(pdf_interp(phi))
                    pdf_val = max(pdf_val, np.finfo(float).tiny)

                    if abs(residual) <= _INVBAT_NEWTON_TOL and (phi_hi - phi_lo) <= _INVBAT_NEWTON_WIDTH_TOL:
                        break

                    if residual > 0.0:
                        phi_hi = min(phi_hi, phi)
                    else:
                        phi_lo = max(phi_lo, phi)

                    step = residual / pdf_val
                    phi_candidate = phi - step
                    if not np.isfinite(phi_candidate) or phi_candidate <= phi_lo or phi_candidate >= phi_hi:
                        phi_candidate = 0.5 * (phi_lo + phi_hi)
                    phi = float(np.clip(phi_candidate, phi_lo, phi_hi))

                theta_vals[idx] = (xi_val + phi) % two_pi

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
            Shape parameter, $-1 \leq \nu \leq 1$.
        lmbd : float
            Skewness parameter, $-1 \leq \lambda \leq 1$.

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
            Shape parameter, $-1 \leq \nu \leq 1$.
        lmbd : float
            Skewness parameter, $-1 \leq \lambda \leq 1$.
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

        unique_vals, unique_idx = np.unique(cumulative, return_index=True)
        inv_cdf_interp = (
            PchipInterpolator(unique_vals, phi[unique_idx], extrapolate=True)
            if unique_vals.size >= 2
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


##########################################
## Helper Functions: inverse_batschelet ##
##########################################


def _tnu(x, nu, xi):
    x_arr = np.asarray(x, dtype=float)
    scalar_input = x_arr.ndim == 0
    phi = np.mod(x_arr - xi + np.pi, 2.0 * np.pi) - np.pi
    phi_flat = np.atleast_1d(phi).astype(float, copy=False)
    results = np.empty_like(phi_flat)

    if abs(nu) <= _INVBAT_NU_TOL:
        results[:] = phi_flat
    else:
        for idx, phi_val in enumerate(phi_flat):
            def _equation(y):
                return y - nu * (1.0 + np.cos(y)) - phi_val

            solution = root_scalar(
                _equation,
                bracket=(-np.pi, np.pi),
                method="brentq",
            )
            if solution.converged:
                y_val = solution.root
            else:  # pragma: no cover - defensive fallback
                y_val = phi_val
            results[idx] = (y_val + np.pi) % (2.0 * np.pi) - np.pi

    if scalar_input:
        return float(results[0])
    return results.reshape(phi.shape)


def _slmbdinv(x, lmbd):
    x_arr = np.asarray(x, dtype=float)
    scalar_input = x_arr.ndim == 0
    x_flat = np.atleast_1d(x_arr).astype(float, copy=False)

    if np.isclose(lmbd, -1.0, atol=_INVBAT_LMBDA_TOL):
        result = x_flat.copy()
    else:
        result = np.empty_like(x_flat)
        for idx, val in enumerate(x_flat):
            def _equation(u):
                return u - 0.5 * (1.0 + lmbd) * np.sin(u) - val

            solution = root_scalar(
                _equation,
                bracket=(-np.pi, np.pi),
                method="brentq",
            )
            if solution.converged:
                u_val = solution.root
            else:  # pragma: no cover - defensive fallback
                u_val = val
            result[idx] = (u_val + np.pi) % (2.0 * np.pi) - np.pi

    if scalar_input:
        return float(result[0])
    return result.reshape(x_arr.shape)


def _A1(kappa):
    return i1(kappa) / i0(kappa)


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


def _invbat_ensure_scalar(value, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(()))
    unique = np.unique(arr)
    if unique.size == 1:
        return float(unique[0])
    raise ValueError(f"Inverse Batschelet parameter '{name}' must be a scalar.")


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
        \delta p + \tfrac{2}{\pi}\beta\gamma p \log(\gamma p), & \alpha = 1.
    \end{cases}
    $$

    Special cases include the wrapped normal (``α=2, β=0``), wrapped Cauchy
    (``α=1, β=0``), and wrapped Lévy (``α=1/2, β=1``).

    Methods
    -------
    pdf(x, delta, alpha, beta, gamma)
        Probability density function via adaptive Fourier series.

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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._series_cache = {}

    def _clear_normalization_cache(self):
        super()._clear_normalization_cache()
        self._series_cache = {}

    def _validate_params(self, delta, alpha, beta, gamma):
        return (
            (0 <= delta <= np.pi * 2)
            and (0 < alpha <= 2)
            and (-1 < beta < 1)
            and (gamma > 0)
        )

    def _argcheck(self, delta, alpha, beta, gamma):
        if self._validate_params(delta, alpha, beta, gamma):
            return True
        else:
            return False

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
            \delta p - \beta \frac{2}{\pi} \log(\gamma p), & \text{if } \alpha = 1
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
        cdf_vals = cdf_vals - anchor
        cdf_vals = np.where(cdf_vals < 0.0, cdf_vals + 1.0, cdf_vals)
        cdf_vals = np.clip(cdf_vals, 0.0, 1.0)

        # Ensure exact endpoints
        two_pi = 2.0 * np.pi
        cdf_vals[np.isclose(theta_flat[0], 0.0, atol=1e-12)] = 0.0
        cdf_vals[np.isclose(theta_flat[0], two_pi, atol=1e-12)] = 1.0

        if scalar_input:
            return float(cdf_vals.reshape(-1)[0])
        return cdf_vals.reshape(x_arr.shape)

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

        q_valid = flat[valid]
        close_zero = np.isclose(q_valid, 0.0, atol=1e-12, rtol=0.0)
        close_one = np.isclose(q_valid, 1.0, atol=1e-12, rtol=0.0)

        two_pi = 2.0 * np.pi

        theta_vals = np.empty_like(q_valid)
        for idx, q_val in enumerate(q_valid):
            if close_zero[idx]:
                theta_vals[idx] = 0.0
                continue
            if close_one[idx]:
                theta_vals[idx] = two_pi
                continue

            lo, hi = 0.0, two_pi
            theta = q_val * two_pi

            for _ in range(_WRAPSTABLE_NEWTON_MAXITER):
                cdf_theta = float(self._cdf(theta, delta_val, alpha_val, beta_val, gamma_val))
                pdf_theta = float(self._pdf(theta, delta_val, alpha_val, beta_val, gamma_val))
                residual = cdf_theta - q_val

                if abs(residual) <= _WRAPSTABLE_NEWTON_TOL and (hi - lo) <= _WRAPSTABLE_NEWTON_WIDTH_TOL:
                    break

                if residual > 0.0:
                    hi = min(hi, theta)
                else:
                    lo = max(lo, theta)

                if pdf_theta <= 0.0 or not np.isfinite(pdf_theta):
                    theta = 0.5 * (lo + hi)
                    continue

                step = residual / pdf_theta
                theta_new = theta - step
                if not np.isfinite(theta_new) or theta_new <= lo or theta_new >= hi:
                    theta = 0.5 * (lo + hi)
                else:
                    theta = theta_new

                if (hi - lo) <= _WRAPSTABLE_NEWTON_WIDTH_TOL:
                    break

            else:  # pragma: no cover - fallback to bisection if Newton fails
                for _ in range(30):
                    theta_mid = 0.5 * (lo + hi)
                    cdf_mid = float(self._cdf(theta_mid, delta_val, alpha_val, beta_val, gamma_val))
                    if cdf_mid > q_val:
                        hi = theta_mid
                    else:
                        lo = theta_mid
                theta = 0.5 * (lo + hi)

            theta_vals[idx] = (theta + two_pi) % two_pi

        result[valid] = theta_vals
        shaped = result.reshape(q_arr.shape)
        if q_arr.ndim == 0:
            return float(shaped)
        return shaped

    def _rvs(self, delta, alpha, beta, gamma, size=None, random_state=None):
        rng = self._init_rng(random_state)

        delta_val = self._scalar_param(delta)
        alpha_val = self._scalar_param(alpha)
        beta_val = self._scalar_param(beta)
        gamma_val = self._scalar_param(gamma)

        if not (0.0 < alpha_val <= 2.0):
            raise ValueError("`alpha` must lie in (0, 2].")
        if not (-1.0 < beta_val < 1.0):
            raise ValueError("`beta` must lie in (-1, 1).")
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
            B1 = (2.0 / np.pi) * gamma_mom * np.log(gamma_mom)
            B2 = (2.0 / np.pi) * (gamma_mom * 2.0) * np.log(gamma_mom * 2.0)
            denom = B2 - 2.0 * B1
            if abs(denom) < 1e-8:
                beta_mom = 0.0
                delta_mom = phi1
            else:
                beta_mom = (phi2 - 2.0 * phi1) / denom
                beta_mom = float(np.clip(beta_mom, beta_bounds[0], beta_bounds[1]))
                delta_mom = phi1 - beta_mom * B1
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
            mu_vals = delta * p + (2.0 / np.pi) * beta * gamma * p * np.log(gamma * p)
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
            raise ValueError("wrapstable parameters must be scalar-valued.")
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

    if abs(alpha - 1.0) > _WRAPSTABLE_ALPHA_TOL:
        tan_term = np.tan(0.5 * np.pi * alpha)
        theta0 = np.arctan(beta * tan_term) / alpha
        factor = (1.0 + (beta * tan_term) ** 2) ** (1.0 / (2.0 * alpha))

        delta_s1 = delta - gamma * beta * tan_term
        part1 = np.sin(alpha * (V + theta0)) / (np.cos(V) ** (1.0 / alpha))
        part2 = (np.cos(V - alpha * (V + theta0)) / W) ** ((1.0 - alpha) / alpha)
        x_s1 = gamma * factor * part1 * part2 + delta_s1
        x = x_s1 + (delta - delta_s1)
    else:
        factor = 2.0 / np.pi
        delta_s1 = delta - factor * beta * gamma * np.log(gamma)
        term = (
            (0.5 * np.pi + beta * V) * np.tan(V)
            - beta * np.log((0.5 * np.pi * W * np.cos(V)) / (0.5 * np.pi + beta * V))
        )
        x_s1 = gamma * factor * term + delta_s1
        x = x_s1 + (delta - delta_s1)

    return x

class katojones_gen(CircularContinuous):
    """
    Kato--Jones (2015) Distribution

    ![katojones](../images/circ-mod-katojones.png)

    Methods
    -------
    pdf(x, mu, gamma, rho, lam)
        Probability density function.
    cdf(x, mu, gamma, rho, lam)
        Cumulative distribution function via adaptive Fourier series.

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
            raise ValueError("katojones parameters must be scalar-valued.")
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
            cdf_flat = np.mod(cdf_raw, 1.0)

        cdf_flat = np.clip(cdf_flat, 0.0, 1.0)
        cdf_flat[np.isclose(flat, 0.0, atol=1e-12)] = 0.0
        cdf_flat[np.isclose(flat, 2.0 * np.pi, atol=1e-12)] = 1.0

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

        def cdf_single(theta):
            value = self._evaluate_cdf_series(theta, mu_val, gamma_val, rho_val, lam_val, series=series)
            value = np.mod(value, 1.0)
            return float(np.clip(value, 0.0, 1.0))

        for idx, q_val in enumerate(flat):
            if not valid[idx]:
                continue
            if np.isclose(q_val, 0.0, atol=1e-12):
                result[idx] = 0.0
                continue
            if np.isclose(q_val, 1.0, atol=1e-12):
                result[idx] = two_pi
                continue

            lo, hi = 0.0, two_pi
            theta = q_val * two_pi

            for _ in range(_KJ_NEWTON_MAXITER):
                cdf_theta = cdf_single(theta)
                pdf_theta = float(self._pdf(theta, mu_val, gamma_val, rho_val, lam_val))
                residual = cdf_theta - q_val

                if abs(residual) <= _KJ_NEWTON_TOL and (hi - lo) <= _KJ_NEWTON_WIDTH_TOL:
                    break

                if residual > 0.0:
                    hi = min(hi, theta)
                else:
                    lo = max(lo, theta)

                if pdf_theta <= 0.0 or not np.isfinite(pdf_theta):
                    theta = 0.5 * (lo + hi)
                    continue

                step = residual / pdf_theta
                theta_candidate = theta - step
                if not np.isfinite(theta_candidate) or theta_candidate <= lo or theta_candidate >= hi:
                    theta = 0.5 * (lo + hi)
                else:
                    theta = theta_candidate

                if (hi - lo) <= _KJ_NEWTON_WIDTH_TOL:
                    break
            else:
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if cdf_single(mid) > q_val:
                        hi = mid
                    else:
                        lo = mid
                theta = 0.5 * (lo + hi)

            result[idx] = theta % two_pi

        if scalar_input:
            return float(result[0])
        return result.reshape(q_arr.shape)

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
        if gamma <= _KJ_GAMMA_TOL or rho <= _KJ_GAMMA_TOL:
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
