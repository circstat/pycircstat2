import numpy as np
import types
from scipy.integrate import quad, quad_vec
from scipy.optimize import minimize, root, brentq
from scipy.special import gamma, i0, i1
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
    "jonespewsey",
    "vonmises_flattopped",
    "jonespewsey_sineskewed",
    "jonespewsey_asym",
    "inverse_batschelet",
    "wrapstable",
    "katojones",
    "katojones_mobius",
]


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

    def freeze(self, *args, **kwds):
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
        rng = self._random_state if random_state is None else random_state
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
        rng = random_state
        if rng is None:
            rng = self._random_state
        if rng is None:
            rng = np.random.default_rng()

        u = rng.uniform(0.0, 1.0, size=size)
        return self._ppf(u, rho)

    def rvs(self, rho, size=None, random_state=None):
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
        rho : float
            Concentration, $0 \le \rho \le 4/\pi^2$.
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
        return super().rvs(rho, size=size, random_state=random_state)

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
    """Cardioid (cosine) Distribution

    ![cardioid](../images/circ-mod-cardioid.png)

    Methods
    -------
    pdf(x, mu, rho)
        Probability density function.

    cdf(x, mu, rho)
        Cumulative distribution function.

    Notes
    -----
    Implementation based on Section 4.3.4 of Pewsey et al. (2014)
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

    Note
    ----
    Implementation based on Section 4.3.5 of Pewsey et al. (2014)
    """

    def _argcheck(self, mu, zeta):
        return 0 <= mu <= 2 * np.pi and zeta > 0

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

    def _cdf(self, x, mu, zeta):
        @np.vectorize
        def _cdf_single(x, mu, zeta):
            integral, _ = quad(self._pdf, a=0, b=x, args=(mu, zeta))
            return integral

        return _cdf_single(x, mu, zeta)

    def cdf(self, x, mu, zeta, *args, **kwargs):
        r"""
        Cumulative distribution function of the Cartwright distribution.

        No closed-form solution is available, so the CDF is computed numerically.

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

    Examples
    --------
    ```
    from pycircstat2.distributions import wrapnorm
    ```

    Notes
    -----
    Implementation based on Section 4.3.7 of Pewsey et al. (2014)
    """

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

    def _cdf(self, x, mu, rho):
        @np.vectorize
        def _cdf_single(x, mu, rho):
            integral, _ = quad(self._pdf, a=0, b=x, args=(mu, rho))
            return integral

        return _cdf_single(x, mu, rho)

    def cdf(self, x, mu, rho, *args, **kwargs):
        """
        Cumulative distribution function of the Wrapped Normal distribution.

        No closed-form solution is available, so the CDF is computed numerically.

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
        @np.vectorize
        def _cdf_single(x, mu, rho):
            integral, _ = quad(self._pdf, a=0, b=x, args=(mu, rho))
            return integral

        return _cdf_single(x, mu, rho)

    def cdf(self, x, mu, rho, *args, **kwargs):
        """
        Cumulative distribution function of the Wrapped Cauchy distribution.

        No closed-form solution is available, so the CDF is computed numerically.

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

    def _rvs(self, mu, rho, size=None, random_state=None):
        """
        Random variate generation for the Wrapped Cauchy distribution.

        Parameters
        ----------

        mu : float
            Mean direction, 0 <= mu <= 2*pi.
        rho : float
            Mean resultant length, 0 <= rho <= 1.
        size : int or tuple, optional
            Number of samples to generate.
        random_state : RandomState, optional
            Random number generator instance.

        Returns
        -------
        samples : ndarray
            Random variates from the Wrapped Cauchy distribution.
        """
        rng = self._random_state if random_state is None else random_state

        if rho == 0:
            return rng.uniform(0, 2 * np.pi, size=size)
        elif rho == 1:
            return np.full(size, mu % (2 * np.pi))
        else:
            from scipy.stats import cauchy

            scale = -np.log(rho)
            samples = cauchy.rvs(loc=mu, scale=scale, size=size, random_state=rng)
            return np.mod(samples, 2 * np.pi)

    def fit(self, data, method="analytical", *args, **kwargs):
        """
        Fit the Wrapped Cauchy distribution to the data.

        Parameters
        ----------
        data : array_like
            Input data (angles in radians).
        method : str, optional
            The approach for fitting the distribution. Options are:
            - "analytical": Compute `rho` and `mu` using closed-form solutions.
            - "numerical": Fit the parameters by minimizing the negative log-likelihood using an optimizer.
            Default is "analytical".

        *args, **kwargs :
            Additional arguments passed to the optimizer (if used).

        Returns
        -------
        rho : float
            Estimated shape parameter.
        mu : float
            Estimated mean direction.
        """
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        data = self._wrap_angles(np.asarray(data, dtype=float))

        # Validate the fitting method
        valid_methods = ["analytical", "numerical"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Available methods are {valid_methods}."
            )

        # Analytical solution for the Von Mises distribution
        mu, rho = circ_mean_and_r(alpha=data)

        # Use analytical estimates for mu and rho
        if method == "analytical":
            return mu, rho
        elif method == "numerical":
            # Numerical optimization
            def nll(params):
                mu, rho = params
                if not self._argcheck(mu, rho):
                    return np.inf
                return -np.sum(self._logpdf(data, mu, rho))

            start_params = [mu, np.clip(rho, 1e-4, 1 - 1e-4)]
            bounds = [(0, 2 * np.pi), (1e-6, 1)]
            algo = kwargs.pop("algorithm", "L-BFGS-B")
            result = minimize(
                nll, start_params, bounds=bounds, method=algo, *args, **kwargs
            )
            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")
            mu, rho = result.x
            return mu, rho
        else:
            raise ValueError(
                "Invalid method. Supported methods are 'analytical' and 'numerical'."
            )


wrapcauchy = wrapcauchy_gen(name="wrapcauchy")


class katojones_gen(CircularContinuous):
    """
    Kato--Jones (2015) Distribution

    ![katojones](../images/circ-mod-katojones.png)

    Methods
    -------
    pdf(x, mu, gamma, rho, lam)
        Probability density function.
    cdf(x, mu, gamma, rho, lam)
        Cumulative distribution function (numeric integration).
    logpdf(x, mu, gamma, rho, lam)
        Logarithm of the probability density function.
    rvs(mu, gamma, rho, lam, size=None, random_state=None)
        Random variates via a wrapped-Cauchy-based composition sampler.
    fit(data, method=\"moments\" | \"mle\", ...)
        Method-of-moments or maximum-likelihood parameter estimation.
    convert_alpha2_beta2(gamma, alpha2, beta2, verify=True)
        Helper to recover (rho, lambda) from second-order trigonometric moments.

    Notes
    -----
    Implements the tractable four-parameter unimodal family proposed by Kato and
    Jones (2015). Parameters control the first two trigonometric moments:
    ``mu`` sets the mean direction, ``gamma`` the mean resultant length, and
    ``rho``/``lam`` encode the magnitude/phase of the second-order moment.
    Feasible parameter tuples satisfy ``0 <= mu < 2*pi``, ``0 <= gamma < 1``,
    ``0 <= rho < 1``, ``0 <= lam < 2*pi`` together with the constraint enforced
    in :py:meth:`_argcheck`.

    Special cases include the uniform distribution (``gamma = 0``), the cardioid
    (``rho = 0``) and the wrapped Cauchy (``lambda = 0`` with ``gamma = rho``).

    References
    ----------
    - Kato, S. and M. C. Jones (2015). *A tractable and interpretable
      four-parameter family of unimodal distributions on the circle*. Biometrika,
      102(1), 181-193.
    """

    _moment_tolerance = 1e-12

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
            return float(pdf)
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

    def _cdf_scalar(self, x, mu, gamma, rho, lam):
        if x <= 0.0:
            return 0.0
        if x >= 2.0 * np.pi:
            return 1.0
        integral, _ = quad(
            self._pdf, 0.0, x, args=(mu, gamma, rho, lam), epsabs=1e-9, epsrel=1e-9
        )
        return float(integral)

    def _cdf(self, x, mu, gamma, rho, lam):
        x_arr = np.asarray(x, dtype=float)
        result = np.vectorize(
            lambda val: self._cdf_scalar(float(val), mu, gamma, rho, lam), otypes=[float]
        )(x_arr)
        if np.isscalar(x):
            return float(result)
        return result

    def cdf(self, x, mu, gamma, rho, lam, *args, **kwargs):
        r"""
        Cumulative distribution function of the Kato--Jones (2015) distribution.

        $$
        G(\theta) = \int_{0}^{\theta} g(t)\,dt
        $$

        where $g(\theta)$ is the density given above. The integral is evaluated
        numerically.

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
        q_arr = np.asarray(q, dtype=float)

        def invert_single(prob):
            if prob <= 0.0:
                return 0.0
            if prob >= 1.0:
                return 2.0 * np.pi

            def objective(theta):
                return self._cdf_scalar(theta, mu, gamma, rho, lam) - prob

            return brentq(objective, 0.0, 2.0 * np.pi, xtol=1e-12, rtol=1e-12, maxiter=200)

        result = np.vectorize(invert_single, otypes=[float])(q_arr)
        if np.isscalar(q):
            return float(result)
        return result

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

    def _fit_moments(self, data):
        data = self._wrap_angles(np.asarray(data, dtype=float))
        if data.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        mu_hat, r1 = circ_mean_and_r(alpha=data)
        centered = angmod(data - mu_hat)
        cos2 = np.cos(2.0 * centered)
        sin2 = np.sin(2.0 * centered)
        alpha2 = float(np.mean(cos2))
        beta2 = float(np.mean(sin2))
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

        return mu_hat, gamma_hat, rho_hat, lam_hat

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

    def _fit_mle(
        self,
        data,
        initial,
        optimizer,
        options,
        **minimize_kwargs,
    ):
        data = self._wrap_angles(np.asarray(data, dtype=float))
        if data.size == 0:
            raise ValueError("`data` must contain at least one observation.")

        if initial is None:
            initial = self._fit_moments(data)

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
            pdf_vals = self._pdf(data, mu, gamma, rho, lam)
            if np.any(pdf_vals <= 0.0) or not np.all(np.isfinite(pdf_vals)):
                return 1e12
            return -np.sum(np.log(pdf_vals))

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

        return mu_hat, gamma_hat, rho_hat, lam_hat

    def fit(
        self,
        data,
        method="moments",
        *,
        initial=None,
        optimizer="L-BFGS-B",
        options=None,
        **kwargs,
    ):
        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        kwargs.pop("floc", None)
        kwargs.pop("fscale", None)

        if method == "moments":
            if kwargs:
                raise TypeError("Unexpected optimizer arguments for method='moments'.")
            return self._fit_moments(data)

        if method != "mle":
            raise ValueError("method must be either 'moments' or 'mle'.")

        options = {} if options is None else dict(options)
        return self._fit_mle(
            data,
            initial=initial,
            optimizer=optimizer,
            options=options,
            **kwargs,
        )


katojones = katojones_gen(name="katojones")

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
        @np.vectorize
        def _cdf_single(x, mu, kappa):
            integral, _ = quad(self._pdf, a=0, b=x, args=(mu, kappa))
            return integral

        return _cdf_single(x, mu, kappa)

    def cdf(self, x, mu, kappa, *args, **kwargs):
        r"""
        Cumulative distribution function of the Von Mises distribution.

        $$
        F(\theta) = \frac{1}{2 \pi I_0(\kappa)}\int_{0}^{\theta} e^{\kappa \cos(\theta - \mu)} dx
        $$

        No closed-form solution is available, so the CDF is computed numerically.

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

    def ppf(self, q, mu, kappa, *args, **kwargs):
        """
        Percent-point function (inverse of the CDF) of the Von Mises distribution.

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
        # Use the random_state attribute or a new default random generator
        rng = self._random_state if random_state is None else random_state

        # Handle size being a tuple
        if size is None:
            size = 1
        num_samples = np.prod(size)  # Total number of samples

        # Best-Fisher algorithm
        a = 1 + np.sqrt(1 + 4 * kappa**2)
        b = (a - np.sqrt(2 * a)) / (2 * kappa)
        r = (1 + b**2) / (2 * b)

        def sample():
            while True:
                u1 = rng.uniform()
                z = np.cos(np.pi * u1)
                f = (1 + r * z) / (r + z)
                c = kappa * (r - f)
                u2 = rng.uniform()
                if u2 < c * (2 - c) or u2 <= c * np.exp(1 - c):
                    break
            u3 = rng.uniform()
            theta = mu + np.sign(u3 - 0.5) * np.arccos(f)
            return theta % (2 * np.pi)

        samples = np.array([sample() for _ in range(num_samples)])
        return samples

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

    def fit(self, data, method="analytical", *args, **kwargs):
        """
        Fit the Von Mises distribution to the given data.

        Parameters
        ----------
        data : array_like
            The data to fit the distribution to. Assumes values are in radians.
        method : str, optional
            The approach for fitting the distribution. Options are:
            - "analytical": Compute `mu` and `kappa` using closed-form solutions.
            - "numerical": Fit the parameters by minimizing the negative log-likelihood using an optimizer.
            Default is "analytical".

            When `method="numerical"`, the optimization algorithm can be specified via `algorithm` in `kwargs`.
            Supported algorithms include any method from `scipy.optimize.minimize`, such as "L-BFGS-B" (default) or "Nelder-Mead".

        *args : tuple, optional
            Additional positional arguments passed to the optimizer (if used).
        **kwargs : dict, optional
            Additional keyword arguments passed to the optimizer (if used).

        Returns
        -------
        kappa : float
            The estimated concentration parameter of the Von Mises distribution.
        mu : float
            The estimated mean direction of the Von Mises distribution.

        Notes
        -----
        - The "analytical" method directly computes the parameters using the circular mean
        and resultant vector length (`r`) for `mu` and `kappa`, respectively.
        - For numerical methods, the negative log-likelihood (NLL) is minimized using `_nnlf` as the objective function.


        Examples
        --------
        ```python
        # MLE fitting using analytical solution
        mu, kappa = vonmises.fit(data, method="analytical")

        # MLE fitting with numerical method using L-BFGS-B
        mu, kappa = vonmises.fit(data, method="L-BFGS-B")
        ```
        """

        kwargs = self._clean_loc_scale_kwargs(kwargs, caller="fit")
        data = self._wrap_angles(np.asarray(data, dtype=float))

        # Validate the fitting method
        valid_methods = ["analytical", "numerical"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Available methods are {valid_methods}."
            )

        # Analytical solution for the Von Mises distribution
        mu, r = circ_mean_and_r(alpha=data)
        kappa = circ_kappa(r=r, n=len(data))

        if method == "analytical":
            if np.isclose(r, 0):
                raise ValueError(
                    "Resultant vector length (r) is zero, e.g. uniform data or low directional bias."
                )
            return mu, kappa
        elif method == "numerical":
            # Use analytical solution as initial guess
            start_params = [mu, kappa]
            bounds = [(0, 2 * np.pi), (0, None)]  # 0 <= mu < 2*pi, kappa > 0,

            algo = kwargs.pop("algorithm", "L-BFGS-B")

            # Define the objective function (NLL) using `_nnlf`
            def nll(params):
                return self._nnlf(params, data)

            # Use the optimizer to minimize NLL
            result = minimize(
                nll, start_params, bounds=bounds, method=algo, *args, **kwargs
            )

            # Extract parameters from optimization result
            if not result.success:
                raise RuntimeError(f"Optimization failed: {result.message}")

            mu, kappa = result.x
            return mu, kappa
        else:
            raise ValueError(
                f"Invalid method '{method}'. Supported methods are 'analytical' and 'numerical'."
            )


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

    def _validate_params(self, mu, kappa, nu):
        return (0 <= mu <= np.pi * 2) and (kappa >= 0) and (-1 <= nu <= 1)

    def _argcheck(self, mu, kappa, nu):
        return bool(self._validate_params(mu, kappa, nu))

    def _pdf(self, x, mu, kappa, nu):
        norm_const = self._get_cached_normalizer(
            lambda: _c_vmft(mu, kappa, nu), mu, kappa, nu
        )
        self._c = norm_const  # retain attribute for existing code paths
        return norm_const * _kernel_vmft(x, mu, kappa, nu)

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
        return super().pdf(x, mu, kappa, nu, *args, **kwargs)

    def _cdf(self, x, mu, kappa, nu):
        @np.vectorize
        def _cdf_single(x, mu, kappa, nu):
            integral, _ = quad(self._pdf, a=0, b=x, args=(mu, kappa, nu))
            return integral

        return _cdf_single(x, mu, kappa, nu)


vonmises_flattopped = vonmises_flattopped_gen(name="vonmises_flattopped")

##############################################
## Helper Functions: Flat-topped von Mises  ##
##############################################


def _kernel_vmft(x, mu, kappa, nu):
    return np.exp(kappa * np.cos(x - mu + nu * np.sin(x - mu)))


def _c_vmft(mu, kappa, nu):
    c = 1 / quad_vec(_kernel_vmft, a=-np.pi, b=np.pi, args=(mu, kappa, nu))[0]
    return c


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

    def _validate_params(self, mu, kappa, psi):
        return (0 <= mu <= np.pi * 2) and (kappa >= 0) and (-np.inf <= psi <= np.inf)

    def _argcheck(self, mu, kappa, psi):
        return bool(self._validate_params(mu, kappa, psi))

    def _pdf(self, x, mu, kappa, psi):
        if np.all(kappa < 0.001):
            return 1 / (2 * np.pi)
        else:
            norm = self._get_cached_normalizer(
                lambda: _c_jonespewsey(mu, kappa, psi), mu, kappa, psi
            )
            self._c = norm
            if np.isclose(np.abs(psi), 0).all():
                return norm * np.exp(kappa * np.cos(x - mu))
            return _kernel_jonespewsey(x, mu, kappa, psi) / norm

    def pdf(self, x, mu, kappa, psi, *args, **kwargs):
        r"""
        Probability density function of the Jones-Pewsey distribution.

        $$
        f(\theta) = \frac{(\cosh(\kappa \psi) + \sinh(\kappa \psi) \cos(\theta - \mu))^{1/\psi}}{2\pi \cosh(\kappa \pi)}
        $$

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
        def vonmises_pdf(x, mu, kappa, psi, c):
            return c * np.exp(kappa * np.cos(x - mu))

        if np.isclose(np.abs(psi), 0).all():
            c = self._get_cached_normalizer(
                lambda: _c_jonespewsey(mu, kappa, psi), mu, kappa, psi
            )
            self._c = c

            @np.vectorize
            def _cdf_single(x, mu, kappa, psi, c):
                integral, _ = quad(vonmises_pdf, a=0, b=x, args=(mu, kappa, psi, c))
                return integral

            return _cdf_single(x, mu, kappa, psi, c)
        else:

            @np.vectorize
            def _cdf_single(x, mu, kappa, psi):
                integral, _ = quad(self._pdf, a=0, b=x, args=(mu, kappa, psi))
                return integral

            return _cdf_single(x, mu, kappa, psi)


jonespewsey = jonespewsey_gen(name="jonespewsey")

####################################
## Helper Functions: Jones-Pewsey ##
####################################


def _kernel_jonespewsey(x, mu, kappa, psi):
    return (np.cosh(kappa * psi) + np.sinh(kappa * psi) * np.cos(x - mu)) ** (
        1 / psi
    ) / (2 * np.pi * np.cosh(kappa * np.pi))


def _c_jonespewsey(mu, kappa, psi):
    if np.all(kappa < 0.001):
        return np.ones_like(kappa) * 1 / 2 / np.pi
    else:
        if np.isclose(np.abs(psi), 0).all():
            return 1 / (2 * np.pi * i0(kappa))
        else:
            c = quad_vec(_kernel_jonespewsey, a=-np.pi, b=np.pi, args=(mu, kappa, psi))[
                0
            ]
            return c


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
        if np.all(kappa < 0.001):
            return 1 / (2 * np.pi) * (1 + lmbd * np.sin(x - xi))
        else:
            norm = self._get_cached_normalizer(
                lambda: _c_jonespewsey(xi, kappa, psi), xi, kappa, psi
            )
            self._c = norm
            if np.isclose(np.abs(psi), 0).all():
                return (
                    norm
                    * np.exp(kappa * np.cos(x - xi))
                    * (1 + lmbd * np.sin(x - xi))
                )
            else:
                return (
                    (1 + lmbd * np.sin(x - xi))
                    * _kernel_jonespewsey(x, xi, kappa, psi)
                    / norm
                )

    def pdf(self, x, xi, kappa, psi, lmbd, *args, **kwargs):
        r"""
        Probability density function of the Sine-Skewed Jones-Pewsey distribution.

        $$
        f(\theta) = \frac{(\cosh(\kappa \psi) + \sinh(\kappa \psi) \cos(\theta - \xi))^{1/\psi}}{2\pi \cosh(\kappa \pi)}
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
        @np.vectorize
        def _cdf_single(x, xi, kappa, psi, lmbd):
            integral, _ = quad(self._pdf, a=0, b=x, args=(xi, kappa, psi, lmbd))
            return integral

        return _cdf_single(x, xi, kappa, psi, lmbd)


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
        norm = self._get_cached_normalizer(
            lambda: _c_jonespewsey_asym(xi, kappa, psi, nu), xi, kappa, psi, nu
        )
        self._c = norm
        return _kernel_jonespewsey_asym(x, xi, kappa, psi, nu) / norm

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
        @np.vectorize
        def _cdf_single(x, xi, kappa, psi, nu):
            integral, _ = quad(self._pdf, a=0, b=x, args=(xi, kappa, psi, nu))
            return integral

        return _cdf_single(x, xi, kappa, psi, nu)


jonespewsey_asym = jonespewsey_asym_gen(name="jonespewsey_asym")


def _kernel_jonespewsey_asym(x, xi, kappa, psi, nu):
    if np.isclose(np.abs(psi), 0).all():
        return np.exp(kappa * np.cos(x - xi + nu * np.cos(x - xi)))
    else:
        return (
            np.cosh(kappa * psi)
            + np.sinh(kappa * psi) * np.cos(x - xi + nu * np.cos(x - xi))
        ) ** (1 / psi)


def _c_jonespewsey_asym(xi, kappa, psi, nu):
    c = quad_vec(
        _kernel_jonespewsey_asym,
        a=-np.pi,
        b=np.pi,
        args=(xi, kappa, psi, nu),
    )[0]
    return c


class inverse_batschelet_gen(CircularContinuous):
    r"""Inverse Batschelet distribution.

    The inverse Batschelet distribution is a flexible circular distribution that allows for
    modeling asymmetric and peaked data. It is defined on the interval $[0, 2\pi)$.

    ![inverse-batschelet](../images/circ-mod-inverse-batschelet.png)

    Methods
    -------
    pdf(x, xi, kappa, psi, nu, lmbd)
        Probability density function.

    cdf(x, xi, kappa, psi, nu, lmbd)
        Cumulative distribution function.


    Note
    ----
    Implementation from 4.3.13 of Pewsey et al. (2014)
    """

    def _validate_params(self, xi, kappa, nu, lmbd):
        return (
            (0 <= xi <= np.pi * 2)
            and (kappa >= 0)
            and (-1 <= nu <= 1)
            and (-1 <= lmbd <= 1)
        )

    def _argcheck(self, xi, kappa, nu, lmbd):
        if not self._validate_params(xi, kappa, nu, lmbd):
            return False
        if np.isclose(lmbd, -1).all():
            self.con1, self.con2 = 0, 0
        else:
            self.con1 = (1 - lmbd) / (1 + lmbd)
            self.con2 = (2 * lmbd) / (1 + lmbd)
        return True

    def _pdf(self, x, xi, kappa, nu, lmbd):
        norm = self._get_cached_normalizer(
            lambda: _c_invbatschelet(kappa, lmbd), kappa, lmbd
        )
        self._c = norm
        arg1 = _tnu(x, nu, xi)
        arg2 = _slmbdinv(arg1, lmbd)

        if np.isclose(lmbd, -1).all():
            return norm * np.exp(kappa * np.cos(arg1 - np.sin(arg1)))
        else:
            return norm * np.exp(kappa * np.cos(self.con1 * arg1 + self.con2 * arg2))

    def pdf(self, x, xi, kappa, nu, lmbd, *args, **kwargs):
        r"""
        Probability density function (PDF) of the inverse Batschelet distribution.

        The PDF is defined as:

        $$
        f(\theta) = c \exp\left(\kappa \cos\left(a \cdot g(\theta, \nu, \xi) + b \cdot s\left(g(\theta, \nu, \xi), \lambda\right)\right)\right)
        $$

        where:

        - $a$: Weight for the angular transformation, defined as:

        $$
        a = \frac{1 - \lambda}{1 + \lambda}
        $$

        - $b$: Weight for the skewness transformation, defined as:

        $$
        b = \frac{2 \lambda}{1 + \lambda}
        $$

        - $g(\theta, \nu, \xi)$: Angular transformation function, which incorporates $\nu$ and the location parameter $\xi$:

        $$
        g(\theta, \nu, \xi) = \theta - \xi - \nu \cdot (1 + \cos(\theta - \xi))
        $$

        - $s(z, \lambda)$: Skewness transformation function, defined as the root of the equation:

        $$
        s(z, \lambda) - 0.5 \cdot (1 + \lambda) \cdot \sin(s(z, \lambda)) = z
        $$

        - $c$: Normalization constant ensuring the PDF integrates to 1, computed as:

        $$
        c = \frac{1}{2\pi \cdot I_0(\kappa) \cdot \left(a - b \cdot \int_{-\pi}^{\pi} \exp(\kappa \cdot \cos(z - (1 - \lambda) \cdot \sin(z) / 2)) dz\right)}
        $$

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF, defined on the interval $[0, 2\pi)$.
        xi : float
            Direction parameter, $0 \leq \xi \leq 2\pi$. This typically represents the mode.
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
        @np.vectorize
        def _cdf_single(x, xi, kappa, nu, lmbd):
            integral, _ = quad(self._pdf, a=0, b=x, args=(xi, kappa, nu, lmbd))
            return integral

        return _cdf_single(x, xi, kappa, nu, lmbd)


inverse_batschelet = inverse_batschelet_gen(name="inverse_batschelet")


##########################################
## Helper Functions: inverse_batschelet ##
##########################################


def _tnu(x, nu, xi):
    phi = x - xi

    def _tnuinv(z, nu):
        return z - nu * (1 + np.cos(z)) - phi

    y = root(_tnuinv, x0=np.zeros_like(x), args=(nu)).x
    y[y > np.pi] -= 2 * np.pi

    if np.isscalar(x):  # Ensure scalar output for scalar input
        return y.item()  # Extract the scalar value
    else:
        return y


def _slmbdinv(x, lmbd):
    if np.isclose(lmbd, -1).all():
        return x
    else:

        def _slmbd(z, lmbd):
            return z - 0.5 * (1 + lmbd) * np.sin(z) - x

        y = root(_slmbd, x0=np.zeros_like(x), args=(lmbd)).x

        if np.isscalar(x):  # Ensure scalar output for scalar input
            return y.item()  # Extract the scalar value
        else:
            return y


def _A1(kappa):
    return i1(kappa) / i0(kappa)


def _c_invbatschelet(kappa, lmbd):
    mult = 2 * np.pi * i0(kappa)
    if np.isclose(lmbd, 1).all():
        K = 1 - _A1(kappa)
        c = 1 / (mult * K)
    else:
        con1 = (1 + lmbd) / (1 - lmbd)
        con2 = (2 * lmbd) / ((1 - lmbd) * mult)

        def kernel(x):
            return np.exp(kappa * np.cos(x - (1 - lmbd) * np.sin(x) / 2))

        intval = quad_vec(kernel, a=-np.pi, b=np.pi)[0]

        c = 1 / (mult * (con1 - con2 * intval))
    return c


class wrapstable_gen(CircularContinuous):
    r"""
    Wrapped Stable Distribution

    - is symmetric around $\delta$ when $\beta = 0$, and to be skewed to the right (left) if $\beta > 0$ ($\beta < 0$).
    - can be reduced to
        - the wrapped normal distribution when $\alpha = 2, \beta = 0$.
        - the wrapped Cauchy distribution when $\alpha = 1, \beta = 0$.
        - the wrappd Lévy distribution when $\alpha = 1/2, \beta = 1$

    ![wrapstable](../images/circ-mod-wrapstable.png)

    References
    ----------
    - Pewsey, A. (2008). The wrapped stable family of distributions as a flexible model for circular data. Computational Statistics & Data Analysis, 52(3), 1516-1523.
    """

    _series_term_limit = 150

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._series_cache = {}
        self._series_indices = np.arange(1, self._series_term_limit, dtype=float)

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
        rho_vals, mu_vals = self._get_series_terms(delta, alpha, beta, gamma)
        p = self._series_indices
        cos_args = p[:, np.newaxis] * x_arr[np.newaxis, ...] - mu_vals[:, np.newaxis]
        series_sum = np.sum(rho_vals[:, np.newaxis] * np.cos(cos_args), axis=0)
        pdf_values = 1 / (2 * np.pi) * (1 + 2 * series_sum)
        if np.isscalar(x):
            return float(pdf_values)
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
        @np.vectorize
        def _cdf_single(x, delta, alpha, beta, gamma):
            integral, _ = quad(self._pdf, a=0, b=x, args=(delta, alpha, beta, gamma))
            return integral

        return _cdf_single(x, delta, alpha, beta, gamma)

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
        p = self._series_indices
        rho_vals = np.exp(-((gamma * p) ** alpha))
        if np.all(alpha == 1):
            mu_vals = delta * p - 2 * beta * gamma * p * np.log(gamma * p) / np.pi
        else:
            mu_vals = delta * p + beta * np.tan(np.pi * alpha / 2) * (
                (gamma * p) ** alpha - gamma * p
            )
        return rho_vals, mu_vals

    @staticmethod
    def _scalar_param(value):
        arr = np.asarray(value, dtype=float)
        if arr.size == 1:
            return float(arr)
        first = float(arr.flat[0])
        if not np.allclose(arr, first):
            raise ValueError("wrapstable parameters must be scalar-valued.")
        return first


wrapstable = wrapstable_gen(name="wrapstable")
