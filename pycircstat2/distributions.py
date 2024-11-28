import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.optimize import minimize, root
from scipy.special import gamma, i0, i1
from scipy.stats import rv_continuous

from .descriptive import circ_kappa, circ_mean_and_r

__all__ = [
    "circularuniform",
    "cardioid",
    "cartwright",
    "wrapnorm",
    "wrapcauchy",
    "vonmises",
    "jonespewsey",
    "vonmises_ext",
    "jonespewsey_sineskewed",
    "jonespewsey_asymext",
    "inverse_batschelet",
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

###########################
## Symmetric Distribtion ##
###########################


class circularuniform_gen(rv_continuous):
    """Continuous Circular Uniform Distribution

    Methods
    -------
    pdf(x)
        Probability density function.

    cdf(x)
        Cumulative distribution function.
    """

    def _pdf(self, x):
        return 1 / np.pi

    def _cdf(self, x):
        return x / (2 * np.pi)


circularuniform = circularuniform_gen(name="circularuniform")


class cardioid_gen(rv_continuous):
    """Cardioid Distribution

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

    def _cdf(self, x, mu, rho):
        return (x + 2 * rho * (np.sin(x - mu) + np.sin(mu))) / (2 * np.pi)


cardioid = cardioid_gen(name="cardioid")


class cartwright_gen(rv_continuous):
    """Cartwright's Power-of-Cosine Distribution

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

    def _cdf(self, x, mu, zeta):
        @np.vectorize
        def _cdf_single(x, mu, zeta):
            return quad(self._pdf, a=0, b=x, args=(mu, zeta))

        return _cdf_single(x, mu, zeta)


cartwright = cartwright_gen(name="cartwright")


class wrapnorm_gen(rv_continuous):
    """Wrapped Normal Distribution

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
        return 0 <= mu <= np.pi * 2 and 0 < rho <= 1

    def _pdf(self, x, mu, rho):
        return (
            1
            + 2
            * np.sum([rho ** (p**2) * np.cos(p * (x - mu)) for p in range(1, 30)], 0)
        ) / (2 * np.pi)

    def _cdf(self, x, mu, rho):
        @np.vectorize
        def _cdf_single(x, mu, rho):
            return quad(self._pdf, a=0, b=x, args=(mu, rho))

        return _cdf_single(x, mu, rho)


wrapnorm = wrapnorm_gen(name="wrapped_normal")


class wrapcauchy_gen(rv_continuous):
    """Wrapped Cauchy Distribution.

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
        return 0 <= mu <= np.pi * 2 and 0 < rho <= 1

    def _pdf(self, x, mu, rho):
        return (1 - rho**2) / (2 * np.pi * (1 + rho**2 - 2 * rho * np.cos(x - mu)))

    def pdf(self, x, *args, **kwargs):
        """
        Probability density function of the Wrapped Cauchy distribution.

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
        return super().pdf(x, *args, **kwargs)

    def _logpdf(self, x, mu, rho):
        return np.log(np.clip(self._pdf(x, mu, rho), 1e-16, None))

    def logpdf(self, x, *args, **kwargs):
        """
        Logarithm of the probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log-PDF.
        rho : float
            Shape parameter, 0 < rho <= 1.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, *args, **kwargs)

    def _cdf(self, x, mu, rho):
        @np.vectorize
        def _cdf_single(x, mu, rho):
            integral, _ = quad(self._pdf, a=0, b=x, args=(mu, rho))
            return integral

        return _cdf_single(x, mu, rho)

    def cdf(self, x, *args, **kwargs):
        """
        Cumulative distribution function of the Wrapped Cauchy distribution.

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
        return super().cdf(x, *args, **kwargs)

    def _rvs(self, mu, rho, size=None, random_state=None):
        """
        Random variate generation for the Wrapped Cauchy distribution.

        Parameters
        ----------
        rho : float
            Shape parameter, 0 <= rho <= 1.
        mu : float
            Mean direction, 0 <= mu <= 2*pi.
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

        # Validate the fitting method
        valid_methods = ["analytical", "numerical"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Available methods are {valid_methods}."
            )

        # Validate the data
        if not np.all((0 <= data) & (data < 2 * np.pi)):
            raise ValueError("Data must be in the range [0, 2π).")

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


class vonmises_gen(rv_continuous):
    """Von Mises Distribution

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
    kappa : float
        The concentration parameter of the distribution (kappa > 0).
    mu : float
        The mean direction of the distribution (0 <= mu <= 2*pi).

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

    def pdf(self, x, *args, **kwargs):
        """
        Probability density function of the Von Mises distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability density function.
        kappa : float
            The concentration parameter of the distribution (kappa > 0).
        mu : float
            The mean direction of the distribution (0 <= mu <= 2*pi).

        Returns
        -------
        pdf_values : array_like
            Probability density function evaluated at `x`.
        """
        return super().pdf(x, *args, **kwargs)

    def _logpdf(self, x, mu, kappa):
        return kappa * np.cos(x - mu) - np.log(2 * np.pi * i0(kappa))

    def logpdf(self, x, *args, **kwargs):
        """
        Logarithm of the probability density function of the Von Mises distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the logarithm of the probability density function.
        kappa : float
            The concentration parameter of the distribution (kappa > 0).
        mu : float
            The mean direction of the distribution (0 <= mu <= 2*pi).

        Returns
        -------
        logpdf_values : array_like
            Logarithm of the probability density function evaluated at `x`.
        """
        return super().logpdf(x, *args, **kwargs)

    def _cdf(self, x, mu, kappa):
        @np.vectorize
        def _cdf_single(x, mu, kappa):
            integral, _ = quad(self._pdf, a=0, b=x, args=(mu, kappa))
            return integral

        return _cdf_single(x, mu, kappa)

    def cdf(self, x, *args, **kwargs):
        """
        Cumulative distribution function of the Von Mises distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the cumulative distribution function.
        kappa : float
            The concentration parameter of the distribution (kappa > 0).
        mu : float
            The mean direction of the distribution (0 <= mu <= 2*pi).

        Returns
        -------
        cdf_values : array_like
            Cumulative distribution function evaluated at `x`.
        """
        return super().cdf(x, *args, **kwargs)

    def ppf(self, q, *args, **kwargs):
        """
        Percent-point function (inverse of the CDF) of the Von Mises distribution.

        Parameters
        ----------
        q : array_like
            Quantiles to evaluate.
        kappa : float
            The concentration parameter of the distribution (kappa > 0).
        mu : float
            The mean direction of the distribution (0 <= mu <= 2*pi).

        Returns
        -------
        ppf_values : array_like
            Values at the given quantiles.
        """
        return super().ppf(q, *args, **kwargs)

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
        # Analytical fitting
        mu, kappa = vonmises.fit(data, method="analytical")

        # Numerical fitting using L-BFGS-B
        mu, kappa = vonmises.fit(data, method="L-BFGS-B")
        """

        # Validate the fitting method
        valid_methods = ["analytical", "numerical"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Available methods are {valid_methods}."
            )

        # Validate the data
        if not np.all((0 <= data) & (data < 2 * np.pi)):
            raise ValueError("Data must be in the range [0, 2π).")

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


class jonespewsey_gen(rv_continuous):
    """Jones-Pewsey Distribution

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

    def __init__(
        self, name="jonespewsey", mu=None, kappa=None, psi=None, *args, **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        if kappa is not None and psi is not None and mu is not None:
            self._c = _c_jonespewsey(mu, kappa, psi)
        else:
            self._c = None

    def _argcheck(self, mu, kappa, psi):
        # we can save a lot of computation if c is automatically computed before others
        # but tbh this is a hack, and I don't have a better way to do it atm.
        return (0 <= mu <= np.pi * 2) and (kappa >= 0) and (-np.inf <= psi <= np.inf)

    def _pdf(self, x, mu, kappa, psi):

        if self._c is None:
            c = _c_jonespewsey(mu, kappa, psi)
        else:
            c = self._c

        if np.all(kappa < 0.001):
            return 1 / (2 * np.pi)
        else:
            if np.isclose(np.abs(psi), 0).all():
                return 1 / (2 * np.pi * i0(kappa)) * np.exp(kappa * np.cos(x - mu))
            else:
                return _kernel_jonespewsey(x, mu, kappa, psi) / c

    def _cdf(self, x, mu, kappa, psi):
        def vonmises_pdf(x, mu, kappa, psi, c):
            return c * np.exp(kappa * np.cos(x - mu))

        if np.isclose(np.abs(psi), 0).all():
            if self._c is None:
                c = _c_jonespewsey(mu, kappa, psi)
            else:
                c = self._c

            @np.vectorize
            def _cdf_single(x, mu, kappa, psi, c):
                return quad(vonmises_pdf, a=0, b=x, args=(mu, kappa, psi, c))

            return _cdf_single(x, mu, kappa, psi, c)
        else:

            @np.vectorize
            def _cdf_single(x, mu, kappa, psi):
                return quad(self._pdf, a=0, b=x, args=(mu, kappa, psi))

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


#########################
## Symmetric Extention ##
#########################


class vonmises_ext_gen(rv_continuous):
    """Flat-topped von Mises Distribution

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

    def _argcheck(self, mu, kappa, nu):
        self.c = _c_vmext(mu, kappa, nu)
        return (0 <= mu <= np.pi * 2) and (kappa >= 0) and (-1 <= nu <= 1)

    def _pdf(self, x, mu, kappa, nu):
        return self.c * _kernel_vmext(x, mu, kappa, nu)

    def _cdf(self, x, mu, kappa, nu):
        @np.vectorize
        def _cdf_single(x, mu, kappa, nu):
            return quad(self._pdf, a=0, b=x, args=(mu, kappa, nu))

        return _cdf_single(x, mu, kappa, nu)


vonmises_ext = vonmises_ext_gen(name="vonmises_ext")

###########################################
## Helper Functions: extended von Mises  ##
###########################################


def _kernel_vmext(x, mu, kappa, nu):
    return np.exp(kappa * np.cos(x - mu + nu * np.sin(x - mu)))


def _c_vmext(mu, kappa, nu):
    c = 1 / quad_vec(_kernel_vmext, a=-np.pi, b=np.pi, args=(mu, kappa, nu))[0]
    return c


###########################
## Sine-Skewed Extention ##
###########################


class jonespewsey_sineskewed_gen(rv_continuous):
    """Sine-Skewed Jones-Pewsey Distribution

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

    def __init__(
        self,
        name="jonespewsey",
        xi=None,
        kappa=None,
        psi=None,
        lmbd=None,
        *args,
        **kwargs,
    ):
        super().__init__(name=name, *args, **kwargs)
        if (
            kappa is not None
            and psi is not None
            and xi is not None
            and lmbd is not None
        ):
            self._c = _c_jonespewsey(xi, kappa, psi)
        else:
            self._c = None

    def _argcheck(self, xi, kappa, psi, lmbd):
        return (
            (0 <= xi <= np.pi * 2)
            and (kappa >= 0)
            and (-np.inf <= psi <= np.inf)
            and (-1 <= lmbd <= 1)
        )

    def _pdf(self, x, xi, kappa, psi, lmbd):

        if self._c is None:
            c = _c_jonespewsey(xi, kappa, psi)
        else:
            c = self._c

        if np.all(kappa < 0.001):
            return 1 / (2 * np.pi) * (1 + lmbd * np.sin(x - xi))
        else:
            if np.isclose(np.abs(psi), 0).all():
                return (
                    1
                    / (2 * np.pi * i0(kappa))
                    * np.exp(kappa * np.cos(x - xi))
                    * (1 + lmbd * np.sin(x - xi))
                )
            else:
                return (
                    (1 + lmbd * np.sin(x - xi))
                    * _kernel_jonespewsey(x, xi, kappa, psi)
                    / c
                )

    def _cdf(self, x, xi, kappa, psi, lmbd):
        @np.vectorize
        def _cdf_single(x, xi, kappa, psi, lmbd):
            return quad(self._pdf, a=0, b=x, args=(xi, kappa, psi, lmbd))

        return _cdf_single(x, xi, kappa, psi, lmbd)


jonespewsey_sineskewed = jonespewsey_sineskewed_gen(name="jonespewsey_sineskewed")

##########################
## Asymmetric Extention ##
##########################


class jonespewsey_asymext_gen(rv_continuous):
    """Asymmetric Extended Jones-Pewsey Distribution

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

    def _argcheck(self, xi, kappa, psi, nu):
        self.c = _c_jonespewsey_asymext(xi, kappa, psi, nu)
        return (
            (0 <= xi <= np.pi * 2)
            and (kappa >= 0)
            and (-np.inf <= psi <= np.inf)
            and (0 <= nu < 1)
        )

    def _pdf(self, x, xi, kappa, psi, nu):
        return _kernel_jonespewsey_asymext(x, xi, kappa, psi, nu) / self.c

    def _cdf(self, x, xi, kappa, psi, nu):
        @np.vectorize
        def _cdf_single(x, xi, kappa, psi, nu):
            return quad(self._pdf, a=0, b=x, args=(xi, kappa, psi, nu))

        return _cdf_single(x, xi, kappa, psi, nu)


jonespewsey_asymext = jonespewsey_asymext_gen(name="jonespewsey_asymext")


def _kernel_jonespewsey_asymext(x, xi, kappa, psi, nu):
    if np.isclose(np.abs(psi), 0).all():
        return np.exp(kappa * np.cos(x - xi + nu * np.cos(x - xi)))
    else:
        return (
            np.cosh(kappa * psi)
            + np.sinh(kappa * psi) * np.cos(x - xi + nu * np.cos(x - xi))
        ) ** (1 / psi)


def _c_jonespewsey_asymext(xi, kappa, psi, nu):

    c = quad_vec(
        _kernel_jonespewsey_asymext, a=-np.pi, b=np.pi, args=(xi, kappa, psi, nu)
    )[0]
    return c


class inverse_batschelet_gen(rv_continuous):
    """Inverse Batschelet distribution.

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

    def _argcheck(self, xi, kappa, nu, lmbd):
        self.c = _c_invbatschelet(kappa, lmbd)
        if np.isclose(lmbd, -1).all():
            self.con1, self.con2 = 0, 0
        else:
            self.con1 = (1 - lmbd) / (1 + lmbd)
            self.con2 = (2 * lmbd) / (1 + lmbd)
        return (
            (0 <= xi <= np.pi * 2)
            and (kappa >= 0)
            and (-1 <= nu <= 1)
            and (-1 <= lmbd <= 1)
        )

    def _pdf(self, x, xi, kappa, nu, lmbd):

        arg1 = _tnu(x, nu, xi)
        arg2 = _slmbdinv(arg1, lmbd)

        if np.isclose(lmbd, -1).all():
            return self.c * np.exp(kappa * np.cos(arg1 - np.sin(arg1)))
        else:
            return self.c * np.exp(kappa * np.cos(self.con1 * arg1 + self.con2 * arg2))

    def _cdf(self, x, xi, kappa, nu, lmbd):
        @np.vectorize
        def _cdf_single(x, xi, kappa, nu, lmbd):
            return quad(self._pdf, a=0, b=x, args=(xi, kappa, nu, lmbd))

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
