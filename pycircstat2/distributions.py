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
    "vonmises_flattopped",
    "jonespewsey_sineskewed",
    "jonespewsey_asym",
    "inverse_batschelet",
    "wrapstable",
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

############################
## Symmetric Distribtions ##
############################


class circularuniform_gen(rv_continuous):
    """Continuous Circular Uniform Distribution

    ![circularuniform](../images/circ-mod-circularuniform.png)

    Methods
    -------
    pdf(x)
        Probability density function.

    cdf(x)
        Cumulative distribution function.
    """

    def _pdf(self, x):
        return 1 / np.pi

    def pdf(self, x, *args, **kwargs):
        r"""
        Probability density function of the Circular Uniform distribution.

        $$
        f(\theta) = \frac{1}{\pi}
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


circularuniform = circularuniform_gen(name="circularuniform")


class triangular_gen(rv_continuous):
    """Triangular Distribution

    ![triangular](../images/circ-mod-triangular.png)

    Methods
    -------
    pdf(x, rho)
        Probability density function.

    cdf(x, rho)
        Cumulative distribution function.

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
        @np.vectorize
        def _cdf_single(x, rho):
            integral, _ = quad(self._pdf, a=0, b=x, args=(rho))
            return integral

        return _cdf_single(x, rho)


triangular = triangular_gen(name="triangular")


class cardioid_gen(rv_continuous):
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


class cartwright_gen(rv_continuous):
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


class wrapnorm_gen(rv_continuous):
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


class wrapcauchy_gen(rv_continuous):
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
        return 0 <= mu <= np.pi * 2 and 0 < rho <= 1

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
                "Invalid method. Supported methods are 'analytical' and " "'numerical'."
            )


wrapcauchy = wrapcauchy_gen(name="wrapcauchy")


class vonmises_gen(rv_continuous):
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


class vonmises_flattopped_gen(rv_continuous):
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
        if self._validate_params(mu, kappa, nu):
            self._c = _c_vmft(mu, kappa, nu)
            return True
        else:
            return False

    def _pdf(self, x, mu, kappa, nu):
        return self._c * _kernel_vmft(x, mu, kappa, nu)

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


class jonespewsey_gen(rv_continuous):
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
        if self._validate_params(mu, kappa, psi):
            self._c = _c_jonespewsey(
                mu, kappa, psi
            )  # Precompute the normalizing constant
            return True
        else:
            return False

    def _pdf(self, x, mu, kappa, psi):

        if np.all(kappa < 0.001):
            return 1 / (2 * np.pi)
        else:
            if np.isclose(np.abs(psi), 0).all():
                return 1 / (2 * np.pi * i0(kappa)) * np.exp(kappa * np.cos(x - mu))
            else:
                return _kernel_jonespewsey(x, mu, kappa, psi) / self._c

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
            c = self._c

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


class jonespewsey_sineskewed_gen(rv_continuous):
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
        if self._validate_params(xi, kappa, psi, lmbd):
            self._c = _c_jonespewsey(xi, kappa, psi)
            return True
        else:
            return False

    def _pdf(self, x, xi, kappa, psi, lmbd):

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
                    / self._c
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


class jonespewsey_asym_gen(rv_continuous):
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
        if self._validate_params(xi, kappa, psi, nu):
            self._c = _c_jonespewsey_asym(xi, kappa, psi, nu)
            return True
        else:
            return False

    def _pdf(self, x, xi, kappa, psi, nu):
        return _kernel_jonespewsey_asym(x, xi, kappa, psi, nu) / self._c

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


class inverse_batschelet_gen(rv_continuous):
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
        if self._validate_params(xi, kappa, nu, lmbd):
            self._c = _c_invbatschelet(kappa, lmbd)
            if np.isclose(lmbd, -1).all():
                self.con1, self.con2 = 0, 0
            else:
                self.con1 = (1 - lmbd) / (1 + lmbd)
                self.con2 = (2 * lmbd) / (1 + lmbd)
            return True
        else:
            return False

    def _pdf(self, x, xi, kappa, nu, lmbd):

        arg1 = _tnu(x, nu, xi)
        arg2 = _slmbdinv(arg1, lmbd)

        if np.isclose(lmbd, -1).all():
            return self._c * np.exp(kappa * np.cos(arg1 - np.sin(arg1)))
        else:
            return self._c * np.exp(kappa * np.cos(self.con1 * arg1 + self.con2 * arg2))

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


class wrapstable_gen(rv_continuous):
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

        def rho_p(p, alpha, gamma):
            return np.exp(-((gamma * p) ** alpha))

        def mu_p(p, alpha, beta, gamma, delta):
            if np.all(alpha == 1):
                mu = delta * p - 2 * beta * gamma * p * np.log(gamma * p) / np.pi
            else:
                mu = delta * p + beta * np.tan(np.pi * alpha / 2) * (
                    (gamma * p) ** alpha - gamma * p
                )
            return mu

        series_sum = 0
        for p in np.arange(1, 150):
            rho = rho_p(p, alpha, gamma)
            mu = mu_p(p, alpha, beta, gamma, delta)
            series_sum += rho * np.cos(p * x - mu)

        pdf_values = 1 / (2 * np.pi) * (1 + 2 * series_sum)

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


wrapstable = wrapstable_gen(name="wrapstable")
