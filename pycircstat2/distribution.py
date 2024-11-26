import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.optimize import root
from scipy.special import gamma, i0, i1
from scipy.stats import rv_continuous

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

    Method
    ------
    pdf(x, rho, mu)
        Probability density function.

    cdf(x, rho, mu)
        Cumulative distribution function.

    Notes
    -----
    1. Implementation from 4.3.4 of Pewsey et al. (2014)
    2. Don't use the `loc` argument from scipy.stats.rv_continous.
       Use `mu` to shift the location.
    """

    def _argcheck(self, rho, mu):
        return 0 <= rho <= 0.5 and 0 <= mu <= np.pi * 2

    def _pdf(self, x, rho, mu):
        return (1 + 2 * rho * np.cos(x - mu)) / 2.0 / np.pi

    def _cdf(self, x, rho, mu):
        return (x + 2 * rho * (np.sin(x - mu) + np.sin(mu))) / (2 * np.pi)


cardioid = cardioid_gen(name="cardioid")


class cartwright_gen(rv_continuous):
    """Cartwright's Power-of-Cosine Distribution

    Method
    ------
    pdf(x, zeta, mu)
        Probability density function.

    cdf(x, zeta, mu)
        Cumulative distribution function.

    Note
    ----
    Implementation from 4.3.5 of Pewsey et al. (2014)
    """

    def _argcheck(self, zeta, mu):
        return zeta > 0 and 0 <= mu <= 2 * np.pi

    def _pdf(self, x, zeta, mu):
        return (
            (2 ** (-1 + 1 / zeta) * (gamma(1 + 1 / zeta)) ** 2)
            * (1 + np.cos(x - mu)) ** (1 / zeta)
            / (np.pi * gamma(1 + 2 / zeta))
        )

    def _cdf(self, x, zeta, mu):
        @np.vectorize
        def _cdf_single(x, zeta, mu):
            return quad(self._pdf, a=0, b=x, args=(zeta, mu))

        return _cdf_single(x, zeta, mu)


cartwright = cartwright_gen(name="cartwright")


class wrapnorm_gen(rv_continuous):
    """Wrapped Normal Distribution

    Methods
    -------
    pdf(x, rho, mu)
        Probability density function.

    cdf(x, rho, mu)
        Cumulative distribution function.

    Note
    ----
    Implementation from 4.3.7 of Pewsey et al. (2014)
    """

    def _argcheck(self, rho, mu):
        return 0 < rho <= 1 and 0 <= mu <= np.pi * 2

    def _pdf(self, x, rho, mu):
        return (
            1
            + 2
            * np.sum([rho ** (p**2) * np.cos(p * (x - mu)) for p in range(1, 30)], 0)
        ) / (2 * np.pi)

    def _cdf(self, x, rho, mu):
        @np.vectorize
        def _cdf_single(x, rho, mu):
            return quad(self._pdf, a=0, b=x, args=(rho, mu))

        return _cdf_single(x, rho, mu)


wrapnorm = wrapnorm_gen(name="wrapped_normal")


class wrapcauchy_gen(rv_continuous):
    """Wrapped Cauchy Distribution

    Methods
    -------
    pdf(x, rho, mu)
        Probability density function.

    cdf(x, rho, mu)
        Cumulative distribution function.

    Note
    ----
    Implementation from 4.3.6 of Pewsey et al. (2014)
    """

    def _argcheck(self, rho, mu):
        return 0 < rho <= 1 and 0 <= mu <= np.pi * 2

    def _pdf(self, x, rho, mu):
        return (1 - rho**2) / (2 * np.pi * (1 + rho**2 - 2 * rho * np.cos(x - mu)))

    def _cdf(self, x, rho, mu):
        @np.vectorize
        def _cdf_single(x, rho, mu):
            return quad(self._pdf, a=0, b=x, args=(rho, mu))

        return _cdf_single(x, rho, mu)


wrapcauchy = wrapcauchy_gen(name="wrapcauchy")


class vonmises_gen(rv_continuous):
    """Von Mises Distribution

    Methods
    -------
    pdf(x, kappa, mu)
        Probability density function.

    cdf(x, kappa, mu)
        Cumulative distribution function.

    ppf(q, kappa, mu)
        Percent-point function (inverse of CDF).

    rvs(kappa, mu, size=None, random_state=None)
        Random variates.

    fit(data, *args, **kwargs)
        Fit the distribution to the data and return the parameters (kappa, mu).

    Note
    ----
    Implementation from 4.3.8 of Pewsey et al. (2014)
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

    def _argcheck(self, kappa, mu):
        return kappa > 0 and 0 <= mu <= np.pi * 2

    def _pdf(self, x, kappa, mu):
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

    def _logpdf(self, x, kappa, mu):
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

    def _cdf(self, x, kappa, mu):
        @np.vectorize
        def _cdf_single(x, kappa, mu):
            integral, _ = quad(self._pdf, a=0, b=x, args=(kappa, mu))
            return integral

        return _cdf_single(x, kappa, mu)

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

    def _rvs(self, kappa, mu, size=None, random_state=None):
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
        kappa = getattr(self, "kappa", None)
        mu = getattr(self, "mu", None)

        # Override instance parameters if provided in args/kwargs
        kappa = kwargs.pop("kappa", kappa)
        mu = kwargs.pop("mu", mu)

        # Ensure required parameters are provided
        if kappa is None or mu is None:
            raise ValueError("Both 'kappa' and 'mu' must be provided.")

        # Call the private _rvs method
        return self._rvs(kappa, mu, size=size, random_state=random_state)

    def fit(self, data, *args, **kwargs):
        """
        Override fit method to fix loc=0 and scale=1 and return only (kappa, mu).
        """
        kwargs["floc"] = 0  # Fix loc to 0
        kwargs["fscale"] = 1  # Fix scale to 1
        params = super().fit(
            data, *args, **kwargs
        )  # Perform fit with fixed loc and scale
        kappa, mu = params[:-2]  # Extract kappa and mu (excluding loc and scale)
        return kappa, mu

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
        (kappa, mu) = self._parse_args(*args, **kwargs)[0]
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
        (kappa, mu) = self._parse_args(*args, **kwargs)[0]
        return 1 - i1(kappa) / i0(kappa)

    def std(self, *args, **kwargs):
        """
        Circular standard deviation of the Von Mises distribution.

        Returns
        -------
        std : float
            The circular standard deviation, derived from `kappa`.
        """
        (kappa, mu) = self._parse_args(*args, **kwargs)[0]
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
        (kappa, mu) = self._parse_args(*args, **kwargs)[0]
        return -np.log(i0(kappa)) + (kappa * i1(kappa)) / i0(kappa)


vonmises = vonmises_gen(name="vonmises")


class jonespewsey_gen(rv_continuous):
    """Jones-Pewsey Distribution

    Methods
    -------
    pdf(x, kappa, psi, mu)
        Probability density function.

    cdf(x, kappa, psi, mu)
        Cumulative distribution function.


    Note
    ----
    Implementation from 4.3.9 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, psi, mu):
        # we can save a lot of computation if c is automatically computed before others
        # but tbh this is a hack, and I don't have a better way to do it atm.
        self._c = _c_jonespewsey(kappa, psi, mu)
        return (kappa >= 0) and (-np.inf <= psi <= np.inf) and (0 <= mu <= np.pi * 2)

    def _pdf(self, x, kappa, psi, mu):

        if np.all(kappa < 0.001):
            return 1 / (2 * np.pi)
        else:
            if np.isclose(np.abs(psi), 0).all():
                return 1 / (2 * np.pi * i0(kappa)) * np.exp(kappa * np.cos(x - mu))
            else:
                return _kernel_jonespewsey(x, kappa, psi, mu) / self._c

    def _cdf(self, x, kappa, psi, mu):
        def vonmises_pdf(x, kappa, psi, mu, c):
            return c * np.exp(kappa * np.cos(x - mu))

        if np.isclose(np.abs(psi), 0).all():
            c = self._c

            @np.vectorize
            def _cdf_single(x, kappa, psi, mu, c):
                return quad(vonmises_pdf, a=0, b=x, args=(kappa, psi, mu, c))

            return _cdf_single(x, kappa, psi, mu, c)
        else:

            @np.vectorize
            def _cdf_single(x, kappa, psi, mu):
                return quad(self._pdf, a=0, b=x, args=(kappa, psi, mu))

            return _cdf_single(x, kappa, psi, mu)


jonespewsey = jonespewsey_gen(name="jonespewsey")

####################################
## Helper Functions: Jones-Pewsey ##
####################################


def _kernel_jonespewsey(x, kappa, psi, mu):
    return (np.cosh(kappa * psi) + np.sinh(kappa * psi) * np.cos(x - mu)) ** (
        1 / psi
    ) / (2 * np.pi * np.cosh(kappa * np.pi))


def _c_jonespewsey(kappa, psi, mu):
    if np.all(kappa < 0.001):
        return np.ones_like(kappa) * 1 / 2 / np.pi
    else:
        if np.isclose(np.abs(psi), 0).all():
            return 1 / (2 * np.pi * i0(kappa))
        else:
            c = quad_vec(_kernel_jonespewsey, a=-np.pi, b=np.pi, args=(kappa, psi, mu))[
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
    pdf(x, kappa, nu, mu)
        Probability density function.

    cdf(x, kappa, nu, mu)
        Cumulative distribution function.

    Note
    ----
    Implementation from 4.3.10 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, nu, mu):
        self.c = _c_vmext(kappa, nu, mu)
        return (kappa >= 0) and (-1 <= nu <= 1) and (0 <= mu <= np.pi * 2)

    def _pdf(self, x, kappa, nu, mu):
        return self.c * _kernel_vmext(x, kappa, nu, mu)

    def _cdf(self, x, kappa, nu, mu):
        @np.vectorize
        def _cdf_single(x, kappa, nu, mu):
            return quad(self._pdf, a=0, b=x, args=(kappa, nu, mu))

        return _cdf_single(x, kappa, nu, mu)


vonmises_ext = vonmises_ext_gen(name="vonmises_ext")

###########################################
## Helper Functions: extended von Mises  ##
###########################################


def _kernel_vmext(x, kappa, nu, mu):
    return np.exp(kappa * np.cos(x - mu + nu * np.sin(x - mu)))


def _c_vmext(kappa, nu, mu):
    c = 1 / quad_vec(_kernel_vmext, a=-np.pi, b=np.pi, args=(kappa, nu, mu))[0]
    return c


###########################
## Sine-Skewed Extention ##
###########################


class jonespewsey_sineskewed_gen(rv_continuous):
    """Sine-Skewed Jones-Pewsey Distribution

    Methods
    -------
    pdf(x, kappa, psi, lmbd, xi)
        Probability density function.

    cdf(x, kappa, psi, lmbd, xi)
        Cumulative distribution function.


    Note
    ----
    Implementation from 4.3.11 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, psi, lmbd, xi):
        # reuse helpers from jonespewsey()
        self.c = _c_jonespewsey(kappa, psi, xi)
        return (
            (kappa >= 0)
            and (-np.inf <= psi <= np.inf)
            and (-1 <= lmbd <= 1)
            and (0 <= xi <= np.pi * 2)
        )

    def _pdf(self, x, kappa, psi, lmbd, xi):

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
                    * _kernel_jonespewsey(x, kappa, psi, xi)
                    / self.c
                )

    def _cdf(self, x, kappa, psi, lmbd, xi):
        @np.vectorize
        def _cdf_single(x, kappa, psi, lmbd, xi):
            return quad(self._pdf, a=0, b=x, args=(kappa, psi, lmbd, xi))

        return _cdf_single(x, kappa, psi, lmbd, xi)


jonespewsey_sineskewed = jonespewsey_sineskewed_gen(name="jonespewsey_sineskewed")

##########################
## Asymmetric Extention ##
##########################


class jonespewsey_asymext_gen(rv_continuous):
    """Asymmetric Extended Jones-Pewsey Distribution

    Methods
    -------
    pdf(x, kappa, psi, nu, xi)
        Probability density function.

    cdf(x, kappa, psi, nu, xi)
        Cumulative distribution function.


    Note
    ----
    Implementation from 4.3.12 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, psi, nu, xi):
        self.c = _c_jonespewsey_asymext(kappa, psi, nu, xi)
        return (
            (kappa >= 0)
            and (-np.inf <= psi <= np.inf)
            and (0 <= nu < 1)
            and (0 <= xi <= np.pi * 2)
        )

    def _pdf(self, x, kappa, psi, nu, xi):
        return _kernel_jonespewsey_asymext(x, kappa, psi, nu, xi) / self.c

    def _cdf(self, x, kappa, psi, nu, xi):
        @np.vectorize
        def _cdf_single(x, kappa, psi, nu, xi):
            return quad(self._pdf, a=0, b=x, args=(kappa, psi, nu, xi))

        return _cdf_single(x, kappa, psi, nu, xi)


jonespewsey_asymext = jonespewsey_asymext_gen(name="jonespewsey_asymext")


def _kernel_jonespewsey_asymext(x, kappa, psi, nu, xi):
    if np.isclose(np.abs(psi), 0).all():
        return np.exp(kappa * np.cos(x - xi + nu * np.cos(x - xi)))
    else:
        return (
            np.cosh(kappa * psi)
            + np.sinh(kappa * psi) * np.cos(x - xi + nu * np.cos(x - xi))
        ) ** (1 / psi)


def _c_jonespewsey_asymext(kappa, psi, nu, xi):

    c = quad_vec(
        _kernel_jonespewsey_asymext, a=-np.pi, b=np.pi, args=(kappa, psi, nu, xi)
    )[0]
    return c


class inverse_batschelet_gen(rv_continuous):
    """Inverse Batschelet distribution.

    Methods
    -------
    pdf(x, kappa, psi, nu, lmbd, xi)
        Probability density function.

    cdf(x, kappa, psi, nu, lmbd, xi)
        Cumulative distribution function.


    Note
    ----
    Implementation from 4.3.13 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, nu, lmbd, xi):
        self.c = _c_invbatschelet(kappa, lmbd)
        if np.isclose(lmbd, -1).all():
            self.con1, self.con2 = 0, 0
        else:
            self.con1 = (1 - lmbd) / (1 + lmbd)
            self.con2 = (2 * lmbd) / (1 + lmbd)
        return (
            (kappa >= 0)
            and (-1 <= nu <= 1)
            and (-1 <= lmbd <= 1)
            and (0 <= xi <= np.pi * 2)
        )

    def _pdf(self, x, kappa, nu, lmbd, xi):

        arg1 = _tnu(x, nu, xi)
        arg2 = _slmbdinv(arg1, lmbd)

        if np.isclose(lmbd, -1).all():
            return self.c * np.exp(kappa * np.cos(arg1 - np.sin(arg1)))
        else:
            return self.c * np.exp(kappa * np.cos(self.con1 * arg1 + self.con2 * arg2))

    def _cdf(self, x, kappa, nu, lmbd, xi):
        @np.vectorize
        def _cdf_single(x, kappa, nu, lmbd, xi):
            return quad(self._pdf, a=0, b=x, args=(kappa, nu, lmbd, xi))

        return _cdf_single(x, kappa, nu, lmbd, xi)


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
