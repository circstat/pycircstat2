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
    pdf(x, rho, mu, scale=1)
        Probability density function.

    cdf(x, rho, mu, scale=1)
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
    pdf(x, zeta, mu, scale=1)
        Probability density function.

    cdf(x, zeta, mu, scale=1)
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
    pdf(x, rho, mu, scale=1)
        Probability density function.

    cdf(x, rho, mu, scale=1)
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


# scipy.stats.wrapcauchy seems broken
# thus we reimplemented it.
class wrapcauchy_gen(rv_continuous):
    """Wrapped Cauchy Distribution

    Methods
    -------
    pdf(x, rho, mu, scale=1)
        Probability density function.

    cdf(x, rho, mu, scale=1)
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


# probably less efficient than scipy.stats.vonmises
# but I would like to keep the parameterization of
# all distribution the same, e.g. pdf(x, *args, mu)
class vonmises(rv_continuous):
    """Von Mises Distribution

    Methods
    -------
    pdf(x, kappa, mu, scale=1)
        Probability density function.

    cdf(x, kappa, mu, scale=1)
        Cumulative distribution function.


    Note
    ----
    Implementation from 4.3.8 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, mu):
        return kappa > 0 and 0 <= mu <= np.pi * 2

    def _pdf(self, x, kappa, mu):
        return np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * i0(kappa))

    def _cdf(self, x, kappa, mu):
        @np.vectorize
        def _cdf_single(x, kappa, mu):
            return quad(self._pdf, a=0, b=x, args=(kappa, mu))

        return _cdf_single(x, kappa, mu)


vonmises = vonmises(name="vonmises")


class jonespewsey_gen(rv_continuous):
    """Jones-Pewsey Distribution

    Methods
    -------
    pdf(x, kappa, psi, mu, scale=1)
        Probability density function.

    cdf(x, kappa, psi, mu, scale=1)
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
    pdf(x, kappa, nu, mu, scale=1)
        Probability density function.

    cdf(x, kappa, nu, mu, scale=1)
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
    pdf(x, kappa, psi, lmbd, xi, scale=1)
        Probability density function.

    cdf(x, kappa, psi, lmbd, xi, scale=1)
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
    pdf(x, kappa, psi, nu, xi, scale=1)
        Probability density function.

    cdf(x, kappa, psi, nu, xi, scale=1)
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
    pdf(x, kappa, psi, nu, lmbd, xi, scale=1)
        Probability density function.

    cdf(x, kappa, psi, nu, lmbd, xi, scale=1)
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
