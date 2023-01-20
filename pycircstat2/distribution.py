import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import root
from scipy.special import gamma, i0, i1
from scipy.stats import rv_continuous, vonmises

###########################
## Symmetric Distribtion ##
###########################


class cardioid_gen(rv_continuous):
    """Cardioid Distribution

    Method
    ------
    pdf(x, rho, loc=0, scale=1)
        Probability density function.

    cdf(x, rho, loc=0, scale=1)
        Cumulative distribution function.

    Notes
    -----
    Implementation from 4.3.4 of Pewsey et al. (2014)
    """

    def _argcheck(self, rho):
        return 0 < rho <= 1 / 2

    def _pdf(self, x, rho):
        return (1 + 2 * rho * np.cos(x)) / 2.0 / np.pi

    def _cdf(self, x, rho):
        return rho * np.sin(x) + x / (2 * np.pi)


cardioid = cardioid_gen(name="cardioid")


class cartwright_gen(rv_continuous):
    """Cartwright's Power-of-Cosine Distribution

    Note
    ----
    Implementation from 4.3.5 of Pewsey et al. (2014)
    """

    def _argcheck(self, zeta):
        return zeta > 0

    def _pdf(self, x, zeta):
        return (
            (2 ** (-1 + 1 / zeta) * (gamma(1 + 1 / zeta)) ** 2)
            * (1 + np.cos(x)) ** (1 / zeta)
            / (np.pi * gamma(1 + 2 / zeta))
        )


cartwright = cartwright_gen(name="cartwright")


class wrapnorm_gen(rv_continuous):
    """Wrapped Normal Distribution

    Note
    ----
    Implementation from 4.3.7 of Pewsey et al. (2014)
    """

    def _argcheck(self, rho):
        return 0 < rho <= 1

    def _pdf(self, x, rho):
        return (
            1 + 2 * np.sum([rho ** (p**2) * np.cos(p * x) for p in range(1, 4)], 0)
        ) / (2 * np.pi)


wrapnorm = wrapnorm_gen(name="wrapped_normal")

# scipy.stats.wrapcauchy seems broken
# thus we reimplemented it.
class wrapcauchy_gen(rv_continuous):

    """Wrapped Cauchy Distribution

    Note
    ----
    Implementation from 4.3.6 of Pewsey et al. (2014)
    """

    def _argcheck(self, rho):
        return 0 < rho <= 1

    def _pdf(self, x, rho):
        return (1 - rho**2) / (2 * np.pi * (1 + rho**2 - 2 * rho * np.cos(x)))

    def _cdf(self, x, rho):
        num = (1 + rho**2) * np.cos(x) - 2 * rho
        den = 1 + rho**2 - 2 * rho * np.cos(x)
        return np.arccos(num / den)


wrapcauchy = wrapcauchy_gen(name="wrapped_cauchy")


class jonespewsey_gen(rv_continuous):

    """Jones-Pewsey Distribution

    Note
    ----
    Implementation from 4.3.9 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, psi):
        return (kappa >= 0) and (-np.inf <= psi <= np.inf)

    def _pdf(self, x, kappa, psi):

        if (kappa < 0.001).all():
            return 1 / (2 * np.pi)
        else:
            if np.isclose(np.abs(psi), 0).all():
                return 1 / (2 * np.pi * i0(kappa)) * np.exp(kappa * np.cos(x))
            else:

                def kernel(x):
                    return (
                        np.cosh(kappa * psi) + np.sinh(kappa * psi) * np.cos(x)
                    ) ** (1 / psi) / (2 * np.pi * np.cosh(kappa * np.pi))

                c = quad_vec(kernel, a=-np.pi, b=np.pi)[0]

                return kernel(x) / c


jonespewsey = jonespewsey_gen(name="jonespewsey")

#########################
## Symmetric Extention ##
#########################


class vonmisesext_gen(rv_continuous):

    """Flat-topped von Mises Distribution

    Note
    ----
    Implementation from 4.3.10 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, nu):
        return (kappa >= 0) and (-1 <= nu <= 1)

    def _pdf(self, x, kappa, nu):
        def kernel(x):
            return np.exp(kappa * np.cos(x + nu * np.sin(x)))

        c = 1 / quad_vec(kernel, a=-np.pi, b=np.pi)[0]

        return c * kernel(x)


vonmisesext = vonmisesext_gen(name="vonmises_ext")

###########################
## Sine-Skewed Extention ##
###########################


class jonespewsey_sineskewed_gen(rv_continuous):

    """Sine-Skewed Jones-Pewsey Distribution

    Note
    ----
    Implementation from 4.3.11 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, psi, lmda):
        return (kappa >= 0) and (-np.inf <= psi <= np.inf) and (-1 <= lmda <= 1)

    def _pdf(self, x, kappa, psi, lmda):

        if (kappa < 0.001).all():
            return 1 / (2 * np.pi) * (1 + lmda * np.sin(x))
        else:
            if np.isclose(np.abs(psi), 0).all():
                return (
                    1
                    / (2 * np.pi * i0(kappa))
                    * np.exp(kappa * np.cos(x))
                    * (1 + lmda * np.sin(x))
                )
            else:

                def kernel(x):
                    return (
                        np.cosh(kappa * psi) + np.sinh(kappa * psi) * np.cos(x)
                    ) ** (1 / psi) / (2 * np.pi * np.cosh(kappa * np.pi))

                c = quad_vec(kernel, a=-np.pi, b=np.pi)[0]

                return (1 + lmda * np.sin(x)) * kernel(x) / c


jonespewsey_sineskewed = jonespewsey_sineskewed_gen(name="jonespewsey_sineskewed")

##########################
## Asymmetric Extention ##
##########################


class jonespewsey_asymext_gen(rv_continuous):

    """Asymmetric Extended Jones-Pewsey Distribution

    Note
    ----
    Implementation from 4.3.12 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, psi, nu):
        return (kappa >= 0) and (-np.inf <= psi <= np.inf) and (0 <= nu < 1)

    def _pdf(self, x, kappa, psi, nu):

        # if np.isclose(np.abs(psi), 0).all():
        if np.isclose(np.abs(psi), 0).all():

            def kernel(x):
                return np.exp(kappa * np.cos(x + nu * np.cos(x)))

            c = quad_vec(kernel, a=-np.pi, b=np.pi)[0]
            return kernel(x) / c
        else:

            def kernel(x):
                return (
                    np.cosh(kappa * psi)
                    + np.sinh(kappa * psi) * np.cos(x + nu * np.cos(x))
                ) ** (1 / psi)

            c = quad_vec(kernel, a=-np.pi, b=np.pi)[0]

            return kernel(x) / c


jonespewsey_asymext = jonespewsey_asymext_gen(name="jonespewsey_asymext")


class inverse_batschelet_gen(rv_continuous):

    """Inverse Batschelet distribution.

    Note
    ----
    Implementation from 4.3.13 of Pewsey et al. (2014)
    """

    def _argcheck(self, kappa, nu, lmbd):
        return (kappa >= 0) and (-1 <= nu <= 1) and (-1 <= lmbd <= 1)

    def _pdf(self, x, kappa, nu, lmbd):
        def _tnu(x, nu):
            def _tnuinv(z, nu):
                return z - nu * (1 + np.cos(z)) - x

            y = root(_tnuinv, x0=np.zeros_like(x), args=(nu)).x
            y[y > np.pi] -= 2 * np.pi
            return y

        def _slmbdinv(x, lmbd):
            if np.isclose(lmbd, -1).all():
                return x
            else:

                def _slmbd(z, lmbd):
                    return z - 0.5 * (1 + lmbd) * np.sin(z) - x

                y = root(_slmbd, x0=np.zeros_like(x), args=(lmbd)).x
                return y

        def _A1(kappa):
            return i1(kappa) / i0(kappa)

        # _c checked
        def _c(kappa, lmbd):
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

        arg1 = _tnu(x, nu)
        arg2 = _slmbdinv(arg1, lmbd)
        c = _c(kappa, lmbd)
        if np.isclose(lmbd, -1).all():
            return c * np.exp(kappa * np.cos(arg1 - np.sin(arg1)))
        else:
            con1 = (1 - lmbd) / (1 + lmbd)
            con2 = (2 * lmbd) / (1 + lmbd)
            return c * np.exp(kappa * np.cos(con1 * arg1 + con2 * arg2))


inverse_batschelet = inverse_batschelet_gen(name="inverse_batschelet")
