import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gamma, i0
from scipy.stats import rv_continuous, vonmises


class cardioid_gen(rv_continuous):
    def _argcheck(self, rho):
        return 0 < rho <= 1 / 2

    def _pdf(self, x, rho):
        return (1 + 2 * rho * np.cos(x)) / 2.0 / np.pi

    def _cdf(self, x, rho):
        return rho * np.sin(x) + x / (2 * np.pi)


cardioid = cardioid_gen(name="cardioid")


class cartwright_gen(rv_continuous):
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
    def _argcheck(self, rho):
        return 0 < rho <= 1

    def _pdf(self, x, rho):
        return (
            1 + 2 * np.sum([rho ** (p**2) * np.cos(p * x) for p in range(1, 4)], 0)
        ) / (2 * np.pi)


wrapnorm = wrapnorm_gen(name="wrapped_normal")

# scipy.stats.wrapcauchy seems broken
class wrapcauchy_gen(rv_continuous):
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


class vonmisesext_gen(rv_continuous):
    def _argcheck(self, kappa, nu):
        return (kappa >= 0) and (-1 <= nu <= 1)

    def _pdf(self, x, kappa, nu):
        def kernel(x):
            return np.exp(kappa * np.cos(x + nu * np.sin(x)))

        c = 1 / quad_vec(kernel, a=-np.pi, b=np.pi)[0]

        return c * kernel(x)

vonmisesext = vonmisesext_gen(name="vonmises")


class jonespewsey_sineskewed_gen(rv_continuous):
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
