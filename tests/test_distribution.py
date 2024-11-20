import numpy as np

from pycircstat2.distribution import (
    cardioid,
    cartwright,
    circularuniform,
    inverse_batschelet,
    jonespewsey,
    jonespewsey_asymext,
    jonespewsey_sineskewed,
    vonmises,
    vonmises_ext,
    wrapcauchy,
    wrapnorm,
)


def test_circularuniform():

    np.testing.assert_approx_equal(circularuniform.cdf(2), 0.3183, significant=5)
    np.testing.assert_approx_equal(circularuniform.ppf(1 / np.pi), 2)


def test_cardioid():

    np.testing.assert_approx_equal(
        cardioid.cdf(np.pi, rho=0.3, mu=np.pi / 2), 0.6909859, significant=5
    )
    np.testing.assert_approx_equal(
        cardioid.ppf(0.6909859, rho=0.3, mu=np.pi / 2), np.pi
    )


def test_cartwright():

    np.testing.assert_approx_equal(
        cartwright.cdf(3 * np.pi / 4, zeta=0.1, mu=np.pi / 2),
        0.9641666531258773,
        significant=5,
    )
    np.testing.assert_approx_equal(
        cartwright.ppf(0.9641666531258773, zeta=0.1, mu=np.pi / 2).round(5),
        3 * np.pi / 4,
        significant=5,
    )


def test_wrapcauchy():

    np.testing.assert_approx_equal(
        wrapcauchy.cdf(np.pi / 6, rho=0.75, mu=np.pi / 2).round(3),
        0.0320,
        significant=3,
    )
    np.testing.assert_approx_equal(
        wrapcauchy.ppf(0.0320, rho=0.75, mu=np.pi / 2),
        np.pi / 6,
        significant=3,
    )


def test_wrapnorm():

    np.testing.assert_approx_equal(
        wrapnorm.cdf(np.pi / 6, rho=0.75, mu=np.pi / 2).round(4),
        0.0645,
        significant=3,
    )
    np.testing.assert_approx_equal(
        wrapnorm.ppf(0.5, rho=0.75, mu=np.pi / 2),
        1.6073,
        significant=4,
    )


def test_vonmises():

    np.testing.assert_approx_equal(
        vonmises.cdf(np.pi / 6, kappa=2.37, mu=np.pi / 2).round(4),
        0.0543,
        significant=3,
    )
    np.testing.assert_approx_equal(
        vonmises.ppf(0.5, kappa=2.37, mu=np.pi / 2),
        1.6139,
        significant=4,
    )


def test_jonespewsey():

    np.testing.assert_approx_equal(
        jonespewsey.cdf(np.pi / 2, kappa=2, psi=-1.5, mu=np.pi / 2).round(7),
        0.4401445,
        significant=7,
    )
    # take a long time to run jonespewsey.ppf()
    # might need to implement the method explicitly
    np.testing.assert_approx_equal(
        jonespewsey.ppf(q=0.4401445, kappa=2, psi=-1.5, mu=np.pi / 2),
        np.pi / 2,
        significant=7,
    )


def test_vonmises_ext():

    np.testing.assert_approx_equal(
        vonmises_ext.cdf(x=3 * np.pi / 4, kappa=2, nu=-0.5, mu=np.pi / 2).round(4),
        0.7120,
        significant=4,
    )
    np.testing.assert_approx_equal(
        vonmises_ext.ppf(q=0.5, kappa=2, nu=-0.5, mu=np.pi / 2).round(4),
        1.7301,
        significant=4,
    )


def test_jonespewsey_sineskewed():

    np.testing.assert_approx_equal(
        jonespewsey_sineskewed.cdf(
            x=3 * np.pi / 2, kappa=2, psi=1, lmbd=0.5, xi=np.pi / 2
        ).round(4),
        0.9446,
        significant=4,
    )
    np.testing.assert_approx_equal(
        jonespewsey_sineskewed.ppf(q=0.5, kappa=2, psi=1, lmbd=0.5, xi=np.pi / 2).round(
            4
        ),
        2.1879,
        significant=4,
    )


def test_jonespewsey_asymext():

    np.testing.assert_approx_equal(
        jonespewsey_asymext.cdf(
            x=np.pi / 2, kappa=2, psi=-1, nu=0.75, xi=np.pi / 2
        ).round(4),
        0.7535,
        significant=4,
    )
    np.testing.assert_approx_equal(
        jonespewsey_asymext.ppf(q=0.5, kappa=2, psi=-1, nu=0.75, xi=np.pi / 2).round(4),
        1.0499,
        significant=4,
    )


def test_inverse_batschelet():

    np.testing.assert_approx_equal(
        inverse_batschelet.cdf(
            x=np.pi / 2, kappa=2, nu=-0.5, lmbd=0.7, xi=np.pi / 2
        ).round(4),
        0.1180,
        significant=4,
    )
    np.testing.assert_approx_equal(
        inverse_batschelet.ppf(q=0.5, kappa=2, nu=-0.5, lmbd=0.7, xi=np.pi / 2).round(
            4
        ),
        2.5138,
        significant=4,
    )
