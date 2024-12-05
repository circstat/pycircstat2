import numpy as np

from pycircstat2.distributions import (
    cardioid,
    cartwright,
    circularuniform,
    inverse_batschelet,
    jonespewsey,
    jonespewsey_asym,
    jonespewsey_sineskewed,
    vonmises,
    vonmises_flattopped,
    wrapcauchy,
    wrapnorm,
)


def test_circularuniform():

    np.testing.assert_approx_equal(circularuniform.cdf(2), 0.3183, significant=5)
    np.testing.assert_approx_equal(circularuniform.ppf(1 / np.pi), 2)


def test_cardioid():

    cd = cardioid(rho=0.3, mu=np.pi / 2)
    np.testing.assert_approx_equal(cd.cdf(np.pi), 0.6909859, significant=5)
    np.testing.assert_approx_equal(cd.ppf(0.6909859), np.pi)


def test_cartwright():

    cw = cartwright(zeta=0.1, mu=np.pi / 2)
    np.testing.assert_approx_equal(
        cw.cdf(3 * np.pi / 4),
        0.9641666531258773,
        significant=5,
    )
    np.testing.assert_approx_equal(
        cw.ppf(0.9641666531258773).round(5),
        3 * np.pi / 4,
        significant=5,
    )


def test_wrapcauchy():

    wc = wrapcauchy(rho=0.75, mu=np.pi / 2)
    # P54, Pewsey, et al. (2014)
    np.testing.assert_approx_equal(
        wc.cdf(np.pi / 6).round(3),
        0.0320,
        significant=3,
    )

    # P54, Pewsey, et al. (2014)
    np.testing.assert_approx_equal(
        wc.ppf(0.0320),
        np.pi / 6,
        significant=3,
    )


def test_wrapnorm():

    wn = wrapnorm(rho=0.75, mu=np.pi / 2)
    np.testing.assert_approx_equal(
        wn.cdf(np.pi / 6).round(4),
        0.0645,
        significant=3,
    )
    np.testing.assert_approx_equal(
        wn.ppf(0.5),
        1.6073,
        significant=4,
    )


def test_vonmises():

    vm = vonmises(kappa=2.37, mu=np.pi / 2)
    np.testing.assert_approx_equal(
        vm.cdf(np.pi / 6).round(4),
        0.0543,
        significant=3,
    )
    np.testing.assert_approx_equal(
        vm.ppf(0.5),
        1.6139,
        significant=4,
    )


def test_jonespewsey():

    jp = jonespewsey(kappa=2, psi=-1.5, mu=np.pi / 2)

    np.testing.assert_approx_equal(
        jp.cdf(np.pi / 2).round(7),
        0.4401445,
        significant=7,
    )
    # take a long time to run jonespewsey.ppf()
    # might need to implement the method explicitly
    np.testing.assert_approx_equal(
        jp.ppf(q=0.4401445),
        np.pi / 2,
        significant=7,
    )


def test_vonmises_flattopped():

    vme = vonmises_flattopped(kappa=2, nu=-0.5, mu=np.pi / 2)
    np.testing.assert_approx_equal(
        vme.cdf(x=3 * np.pi / 4).round(4),
        0.7120,
        significant=4,
    )
    np.testing.assert_approx_equal(
        vme.ppf(q=0.5).round(4),
        1.7301,
        significant=4,
    )


def test_jonespewsey_sineskewed():

    jps = jonespewsey_sineskewed(kappa=2, psi=1, lmbd=0.5, xi=np.pi / 2)

    np.testing.assert_approx_equal(
        jps.cdf(x=3 * np.pi / 2).round(4),
        0.9446,
        significant=4,
    )
    np.testing.assert_approx_equal(
        jps.ppf(q=0.5).round(4),
        2.1879,
        significant=4,
    )


def test_jonespewsey_asym():

    jpa = jonespewsey_asym(kappa=2, psi=-1, nu=0.75, xi=np.pi / 2)
    np.testing.assert_approx_equal(
        jpa.cdf(x=np.pi / 2).round(4),
        0.7535,
        significant=4,
    )
    np.testing.assert_approx_equal(
        jpa.ppf(q=0.5).round(4),
        1.0499,
        significant=4,
    )


def test_inverse_batschelet():

    ib = inverse_batschelet(kappa=2, nu=-0.5, lmbd=0.7, xi=np.pi / 2)
    np.testing.assert_approx_equal(
        ib.cdf(x=np.pi / 2).round(4),
        0.1180,
        significant=4,
    )
    np.testing.assert_approx_equal(
        ib.ppf(q=0.5).round(4),
        2.5138,
        significant=4,
    )
