import numpy as np

from pycircstat2.utils import angrange, angular_distance, data2rad, rad2data, time2float


def test_data2rad():

    # Ch26 Example 1.1 (Zar, 2010, P647)
    a = data2rad(data=6, k=24)
    np.testing.assert_approx_equal(np.rad2deg(a), 90.0, significant=1)

    # Ch26 Example 1.2 (Zar, 2010, P647)
    a = data2rad(data=6.25, k=24)
    np.testing.assert_approx_equal(np.rad2deg(a), 93.75, significant=2)

    # Ch26 Example 1.3 (Zar, 2010, P647)
    a = data2rad(data=45, k=365)
    np.testing.assert_approx_equal(np.rad2deg(a), 44.38, significant=2)


def test_rad2data():

    # Ch26 (Zar, 2010, P653)
    a = np.deg2rad(270)
    np.testing.assert_approx_equal(rad2data(rad=a, k=24), 18, significant=0)


def test_angrange():

    np.testing.assert_almost_equal(np.rad2deg(angrange(rad=np.deg2rad(0))), 0)
    np.testing.assert_almost_equal(np.rad2deg(angrange(rad=np.deg2rad(90))), 90)
    np.testing.assert_almost_equal(np.rad2deg(angrange(rad=np.deg2rad(180))), 180)
    np.testing.assert_almost_equal(np.rad2deg(angrange(rad=np.deg2rad(360))), 0)
    np.testing.assert_almost_equal(np.rad2deg(angrange(rad=np.deg2rad(361))), 1)
    np.testing.assert_almost_equal(np.rad2deg(angrange(rad=np.deg2rad(-1))), 359)


def test_time2float():

    np.testing.assert_almost_equal(time2float(x="6:15"), 6.25)


def test_angular_distance():

    np.testing.assert_almost_equal(
        np.rad2deg(angular_distance(a=np.deg2rad(95), b=np.deg2rad(120))).round(3), 25
    )
    np.testing.assert_almost_equal(
        np.rad2deg(angular_distance(a=np.deg2rad(340), b=np.deg2rad(30))).round(3), 50
    )
    np.testing.assert_almost_equal(
        np.rad2deg(angular_distance(a=np.deg2rad(190), b=np.deg2rad(5))).round(3), 175
    )
