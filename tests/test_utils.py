import numpy as np

from pycircstat2.utils import (
    angmod,
    angular_distance,
    data2rad,
    rad2data,
    rotate_data,
    time2float,
)


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


def test_angmod():
    # Test case 1: Default range [0, 2π)
    assert np.isclose(angmod(-np.pi), np.pi), "Failed default bounds with -π"
    assert np.isclose(angmod(0), 0), "Failed default bounds with 0"
    assert np.isclose(angmod(2 * np.pi), 0), "Failed default bounds with 2π"
    assert np.isclose(angmod(3 * np.pi), np.pi), "Failed default bounds with 3π"
    assert np.isclose(angmod(-3 * np.pi), np.pi), "Failed default bounds with -3π"

    # Test case 2: Custom bounds [-π, π)
    assert np.isclose(
        angmod(-np.pi, [-np.pi, np.pi]), -np.pi
    ), "Failed bounds [-π, π) with -π"
    assert np.isclose(angmod(0, [-np.pi, np.pi]), 0), "Failed bounds [-π, π) with 0"
    assert np.isclose(
        angmod(2 * np.pi, [-np.pi, np.pi]), 0
    ), "Failed bounds [-π, π) with 2π"
    assert np.isclose(
        angmod(3 * np.pi, [-np.pi, np.pi]), -np.pi
    ), "Failed bounds [-π, π) with 3π"
    assert np.isclose(
        angmod(-3 * np.pi, [-np.pi, np.pi]), -np.pi
    ), "Failed bounds [-π, π) with -3π"

    # Test case 3: Custom bounds [0, 1)
    assert np.isclose(angmod(-1.5, [0, 1]), 0.5), "Failed bounds [0, 1) with -1.5"
    assert np.isclose(angmod(0.25, [0, 1]), 0.25), "Failed bounds [0, 1) with 0.25"
    assert np.isclose(angmod(1.75, [0, 1]), 0.75), "Failed bounds [0, 1) with 1.75"

    # Test case 4: Array input with default bounds [0, 2π)
    angles = np.array([-3 * np.pi, -np.pi, 0, np.pi, 3 * np.pi])
    expected = np.array([np.pi, np.pi, 0, np.pi, np.pi])
    np.testing.assert_allclose(
        angmod(angles),
        expected,
        err_msg="Failed array input with default bounds [0, 2π)",
    )

    # Test case 5: Array input with cu


def test_time2float():

    np.testing.assert_almost_equal(time2float(x="6:15"), 6.25)


def test_angular_distance():

    np.testing.assert_almost_equal(
        np.rad2deg(angular_distance(a=np.deg2rad(95), b=np.deg2rad(120))).round(3),
        25,
    )
    np.testing.assert_almost_equal(
        np.rad2deg(angular_distance(a=np.deg2rad(340), b=np.deg2rad(30))).round(3),
        50,
    )
    np.testing.assert_almost_equal(
        np.rad2deg(angular_distance(a=np.deg2rad(190), b=np.deg2rad(5))).round(3),
        175,
    )

def test_rotate_data():
    # Test case 1: Rotating angles in radians
    alpha = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    rotated = rotate_data(alpha, np.pi / 4, unit="radian")
    expected = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
    np.testing.assert_allclose(rotated, expected, atol=1e-6)

    # Test case 2: Rotating angles in degrees
    alpha_deg = np.array([0., 90., 180., 270.])
    rotated_deg = rotate_data(alpha_deg, 45, unit="degree")
    expected_deg = np.array([45., 135., 225., 315.])
    np.testing.assert_allclose(rotated_deg, expected_deg, atol=1e-6)

    # Test case 3: Rotating angles in hours
    alpha_hour = np.array([0., 6., 12., 18.])
    rotated_hour = rotate_data(alpha_hour, 3, unit="hour")
    expected_hour = np.array([3., 9., 15., 21.])
    np.testing.assert_allclose(rotated_hour, expected_hour, atol=1e-6)

    # Test case 4: Rotation that wraps around the cycle
    rotated_wrap = rotate_data(np.array([350.]), 20, unit="degree")
    expected_wrap = np.array([10])
    np.testing.assert_allclose(rotated_wrap, expected_wrap, atol=1e-6)

    rotated_wrap_hour = rotate_data(np.array([22.]), 5, unit="hour")
    expected_wrap_hour = np.array([3.])
    np.testing.assert_allclose(rotated_wrap_hour, expected_wrap_hour, atol=1e-6)

    # Test case 5: Rotating an empty array should return an empty array
    np.testing.assert_array_equal(rotate_data(np.array([]), 45., unit="degree"), np.array([]))

    # Test case 6: Rotating by 0 should return the same values
    alpha_no_rotation = np.array([10., 50., 120.])
    np.testing.assert_allclose(
        rotate_data(alpha_no_rotation, 0, unit="degree"), alpha_no_rotation
    )
