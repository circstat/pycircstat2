import numpy as np

from pycircstat2 import Circular, load_data
from pycircstat2.hypothesis import V_test, one_sample_test, rayleigh_test


def test_rayleigh_test():

    # Ch27 Example 1 (Zar, 2010, P667)
    # Using data from Ch26 Example 2.
    data_zar_ex2_ch26 = load_data("D1", source="zar_2010")
    circ_zar_ex1_ch27 = Circular(data_zar_ex2_ch26["θ"].values)

    # computed directly from r and n
    z, p = rayleigh_test(n=circ_zar_ex1_ch27.n, r=circ_zar_ex1_ch27.r)
    np.testing.assert_approx_equal(z, 5.448, significant=3)
    assert 0.001 < p < 0.002

    # computed directly from alpha
    z, p = rayleigh_test(alpha=circ_zar_ex1_ch27.alpha)
    np.testing.assert_approx_equal(z, 5.448, significant=3)
    assert 0.001 < p < 0.002


def test_V_test():

    # Ch27 Example 2 (Zar, 2010, P669)
    data_zar_ex2_ch27 = load_data("D7", source="zar_2010")
    circ_zar_ex2_ch27 = Circular(data_zar_ex2_ch27["θ"].values)

    # computed directly from r and n
    V, u, p = V_test(
        mean=circ_zar_ex2_ch27.mean,
        n=circ_zar_ex2_ch27.n,
        r=circ_zar_ex2_ch27.r,
        angle=90,
        unit="degree",
    )

    np.testing.assert_approx_equal(V, 9.498, significant=3)
    np.testing.assert_approx_equal(u, 4.248, significant=3)
    assert p < 0.0005

    # computed directly from alpha
    V, u, p = V_test(
        alpha=circ_zar_ex2_ch27.alpha,
        angle=90,
        unit="degree",
    )

    np.testing.assert_approx_equal(V, 9.498, significant=3)
    np.testing.assert_approx_equal(u, 4.248, significant=3)
    assert p < 0.0005


def test_one_sample_test():

    # Ch27 Example 3 (Zar, 2010, P669)
    # Using data from Ch27 Example 2
    data_zar_ex2_ch27 = load_data("D7", source="zar_2010")
    circ_zar_ex3_ch27 = Circular(data_zar_ex2_ch27["θ"].values, unit="degree")

    # # computed directly from lb and ub
    # reject_null = one_sample_test(
    #     lb=circ_zar_ex3_ch27.mean_lb,
    #     ub=circ_zar_ex3_ch27.mean_ub,
    #     angle=90,
    #     unit="degree",
    # )

    # assert reject_null is False

    # computed directly from alpha
    reject_null = one_sample_test(
        alpha=circ_zar_ex3_ch27.alpha, angle=90, unit="degree"
    )

    assert reject_null is False


# # # def test_symmetry_median_test():
