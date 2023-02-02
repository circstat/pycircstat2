import numpy as np

from pycircstat2 import Circular, load_data
from pycircstat2.hypothesis import (
    V_test,
    batschelet_test,
    chisquare_test,
    kuiper_test,
    omnibus_test,
    one_sample_test,
    rao_spacing_test,
    rayleigh_test,
    symmetry_test,
    wallraff_test,
    watson_test,
    watson_u2_test,
    watson_williams_test,
    wheeler_watson_test,
)


def test_rayleigh_test():

    # Ch27 Example 1 (Zar, 2010, P667)
    # Using data from Ch26 Example 2.
    data_zar_ex2_ch26 = load_data("D1", source="zar")
    circ_zar_ex1_ch27 = Circular(data=data_zar_ex2_ch26["θ"].values)

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
    data_zar_ex2_ch27 = load_data("D7", source="zar")
    circ_zar_ex2_ch27 = Circular(data=data_zar_ex2_ch27["θ"].values)

    # computed directly from r and n
    V, u, p = V_test(
        angle=np.deg2rad(90),
        mean=circ_zar_ex2_ch27.mean,
        n=circ_zar_ex2_ch27.n,
        r=circ_zar_ex2_ch27.r,
    )

    np.testing.assert_approx_equal(V, 9.498, significant=3)
    np.testing.assert_approx_equal(u, 4.248, significant=3)
    assert p < 0.0005

    # computed directly from alpha
    V, u, p = V_test(
        alpha=circ_zar_ex2_ch27.alpha,
        angle=np.deg2rad(90),
    )

    np.testing.assert_approx_equal(V, 9.498, significant=3)
    np.testing.assert_approx_equal(u, 4.248, significant=3)
    assert p < 0.0005


def test_one_sample_test():

    # Ch27 Example 3 (Zar, 2010, P669)
    # Using data from Ch27 Example 2
    data_zar_ex2_ch27 = load_data("D7", source="zar")
    circ_zar_ex3_ch27 = Circular(data=data_zar_ex2_ch27["θ"].values, unit="degree")

    # computed directly from lb and ub
    reject_null = one_sample_test(
        lb=circ_zar_ex3_ch27.mean_lb,
        ub=circ_zar_ex3_ch27.mean_ub,
        angle=np.deg2rad(90),
    )

    assert reject_null is False

    # computed directly from alpha
    reject_null = one_sample_test(alpha=circ_zar_ex3_ch27.alpha, angle=np.deg2rad(90))

    assert reject_null is False


def test_omnibus_test():

    data_zar_ex4_ch27 = load_data("D8", source="zar")
    circ_zar_ex4_ch27 = Circular(data=data_zar_ex4_ch27["θ"].values, unit="degree")

    A, pval = omnibus_test(alpha=circ_zar_ex4_ch27.alpha, scale=1)

    np.testing.assert_approx_equal(pval, 0.0043, significant=2)


def test_batschelet_test():

    data_zar_ex5_ch27 = load_data("D8", source="zar")
    circ_zar_ex5_ch27 = Circular(data=data_zar_ex5_ch27["θ"].values, unit="degree")

    C, pval = batschelet_test(
        angle=np.deg2rad(45),
        alpha=circ_zar_ex5_ch27.alpha,
    )
    np.testing.assert_equal(C, 5)
    np.testing.assert_approx_equal(pval, 0.00661, significant=3)


def test_chisquare_test():

    d2 = load_data("D2", source="zar")
    c2 = Circular(data=d2["θ"].values, w=d2["w"].values)

    χ2, pval = chisquare_test(c2.w)
    np.testing.assert_approx_equal(χ2, 66.543, significant=3)
    assert pval < 0.001


def test_symmetry_test():

    data_zar_ex6_ch27 = load_data("D9", source="zar")
    circ_zar_ex6_ch27 = Circular(data=data_zar_ex6_ch27["θ"].values, unit="degree")

    d, p = symmetry_test(median=circ_zar_ex6_ch27.median, alpha=circ_zar_ex6_ch27.alpha)
    assert p > 0.5


def test_watson_williams_test():

    data = load_data("D10", source="zar")
    s1 = Circular(data=data[data["sample"] == 1]["θ"].values)
    s2 = Circular(data=data[data["sample"] == 2]["θ"].values)
    F, pval = watson_williams_test(circs=[s1, s2])

    np.testing.assert_approx_equal(F, 1.61, significant=3)
    np.testing.assert_approx_equal(pval, 0.22, significant=2)

    data = load_data("D11", source="zar")
    s1 = Circular(data=data[data["sample"] == 1]["θ"].values)
    s2 = Circular(data=data[data["sample"] == 2]["θ"].values)
    s3 = Circular(data=data[data["sample"] == 3]["θ"].values)

    F, pval = watson_williams_test(circs=[s1, s2, s3])

    np.testing.assert_approx_equal(F, 1.86, significant=3)
    np.testing.assert_approx_equal(pval, 0.19, significant=2)


def test_watson_u2_test():

    d = load_data("D12", source="zar")
    c0 = Circular(data=d[d["sample"] == 1]["θ"].values)
    c1 = Circular(data=d[d["sample"] == 2]["θ"].values)
    U2, pval = watson_u2_test(circs=[c0, c1])

    np.testing.assert_approx_equal(U2, 0.1458, significant=3)
    assert 0.1 < pval < 0.2

    d = load_data("D13", source="zar")
    c0 = Circular(
        data=d[d["sample"] == 1]["θ"].values, w=d[d["sample"] == 1]["w"].values
    )
    c1 = Circular(
        data=d[d["sample"] == 2]["θ"].values, w=d[d["sample"] == 2]["w"].values
    )
    U2, pval = watson_u2_test(circs=[c0, c1])

    np.testing.assert_approx_equal(U2, 0.0612, significant=3)
    assert pval > 0.5


def test_wheeler_watson_test():
    d = load_data("D12", source="zar")
    c0 = Circular(data=d[d["sample"] == 1]["θ"].values)
    c1 = Circular(data=d[d["sample"] == 2]["θ"].values)

    W, pval = wheeler_watson_test(circs=[c0, c1])
    np.testing.assert_approx_equal(W, 3.678, significant=3)
    assert 0.1 < pval < 0.25


def test_wallraff_test():

    d = load_data("D14", source="zar")
    c0 = Circular(data=d[d["sex"] == "male"]["θ"].values)
    c1 = Circular(data=d[d["sex"] == "female"]["θ"].values)
    U, pval = wallraff_test(angle=np.deg2rad(135), circs=[c0, c1])
    np.testing.assert_approx_equal(U, 18.5, significant=3)
    assert pval > 0.20

    from pycircstat2.utils import time2float
    d = load_data("D15", source="zar")
    c0 = Circular(data=time2float(d[d["sex"] == "male"]["time"].values))
    c1 = Circular(data=time2float(d[d["sex"] == "female"]["time"].values))
    U, pval = wallraff_test(angle=np.deg2rad(time2float(['7:55', '8:15'])), circs=[c0, c1], verbose=True)
    np.testing.assert_equal(U, 13)
    assert pval > 0.05


def test_kuiper_test():

    d = load_data("B5", source="fisher")["θ"].values
    c = Circular(data=d, unit="degree", n_intervals=180)
    V, pval = kuiper_test(alpha=c.alpha)
    np.testing.assert_approx_equal(V, 1.5864, significant=3)
    assert pval > 0.05


def test_watson_test():

    pigeon = np.array([20, 135, 145, 165, 170, 200, 300, 325, 335, 350, 350, 350, 355])
    c_pigeon = Circular(data=pigeon)
    U2, pval = watson_test(alpha=c_pigeon.alpha, n_simulation=9999)
    np.testing.assert_approx_equal(U2, 0.137, significant=3)
    assert pval > 0.10


def test_rao_spacing_test():
    pigeon = np.array([20, 135, 145, 165, 170, 200, 300, 325, 335, 350, 350, 350, 355])
    c_pigeon = Circular(data=pigeon)
    U, pval = rao_spacing_test(alpha=c_pigeon.alpha, n_simulation=9999)
    np.testing.assert_approx_equal(U, 161.92308, significant=3)
    assert 0.05 < pval < 0.10
