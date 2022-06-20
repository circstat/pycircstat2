import numpy as np
from pycircstat2.base import Circular
from pycircstat2.data import data


def test_circular():

    # Ch26.2 Example 3 (Zar, 2010)
    data_zar_ex3_ch26 = data("zar_ex3_ch26")
    circ_zar_ex3_ch26 = Circular(
        data=data_zar_ex3_ch26["data"], w=data_zar_ex3_ch26["w"]
    )

    np.testing.assert_approx_equal(circ_zar_ex3_ch26.n, 105, significant=1)
    np.testing.assert_approx_equal(
        np.rad2deg(circ_zar_ex3_ch26.bin_size), 30, significant=1
    )


#     # Ch26.4 Example 4 (Zar, 2010) / Ungrouped data
#     # Using data from Ch26 Example 2
#     data_zar_ex4_ch26 = data("zar_ex2_ch26")
#     circ_zar_ex4_ch26 = Circular(data_zar_ex4_ch26, unit="degree")

#     np.testing.assert_approx_equal(circ_zar_ex4_ch26.n, 8, significant=1)
#     np.testing.assert_approx_equal(circ_zar_ex4_ch26.r, 0.82522, significant=5)
#     np.testing.assert_approx_equal(circ_zar_ex4_ch26.cos_a_bar, -0.15623, significant=5)
#     np.testing.assert_approx_equal(circ_zar_ex4_ch26.sin_a_bar, 0.98772, significant=5)
#     np.testing.assert_approx_equal(
#         circ_zar_ex4_ch26.angular_mean_deg, 99, significant=1.0
#     )

#     # Ch26.4 Example 5 (Zar, 2010) / grouped data
#     # Using data from Ch26 Example 3
#     circ_zar_ex5_ch26 = Circular(data_zar_ex3_ch26, unit="degree")
#     np.testing.assert_approx_equal(circ_zar_ex5_ch26.Y, 0.17413, significant=5)
#     np.testing.assert_approx_equal(circ_zar_ex5_ch26.X, -0.52238, significant=5)
#     np.testing.assert_approx_equal(circ_zar_ex5_ch26.r, 0.55064, significant=5)
#     np.testing.assert_approx_equal(circ_zar_ex5_ch26.cos_a_bar, -0.94868, significant=5)
#     np.testing.assert_approx_equal(circ_zar_ex5_ch26.sin_a_bar, 0.31623, significant=5)
#     np.testing.assert_approx_equal(
#         circ_zar_ex5_ch26.angular_mean_deg, 162.0, significant=1.0
#     )

#     # Angular dispersion from Ch26.5 (Zar, 2010)
#     # Part of Ch26 Example 4, using data from Ch26 Example 2
#     np.testing.assert_approx_equal(
#         circ_zar_ex4_ch26.angular_deviation_deg, 34.0, significant=1.0
#     )
#     np.testing.assert_approx_equal(
#         circ_zar_ex4_ch26.circular_standard_deviation_deg, 36.0, significant=1.0
#     )
#     np.testing.assert_approx_equal(
#         circ_zar_ex5_ch26.angular_deviation_deg, 54.0, significant=1.0
#     )
#     np.testing.assert_approx_equal(
#         circ_zar_ex5_ch26.circular_standard_deviation_deg, 63.0, significant=1.0
#     )

#     # Ch26 Example 6 (Zar, 2010) / Confidence Intervel
#     # Using data from Ch26 Example 2
#     # The value of R printed in the book (6.60108) was inaccurate!
#     np.testing.assert_approx_equal(circ_zar_ex4_ch26.R, 6.60174, significant=5)
#     np.testing.assert_approx_equal(circ_zar_ex4_ch26.d_deg, 31.0, significant=2)

#     # Ch27 Example 2 (Zar, 2010)
#     data_zar_ex2_ch27 = data("zar_ex2_ch27")
#     circ_zar_ex2_ch27 = Circular(data_zar_ex2_ch27, unit="degree")
#     np.testing.assert_approx_equal(circ_zar_ex2_ch27.X, -0.06722, significant=4)
#     np.testing.assert_approx_equal(circ_zar_ex2_ch27.Y, 0.94976, significant=5)
#     np.testing.assert_approx_equal(circ_zar_ex2_ch27.r, 0.95214, significant=5)
#     np.testing.assert_approx_equal(circ_zar_ex2_ch27.R, 9.5214, significant=5)
#     np.testing.assert_approx_equal(
#         circ_zar_ex2_ch27.angular_mean_deg, 94.0, significant=1
#     )
#     np.testing.assert_approx_equal(circ_zar_ex2_ch27.d_deg, 13.0, significant=2)
