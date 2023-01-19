import numpy as np

from pycircstat2.base import Circular
from pycircstat2.utils import load_data


def test_Circular():

    # Ch26.2 Example 3 (Zar, 2010)
    data_zar_ex3_ch26 = load_data("D2", source="zar")
    circ_zar_ex3_ch26 = Circular(
        data=data_zar_ex3_ch26["θ"].values,
        w=data_zar_ex3_ch26["w"].values,
        unit="degree",
    )

    np.testing.assert_approx_equal(circ_zar_ex3_ch26.n, 105, significant=1)
    np.testing.assert_approx_equal(
        np.rad2deg(circ_zar_ex3_ch26.bin_size), 30, significant=1
    )
