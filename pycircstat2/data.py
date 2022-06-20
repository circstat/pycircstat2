import numpy as np


def data(name):

    __name__ = [
        "zar_ex2_ch26",
        "zar_ex3_ch26",
        "zar_ex4_ch26",
        "zar_ex5_ch26",
        "zar_ex6_ch26",
        "zar_ex2_ch27",
        "zar_ex3_ch27",
        "zar_ex4_ch27",
        "mardia_ex1_ch1",
        "mallard",
    ]

    if name in [
        "zar_ex2_ch26",
        "zar_ex4_ch26",
        "zar_ex6_ch26",
        "zar_ex1_ch27",
        "zar_ex24_ch27",
    ]:

        # ungrouped data in degree
        d = {
            "data": np.array([45, 55, 81, 96, 110, 117, 132, 154]),
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex3_ch26", "zar_ex5_ch26", "zar_ex23_ch27"]:

        # grouped data
        d = {
            "data": np.arange(15, 360, 30),  # data (degree): interval centers
            "w": np.array([0, 6, 9, 13, 15, 22, 17, 12, 8, 3, 0, 0]),  # frequency
            "unit": "degree",
            "k": 360,
        }

    elif name == "zar_ex7_ch26":

        # ungropped data / axial data
        d = {
            "data": np.array(
                [35, 40, 40, 45, 45, 55, 60, 215, 220, 225, 225, 235, 235, 240, 245]
            ),
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex2_ch27", "zar_ex3_ch27"]:

        # ungrouped data in degree
        d = {
            "data": np.array([66, 75, 86, 88, 88, 93, 97, 101, 118, 130]),
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex4_ch27", "zar_ex5_ch27"]:

        # ungrouped data in degree
        d = {
            "data": np.array(
                [
                    10,
                    15,
                    25,
                    30,
                    30,
                    30,
                    35,
                    45,
                    50,
                    60,
                    75,
                    80,
                    100,
                    110,
                    255,
                    270,
                    280,
                    280,
                    300,
                    320,
                    330,
                    350,
                    350,
                    355,
                ]
            ),
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex6_ch27"]:

        # ungrouped data in degree
        d = {
            "data": np.array([97, 104, 121, 159, 164, 172, 195, 213]),
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex7_ch27"]:

        # two-sample ungrouped data in degree
        d = {
            "s0": {"data": np.array([94, 65, 45, 52, 38, 47, 73, 82, 90, 40, 87])},
            "s1": {"data": np.array([77, 70, 61, 45, 50, 35, 48, 65, 36])},
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex8_ch27"]:

        # three-sample ungropped data in degree
        d = {
            "s0": {"data": np.array([135, 145, 125, 140, 165, 170])},
            "s1": {"data": np.array([150, 130, 175, 190, 180, 220])},
            "s2": {"data": np.array([140, 165, 185, 125, 175, 175, 140])},
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex9_ch27", "zar_ex11_ch27"]:

        # two-sample ungrouped data in degree
        d = {
            "s0": {"data": np.array([35, 45, 50, 55, 60, 70, 85, 95, 105, 120])},
            "s1": {
                "data_s1": np.array(
                    [75, 80, 90, 100, 110, 130, 135, 140, 150, 155, 165]
                )
            },
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex10_ch27"]:

        # two-sample grouped data in degree

        d = {
            "s0": {
                "data": np.array([40, 45, 50, 55, 70, 80, 95, 105, 110, 120]),
                "w": np.array([1, 1, 1, 1, 1, 2, 1, 1, 2, 1]),
            },
            "s1": {
                "data": np.array([30, 35, 50, 60, 65, 75, 80, 90, 100]),
                "w": np.array([1, 1, 1, 1, 2, 1, 1, 1, 1]),
            },
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex12_ch27"]:

        # two-sample ungrouped data in degree
        d = {
            "s0": {
                "data": np.array([145, 155, 130, 145, 145, 160, 140]),
            },
            "s1": {
                "data": np.array([160, 135, 145, 150, 125, 120]),
            },
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex13_ch27"]:

        # two-sample upgrouped data in time(hh:mm)
        raise NotSupportedError(f"{name} is Not ready yet.")

    elif name in ["zar_ex18_ch27", "zar_ex19_ch27"]:

        # paired two-sample upgrouped data
        d = {
            "s0": {"data": np.array([105, 120, 135, 95, 155, 170, 160, 155, 120, 115])},
            "s1": {
                "data": np.array([205, 210, 235, 245, 260, 255, 240, 245, 210, 200])
            },
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex20_ch27"]:

        # two sample ungropped data in degree
        # angular-angular correlation
        d = {
            "s0": {"data": np.array([145, 190, 310, 210, 80])},
            "s1": {"data": np.array([120, 180, 330, 225, 55])},
            "unit": "degree",
            "k": 360,
        }

    elif name in ["zar_ex21_ch27"]:

        # angular-linear correlation
        d = {
            "direction": {
                "data": np.array([190, 160, 210, 225, 220, 140, 120]),
                "unit": "degree",
                "k": 360,
            },
            "distance": {
                "data": np.array([48, 55, 26, 23, 22, 62, 64]),
                "unit": "km",
            },
        }

    elif name in ["zar_ex22_ch27"]:

        # paired one-sample repeated measure in degree
        d = {
            "evening": {"data": np.array([30, 10, 350, 0, 340, 330, 20, 30])},
            "morning": {"data": np.array([60, 50, 10, 350, 330, 0, 40, 70])},
            "unit": "degree",
            "k": 360,
        }

        return (evening, morning)

    elif name in ["mardia_ex1_ch1"]:

        # ungrouped data in degree
        d = {
            "data": np.array([43, 45, 52, 61, 75, 88, 88, 279, 357]),
            "unit": "degree",
            "k": 360,
        }

    elif name in ["mardia_tb1_ch1", "mallard"]:

        # Vanishing angles of 714 British mallards (matthews, 1961)
        # Mardia (1972) Table 1.1
        # grouped data (degree and frequency)
        d = {
            "data": np.arange(10, 360, 20),  # data (degree): bin centers
            "w": np.array(
                [40, 22, 20, 9, 6, 3, 3, 1, 6, 3, 11, 22, 24, 58, 136, 138, 143, 69]
            ),
            "unit": "degree",
            "k": 360,
        }

    elif name in ["mardia_tb2_ch1"]:

        # orientations of sand-grains of recent Gulf Coasat beach (Curray, 1956)
        # Mardia (1972) Table 1.2
        # grouped data in degree
        d = {
            "data": np.arange(5, 175, 10),
            "w": np.array(
                [
                    244,
                    262,
                    246,
                    290,
                    284,
                    314,
                    326,
                    340,
                    371,
                    401,
                    382,
                    332,
                    322,
                    295,
                    230,
                    256,
                    263,
                    281,
                ]
            ),
            "unit": "degree",
            "k": 360,
        }

    elif name in ["mardia_tb3_ch1"]:

        # number of occasions on which thunder was heard at Kew in summers of 1910-1935 (Bishop, 1947)
        # Mardia (1972) Table 1.3
        d = {
            "data": np.arange(1, 24, 2),
            "w": np.array([26, 24, 14, 15, 14, 65, 133, 149, 122, 80, 61, 22]),
            "unit": "degree",
            "k": 360,
        }

    elif name in ["mardia_tb4_ch1"]:

        # the number of occurrences of rainfall of 1'' or more per hour in the USA, 1908-37 (Dyck and Mattice, 1941)
        # Mardia (1972) Table 1.4
        d = {
            "data": np.arange(1, 13, 1),  # month
            "unit": "month",
            "k": 12,
            "w_unadjusted": np.array(
                [101, 94, 232, 406, 685, 1225, 1478, 1384, 907, 383, 195, 145]
            ),  # unadjusted
            "w_adjusted": np.array(
                [100, 103, 229, 414, 676, 1248, 1458, 1365, 924, 278, 199, 143]
            ),  # adjusted
        }

    elif name in ["mardia_tb5_ch1"]:
        # Ayimuths of cross-beds in the upper Kamthi river (Sengupta and Rao, 1966)
        # Mardia (1972) Table 1.5
        d = {
            "data": np.arange(10, 350, 10),
            "w": np.array(
                [75, 75, 15, 25, 7, 3, 3, 0, 0, 0, 21, 8, 24, 16, 36, 75, 90, 107]
            ),
            "unit": "degree",
            "k": 360,
        }

    elif name == "frog":

        d = {
            "data": np.array(
                [104, 110, 117, 121, 127, 130, 136, 144, 152, 178, 184, 192, 200, 316]
            ),
            "unit": "degree",
            "k": 360,
        }

    elif name in [
        "fisher_ex1_ch2",
        "fisher_ex3_ch2",
        "fisher_ex4_ch2",
        "fisher_ex5_ch2",
        "fisher_ex6_ch2",
        "fisher_ex8_ch2",
        "fisher_ex10_ch2",
    ]:

        d = {
            "data": np.array(
                [
                    11.00,
                    17.00,
                    23.15,
                    10.10,
                    12.00,
                    8.45,
                    16.00,
                    10.00,
                    15.30,
                    20.20,
                    4.00,
                    12.00,
                    2.20,
                    12.00,
                    5.30,
                    7.30,
                    12.00,
                    16.00,
                    16.00,
                    1.30,
                    11.05,
                    16.00,
                    19.00,
                    17.45,
                    20.00,
                    21.00,
                    12.00,
                    12.00,
                    18.00,
                    22.00,
                    # not finish entering...
                ]
            ),
            "unit": "hour",
            "k": 24,
        }

    elif name in ["fisher_ex2_ch2", "fisher_ex7_ch2", "fisher_ex8_ch2"]:

        # Long-axis orientations of deldspar laths
        # type: Axes.
        # Source: Smith (1988, set 28-6-1co.prn); data kindly supplied by Ms Nicola Smith.
        d = {
            "data": np.array(
                [
                    176,
                    162,
                    49,
                    174,
                    174,
                    49,
                    54,
                    63,
                    59,
                    61,
                    66,
                    104,
                    97,
                    58,
                    121,
                    5,
                    178,
                    3,
                    168,
                    0,
                    18,
                    39,
                    140,
                    63,
                    55,
                    170,
                    169,
                    37,
                    152,
                    73,
                    53,
                    176,
                    72,
                    170,
                    113,
                    56,
                    87,
                    161,
                    164,
                    21,
                    50,
                    6,
                    59,
                    140,
                    54,
                    64,
                    56,
                    38,
                    61,
                    143,
                    51,
                    144,
                    148,
                    44,
                    60,
                    98,
                    86,
                    145,
                    38,
                    168,
                    39,
                    134,
                    68,
                    57,
                    129,
                    68,
                    132,
                    82,
                    54,
                    119,
                    131,
                    50,
                    93,
                    160,
                    127,
                    124,
                    65,
                    108,
                    52,
                    61,
                    86,
                    37,
                    132,
                    83,
                    163,
                    58,
                    144,
                    29,
                    80,
                    172,
                    144,
                    138,
                    10,
                    45,
                    137,
                    11,
                    145,
                    103,
                    69,
                    124,
                    54,
                    121,
                    1,
                    39,
                    111,
                    153,
                    13,
                    5,
                    5,
                    107,
                    104,
                    39,
                    133,
                    36,
                    63,
                    4,
                    21,
                    51,
                    30,
                    52,
                    90,
                    143,
                    13,
                    50,
                    109,
                    12,
                    170,
                    5,
                    14,
                    91,
                    132,
                    12,
                    1,
                ]
            ),
            "unit": "degree",
            "k": 360,
        }

    elif name in ["fisher_ex9_ch2", "fisher_ex25_ch4"]:

        # Measurements of the directions taken by 76 turtiles after treatment.
        # Type: Vectors
        # source: Stephens (1969b).
        d = {
            "data": np.array(
                [
                    8,
                    9,
                    13,
                    13,
                    14,
                    18,
                    22,
                    27,
                    30,
                    34,
                    38,
                    38,
                    40,
                    44,
                    45,
                    47,
                    48,
                    48,
                    48,
                    48,
                    50,
                    53,
                    56,
                    57,
                    58,
                    58,
                    61,
                    63,
                    64,
                    64,
                    64,
                    65,
                    65,
                    68,
                    70,
                    73,
                    78,
                    78,
                    78,
                    83,
                    83,
                    88,
                    88,
                    88,
                    90,
                    92,
                    92,
                    93,
                    95,
                    96,
                    98,
                    100,
                    103,
                    106,
                    113,
                    118,
                    138,
                    153,
                    153,
                    155,
                    204,
                    215,
                    223,
                    226,
                    237,
                    238,
                    243,
                    244,
                    250,
                    251,
                    257,
                    268,
                    285,
                    319,
                    343,
                    350,
                ]
            ),
            "unit": "degree",
            "k": 360,
        }

    elif name in [
        "fisher_ex3_ch4",
        "fisher_ex6_ch4",
        "fisher_ex13_ch4",
        "fisher_ex14_ch4",
        "fisher_ex16_ch4",
        "fisher_ex17_ch4",
        "fisher_ex1_ch5",
        "fisher_ex11_ch5",
    ]:

        d = {
            "s0": {
                "data": np.array(
                    [
                        284,
                        311,
                        334,
                        320,
                        294,
                        137,
                        123,
                        166,
                        143,
                        127,
                        244,
                        243,
                        152,
                        242,
                        143,
                        186,
                        263,
                        234,
                        209,
                        267,
                        315,
                        329,
                        235,
                        235,
                        38,
                        241,
                        319,
                        308,
                        127,
                        217,
                        245,
                        169,
                        161,
                        263,
                        209,
                        228,
                        168,
                        98,
                        278,
                        154,
                        279,
                    ]
                )
            },
            "s1": {
                "data": np.array(
                    [
                        294,
                        301,
                        329,
                        315,
                        277,
                        281,
                        254,
                        245,
                        272,
                        242,
                        177,
                        257,
                        177,
                        229,
                        250,
                        166,
                        232,
                        245,
                        224,
                        186,
                        257,
                        267,
                        241,
                        239,
                        287,
                        229,
                        290,
                        214,
                        215,
                        224,
                    ]
                )
            },
            "s2": {
                "data": np.array(
                    [
                        163,
                        275,
                        218,
                        287,
                        313,
                        322,
                        236,
                        254,
                        239,
                        286,
                        268,
                        245,
                        211,
                        271,
                        151,
                        309,
                        27,
                        224,
                        181,
                        220,
                        217,
                        192,
                        283,
                        216,
                        231,
                        147,
                        163,
                        155,
                        203,
                    ]
                )
            },
        }

    else:
        raise ValueError(f"List of data sets: {__name__}")

    return d
