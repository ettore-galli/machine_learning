# pylint: disable=too-many-lines
import numpy as np

from pytest import approx

from perceptron.perceptron_algorithm import (
    averaged_perceptron,
    averaged_perceptron_legacy,
    d_split_j,
    d_split_j_looper,
    eval_classifier,
    eval_learning_alg,
    offset_perceptron_step,
    perceptron,
    Data,
    Labels,
    Params,
    Hook,
    Sample,
    Label,
    Theta,
    ThetaZero,
    xval_learning_alg,
)


def log_classifier_step(sample: Sample, label: Label, theta: Theta, theta_0=ThetaZero):
    print(sample, label, theta, theta_0)


def test_perceptron():
    data: Data = np.array([[2, 3, 9, 12], [5, 1, 6, 5]])
    labels: Labels = np.array([[1, -1, 1, -1]])
    params: Params = {"T": 100}
    hook: Hook = log_classifier_step
    classifier = perceptron(
        data=data,
        labels=labels,
        params=params,
        perceptron_step=offset_perceptron_step,
        hook=hook,
    )

    classifier_coefficents = classifier.get_classifier_coefficents()

    print([x.tolist() for x in classifier_coefficents])

    np.array_equal(classifier_coefficents[0], np.array([-9.0, 18.0]))
    assert classifier_coefficents[1] == approx(2.0)


def test_averaged_perceptron():
    data: Data = np.array([[2, 3, 9, 12], [5, 1, 6, 5]])
    labels: Labels = np.array([[1, -1, 1, -1]])
    params: Params = {"T": 100}
    hook: Hook = log_classifier_step
    classifier = averaged_perceptron(
        data=data,
        labels=labels,
        params=params,
        perceptron_step=offset_perceptron_step,
        hook=hook,
    )
    classifier_coefficents = classifier.get_classifier_coefficents()
    print([x.tolist() for x in classifier_coefficents])

    np.array_equal(classifier_coefficents[0], np.array([-9.0, 18.0]))

    assert classifier_coefficents[1] == approx(1.9425)


def test_averaged_perceptron_legacy():
    data: Data = np.array([[2, 3, 9, 12], [5, 1, 6, 5]])
    labels: Labels = np.array([[1, -1, 1, -1]])
    params: Params = {"T": 100}
    hook: Hook = log_classifier_step
    classifier = averaged_perceptron_legacy(
        data=data,
        labels=labels,
        params=params,
        perceptron_step=offset_perceptron_step,
        hook=hook,
    )
    classifier_coefficents = classifier.get_classifier_coefficents()
    print([x.tolist() for x in classifier_coefficents])

    np.array_equal(classifier_coefficents[0], np.array([-9.0, 18.0]))

    assert classifier_coefficents[1] == approx(1.9425)


def test_eval_classifier():
    data_train: Data = np.array([[2, 3, 9, 12], [5, 1, 6, 5]])
    labels_train: Labels = np.array([[1, -1, 1, -1]])
    data_test: Data = np.array([[2, 3, 9, 12], [5, 1, 6, 5]])
    labels_test: Labels = np.array([[1, -1, 1, 1]])

    assert eval_classifier(
        learner=perceptron,
        data_train=data_train,
        labels_train=labels_train,
        data_test=data_test,
        labels_test=labels_test,
    ) == approx(0.75)


def test_eval_learning_alg():
    data_train = np.array(
        [
            [
                -2.04297103,
                -1.85361169,
                -2.65467827,
                -1.23013149,
                -0.31934782,
                1.33112127,
                2.3297942,
                1.47705445,
                -1.9733787,
                -2.35476882,
            ],
            [
                -2.1071566,
                -3.06450052,
                -3.43898434,
                0.71320285,
                1.51214693,
                4.14295175,
                4.73681233,
                -2.80366981,
                1.56182223,
                0.07061724,
            ],
        ]
    )

    labels_train = np.array([[-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])

    data_test = np.array(
        [
            [
                -4.97193554,
                3.49851995,
                4.00302943,
                0.83369183,
                0.41371989,
                4.37614714,
                1.03536965,
                1.2354608,
                -0.7933465,
                -3.85456759,
            ],
            [
                -0.92053415,
                -3.61953464,
                0.39577344,
                -3.03202474,
                -4.90408303,
                -0.10239158,
                -1.35546287,
                1.31372748,
                -1.97924525,
                -3.72545813,
            ],
        ]
    )
    labels_test = np.array([[-1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0]])

    def make_gen_data():
        progress = {"step": 0}

        def gen_data(size):
            _ = size
            gendata = (
                (data_train, labels_train)
                if progress["step"] == 0
                else (data_test, labels_test)
            )

            progress["step"] += 1
            return gendata

        return gen_data

    assert eval_learning_alg(
        learner=perceptron, data_gen=make_gen_data(), n_train=5, n_test=5, it=10
    ) == approx(0.5)


def test_d_split_j():
    data = np.array(
        [
            [100 + i + 1 for i in range(60)],
            [200 + i + 1 for i in range(60)],
            [300 + i + 1 for i in range(60)],
        ]
    )
    d_j, d_minus_j = d_split_j(data, 5, 2)

    assert np.array_equal(
        d_j,
        np.array(
            [
                [125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136],
                [225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236],
                [325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336],
            ]
        ),
    )

    assert np.array_equal(
        d_minus_j,
        np.array(
            [
                [
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                    124,
                    137,
                    138,
                    139,
                    140,
                    141,
                    142,
                    143,
                    144,
                    145,
                    146,
                    147,
                    148,
                    149,
                    150,
                    151,
                    152,
                    153,
                    154,
                    155,
                    156,
                    157,
                    158,
                    159,
                    160,
                ],
                [
                    201,
                    202,
                    203,
                    204,
                    205,
                    206,
                    207,
                    208,
                    209,
                    210,
                    211,
                    212,
                    213,
                    214,
                    215,
                    216,
                    217,
                    218,
                    219,
                    220,
                    221,
                    222,
                    223,
                    224,
                    237,
                    238,
                    239,
                    240,
                    241,
                    242,
                    243,
                    244,
                    245,
                    246,
                    247,
                    248,
                    249,
                    250,
                    251,
                    252,
                    253,
                    254,
                    255,
                    256,
                    257,
                    258,
                    259,
                    260,
                ],
                [
                    301,
                    302,
                    303,
                    304,
                    305,
                    306,
                    307,
                    308,
                    309,
                    310,
                    311,
                    312,
                    313,
                    314,
                    315,
                    316,
                    317,
                    318,
                    319,
                    320,
                    321,
                    322,
                    323,
                    324,
                    337,
                    338,
                    339,
                    340,
                    341,
                    342,
                    343,
                    344,
                    345,
                    346,
                    347,
                    348,
                    349,
                    350,
                    351,
                    352,
                    353,
                    354,
                    355,
                    356,
                    357,
                    358,
                    359,
                    360,
                ],
            ]
        ),
    )

    d_j_2, d_minus_j_2 = d_split_j(np.array([[11, 12, 13], [21, 22, 23]]), 1, 0)
    assert np.array_equal(d_j_2, np.array([[11, 12, 13], [21, 22, 23]]))
    assert np.array_equal(d_minus_j_2, np.ndarray((2, 0)))


def test_d_split_j_looper():
    data = np.array(
        [
            [100 + i + 1 for i in range(60)],
            [200 + i + 1 for i in range(60)],
            [300 + i + 1 for i in range(60)],
        ]
    )
    labels = np.array(
        [
            [1000 + i + 1 for i in range(60)],
        ]
    )

    for data_test, data_train, labels_test, labels_train in d_split_j_looper(
        data, labels, 7
    ):
        print("-" * 78)
        print(data_test)
        print(data_train)
        print(labels_test)
        print(labels_train)


def test_xval_learning_alg():
    data = np.array(
        [
            [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
        ]
    )
    labels = np.array(
        [
            [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1],
        ]
    )

    p = xval_learning_alg(perceptron, data, labels, k=5)

    print(repr(p))


def test_xval_learning_alg_2():
    big_data, big_data_labels = (
        np.array(
            [
                [
                    -2.04297103,
                    -1.85361169,
                    -2.65467827,
                    -1.23013149,
                    -0.31934782,
                    1.33112127,
                    2.3297942,
                    1.47705445,
                    -1.9733787,
                    -2.35476882,
                    -4.97193554,
                    3.49851995,
                    4.00302943,
                    0.83369183,
                    0.41371989,
                    4.37614714,
                    1.03536965,
                    1.2354608,
                    -0.7933465,
                    -3.85456759,
                    3.22134658,
                    -3.39787483,
                    -1.31182253,
                    -2.61363628,
                    -1.14618119,
                    -0.2174626,
                    1.32549116,
                    2.54520221,
                    0.31565661,
                    2.24648287,
                    -3.33355258,
                    -0.98689271,
                    -0.24876636,
                    -3.16008017,
                    1.22353111,
                    4.77766994,
                    -1.81670773,
                    -3.58939471,
                    -2.16268851,
                    2.88028351,
                    -3.42297827,
                    -2.74992813,
                    -0.40293356,
                    -3.45377267,
                    0.62400624,
                    -0.35794507,
                    -4.1648704,
                    -1.08734116,
                    0.22367444,
                    1.09067619,
                    1.28738004,
                    2.07442478,
                    4.61951855,
                    4.47029706,
                    2.86510481,
                    4.12532285,
                    0.48170777,
                    0.60089857,
                    4.50287515,
                    2.95549453,
                    4.22791451,
                    -1.28022286,
                    2.53126681,
                    2.41887277,
                    -4.9921717,
                    4.15022718,
                    0.49670572,
                    2.0268248,
                    -4.63475897,
                    -4.20528418,
                    1.77013481,
                    -3.45389325,
                    1.0238472,
                    -1.2735185,
                    4.75384686,
                    1.32622048,
                    -0.13092625,
                    1.23457116,
                    -1.69515197,
                    2.82027615,
                    -1.01140935,
                    3.36451016,
                    4.43762708,
                    -4.2679604,
                    4.76734154,
                    -4.14496071,
                    -4.38737405,
                    -1.13214501,
                    -2.89008477,
                    3.22986894,
                    1.84103699,
                    -3.91906092,
                    -2.8867831,
                    2.31059245,
                    -3.62773189,
                    -4.58459406,
                    -4.06343392,
                    -3.10927054,
                    1.09152472,
                    2.99896855,
                ],
                [
                    -2.1071566,
                    -3.06450052,
                    -3.43898434,
                    0.71320285,
                    1.51214693,
                    4.14295175,
                    4.73681233,
                    -2.80366981,
                    1.56182223,
                    0.07061724,
                    -0.92053415,
                    -3.61953464,
                    0.39577344,
                    -3.03202474,
                    -4.90408303,
                    -0.10239158,
                    -1.35546287,
                    1.31372748,
                    -1.97924525,
                    -3.72545813,
                    1.84834303,
                    -0.13679709,
                    1.36748822,
                    -2.92886952,
                    -2.48367819,
                    -0.0894489,
                    -2.99090327,
                    0.35494698,
                    0.94797491,
                    4.20393035,
                    -3.14009852,
                    -4.86292242,
                    3.2964068,
                    -0.9911453,
                    4.39465,
                    3.64956975,
                    -0.72225648,
                    -0.15864119,
                    -2.0340774,
                    -4.00758749,
                    0.8627915,
                    3.73237594,
                    -0.70011824,
                    1.07566463,
                    -4.05063547,
                    -3.98137177,
                    4.82410619,
                    2.5905222,
                    0.34188269,
                    -1.44737803,
                    3.27583966,
                    2.06616486,
                    -4.43584161,
                    0.27795053,
                    4.37207651,
                    -4.48564119,
                    0.7183541,
                    1.59374552,
                    -0.13951634,
                    0.67825519,
                    -4.02423434,
                    4.15893861,
                    -1.52110278,
                    2.1320374,
                    3.31118893,
                    -4.04072252,
                    2.41403912,
                    -1.04635499,
                    3.39575642,
                    2.2189097,
                    4.78827245,
                    1.19808069,
                    3.10299723,
                    0.18927394,
                    0.14437543,
                    -4.17561642,
                    0.6060279,
                    0.22693751,
                    -3.39593567,
                    1.14579319,
                    3.65449494,
                    -1.27240159,
                    0.73111639,
                    3.48806017,
                    2.48538719,
                    -1.83892096,
                    1.42819622,
                    -1.37538641,
                    3.4022984,
                    0.82757044,
                    -3.81792516,
                    2.77707152,
                    -1.49241173,
                    2.71063994,
                    -3.33495679,
                    -4.00845675,
                    0.719904,
                    -2.3257032,
                    1.65515972,
                    -1.90859948,
                ],
            ]
        ),
        np.array(
            [
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                ]
            ]
        ),
    )

    p = xval_learning_alg(perceptron, data=big_data, labels=big_data_labels, k=5)

    print(repr(p))
