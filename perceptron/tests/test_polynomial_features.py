from perceptron.polynomial_features import (
    polynomial_features,
    cross_product,
    concat_combiner,
    numerical_polynomial_features
)


def test_cross_product():
    got = cross_product(alfa=["a", "b"], beta=["x", "y"], combiner=concat_combiner)
    want = ["ax", "bx", "ay", "by"]

    assert sorted(got) == sorted(want)


def test_unique_cross_product():
    got = cross_product(alfa=["a", "b", "c"], beta=["a", "b"], combiner=concat_combiner)
    want = ["aa", "ab", "ba", "bb", "ca", "cb"]

    assert sorted(got) == sorted(want)


def test_polynomial_features():
    data = ["a", "b"]
    got = polynomial_features(data, degree=3, combiner=concat_combiner, one="1")
    want = [
        "1",
        "a1",
        "b1",
        "aa1",
        "ab1",
        "bb1",
        "aaa1",
        "aab1",
        "abb1",
        "bab1",
        "bbb1",
    ]
    assert sorted(got) == sorted(want)


def test_numerical_polynomial_features():
    data = [1.25, 2.35]
    got = numerical_polynomial_features(data, degree=3)
    want = [
        1.0,
        1.25,
        1.5625,
        1.953125,
        2.35,
        2.9375,
        3.671875,
        5.522500000000001,
        6.903125,
        6.903125000000001,
        12.977875000000003,
    ]

    assert sorted(got) == sorted(want)
