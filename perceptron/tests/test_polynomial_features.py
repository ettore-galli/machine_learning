from functools import reduce
from perceptron.polynomial_features import (
    polynomial_features,
    concat_combiner,
    numerical_polynomial_features,
    self_crossproduct_indices,
    unique_combinations,
    unique_combinations_indices,
)


def test_self_crossproduct_indices():
    got = list(self_crossproduct_indices(items=3, order=3))
    want = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 2, 0),
        (0, 2, 1),
        (0, 2, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 0),
        (1, 2, 1),
        (1, 2, 2),
        (2, 0, 0),
        (2, 0, 1),
        (2, 0, 2),
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 0),
        (2, 2, 1),
        (2, 2, 2),
    ]
    assert sorted(got) == sorted(want)


def test_unique_combinations_indices():
    got = list(unique_combinations_indices(items=3, order=3))
    want = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 1),
        (0, 1, 2),
        (0, 2, 2),
        (1, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 2, 2),
    ]

    assert sorted(got) == sorted(want)


def test_unique_combinations_indices_other():
    assert list(unique_combinations_indices(items=2, order=3)) == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]


def test_unique_combinations():
    got = unique_combinations(items=["x", "y", "z"], order=3)
    want = [
        ("x", "x", "x"),
        ("x", "x", "y"),
        ("x", "x", "z"),
        ("x", "y", "y"),
        ("x", "y", "z"),
        ("x", "z", "z"),
        ("y", "y", "y"),
        ("y", "y", "z"),
        ("y", "z", "z"),
        ("z", "z", "z"),
    ]

    def sort_key(item):
        return tuple(reduce(lambda a, b: a + b, item, ""))

    def compare_results(got, want):
        assert sorted([tuple(gotitem) for gotitem in got], key=sort_key) == (
            sorted(want, key=sort_key)
        )

    compare_results(got, want)

    compare_results(
        unique_combinations(items=["a", "b"], order=2),
        [("a", "a"), ("a", "b"), ("b", "b")],
    )

    compare_results(
        unique_combinations(items=["a", "b"], order=3),
        [("a", "a", "a"), ("a", "a", "b"), ("a", "b", "b"), ("b", "b", "b")],
    )


def test_polynomial_features():
    data = "ab"
    got = polynomial_features(data, degree=3, combiner=concat_combiner, one="1")
    want = [
        "1",
        "a",
        "b",
        "aa",
        "ab",
        "bb",
        "aaa",
        "aab",
        "abb",
        "bbb",
    ]
    print("got:", got)
    assert sorted(got) == sorted(want)


def test_polynomial_features_large():
    data = ["x", "y"]
    got = polynomial_features(data, degree=10, combiner=concat_combiner, one="")
    want = [
        "",
        "x",
        "y",
        "xx",
        "xy",
        "yy",
        "xxx",
        "xxy",
        "xyy",
        "yyy",
        "xxxx",
        "xxxy",
        "xxyy",
        "xyyy",
        "yyyy",
        "xxxxx",
        "xxxxy",
        "xxxyy",
        "xxyyy",
        "xyyyy",
        "yyyyy",
        "xxxxxx",
        "xxxxxy",
        "xxxxyy",
        "xxxyyy",
        "xxyyyy",
        "xyyyyy",
        "yyyyyy",
        "xxxxxxx",
        "xxxxxxy",
        "xxxxxyy",
        "xxxxyyy",
        "xxxyyyy",
        "xxyyyyy",
        "xyyyyyy",
        "yyyyyyy",
        "xxxxxxxx",
        "xxxxxxxy",
        "xxxxxxyy",
        "xxxxxyyy",
        "xxxxyyyy",
        "xxxyyyyy",
        "xxyyyyyy",
        "xyyyyyyy",
        "yyyyyyyy",
        "xxxxxxxxx",
        "xxxxxxxxy",
        "xxxxxxxyy",
        "xxxxxxyyy",
        "xxxxxyyyy",
        "xxxxyyyyy",
        "xxxyyyyyy",
        "xxyyyyyyy",
        "xyyyyyyyy",
        "yyyyyyyyy",
        "xxxxxxxxxx",
        "xxxxxxxxxy",
        "xxxxxxxxyy",
        "xxxxxxxyyy",
        "xxxxxxyyyy",
        "xxxxxyyyyy",
        "xxxxyyyyyy",
        "xxxyyyyyyy",
        "xxyyyyyyyy",
        "xyyyyyyyyy",
        "yyyyyyyyyy",
    ]

    assert sorted(got) == sorted(want)


def test_numerical_polynomial_features():
    data = [3.0, 4.0, 5.0]
    got = numerical_polynomial_features(data, degree=4)
    want = [
        1.0,
        3.0,
        4.0,
        5.0,
        9.0,
        12.0,
        15.0,
        16.0,
        20.0,
        25.0,
        27.0,
        36.0,
        45.0,
        48.0,
        60.0,
        75.0,
        64.0,
        80.0,
        100.0,
        125.0,
        81.0,
        108.0,
        135.0,
        144.0,
        180.0,
        225.0,
        192.0,
        240.0,
        300.0,
        375.0,
        256.0,
        320.0,
        400.0,
        500.0,
        625.0,
    ]

    assert sorted(got) == sorted(want)
