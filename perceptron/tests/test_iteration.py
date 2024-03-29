from itertools import accumulate, takewhile
from perceptron.iteration import (
    accumulate_iterate_while,
    iterate_while,
)


def test_iteration_ideas():
    assert list(list(accumulate(range(16), lambda t, _: t * 2, initial=1))) == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
    ]
    assert list(
        takewhile(
            lambda k: k < 1000,
            accumulate(range(10000000000), lambda t, _: t * 2, initial=1),
        )
    ) == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    assert (
        list(
            takewhile(
                lambda k: k < 1000,
                accumulate(range(10000000000), lambda t, _: t * 2, initial=1),
            )
        )[-1]
        == 512
    )


def test_accumulate_iterate_while():
    result = accumulate_iterate_while(
        initial=1,
        iteration_function=lambda x: x * 2,
        while_predicate=lambda x: x < 1000,
        maximum_iterations=1000000000,
    )
    assert result == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def test_iterate_while():
    result = iterate_while(
        initial=1,
        iteration_function=lambda x: x * 2,
        while_predicate=lambda x: x < 1000,
        maximum_iterations=1000000000,
    )
    assert result == 512
