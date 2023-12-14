from itertools import accumulate, takewhile
from perceptron.iteration import (
    accumulate_iterated_while,
    reduce_while,
    reduce_until,
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
    assert [
        q
        for q in takewhile(
            lambda k: k < 1000,
            accumulate(range(10000000000), lambda t, _: t * 2, initial=1),
        )
    ] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    assert [
        q
        for q in takewhile(
            lambda k: k < 1000,
            accumulate(range(10000000000), lambda t, _: t * 2, initial=1),
        )
    ][-1] == 512


def test_reduce_while():
    def reducer_function(acc, cur):
        return acc + cur

    def while_predicate(acc, _):
        return acc < 11

    assert (
        reduce_while(
            function=reducer_function,
            sequence=range(1000000000),
            initial=0,
            predicate=while_predicate,
        )
        == 15
    )


def test_reduce_until():
    def reducer_function(acc, cur):
        return acc + cur

    def until_predicate(acc, _):
        return acc > 11

    assert (
        reduce_until(
            function=reducer_function,
            sequence=range(1000000000),
            initial=0,
            predicate=until_predicate,
        )
        == 15
    )


def test_accumulate_iterated_while():
    result = accumulate_iterated_while(
        initial=1,
        iteration_function=lambda x: x * 2,
        while_predicate=lambda x: x < 1000,
        maximum_iterations=1000000000,
    )
    assert result == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
