from perceptron.iteration import reduce_while, reduce_until


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
