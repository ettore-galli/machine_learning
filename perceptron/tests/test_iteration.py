from perceptron.iteration import reduce_while


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
            while_predicate=while_predicate,
        )
        == 15
    )
