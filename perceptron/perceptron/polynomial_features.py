from functools import reduce

from typing import Any, Callable, Generator, Iterable, List, Tuple


def concat_combiner(alfa: Any, beta: Any) -> Any:
    return alfa + beta


def concat_couples_combiner(alfa: Tuple, beta: Tuple) -> Any:
    return tuple(list(alfa) + list(beta))


def product_combiner(alfa: Any, beta: Any) -> Any:
    return alfa * beta


def unique_combinations_indices(
    items: int,
    order: int,
) -> Generator:
    root = tuple(range(items))

    def tuplize(item):
        return item if isinstance(item, tuple) else (item,)

    def combinations_couple(alfa: Tuple, beta: Tuple):
        return (
            (tuplize(a) + tuplize(b) for a in alfa for b in beta)
            if alfa and beta
            else (tuplize(item) for item in beta or alfa)
        )

    def comb_reduce(acc, _):
        return combinations_couple(acc, root)

    item: Generator
    for item in reduce(comb_reduce, range(order), ()):
        if sorted(item) == list(item):
            yield item


def unique_combinations(
    items: List,
    order: int,
) -> Generator:
    for indices in unique_combinations_indices(items=len(items), order=order):
        yield (items[index] for index in indices)


def polynomial_features(
    data: Iterable, degree: int, one: Any = 1, combiner: Callable = product_combiner
) -> List:
    features = [one]
    for degree_prog in range(degree):
        product = [
            reduce(combiner, item)
            for item in unique_combinations(items=list(data), order=degree_prog + 1)
        ]
        features.extend(product)
    return features


def numerical_polynomial_features(data: List, degree: int) -> List:
    return polynomial_features(
        data=data, degree=degree, one=1.0, combiner=lambda alfa, beta: alfa * beta
    )
