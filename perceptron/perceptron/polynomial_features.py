from functools import reduce
from typing import Any, Callable, List


def concat_combiner(alfa: Any, beta: Any) -> Any:
    return alfa + beta


def product_combiner(alfa: Any, beta: Any) -> Any:
    return alfa * beta


def cross_product(
    alfa: List,
    beta: List,
    combiner: Callable[[Any, Any], Any] = concat_combiner,
) -> List:
    combinations = [(a, b) for a in range(len(alfa)) for b in range(len(beta))]
    return [combiner(alfa[a], beta[b]) for a, b in combinations]


def unique_cross_product(
    alfa: List,
    beta: List,
    combiner: Callable[[Any, Any], Any] = concat_combiner,
) -> List:
    combinations_base = [(a, b) for a in range(len(alfa)) for b in range(len(beta))]
    combinations: List = reduce(
        lambda acc, cur: (acc + [cur])
        if (cur not in acc and (cur[1], cur[0]) not in acc)
        else acc,
        combinations_base,
        [],
    )
    return [combiner(alfa[a], beta[b]) for a, b in combinations]


def polynomial_features(
    data: List, degree: int, one: Any = 1, combiner: Callable = product_combiner
) -> List:
    features = [one]
    product = features
    for _ in range(degree):
        product = unique_cross_product(data, product, combiner=combiner)
        features.extend(product)

    return features


def numerical_polynomial_features(data: List, degree: int) -> List:
    return polynomial_features(
        data=data, degree=degree, one=1.0, combiner=lambda alfa, beta: alfa * beta
    )
