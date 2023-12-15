from itertools import accumulate, takewhile
from typing import Any, Callable, List

MAXIMUM_ITERATIONS_EVER = 1000000000000000


def accumulate_iterate_while(
    initial: Any,
    iteration_function: Callable[[Any], Any],
    while_predicate: Callable[[Any], bool],
    maximum_iterations: int = MAXIMUM_ITERATIONS_EVER,
    evaluate_predicate_post: bool = False,
) -> List[Any]:
    return list(
        takewhile(
            while_predicate,
            accumulate(
                iterable=range(1 if evaluate_predicate_post else 0, maximum_iterations),
                func=lambda accumulator, _: iteration_function(accumulator),
                initial=(
                    iteration_function(initial) if evaluate_predicate_post else initial
                ),
            ),
        )
    ) or [initial]


def iterate_while(
    initial: Any,
    iteration_function: Callable[[Any], Any],
    while_predicate: Callable[[Any], bool],
    maximum_iterations: int = 10000000000,
    evaluate_predicate_post: bool = False,
) -> Any:
    return accumulate_iterate_while(
        initial=initial,
        iteration_function=iteration_function,
        while_predicate=while_predicate,
        maximum_iterations=maximum_iterations,
        evaluate_predicate_post=evaluate_predicate_post,
    )[-1]
