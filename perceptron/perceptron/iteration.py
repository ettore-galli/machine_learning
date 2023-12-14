from itertools import accumulate, takewhile
from typing import Any, Iterable, Callable, List, Optional


def __interruptable_reduce(
    function: Callable[[Any, Any], Any],
    sequence: Iterable,
    initial: Any,
    while_predicate: Optional[Callable[[Any, Any], bool]],
    until_predicate: Optional[Callable[[Any, Any], bool]],
):
    it = iter(sequence)

    accumulator = initial

    for current in it:
        if while_predicate and not while_predicate(accumulator, current):
            break

        accumulator = function(accumulator, current)

        if until_predicate and until_predicate(accumulator, current):
            break

    return accumulator


def reduce_while(
    function: Callable[[Any, Any], Any],
    sequence: Iterable,
    initial: Any,
    predicate: Callable[[Any, Any], bool],
):
    return __interruptable_reduce(
        function=function,
        sequence=sequence,
        initial=initial,
        while_predicate=predicate,
        until_predicate=None,
    )


def reduce_until(
    function: Callable[[Any, Any], Any],
    sequence: Iterable,
    initial: Any,
    predicate: Callable[[Any, Any], bool],
):
    return __interruptable_reduce(
        function=function,
        sequence=sequence,
        initial=initial,
        while_predicate=None,
        until_predicate=predicate,
    )


def accumulate_iterated_while(
    initial: Any,
    iteration_function: Callable[[Any], Any],
    while_predicate: Callable[[Any], bool],
    maximum_iterations: int = 10000000000,
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
