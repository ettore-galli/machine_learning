from typing import Any, Iterable, Callable, Optional


def reduce_conditioned(
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
    while_predicate: Callable[[Any, Any], bool],
):
    return reduce_conditioned(
        function=function,
        sequence=sequence,
        initial=initial,
        while_predicate=while_predicate,
        until_predicate=None,
    )


def reduce_until(
    function: Callable[[Any, Any], Any],
    sequence: Iterable,
    initial: Any,
    until_predicate: Callable[[Any, Any], bool],
):
    return reduce_conditioned(
        function=function,
        sequence=sequence,
        initial=initial,
        while_predicate=None,
        until_predicate=until_predicate,
    )
