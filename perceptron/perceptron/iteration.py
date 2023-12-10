from typing import Any, Iterable, Callable


def reduce_while(
    function: Callable[[Any, Any], Any],
    sequence: Iterable,
    initial: Any,
    while_predicate: Callable[[Any, Any], bool],
):
    """
    reduce_while(function, iterable, initial, while_predicate) -> value

    Just like functools.reduce but accepting a predicate function that controls
    permanence in the cycle.

    Useful for leaving when a certain condition is reached
    """

    it = iter(sequence)

    accumulator = initial

    for current in it:
        accumulator = function(accumulator, current)
        if not while_predicate(accumulator, current):
            break
    return accumulator
