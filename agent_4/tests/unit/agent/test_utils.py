from pytest import mark

from agent.utils import calculator


@mark.parametrize(
    "expression, expected_result",
    [("7*3", "21"), ("ermenegildo", "Errore: name 'ermenegildo' is not defined")],
)
def test_calculator(expression, expected_result):
    assert calculator(expression) == expected_result
