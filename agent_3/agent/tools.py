from langchain.tools import tool

from agent_3.agent.utils import calculator


@tool
def calculator_tool(expression: str) -> str:
    """Valuta un'espressione matematica semplice."""
    return calculator(expression=expression)
