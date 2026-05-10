from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Valuta un'espressione matematica semplice."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Errore nel calcolo: {e}"
