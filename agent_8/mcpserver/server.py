from mcp.server import MCPServer

mcp = MCPServer("CoderTools")


@mcp.tool()
def calculate(expression: str) -> str:
    """Valuta un'espressione matematica Python sicura (es. '2 + 3 * 4')."""
    try:
        # Valutazione molto limitata e sicura
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Errore: caratteri non consentiti"
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:  # noqa: BLE001
        return f"Errore: {e}"


@mcp.tool()
def reverse_string(text: str) -> str:
    """Inverte una stringa."""
    return text[::-1]


@mcp.tool()
def word_count(text: str) -> int:
    """Conta le parole in un testo."""
    return len(text.split())


@mcp.tool()
def root(number: str) -> float:
    """calcola la radice quadrata."""
    try:
        return float(number) ** 0.5
    except ValueError:
        return -1.234567


@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@mcp.prompt()
def summarize(text: str) -> str:
    """Summarize a piece of text in one sentence."""
    return f"Summarize the following text in one sentence:\n\n{text}"


if __name__ == "__main__":
    mcp.run()  # transport stdio di default
