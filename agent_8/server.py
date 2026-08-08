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
    except Exception as e:
        return f"Errore: {e}"

@mcp.tool()
def reverse_string(text: str) -> str:
    """Inverte una stringa."""
    return text[::-1]

@mcp.tool()
def word_count(text: str) -> int:
    """Conta le parole in un testo."""
    return len(text.split())

if __name__ == "__main__":
    mcp.run()   # transport stdio di default