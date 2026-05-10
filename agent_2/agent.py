import os

from llama_cpp import Llama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List


# ------------------------------------------------------------
# 1) Stato dell’agente
# ------------------------------------------------------------
class AgentState(TypedDict):
    messages: List[dict]


# ------------------------------------------------------------
# 2) Modello locale GGUF
# ------------------------------------------------------------
llm = Llama(
    model_path=os.getenv("MODEL_PATH"),
    n_ctx=4096,
    temperature=0.2,
    max_tokens=512,
)


def call_llm(messages: List[dict]) -> dict:
    """Invoca il modello locale in stile Chat."""
    prompt = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        prompt += f"{role.upper()}: {content}\n"

    out = llm(prompt)
    text = out["choices"][0]["text"]
    return {"role": "assistant", "content": text}


# ------------------------------------------------------------
# 3) Tool (calculator)
# ------------------------------------------------------------
@tool
def calculator_tool(expression: str) -> str:
    """Valuta un'espressione matematica semplice."""
    return calculator(expression=expression)


def calculator(expression: str) -> str:
    print("***** USO lA CALCOLATRICE *****")
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Errore: {e}"


# ------------------------------------------------------------
# 4) Nodo LLM
# ------------------------------------------------------------
def llm_node(state: AgentState):
    msg = call_llm(state["messages"])
    return {"messages": state["messages"] + [msg]}


# ------------------------------------------------------------
# 5) Nodo tool
# ------------------------------------------------------------
def tool_node(state: AgentState):
    last = state["messages"][-1]["content"]
    print("***** USO IL TOOL NODE *****")
    # estrai input tool (didattico)
    if "calculator(" in last:
        expr = last.split("calculator(")[1].split(")")[0]
        try:
            result = calculator(expr)
        except:
            result = 0
        return {"messages": state["messages"] + [{"role": "tool", "content": result}]}

    return state


def get_next_tool(state: AgentState) -> str:
    return "tool" if "calculator(" in state["messages"][-1]["content"] else END


# ------------------------------------------------------------
# 6) Grafo dell’agente
# ------------------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    get_next_tool,
    {"tool": "tool", END: END},
)

graph.add_edge("tool", "llm")

agent_executor = graph.compile()

# Genera una chiamata alla funzione calculator() che calcola (23 + 5) * 2. Non aggiungere testo.
