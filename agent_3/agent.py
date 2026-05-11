import os

from llama_cpp import Llama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any


# ------------------------------------------------------------
# 1) Stato dell’agente
# ------------------------------------------------------------
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]


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
    """Invoca il modello con il prompt utente con istruzioni."""
    prompt = ""
    for m in messages:
        prompt += f"{m['role'].upper()}: {m['content']}\n"

    prompt += """
ISTRUZIONI:
- Verifica se il prompt è un'operazione aritmetica. Se è un'operazione aritmetica allora usa un tool come spiegato sotto.
- Se devi usare un tool, rispondi SOLO nel formato:
  <tool name="calculator">ESPRESSIONE</tool>
- Se devi rispondere all’utente, usa:
  <assistant>TESTO</assistant>
"""

    out = llm(prompt)
    text = out["choices"][0]["text"].strip()
    return {"role": "assistant", "content": text}


def check_user_request_via_llm(messages: List[dict]) -> dict:
    """Invoca il modello con icercando di capire la richiesta utente"""

    all_user_input = ", ".join(message["content"] for message in messages if message["role"]=="user")

    prompt = (
        f"given the following prompt: [{all_user_input}]: "
        f"if it is a calculation, respond with '[CALC: {all_user_input}]' "
        f"otherwise respond with '[OTHER: {all_user_input}])' "
    )

    out = llm(prompt)
    text = out["choices"][0]["text"].strip()
    return {"role": "assistant", "content": text}


# ------------------------------------------------------------
# 3) Tool (calculator)
# ------------------------------------------------------------
@tool
def calculator_tool(expression: str) -> str:
    """Valuta un'espressione matematica semplice."""
    return calculator(expression=expression)


def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Errore: {e}"


# ------------------------------------------------------------
# 4) Nodo LLM
# ------------------------------------------------------------
def check_user_request_via_llm_node(state: AgentState):
    msg = check_user_request_via_llm(state["messages"])
    return {"messages": state["messages"] + [msg]}


# ------------------------------------------------------------
# 5) Nodo tool
# ------------------------------------------------------------
def tool_node(state: AgentState):
    last = state["messages"][-1]["content"]

    # estrai tool call strutturata
    if "<tool" in last:
        name = last.split('name="')[1].split('"')[0]
        expr = last.split(">")[1].split("<")[0]

        if name == "calculator":
            result = calculator(expr)

            return {
                "messages": state["messages"]
                + [{"role": "tool", "content": result, "tool_name": name}]
            }

    return state


def get_next_tool(state: AgentState) -> str:
    return "tool" if "calculator(" in state["messages"][-1]["content"] else END


def route(state: AgentState):
    last = state["messages"][-1]["content"]

    if last.startswith("<tool"):
        return "tool"

    return END


# ------------------------------------------------------------
# 6) Grafo dell’agente
# ------------------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("llm", check_user_request_via_llm_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    route,
    {
        "tool": "tool",
        END: END,
    },
)

graph.add_edge("tool", "llm")

agent_executor = graph.compile()

# Genera una chiamata alla funzione calculator() che calcola (23 + 5) * 2. Non aggiungere testo.
