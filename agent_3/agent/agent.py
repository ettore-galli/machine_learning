import os

from llama_cpp import Llama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import List

from agent.utils import calculator
from agent.base import AgentState

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

    all_user_input = ", ".join(
        message["content"] for message in messages if message["role"] == "user"
    )

    prompt = (
        f"""
        Sei un classificatore deterministico. Devi produrre SOLO una delle due forme:

        CALC <input>
        OTHER <input>

        Regole:
        - Considera "espressione aritmetica valida" solo se l’input contiene esclusivamente cifre, spazi, + - * / e parentesi.
        - Se l’input è una espressione aritmetica valida, rispondi esattamente:
        CALC <input>

        - Altrimenti rispondi esattamente:
        OTHER <input>

        Vincoli assoluti:
        - Non aggiungere testo.
        - Non spiegare.
        - Non riformattare l’input.
        - Non generare contenuti in stile web, Quora, FAQ, articoli, ricerche o suggerimenti.
        - Non completare l’espressione.
        - Non risolvere l’espressione.
        - Non interpretare l’intento dell’utente.
        - Non generare nulla oltre alle due forme consentite.

        input: {all_user_input}

        """
    )

    out = llm(prompt)
    text = out["choices"][0]["text"].strip()
    return {"role": "assistant", "content": text}


# ------------------------------------------------------------
# 4) Nodo LLM
# ------------------------------------------------------------
def check_user_request_via_llm_node(state: AgentState):
    msg = check_user_request_via_llm(state["messages"])
    return {"messages": state["messages"] + [msg]}


# ------------------------------------------------------------
# 5) Nodo tool
# ------------------------------------------------------------
def calculator_node(state: AgentState):
    last = state["messages"][-1]["content"]

    # estrai tool call strutturata
    # if last.strip().startswith("CALC"):
    name = "calculator"
    expr = last.split("CALC")[1].split("\n")[0]

    if name == "calculator":
        result = calculator(expr)

    return {
        "messages": state["messages"]
        + [{"role": "tool", "content": result, "tool_name": name}]
    }

    # return state


# def get_next_tool(state: AgentState) -> str:
#     return "tool" if "calculator(" in state["messages"][-1]["content"] else END


def route(state: AgentState):
    last = state["messages"][-1]["content"]

    if "CALC" in last:
        return "calculator_tool"

    return END


# ------------------------------------------------------------
# 6) Grafo dell’agente
# ------------------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("llm_check_request", check_user_request_via_llm_node)
graph.add_node("calculator_tool", calculator_node)

graph.set_entry_point("llm_check_request")

graph.add_conditional_edges(
    "llm_check_request",
    route,
    {
        "calculator_tool": "calculator_tool",
        END: END,
    },
)

graph.add_edge("calculator_tool", "llm_check_request")

agent_executor = graph.compile()

# Genera una chiamata alla funzione calculator() che calcola (23 + 5) * 2. Non aggiungere testo.
