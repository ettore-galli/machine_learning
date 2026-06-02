import os

from llama_cpp import Llama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import List
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from agent.utils import calculator
from agent.base import AgentState
from agent.ollama_client import OllamaClient


def extract_message_from_completion(completion: ChatCompletion) -> str:
    return completion.choices[0].message.content


def openai_chat(system_prompt: str, user_prompt: str) -> ChatCompletion:
    client = OpenAI(
        base_url=f'{os.getenv("LLAMA_CPP_SERVER_URL")}/v1', api_key="not-needed"
    )

    return client.chat.completions.create(
        model=os.getenv("LLAMA_CPP_SERVER_MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )


def check_user_request_via_llm(messages: List[dict]) -> dict:
    """Invoca il modello con icercando di capire la richiesta utente"""

    all_user_input = ", ".join(
        message["content"] for message in messages if message["role"] == "user"
    )

    system_prompt = "Rispondi CALC + il prompt utente se l’input contiene esclusivamente cifre (0-9), spazi, + - * / e parentesi ( ) altrimenti rispondi OTHER"
    user_prompt = all_user_input

    out = openai_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return out


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


def route(state: AgentState):
    last = state["messages"][-1].choices[0].message.content

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

agent_executor_legacy_dont_use_this = graph.compile()

# Genera una chiamata alla funzione calculator() che calcola (23 + 5) * 2. Non aggiungere testo.
