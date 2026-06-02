import os
from typing import List
from openai import OpenAI
from langgraph.graph import StateGraph, END
from agent.base import AgentState
from agent.utils import calculator


def openai_chat(system_prompt: str, user_prompt: str) -> str:
    client = OpenAI(
        base_url=f'{os.getenv("LLAMA_CPP_SERVER_URL")}/v1',
        api_key="not-needed",
    )

    completion = client.chat.completions.create(
        model=os.getenv("LLAMA_CPP_SERVER_MODEL"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    return completion.choices[0].message.content


def check_user_request_via_llm(messages: List[dict]) -> dict:
    user_text = ", ".join(m["content"] for m in messages if m["role"] == "user")

    system_prompt = (
        "Se l’input contiene SOLO cifre, spazi, + - * / e parentesi, "
        "rispondi esattamente: CALC <expr>. "
        "Altrimenti rispondi esattamente: OTHER"
    )

    result = openai_chat(system_prompt, user_text)

    # Normalizziamo in formato LangGraph
    return {"role": "assistant", "content": result}


def check_user_request_via_llm_node(state: AgentState):
    msg = check_user_request_via_llm(state["messages"])
    return {"messages": state["messages"] + [msg]}


def calculator_node(state: AgentState):
    last = state["messages"][-1]["content"]

    expr = last.replace("CALC", "").strip()
    result = calculator(expr)

    return {
        "messages": state["messages"]
        + [{"role": "tool", "tool_name": "calculator", "content": str(result)}]
    }


def final_answer_node(state: AgentState):
    last = state["messages"][-1]["content"]

    result = f"Risposta finale: {last}"

    return {
        "messages": state["messages"]
        + [{"role": "tool", "tool_name": "calculator", "content": result}]
    }


def route(state: AgentState):
    last_msg = state["messages"][-1]

    # Se il tool ha già risposto → fine
    if last_msg["role"] == "tool":
        return "final"

    # Altrimenti routing normale
    content = last_msg["content"]
    if content.startswith("CALC"):
        return "calculator_tool"

    return END


graph = StateGraph(AgentState)

graph.add_node("llm_check_request", check_user_request_via_llm_node)
graph.add_node("calculator_tool", calculator_node)
graph.add_node("final", final_answer_node)

graph.set_entry_point("llm_check_request")

graph.add_conditional_edges(
    "llm_check_request",
    route,
    {
        "calculator_tool": "calculator_tool",
        END: END,
    },
)

graph.add_edge("calculator_tool", "final")

agent_executor = graph.compile()
