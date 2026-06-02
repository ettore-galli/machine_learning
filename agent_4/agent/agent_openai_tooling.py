import json
from typing import Any
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from langgraph.graph import StateGraph, END
from agent.base import AgentSettings, AgentState
from agent.utils import calculator, meteo

DEFAULT_NO_RESPONSE: str = "Dati insufficienti per una risposta significativa"

SYSTEM_PROMPT = """
Se l'utente chiede un calcolo matematico, NON rispondere direttamente.
Invece, restituisci ESATTAMENTE un JSON di questo tipo:

{"tool": "calculator", "args": {"expr": "<espressione>"}}

Esempi:
- Input: "quanto fa 2+2?"
  Output: {"tool": "calculator", "args": {"expr": "2+2"}}

Se l'utente chiede informazioni sul meteo, NON rispondere direttamente.
Invece restituisci ESATTAMENTE un JSON del tipo:
{"tool": "weather", "args": {"city": "<nome città>"}, "when": "<quando>"}}  

Esempi:
- Input: "Che tempo farà a Firenze domani?"
  Output: {"tool": "meteo", "args": {"city":"Firenze", "when": "Domani"}}
- Input: "Dammi il meteo di Torino"
  Output: {"tool": "meteo", "args": {"city":"Firenze", "when": "Oggi"}}

Se invece la richiesta NON richiede il tool, rispondi normalmente in italiano,
con un testo libero (non JSON).
"""

KNOWN_CALC_TOOLS = ["calculator", "calc", "calculus"]
KNOWN_METEO_TOOLS = ["weather", "meteo", "tempo"]


agent_settings: AgentSettings = AgentSettings.load()

client: OpenAI = OpenAI(
    base_url=f"{agent_settings.llama_cpp_server_url}/v1",
    api_key="not-needed",
)


def to_openai_message(msg: dict[str, Any]) -> ChatCompletionMessageParam:
    role = msg["role"]
    content = msg["content"]

    if role == "system":
        return ChatCompletionSystemMessageParam(role="system", content=content)

    if role == "user":
        return ChatCompletionUserMessageParam(role="user", content=content)

    if role == "assistant":
        return ChatCompletionAssistantMessageParam(role="assistant", content=content)

    if role == "tool":
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": msg["tool_call_id"],
        }

    raise ValueError(f"Ruolo sconosciuto: {role}")


def llm_node(state: AgentState):
    # 1. Storia vecchia (tutti i messaggi tranne l'ultimo)
    history = [to_openai_message(m) for m in state["messages"][:-1]]

    # 2. System prompt (sempre fresco)
    system_msg = ChatCompletionSystemMessageParam(role="system", content=SYSTEM_PROMPT)

    # 3. User prompt attuale (ultimo messaggio)
    user_msg = to_openai_message(state["messages"][-1])

    # Ordine corretto: storia → system → user
    messages = history + [system_msg, user_msg]

    completion = client.chat.completions.create(
        model=agent_settings.llama_cpp_server_model,
        messages=messages,
        temperature=0,
    )

    msg = completion.choices[0].message
    msg_dict = msg.model_dump()

    return {"messages": state["messages"] + [msg_dict]}


def calculator_tool_node(state: AgentState) -> dict[str, Any]:
    last = state["messages"][-1]
    content = last.get("content", "")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Il modello ha sbagliato formato: fallback
        result_text = f"Errore: output non valido per il tool: {content}"
        final_msg = {"role": "assistant", "content": result_text}
        return {"messages": state["messages"] + [final_msg]}

    if data.get("tool") not in KNOWN_CALC_TOOLS:
        result_text = f"Errore: tool richiesto sconosciuto: {data.get('tool')}"
        final_msg = {"role": "assistant", "content": result_text}
        return {"messages": state["messages"] + [final_msg]}

    args = data.get("args", {})
    expr = args.get("expr", "")

    result = calculator(expr)
    final_msg = {
        "role": "assistant",
        "content": str(result),
    }

    # Qui, per la tua specifica, la risposta finale È la risposta del tool
    return {"messages": state["messages"] + [final_msg]}


def meteo_tool_node(state: AgentState) -> dict[str, Any]:
    last = state["messages"][-1]
    content = last.get("content", "")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Il modello ha sbagliato formato: fallback
        result_text = f"Errore: output non valido per il tool: {content}"
        final_msg = {"role": "assistant", "content": result_text}
        return {"messages": state["messages"] + [final_msg]}

    if data.get("tool") not in KNOWN_METEO_TOOLS:
        result_text = f"Errore: tool richiesto sconosciuto: {data.get('tool')}"
        final_msg = {"role": "assistant", "content": result_text}
        return {"messages": state["messages"] + [final_msg]}

    args = data.get("args", {})
    city = str(args.get("city", "")).lower()
    when = str(args.get("when", "")).lower()

    result = meteo(city, when)

    final_msg = {
        "role": "assistant",
        "content": str(result),
    }

    # Qui, per la tua specifica, la risposta finale È la risposta del tool
    return {"messages": state["messages"] + [final_msg]}


def final_answer_node(state: AgentState) -> dict[str, Any]:
    last = state["messages"][-1]
    content = last.get("content", DEFAULT_NO_RESPONSE)

    final_msg = {
        "role": "assistant",
        "content": content,
    }

    return {"messages": state["messages"] + [final_msg]}


def route_from_llm(state: AgentState) -> str:
    last = state["messages"][-1]
    content = last.get("content", "")

    # Proviamo a interpretare il contenuto come JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Non è JSON → risposta diretta del modello
        return "final"

    tool_name = data.get("tool")

    if tool_name.lower().strip() in KNOWN_CALC_TOOLS:
        return "calculator_tool"

    if tool_name.lower().strip() in KNOWN_METEO_TOOLS:
        return "meteo_tool"

    # Tool sconosciuto o assente → risposta diretta
    return "final"


graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("calculator_tool", calculator_tool_node)
graph.add_node("meteo_tool", meteo_tool_node)
graph.add_node("final", final_answer_node)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    route_from_llm,
    {
        "calculator_tool": "calculator_tool",
        "meteo_tool": "meteo_tool",
        "final": "final",
    },
)

# Il nodo calculator_tool produce già la risposta finale
# quindi non serve un ulteriore passaggio: possiamo andare direttamente a END
graph.add_edge("calculator_tool", END)

agent_executor = graph.compile()
