from datetime import datetime
from typing import Tuple

print(f"Pre import ollama {datetime.now()}")
import ollama

print(f"Post import ollama {datetime.now()}")

ACTION_IDENTIFIER = "ACTION"


# --- DEFINIZIONE DEGLI STRUMENTI (TOOLS) ---
# Queste sono le funzioni che l'agente può decidere di "chiamare"
def get_weather(input_city: str):
    """Restituisce il meteo per una città specifica."""
    meteo = {
        "milano": 35,
        "roma": 34,
        "mulazzano": 38,
        "lodi": 32,
    }

    query_city = input_city.lower().strip().strip("\"'")
    
    temperature = meteo.get(query_city, sum(meteo.values()) / len(meteo.values()))
    meteo_response = f"{input_city}: {temperature} °C ({32 + temperature*9/5} °F)"

    return meteo_response


def calculate_sum(a: float, b: float):
    """Esegue l'addizione di due numeri."""
    return a + b


def needs_tool_use(response: str) -> bool:
    return response.startswith(ACTION_IDENTIFIER)


def get_tool_payload(response: str) -> Tuple[str, ...]:
    action_payload = response.split(":")[1].strip()
    return tuple(action_payload.split(","))


# --- LOGICA DELL'AGENTE ---
def simple_agent(user_prompt):
    # Definiamo le capacità dell'agente nel System Prompt
    system_prompt = """
    Sei un assistente utile. Hai a disposizione i seguenti strumenti:

    1. get_weather(city): per ottenere il meteo.
    2. calculate_sum(a, b): per sommare due numeri.

    Se l'utente chiede il meteo o una somma, rispondi nel seguente formato:

    ACTION: nome_funzione, parametro1, parametro2, ecc ecc

    Ad esempio

    ACTION: get_weather, "Mulazzano"
    ACTION: calculate_sum, 17, 34

    """

    # Primo passaggio: L'LLM decide cosa fare
    response = ollama.generate(
        model="Qwen2.5-Coder:7b", prompt=f"{system_prompt}\nUtente: {user_prompt}"
    )
    decision = response["response"]

    print(f"--- Ragionamento Agente ---\n{decision}")

    if needs_tool_use(response=decision):
        print("*** USO TOOL ***")
        tool_payload = get_tool_payload(decision)

        # Logica semplificata di "esecuzione azione" (Parsing manuale per didattica)
        if "calculate_sum" in decision:
            print("*** USO TOOL SUM ***")
            # Estraiamo i numeri (in un progetto reale useresti una Regex o
            # Pydantic)
            # Per semplicità, simuliamo l'esecuzione della funzione

            _, addendum_a, addendum_2 = tool_payload

            result = calculate_sum(
                float(addendum_a), float(addendum_a)
            )  # Esempio statico

            print(f"Esecuzione Tool: {result}")

            return f"Il risultato del calcolo è {result}."

        elif "get_weather" in decision:
            print("*** USO TOOL METEO ***")

            _, city = tool_payload

            result = get_weather(city)
            print(f"Esecuzione Tool: {result}")
            return f"Il meteo attuale a {city} è {result}."
        else:
            print("*** TOOL SCONOSCIUTO ***")
            return f"Unrecognized tool for {response['response']}"

    else:
        return response["response"]


# --- TEST ---
if __name__ == "__main__":
    ini = datetime.now()
    print(f"INI: {ini}")
    for city in ["Mulazzano", "Milano", "Casalmaiocco"]:
        domanda = f"Che tempo fa a {city}?"
        print(f"Domanda 1: {domanda}")
        print("Risposta:", simple_agent(domanda))
        print("-" * 30)

    domanda = f"Quanto fa 17.2 + 51?"
    print(f"Domanda 2: {domanda}")
    print("Risposta:", simple_agent(domanda))
    end = datetime.now()
    print(f"END: {end}")
    print(f"DUR: {end-ini}")
