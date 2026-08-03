from datetime import datetime
print(f"Pre import ollama {datetime.now()}")
import ollama
print(f"Post import ollama {datetime.now()}")



# --- DEFINIZIONE DEGLI STRUMENTI (TOOLS) ---
# Queste sono le funzioni che l'agente può decidere di "chiamare"
def get_weather(city: str):
    """Restituisce il meteo per una città specifica."""
    # In un caso reale, qui ci sarebbe una chiamata ad API meteo
    if "roma" in city.lower():
        return "22°C, Soleggiato"
    return "18°C, Nuvoloso"


def calculate_sum(a: float, b: float):
    """Esegue l'addizione di due numeri."""
    return a + b


# --- LOGICA DELL'AGENTE ---
def simple_agent(user_prompt):
    # Definiamo le capacità dell'agente nel System Prompt
    system_prompt = """
    Sei un assistente utile. Hai a disposizione i seguenti strumenti:
    1. get_weather(city): per ottenere il meteo.
    2. calculate_sum(a, b): per sommare due numeri.

    Se l'utente chiede il meteo o una somma, rispondi nel seguente formato:
    ACTION: nome_funzione(parametri)
    Risultato della funzione: [risultato]
    Risposta finale all'utente.
    """

    # Primo passaggio: L'LLM decide cosa fare
    response = ollama.generate(
        model="Qwen2.5-Coder:7b", prompt=f"{system_prompt}\nUtente: {user_prompt}"
    )
    decision = response["response"]

    print(f"--- Ragionamento Agente ---\n{decision}")

    # Logica semplificata di "esecuzione azione" (Parsing manuale per didattica)
    if "calculate_sum" in decision:
        # Estraiamo i numeri (in un progetto reale useresti una Regex o
        # Pydantic)
        # Per semplicità, simuliamo l'esecuzione della funzione
        result = calculate_sum(10, 20)  # Esempio statico
        print(f"Esecuzione Tool: {result}")
        return f"Il risultato del calcolo è {result}."

    elif "get_weather" in decision:
        result = get_weather("Roma")
        print(f"Esecuzione Tool: {result}")
        return f"Il meteo attuale a Roma è {result}."

    else:
        return response["response"]


# --- TEST ---
if __name__ == '__main__':
    ini = datetime.now()
    print(f"INI: {ini}")
    print("Domanda 1: Che tempo fa a Roma?")
    print("Risposta:", simple_agent("Che tempo fa a Roma?"))
    print("-" * 30)
    print("Domanda 2: Quanto fa 10 + 20?")
    print("Risposta:", simple_agent("Quanto fa 10 + 20?"))
    end = datetime.now()
    print(f"END: {end}")
    print(f"DUR: {end-ini}")
