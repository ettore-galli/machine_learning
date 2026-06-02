def calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Errore: {e}"


def meteo(city: str, when: str) -> str:
    meteo_map = {"milano": "sereno", "roma": "piovoso"}

    return f"{when} a {city} sarà {meteo_map.get(city, 'nuvoloso a tratti')}"
