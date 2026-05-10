#!/usr/bin/env python3
"""
check_env.py
Verifica ambiente, pacchetti, versioni e import critici
per un progetto LangChain 0.3.x + LangGraph 0.2.x + llama-cpp-python.
"""

import sys
import os
import importlib
import subprocess


REQUIRED_PACKAGES = {
    "langchain": ">=0.3.0",
    "langchain-community": ">=0.3.0",
    "langgraph": ">=0.2.59",
    "llama-cpp-python": ">=0.2.90",
    "python-dotenv": ">=1.0.1",
}


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_venv():
    print_header("1) Verifica Virtual Environment")

    if sys.prefix == sys.base_prefix:
        print("❌ Nessuna venv attiva")
        print("   Attiva la venv prima di continuare:")
        print("   source .venv/bin/activate")
        return False

    print(f"✔ Venv attiva: {sys.prefix}")
    return True


def check_package_installed(pkg):
    """Usa pip show per verificare se un pacchetto è installato."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_packages():
    print_header("2) Verifica pacchetti installati")

    all_ok = True

    for pkg, version in REQUIRED_PACKAGES.items():
        installed = check_package_installed(pkg)
        if installed:
            print(f"✔ {pkg} {version} — INSTALLATO")
        else:
            print(f"❌ {pkg} {version} — NON INSTALLATO")
            all_ok = False

    return all_ok


def try_import(module_name, symbol=None):
    """Prova un import e ritorna True/False."""
    try:
        module = importlib.import_module(module_name)
        if symbol:
            getattr(module, symbol)
        return True
    except Exception as e:
        print(f"❌ Import fallito: {module_name}{'.' + symbol if symbol else ''}")
        print(f"   Errore: {e}")
        return False


def check_imports():
    print_header("3) Verifica import critici")

    results = {
        "llama_cpp": try_import("llama_cpp", "Llama"),
        "langchain": try_import("langchain"),
        "langchain_community": try_import("langchain_community"),
        "langgraph": try_import("langgraph"),
    }

    if all(results.values()):
        print("✔ Tutti gli import funzionano")
        return True

    print("❌ Alcuni import NON funzionano")
    return False


def main():
    print_header("CHECK AMBIENTE LANGCHAIN + LLAMA_CPP")

    ok_venv = check_venv()
    ok_packages = check_packages()
    ok_imports = check_imports()

    print_header("RISULTATO FINALE")

    if ok_venv and ok_packages and ok_imports:
        print("🎉 Ambiente OK — puoi avviare il progetto senza problemi")
    else:
        print("⚠ Ambiente NON pronto — correggi gli errori sopra")


if __name__ == "__main__":
    main()
