import os

from langchain.tools import tool


@tool
def get_downloads_directory() -> list[str]:
    """Get the directory listing of the Downloads directory"""
    print("STO USANDO IL TOOL get_downloads_directory()")
    return os.listdir("/Users/ettoregalli/Downloads")[:10] + ["altro..."]


@tool
def superbeta(input: str) -> float | str:
    """Calculate the superbeta funcion of a number"""
    print("STO USANDO IL TOOL superbeta()")
    try:
        return float(input) + 44
    except:  # noqa: E722
        return f"super-beta-of({input.strip().upper()})"
