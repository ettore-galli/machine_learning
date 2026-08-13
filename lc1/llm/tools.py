import os

from langchain.tools import tool


@tool
def get_downloads_directory() -> dict[str, str]:
    """Get the directory listing of the Downloads directory"""
    print("STO USANDO IL TOOL get_downloads_directory()")
    return {"path": os.path.expanduser("~/Downloads")}


@tool
def superbeta(input: str) -> float | str:
    """Calculate the superbeta funcion of a number"""
    print("STO USANDO IL TOOL superbeta()")
    try:
        return float(input) + 44
    except:  # noqa: E722
        return f"super-beta-of({input.strip().upper()})"
