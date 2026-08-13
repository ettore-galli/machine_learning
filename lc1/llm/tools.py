import os

from langchain.tools import tool


@tool
def get_downloads_directory() -> dict[str, str]:
    """Get the directory listing of the Downloads directory"""
    print("STO USANDO IL TOOL get_downloads_directory()")
    return {"path": os.path.expanduser("~/Downloads")}
