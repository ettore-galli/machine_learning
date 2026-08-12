import os

from langchain.tools import tool


@tool
def get_downloads_directory() -> list[str]:
    """Get the directory listing of the Downloads directory"""
    print("STO USANDO IL TOOL get_downloads_directory()")
    return os.listdir("/Users/ettoregalli/Downloads/")
