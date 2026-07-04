from typing import List

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


def create_web_search_tool():
    return DuckDuckGoSearchRun()


@tool
def calculate_average(values: List[float | int]):
    "Calculate the average of a list of numbers."

    ave = sum(values) / len(values) if values else 0.0
    return ave
