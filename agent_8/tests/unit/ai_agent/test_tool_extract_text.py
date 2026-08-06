import os
from pathlib import Path

from ai_agent.tools import extract_text

DATA = Path(os.path.dirname(__file__), "data")
PDF_1 = Path(DATA, "test-pdf.pdf")


def test_extract_test():
    assert extract_text(PDF_1) == "Questa è una prova di file PDF"
