import os

from ai_agent.tools import scan_folder


def test_scan_folder():
    print(scan_folder(os.path.dirname(__file__)))
    assert sorted(
        [item["name"] for item in scan_folder(os.path.dirname(__file__))]
    ) == sorted(
        [
            "test_tool_extract_text.py",
            "__init__.py",
            "test_tool_scan_folder.py",
        ]
    )
