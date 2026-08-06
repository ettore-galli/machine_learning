import mimetypes
import os

import pdfplumber


def scan_folder(path):
    files = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            mime, _ = mimetypes.guess_type(full)
            files.append({"name": name, "path": full, "mime": mime})
    return files


def extract_text(path):
    mime, _ = mimetypes.guess_type(path)

    if mime == "application/pdf":
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    if mime and mime.startswith("text"):
        return path.read_text(encoding="utf-8")

    return None
