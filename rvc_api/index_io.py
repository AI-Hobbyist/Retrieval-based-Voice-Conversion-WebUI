"""FAISS's Windows filename API cannot read Unicode paths; use Python IO."""
from pathlib import Path


def read_index(path):
    import faiss
    with Path(path).open("rb") as source:
        return faiss.read_index(faiss.PyCallbackIOReader(source.read))
