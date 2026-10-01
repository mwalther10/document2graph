"""JSON round-trip for a ``DocumentGraph``.

Export used to be one-way (GEXF), and GEXF drops everything a retrieval harness
needs: bbox, page_no, charspan, sequence_no, parent refs, table serializations.
These two functions close the loop, so a corpus is parsed once and every
condition of an ablation reads the graph back off disk.
"""

import os

from .document_graph import DocumentGraph


def save_graph(graph: DocumentGraph, path: str) -> str:
    """Write a document graph as JSON, creating the parent directory."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(graph.model_dump_json(indent=2))
    return path


def load_graph(path: str) -> DocumentGraph:
    """Read a document graph written by :func:`save_graph`."""
    with open(path) as f:
        return DocumentGraph.model_validate_json(f.read())
