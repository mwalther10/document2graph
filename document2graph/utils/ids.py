"""Stable identifiers shared by every representation of a document.

Both extractors derive the same ``document_id`` for the same PDF, and derive it
the same way on every run, so that graph nodes and baseline chunks can be joined
on it and so that a re-run updates a corpus instead of duplicating it.
"""

import os
import uuid

# namespace for all document2graph ids; changing it re-keys every corpus
DOCUMENT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "document2graph")


def clean_filename(filename: str) -> str:
    """The bare document name: no directory, no extension."""
    return os.path.splitext(os.path.basename(filename))[0]


def document_id_for(filename: str) -> str:
    """Deterministic document id derived from the document's name.

    Derived from the name rather than the file's content because both extractors
    have the name before parsing, which keeps ids stable across the docling
    cache. The trade-off is that renaming a PDF re-keys it.
    """
    return str(uuid.uuid5(DOCUMENT_NAMESPACE, clean_filename(filename)))


def chunk_id_for(document_id: str, index: int, variant: str = "baseline") -> str:
    """Deterministic chunk id, so two runs of the same condition are comparable
    chunk for chunk."""
    return str(uuid.uuid5(DOCUMENT_NAMESPACE, f"{document_id}:{variant}:{index}"))
