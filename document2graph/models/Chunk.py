from pydantic import BaseModel
from typing import Any, Sequence

from .Provenance import Provenance

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str = ""
    meta: dict[str, Any] | None = None
    # the PDF locations this chunk was merged from, lifted out of meta["doc_items"]
    provenance: list[Provenance] = []
    text: str
    embedding: Sequence[float] | None = None
    baseline_description: str | None = None