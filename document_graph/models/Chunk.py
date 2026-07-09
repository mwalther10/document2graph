from pydantic import BaseModel
from typing import Any, Sequence

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    meta: dict[str, Any] | None = None
    text: str
    embedding: Sequence[float] | None = None
    baseline_description: str | None = None