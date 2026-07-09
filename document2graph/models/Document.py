from pydantic import BaseModel
from .DocumentMetadata import DocumentMetadata


class Document(BaseModel):
    document_id: str = ""
    document_type: str = ""
    filename: str = ""
    title: str = ""
    metadata: DocumentMetadata = DocumentMetadata()