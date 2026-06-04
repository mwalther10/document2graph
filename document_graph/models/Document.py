from pydantic import BaseModel

class Document(BaseModel):
    document_id: str = ""
    document_version: str = ""
    document_type: str = ""
    filename: str = ""
    title: str = ""
    authors: str = ""
    institutions: str = ""
    bibliography: str = ""
    correspondence: str = ""