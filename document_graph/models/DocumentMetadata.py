from pydantic import BaseModel


class MetadataFieldConfig(BaseModel):
    label: str               # substring to search for in the document text
    pages: tuple[int, int]   # inclusive 1-based page range to search on


class MetadataExtractionConfig(BaseModel):
    title_page: int = 2
    version: MetadataFieldConfig | None = MetadataFieldConfig(label="version", pages=(1, 1))
    authors: MetadataFieldConfig | None = MetadataFieldConfig(label="korrespondenzadresse", pages=(2, 2))
    institutions: MetadataFieldConfig | None = MetadataFieldConfig(label="institute", pages=(2, 2))
    bibliography: MetadataFieldConfig | None = MetadataFieldConfig(label="bibliografie", pages=(2, 2))
    correspondence: MetadataFieldConfig | None = MetadataFieldConfig(label="korrespondenzadresse", pages=(2, 2))


class DocumentMetadata(BaseModel):
    version: str = ""
    authors: str = ""
    institutions: str = ""
    bibliography: str = ""
    correspondence: str = ""
