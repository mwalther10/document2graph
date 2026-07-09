from pydantic import BaseModel, Field
from docling.datamodel.pipeline_options import PdfPipelineOptions
from .DocumentMetadata import MetadataExtractionConfig
from .EdgeWeightConfig import EdgeWeightConfig


class ExtractorConfig(BaseModel):
    pdf_path: str
    data_path: str
    pdfPipelineOptions: PdfPipelineOptions = PdfPipelineOptions()
    document_type: str = ""
    save_json: bool = True
    metadata_config: MetadataExtractionConfig = Field(default_factory=MetadataExtractionConfig)
    edge_weights: EdgeWeightConfig = Field(default_factory=EdgeWeightConfig)