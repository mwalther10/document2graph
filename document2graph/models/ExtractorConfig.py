from pydantic import BaseModel, Field
from docling.datamodel.pipeline_options import PdfPipelineOptions
from .ChunkerConfig import ChunkerConfig
from .DocumentMetadata import MetadataExtractionConfig
from .EdgeWeightConfig import EdgeWeightConfig
from .PipelineFlags import PipelineFlags


class ExtractorConfig(BaseModel):
    pdf_path: str
    data_path: str
    # default_factory: a shared instance would let one config mutate every other
    pdfPipelineOptions: PdfPipelineOptions = Field(default_factory=PdfPipelineOptions)
    document_type: str = ""
    save_json: bool = True
    metadata_config: MetadataExtractionConfig = Field(default_factory=MetadataExtractionConfig)
    edge_weights: EdgeWeightConfig = Field(default_factory=EdgeWeightConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    flags: PipelineFlags = Field(default_factory=PipelineFlags)
    # reuse the docling json under <data_path>/raw_texts/ instead of re-parsing the
    # PDF; ablations that only change post-parse behaviour then parse a corpus once
    use_docling_cache: bool = True
    refresh_docling_cache: bool = False