from pydantic import BaseModel
from docling.datamodel.pipeline_options import PdfPipelineOptions
class ExtractorConfig(BaseModel):
    pdf_path: str
    data_path: str
    pdfPipelineOptions: PdfPipelineOptions = PdfPipelineOptions()
    document_type: str = "Praxisempfehlung"
    save_json: bool = True