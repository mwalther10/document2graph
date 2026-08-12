import os
from .base_extractor import Extractor
from ..models.ChunkerConfig import ChunkerConfig
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

class HybridChunkerExtractor(Extractor):
    def __init__(self, source: str, config: ChunkerConfig = ChunkerConfig(), pipeline_options: PdfPipelineOptions = PdfPipelineOptions()):
        super().__init__(source, pipeline_options)
        self.config = config
        self.merge_peers = config.merge_peers
        self.chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(config.tokenizer, max_tokens=config.max_tokens),
            merge_peers=config.merge_peers)

    def extract_and_chunk(self, save_dir: str, filename: str):
        # First, we extract the document using the base extractor logic
        docling_doc = self.converter.convert(self.source).document
        # Then, we apply the hybrid chunker to the extracted document
        self.logger.info(f"Saving parsed pdf as {filename}.json")

        os.makedirs(save_dir, exist_ok=True)

        docling_doc.save_as_json(
            f"{save_dir}/{filename}_baseline_docling_doc.json"
        )
        yield from self.chunker.chunk(docling_doc)