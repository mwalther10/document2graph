import os
from .base_extractor import Extractor
from ..models.ChunkerConfig import ChunkerConfig
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

class HybridChunkerExtractor(Extractor):
    def __init__(self, source: str, config: ChunkerConfig | None = None,
                 pipeline_options: PdfPipelineOptions | None = None,
                 cache_dir: str | None = None, refresh: bool = False):
        super().__init__(source, pipeline_options, cache_dir=cache_dir, refresh=refresh)
        config = config or ChunkerConfig()
        self.config = config
        self.merge_peers = config.merge_peers
        self.chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer.from_pretrained(config.tokenizer, max_tokens=config.max_tokens),
            merge_peers=config.merge_peers)

    def extract_and_chunk(self, save_dir: str, filename: str):
        # the document was parsed (or loaded from the cache) by the base extractor;
        # chunk that one rather than converting the same PDF a second time
        self.logger.info(f"Saving parsed pdf as {filename}.json")

        os.makedirs(save_dir, exist_ok=True)

        self.doc.save_as_json(
            f"{save_dir}/{filename}_baseline_docling_doc.json"
        )
        yield from self.chunker.chunk(self.doc)