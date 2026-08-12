import os
import uuid

from ..utils.hybrid_chunker_extractor import HybridChunkerExtractor
from tqdm import tqdm
import json
from ..models.Chunk import Chunk
from ..models.ExtractorConfig import ExtractorConfig
from typing import List

from ..utils.log import Log

_log = Log("BaselineExtractor")

class BaselineExtractor:
    def __init__(self, config: ExtractorConfig):
        self.logger = _log.logger
        self.pdf_path = config.pdf_path
        self.data_path = config.data_path
        self.raw_text_save_dir = os.path.join(self.data_path, "raw_texts/")
        self.chunk_save_dir = os.path.join(self.data_path, "baseline_chunks/")
        self.pdfPipelineOptions = config.pdfPipelineOptions
        self.save_json = config.save_json
        self.document_type = config.document_type
        self.chunker_config = config.chunker
    
    def _get_clean_filename(self, filename: str) -> str:
        return os.path.splitext(os.path.basename(filename))[0]

    def extract_baseline_chunks(self, filename: str, baseline_description: str, enrich: bool = False) -> dict[str, List[Chunk]]:
        """Extract baseline chunks from the raw text files for a given document."""
        sample = os.path.join(self.pdf_path, filename)
        clean_filename = self._get_clean_filename(sample)
        baseline_extractor = HybridChunkerExtractor(
            source=sample,
            config=self.chunker_config,
            pipeline_options=self.pdfPipelineOptions
        )
        chunks = []
        enriched_chunks = []
        for chunk in baseline_extractor.extract_and_chunk(save_dir=self.raw_text_save_dir, filename=clean_filename):
            meta = chunk.meta.export_json_dict()

            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=filename,
                text=chunk.text,
                meta=meta,
                baseline_description=baseline_description
            ))
            if enrich:
                enriched_text = baseline_extractor.chunker.contextualize(chunk)
                enriched_chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=filename,
                    text=enriched_text,
                    meta=meta,
                    baseline_description=f"Enriched; {baseline_description}"
                ))
        return {"baseline": chunks, "enriched": enriched_chunks}
    
    def generate_baseline_chunks(self, return_chunks: bool = False) -> dict[str, List[Chunk]] | None:
        """Generate baseline chunks for all documents in the pdf_path."""
        all_chunks = {}
        for filename in tqdm(os.listdir(self.pdf_path)):
            clean_filename = self._get_clean_filename(filename)
            if filename.endswith(".pdf"):
                self.logger.info(f"Extracting baseline chunks for {filename}...")
                baseline_description = f"Baseline chunks for {filename}"
                chunks = self.extract_baseline_chunks(filename, baseline_description, enrich=True)
                all_chunks[filename] = chunks

                # by default, the chunks are saved as json
                if self.save_json:
                    os.makedirs(self.chunk_save_dir, exist_ok=True)
                    with open(f"{self.chunk_save_dir}/{clean_filename}_baseline_chunks.json", "w") as f:
                        json.dump([chunk.model_dump() for chunk in chunks["baseline"]], f, indent=4)
                    if len(chunks["enriched"]) > 0:
                        with open(f"{self.chunk_save_dir}/{clean_filename}_enriched_chunks.json", "w") as f:
                            json.dump([chunk.model_dump() for chunk in chunks["enriched"]], f, indent=4)
            

        return all_chunks if return_chunks else None