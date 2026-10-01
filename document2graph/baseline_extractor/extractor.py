import os

from ..utils.hybrid_chunker_extractor import HybridChunkerExtractor
from tqdm import tqdm
import json
from ..models.Chunk import Chunk
from ..models.ExtractorConfig import ExtractorConfig
from typing import List

from ..utils.ids import chunk_id_for, clean_filename as _clean_filename, document_id_for
from ..utils.log import Log
from ..utils.provenance import provenance_from_chunk_meta

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
        self.use_docling_cache = config.use_docling_cache
        self.refresh_docling_cache = config.refresh_docling_cache
    
    def _get_clean_filename(self, filename: str) -> str:
        return _clean_filename(filename)

    def extract_baseline_chunks(self, filename: str, baseline_description: str, enrich: bool = False) -> dict[str, List[Chunk]]:
        """Extract baseline chunks from the raw text files for a given document."""
        sample = os.path.join(self.pdf_path, filename)
        clean_filename = self._get_clean_filename(sample)
        baseline_extractor = HybridChunkerExtractor(
            source=sample,
            config=self.chunker_config,
            pipeline_options=self.pdfPipelineOptions,
            # the same cache the graph extractor writes, so a document is parsed
            # once no matter which representation is built first
            cache_dir=self.raw_text_save_dir if self.use_docling_cache else None,
            refresh=self.refresh_docling_cache,
        )
        # the same id the graph side derives, so chunks and nodes join on it
        document_id = document_id_for(filename)
        chunks = []
        enriched_chunks = []
        for i, chunk in enumerate(baseline_extractor.extract_and_chunk(save_dir=self.raw_text_save_dir, filename=clean_filename)):
            meta = chunk.meta.export_json_dict()
            provenance = provenance_from_chunk_meta(meta)

            chunks.append(Chunk(
                chunk_id=chunk_id_for(document_id, i),
                document_id=document_id,
                filename=filename,
                text=chunk.text,
                meta=meta,
                provenance=provenance,
                baseline_description=baseline_description
            ))
            if enrich:
                enriched_text = baseline_extractor.chunker.contextualize(chunk)
                enriched_chunks.append(Chunk(
                    chunk_id=chunk_id_for(document_id, i, variant="enriched"),
                    document_id=document_id,
                    filename=filename,
                    text=enriched_text,
                    meta=meta,
                    provenance=provenance,
                    baseline_description=f"Enriched; {baseline_description}"
                ))
        return {"baseline": chunks, "enriched": enriched_chunks}
    
    def generate_baseline_chunks(self, return_chunks: bool = False) -> dict[str, dict[str, List[Chunk]]] | None:
        """Generate baseline chunks for all documents in the pdf_path.

        Returns ``{filename: {"baseline": [...], "enriched": [...]}}`` when
        ``return_chunks`` is set, otherwise ``None``."""
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