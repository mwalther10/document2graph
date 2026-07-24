"""Local debugging harness for SnippetGraphConstructor.

Reuses the docling JSON cached under tests/.test-data/raw_texts/ (written by the
extractor / test runs), so repeated runs skip the slow docling pipeline and go
straight to graph construction. Only the first run of a new PDF converts it.

Usage:
    uv run python scripts/debug_snippet_graph.py
    uv run python scripts/debug_snippet_graph.py --pdf tests/.test-pdf/<file>.pdf
    uv run python scripts/debug_snippet_graph.py --refresh          # force re-conversion
    uv run python scripts/debug_snippet_graph.py --log-level INFO   # less noise
"""

import argparse
import glob
import logging
import os
import sys

import coloredlogs
import networkx as nx
from docling_core.types.doc.document import DoclingDocument
from docling_parse.pdf_parser import DoclingPdfParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document2graph.document2graph_extractor.snippet_graph_constructor import (
    ROOT_NODE_ID,
    SnippetGraphConstructor,
)
from document2graph.models import MetadataExtractionConfig, MetadataFieldConfig

PDF_DIR = "tests/.test-pdf/"
CACHE_DIR = "tests/.test-data/raw_texts/"


def load_docling_doc(pdf_path: str, refresh: bool) -> DoclingDocument:
    """Load the DoclingDocument from the json cache, converting the PDF only on a cache miss."""
    clean_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    cache_file = os.path.join(CACHE_DIR, f"{clean_filename}_docling_doc.json")
    if os.path.isfile(cache_file) and not refresh:
        logging.getLogger("debug").info(f"Loading cached docling doc from {cache_file}")
        return DoclingDocument.load_from_json(cache_file)

    logging.getLogger("debug").info(f"No cache for {clean_filename}, running docling conversion (slow)...")
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat

    # force CPU: docling's layout model needs float64, which MPS does not support
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU)
    )
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    })
    doc = converter.convert(pdf_path).document
    os.makedirs(CACHE_DIR, exist_ok=True)
    doc.save_as_json(cache_file)
    return doc


def print_summary(constructor: SnippetGraphConstructor, graph) -> None:
    meta = constructor.document_metadata
    node_by_id = {n.snippet_id: n for n in graph.text_nodes + graph.image_nodes + graph.table_nodes}

    print("\n=== Document metadata ===")
    print(f"title:    {meta.title!r}")
    print(f"metadata: {meta.metadata.model_dump()}")

    print("\n=== Page regions ===")
    region_counts: dict[str, int] = {}
    for node in graph.text_nodes:
        region_counts[node.region] = region_counts.get(node.region, 0) + 1
    for region, count in sorted(region_counts.items(), key=lambda kv: -kv[1]):
        print(f"{count:5d}  {region}")

    print("\n=== Heading styles (font, height) -> level ===")
    levels = constructor.levels
    headings = [n for n in graph.text_nodes if n.level in levels.header_levels()]
    by_style: dict[tuple[str, int, str], list[str]] = {}
    for node in headings:
        key = (node.font_key or "-", node.level, node.region)
        by_style.setdefault(key, []).append(node.text)
    for (font, level, region), texts in sorted(by_style.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  L{level} {font:6} {region:12} n={len(texts):3}  {[t[:34] for t in texts[:3]]}")
    print(f"\ntext body levels: {dict(levels.text_body_levels)}  labels: {dict(levels.text_level_to_label)}")

    print("\n=== Heading tree ===")
    for node in sorted(headings, key=lambda n: n.sequence_no):
        marker = "" if node.region == "body" else f" [{node.region}]"
        print(f"  {'  ' * node.level}L{node.level}{marker} {node.text[:78]}")

    print("\n=== Root ===")
    if graph.root_id == ROOT_NODE_ID:
        print(f"synthetic fallback root: {graph.root_id}")
    else:
        root = node_by_id[graph.root_id]
        print(f"title node root: {graph.root_id} (level={root.level}, label={root.level_label!r}) -> {root.text[:100]!r}")

    print("\n=== Nodes per level_label ===")
    label_counts: dict[str, int] = {}
    for node in node_by_id.values():
        label_counts[node.level_label] = label_counts.get(node.level_label, 0) + 1
    for label, count in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        print(f"{count:5d}  {label!r}")

    print("\n=== Nodes attached directly to the root ===")
    for parent_id, child_id, weight in graph.edges:
        if parent_id == graph.root_id:
            child = node_by_id[child_id]
            print(f"  {child_id} (w={weight}, level={child.level}, {child.level_label!r}) {getattr(child, 'text', getattr(child, 'caption_text', ''))[:80]!r}")

    print("\n=== Reference edges (mentioning text -> media) ===")
    for source_id, target_id, weight in graph.reference_edges:
        source = node_by_id[source_id]
        print(f"  {source_id} -> {target_id} (w={weight}) {source.text[:70]!r}")
    if not graph.reference_edges:
        print("  (none)")

    nx_graph = nx.DiGraph()
    nx_graph.add_node(graph.root_id)
    nx_graph.add_nodes_from(node_by_id)
    nx_graph.add_edges_from((p, c) for p, c, _ in graph.edges)
    print("\n=== Graph shape ===")
    print(f"nodes: {nx_graph.number_of_nodes()}, tree edges: {nx_graph.number_of_edges()}, "
          f"reference edges: {len(graph.reference_edges)}, "
          f"weakly connected (tree only): {nx.is_weakly_connected(nx_graph)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_pdf = next(iter(sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))), None)
    parser.add_argument("--pdf", default=default_pdf, help=f"PDF to process (default: first PDF in {PDF_DIR})")
    parser.add_argument("--refresh", action="store_true", help="ignore the docling json cache and re-convert")
    parser.add_argument("--log-level", default="DEBUG", help="log level for all loggers (default: DEBUG)")
    parser.add_argument("--save-gexf", default="", help="optionally write the graph to this .gexf path")
    args = parser.parse_args()
    if not args.pdf:
        parser.error(f"no PDF found in {PDF_DIR}, pass one with --pdf")

    # re-install over the INFO handler set up at import time in utils/log.py
    coloredlogs.install(level=args.log_level, isatty=True)

    docling_doc = load_docling_doc(args.pdf, refresh=args.refresh)
    pdf_doc = DoclingPdfParser().load(path_or_stream=args.pdf)

    # mirrors the metadata config used in the tests for the German test PDFs
    metadata_config = MetadataExtractionConfig(
        title_page=1,
        version=None,
        authors=MetadataFieldConfig(label="autoren", pages=(1, 1)),
        institutions=MetadataFieldConfig(label="institute", pages=(1, 1)),
        bibliography=MetadataFieldConfig(label="bibliografie", pages=(1, 1)),
        correspondence=MetadataFieldConfig(label="korrespondenzadresse", pages=(1, 1)),
    )

    constructor = SnippetGraphConstructor(
        pdf_doc,
        docling_doc,
        filename=os.path.splitext(os.path.basename(args.pdf))[0],
        document_type="debug",
        metadata_config=metadata_config,
    )
    graph = constructor.get_graph(save_to=args.save_gexf)
    print_summary(constructor, graph)


if __name__ == "__main__":
    main()
