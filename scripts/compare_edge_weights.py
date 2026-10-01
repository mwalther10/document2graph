"""Compare edge weight metrics on the same document.

Builds the graph once and then evaluates every metric against the same
``EdgeContext``, so a sweep costs one parse plus one cheap pass per metric
rather than a full pipeline run each. Reuses the docling JSON cached under
tests/.test-data/raw_texts/, so repeated runs skip docling entirely.

Usage:
    uv run python scripts/compare_edge_weights.py
    uv run python scripts/compare_edge_weights.py --pdf tests/.test-pdf/<file>.pdf
    uv run python scripts/compare_edge_weights.py --metrics uniform structural need_supply
    uv run python scripts/compare_edge_weights.py --top 10       # show the top edges per metric
    uv run python scripts/compare_edge_weights.py --with-surprisal   # loads a causal LM
"""

import argparse
import glob
import os
import statistics as st
import sys

import coloredlogs
from docling_core.types.doc.document import DoclingDocument
from docling_parse.pdf_parser import DoclingPdfParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document2graph.document2graph_extractor.snippet_graph_constructor import SnippetGraphConstructor
from document2graph.edge_weights import EdgeContext, compute_edge_weights
from document2graph.models import (
    EdgeWeightConfig,
    MetadataExtractionConfig,
    NeedConfig,
    SimilarityConfig,
    SupplyConfig,
    SupplyMetricName,
)
from document2graph.utils.base_extractor import Extractor

PDF_DIR = "tests/.test-pdf/"
CACHE_DIR = "tests/.test-data/raw_texts/"

# one representative configuration per metric; --backend and --supply override
# the content settings, so the sweep stays a one-variable comparison
def build_configs(backend: str, supply: SupplyMetricName) -> dict[str, EdgeWeightConfig]:
    similarity = SimilarityConfig(backend=backend)
    supply_config = SupplyConfig(metric=supply, similarity=similarity)
    return {
        "structural": EdgeWeightConfig(metric="structural"),
        "uniform": EdgeWeightConfig(metric="uniform"),
        "similarity": EdgeWeightConfig(metric="similarity", similarity=similarity),
        "inverted_similarity": EdgeWeightConfig(metric="inverted_similarity", similarity=similarity),
        "need_supply/reference_density": EdgeWeightConfig(
            metric="need_supply", supply=supply_config,
            need=NeedConfig(estimator="reference_density")),
        "need_supply/syntactic": EdgeWeightConfig(
            metric="need_supply", supply=supply_config,
            need=NeedConfig(estimator="syntactic")),
        "need_supply/blend": EdgeWeightConfig(
            metric="need_supply", supply=supply_config,
            need=NeedConfig(estimator="blend",
                            blend_weights={"reference_density": 0.5, "syntactic": 0.5})),
        "need_supply/surprisal": EdgeWeightConfig(
            metric="need_supply", supply=supply_config,
            need=NeedConfig(estimator="surprisal")),
    }


def load_docling_doc(pdf_path: str, refresh: bool) -> DoclingDocument:
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    # force CPU: docling's layout model needs float64, which MPS does not support
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU)
    )
    return Extractor(pdf_path, pipeline_options, cache_dir=CACHE_DIR, refresh=refresh).doc


def describe(name: str, weights: dict, ctx: EdgeContext, top: int) -> None:
    values = list(weights.values())
    spread = st.pstdev(values) if len(values) > 1 else 0.0
    zeros = sum(1 for value in values if value == 0.0)
    print(f"{name:<32} mean={st.mean(values):.3f} sd={spread:.3f} "
          f"min={min(values):.3f} max={max(values):.3f} zero={zeros:<4} "
          f"distinct={len(set(round(v, 4) for v in values))}")
    for (parent_id, child_id), weight in sorted(weights.items(), key=lambda kv: -kv[1])[:top]:
        print(f"      {weight:.3f} [{ctx.relation((parent_id, child_id)):<18}] "
              f"{ctx.text(parent_id)[:30]!r:<34} -> {ctx.text(child_id)[:46]!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_pdf = next(iter(sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))), None)
    parser.add_argument("--pdf", default=default_pdf, help=f"PDF to process (default: first in {PDF_DIR})")
    parser.add_argument("--refresh", action="store_true", help="ignore the docling json cache")
    parser.add_argument("--metrics", nargs="*", default=None, help="subset of configurations to run")
    parser.add_argument("--backend", default="bm25", choices=["bm25", "embedding", "blend"],
                        help="similarity backend (default: bm25, needs no model download)")
    parser.add_argument("--supply", default="composite",
                        choices=["uniform", "structural", "similarity", "inverted_similarity",
                                 "surprisal", "antecedent_coverage", "complementarity", "composite"],
                        help="supply metric for the need_supply configurations (default: composite)")
    parser.add_argument("--with-surprisal", action="store_true",
                        help="include the surprisal configuration (downloads and runs a causal LM)")
    parser.add_argument("--top", type=int, default=0, help="show this many highest-weighted edges per metric")
    parser.add_argument("--log-level", default="WARNING", help="log level (default: WARNING)")
    args = parser.parse_args()
    if not args.pdf:
        parser.error(f"no PDF found in {PDF_DIR}, pass one with --pdf")
    coloredlogs.install(level=args.log_level, isatty=True)

    docling_doc = load_docling_doc(args.pdf, refresh=args.refresh)
    pdf_doc = DoclingPdfParser().load(path_or_stream=args.pdf)
    constructor = SnippetGraphConstructor(
        pdf_doc, docling_doc, os.path.splitext(os.path.basename(args.pdf))[0],
        document_type="", metadata_config=MetadataExtractionConfig(title_page=1),
    )
    graph = constructor.get_graph()

    nodes = graph.text_nodes + graph.image_nodes + graph.table_nodes
    all_edges = graph.edges + graph.reference_edges
    ctx = constructor.build_edge_context(nodes, graph.root_id, [(p, c) for p, c, _ in all_edges])
    print(f"{os.path.basename(args.pdf)}: {len(nodes)} nodes, {len(ctx.edges)} edges "
          f"(similarity backend: {args.backend}, supply: {args.supply})\n")

    configs = build_configs(args.backend, args.supply)
    if not args.with_surprisal:
        configs.pop("need_supply/surprisal")
    if args.metrics:
        missing = set(args.metrics) - set(configs)
        if missing:
            parser.error(f"unknown metrics {sorted(missing)}; available: {sorted(configs)}")
        configs = {name: configs[name] for name in args.metrics}

    for name, config in configs.items():
        describe(name, compute_edge_weights(ctx, config), ctx, args.top)


if __name__ == "__main__":
    main()
