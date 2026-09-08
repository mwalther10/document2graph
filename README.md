# document2graph

Extract hierarchical document graphs and baseline chunks from PDF files. The package parses PDFs using [Docling](https://github.com/DS4SD/docling) and produces two complementary representations:

- **Document graph** — text, image, and table nodes connected by weighted structural edges, capturing the layout hierarchy of the document. Every graph is guaranteed to be connected: nodes without a resolvable parent are attached to a synthetic document root node.
- **Baseline chunks** — flat, semantically merged chunks suitable for retrieval pipelines, with optional context enrichment.

## Installation

```bash
pip install document2graph
```

Or from source:

```bash
pip install -e .
```

For development dependencies (pytest):

```bash
pip install -e ".[dev]"
```

Optional extras: `embeddings` (semantic edge weights), `neo4j` (writing graphs to a
Neo4j instance).

Requires Python 3.11+.

## Quick start

### Configuration

Both extractors share the same `ExtractorConfig`:

```python
from document2graph.models import ExtractorConfig
from docling.datamodel.pipeline_options import PdfPipelineOptions

config = ExtractorConfig(
    pdf_path="data/pdfs/",        # directory containing input PDFs
    data_path="data/output/",     # root directory for all output files
    save_json=True,               # write results to disk as JSON (default: True)
    document_type="report",       # arbitrary label stored in document metadata
    pdfPipelineOptions=PdfPipelineOptions(),  # optional Docling pipeline config
)
```

#### Metadata extraction

The `metadata_config` field controls how document metadata (title, authors, version, etc.) is
located inside each PDF. Each field is described by a search string and an inclusive page range
to search on. Fields set to `None` are skipped.

```python
from document2graph.models import ExtractorConfig, MetadataExtractionConfig, MetadataFieldConfig

config = ExtractorConfig(
    pdf_path="data/pdfs/",
    data_path="data/output/",
    metadata_config=MetadataExtractionConfig(
        title_page=1,
        version=MetadataFieldConfig(label="edition", pages=(1, 1)),
        authors=MetadataFieldConfig(label="authors", pages=(1, 2)),
        institutions=MetadataFieldConfig(label="affiliations", pages=(1, 2)),
        bibliography=None,        # not present in this document type
        correspondence=MetadataFieldConfig(label="contact", pages=(2, 3)),
    ),
)
```

The default `MetadataExtractionConfig()` targets German-language Praxisempfehlungen PDFs
(version on page 1, authors/institutions/bibliography/correspondence on page 2).

#### Edge weights

Every edge in the document graph carries a `weight` attribute reflecting the strength of the
structural connection. The defaults can be overridden per category via `edge_weights`:

```python
from document2graph.models import ExtractorConfig, EdgeWeightConfig

config = ExtractorConfig(
    pdf_path="data/pdfs/",
    data_path="data/output/",
    edge_weights=EdgeWeightConfig(
        section=1.0,             # heading -> subheading
        text=0.8,                # heading/body -> body text, subtext, footnotes
        list_item=0.6,           # anchor text -> grouped list item (bullets)
        media=0.9,               # referencing text -> image/table (matched via caption)
        unreferenced_media=0.3,  # fallback heading -> image/table without a caption match
        root=0.1,                # document root -> otherwise disconnected node
    ),
)
```

#### Relevancy weights

Optionally, a content-based relevancy weight can be combined with the structural weight
of each edge. Relevancy weights are computed from the parent and child snippet texts and
normalized to [0, 1]: 0 means parent and child are highly similar (the parent adds little
new context for the child), 1 means the parent is very relevant to consider when looking
at the child.

Three metrics are available:

- `bm25` — lexical relevancy: Okapi BM25, clipped at zero and normalized by the maximum
  BM25 score across the document tree. `bm25_scoring` controls the query/document
  asymmetry: `child_query` (default, child text as query against the parent text),
  `symmetric_mean`, or `symmetric_max` (mean/max of both directions)
- `embedding` — semantic relevancy: cosine similarity of sentence embeddings from a
  configurable model (requires the `embeddings` extra: `pip install document2graph[embeddings]`)
- `blend` — linear blend `alpha * embedding + (1 - alpha) * bm25`

```python
from document2graph.models import EdgeWeightConfig, RelevancyWeightConfig

edge_weights = EdgeWeightConfig(
    relevancy=RelevancyWeightConfig(
        enabled=True,
        metric="blend",           # "bm25", "embedding", or "blend"
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        alpha=0.5,                # semantic share: 1.0 = fully semantic, 0.0 = fully lexical
        combination="mean",       # combine with the structural weight: "mean" or "multiply"
    ),
)
```

#### Connectivity guarantee

Each graph contains a synthetic root node (`#/document-root`, labelled with the document
title). Any node whose parent cannot be resolved — as well as any component that would
otherwise be disconnected — is attached to this root with the `root` edge weight, so every
document graph is a single (weakly) connected component.

Extracted metadata is available on the `Document` object under the `metadata` sub-object:

```python
result = extractor.run("my_document.pdf")
doc = result["document_metadata"]

print(doc.title)
print(doc.metadata.authors)
print(doc.metadata.version)
```

#### Pipeline flags

Graph construction repairs what the PDF parser leaves ambiguous: reading order,
paragraphs split across columns, page regions, unlinked captions. Each step can be
switched off individually to measure what it contributes. All of them are applied
*after* parsing, so a sweep over several settings reuses one parse of the corpus.

```python
from document2graph.models import ExtractorConfig, PipelineFlags

config = ExtractorConfig(
    pdf_path="./pdfs",
    data_path="./data",
    flags=PipelineFlags(
        repair_reading_order=True,       # sort snippets by page block instead of docling's order
        stitch_continuations=True,       # rejoin a paragraph split across a column/page break
        assign_regions=True,             # body / front_matter / sidebar / figure
        recover_captions=True,           # find captions docling left unlinked to their figure
        filter_decorative_pictures=True, # drop logos and rules
        level_source="hybrid",           # "hybrid" | "typography" | "docling"
    ),
)
```

`level_source` selects how heading levels are derived:

- `hybrid` (default) — headings are grouped into styles by (embedded font, height
  cluster) and ranked by how they nest in reading order
- `typography` — font height alone; headings of the same apparent size stay siblings
- `docling` — docling's own `SectionHeaderItem.level`

#### Parsing once

Parsing a PDF is by far the most expensive step, so the parsed document is cached as
JSON under `<data_path>/raw_texts/` and reused on the next run — by both extractors,
so a document is parsed once no matter which representation is built first.

```python
config = ExtractorConfig(
    pdf_path="./pdfs",
    data_path="./data",
    use_docling_cache=True,      # default; set False to always re-parse
    refresh_docling_cache=False, # True re-parses and overwrites the cache
)
```

### Document graph extraction

Processes every PDF in `pdf_path` and builds a NetworkX graph per document.

```python
from document2graph.document2graph_extractor import DocumentGraphExtractor

extractor = DocumentGraphExtractor(config)

# Process all PDFs and save graphs + snippets to disk
extractor.generate_snippets()

# Or collect Snippet objects in memory as well
snippets = extractor.generate_snippets(return_snippets=True)
```

To process a single file and get the raw graph components:

```python
result = extractor.run("my_document.pdf")

text_nodes   = result["text_nodes"]
image_nodes  = result["image_nodes"]
table_nodes  = result["table_nodes"]
edges        = result["edges"]        # list of (parent_id, child_id, weight) tuples
metadata     = result["document_metadata"]
```

Graphs are saved as `.gexf` files under `<data_path>/nx_graphs/` and can be loaded with NetworkX:

```python
import networkx as nx
G = nx.read_gexf("data/output/nx_graphs/my_document.gexf")
```

### Loading a graph back

GEXF is a view for graph tools: it drops bounding boxes, page numbers, char spans and
the table serializations. The full graph is written next to it as JSON under
`<data_path>/graphs/` and can be read back without re-parsing the PDF — which is what
makes it practical to run many retrieval conditions over the same corpus.

```python
from document2graph import load_graph

graph = load_graph("data/output/graphs/my_document_graph.json")

graph.document_id          # stable id, shared with the document's baseline chunks
graph.root_id              # snippet_id of the document root
for snippet in graph.snippets:
    snippet.snippet_id     # globally unique across the corpus
    snippet.local_ref      # the document-local docling ref, e.g. "#/texts/12"
    snippet.parent_id      # snippet_id of the parent, or None
    snippet.provenance     # [(page_no, bbox, charspan), ...]
graph.edges                # [(parent snippet_id, child snippet_id, weight), ...]
graph.reference_edges      # [(mentioning text, media item, weight), ...]
```

`DocumentGraph.from_snippet_graph(graph, document)` builds the same object from a
`DocumentGraphExtractor.run()` result in memory; `extractor.run(...)["document_graph"]`
returns it directly.

### Writing to Neo4j

Requires the `neo4j` extra. The driver and its credentials belong to the caller.

```python
from neo4j import GraphDatabase
from document2graph import ensure_neo4j_constraints, write_graph_to_neo4j

with GraphDatabase.driver(uri, auth=(user, password)) as driver:
    ensure_neo4j_constraints(driver)   # once per database
    write_graph_to_neo4j(driver, graph)
```

Documents become `(:Document)-[:HAS_ROOT]->(:Snippet)`, the hierarchy becomes
`(:Snippet)-[:HAS_CHILD {weight}]->(:Snippet)` and mentions become
`[:REFERENCES {weight}]`. Writing the same document again updates it in place: every id
is derived deterministically from the document's name.

### Identifiers and provenance

Both extractors derive the same `document_id` for a given PDF, and derive it the same
way on every run, so graph nodes and baseline chunks can be joined and a re-run updates
a corpus instead of duplicating it.

```python
from document2graph import document_id_for
document_id_for("my_document.pdf")  # stable uuid5, independent of directory and extension
```

Every unit — a graph node, a `Snippet`, a `GraphSnippet` or a baseline `Chunk` — exposes
its location in the PDF the same way, as a list of `Provenance(page_no, bbox, charspan)`.
A unit has more than one entry when it spans more than one region: a paragraph stitched
back together across a column break, a table continued on the next page, or a chunk
merged from several document items.

### Baseline chunk extraction

Produces flat, retrieval-ready chunks using Docling's hybrid chunker.

```python
from document2graph.baseline_extractor import BaselineExtractor

extractor = BaselineExtractor(config)

# Process all PDFs; saves JSON to disk
extractor.generate_baseline_chunks()

# Or return chunks in memory
all_chunks = extractor.generate_baseline_chunks(return_chunks=True)
# all_chunks["my_document.pdf"]["baseline"]  -> List[Chunk]
# all_chunks["my_document.pdf"]["enriched"]  -> List[Chunk] (context-enriched)
```

To process a single file:

```python
chunks = extractor.extract_baseline_chunks(
    filename="my_document.pdf",
    baseline_description="Q1 report chunks",
    enrich=True,   # also produce context-enriched variants
)
```

#### Chunker settings

`max_tokens` bounds the size of a baseline chunk and should match the context
window of the model that later embeds it — set it too high and the chunker
emits chunks the retriever silently truncates. `tokenizer` only determines how
chunk length is measured, so it should be the tokenizer of that same model.

```python
from document2graph.models import ExtractorConfig, ChunkerConfig

config = ExtractorConfig(
    pdf_path="./pdfs",
    data_path="./data",
    chunker=ChunkerConfig(
        tokenizer="intfloat/multilingual-e5-large",
        max_tokens=512,
        merge_peers=True,   # merge adjacent chunks under one heading while they fit
    ),
)
```

## Output structure

```
<data_path>/
├── raw_texts/          # parsed Docling documents (also the parse cache)
├── nx_graphs/          # .gexf graph files (one per document)
├── graphs/             # <filename>_graph.json (full graph, reloadable with load_graph)
├── snippets/           # <filename>_snippets.json (document graph output)
└── baseline_chunks/    # <filename>_baseline_chunks.json
```

## Data models

| Model | Key fields |
|---|---|
| `Snippet` | `snippet_id`, `type` (text/image/table), `document_id`, `sequence_no`, `label`, `level`, `region` (body/front_matter/sidebar/figure), `page_no`, `bbox`, `charspan`, `provenance`, `text` |
| `Chunk` | `chunk_id`, `document_id`, `filename`, `text`, `meta`, `provenance`, `baseline_description`, `embedding` |
| `Provenance` | `page_no`, `bbox`, `charspan` |
| `DocumentGraph` | `document_id`, `filename`, `title`, `root_id`, `snippets: list[GraphSnippet]`, `edges`, `reference_edges` |
| `GraphSnippet` | everything on `Snippet` plus `local_ref`, `parent_local_ref`, `snippet_type`, `line_heights`, `font_key`, `level_height`, `extracted_at` |
| `Document` | `document_id`, `document_type`, `filename`, `title`, `metadata: DocumentMetadata` |
| `DocumentMetadata` | `version`, `authors`, `institutions`, `bibliography`, `correspondence` |
| `MetadataExtractionConfig` | `title_page`, `version`, `authors`, `institutions`, `bibliography`, `correspondence` (each a `MetadataFieldConfig`) |
| `MetadataFieldConfig` | `label` (search string), `pages` (inclusive 1-based page range, e.g. `(1, 3)`) |
| `ChunkerConfig` | `tokenizer` (HF tokenizer name), `max_tokens`, `merge_peers` |
| `PipelineFlags` | `repair_reading_order`, `stitch_continuations`, `assign_regions`, `recover_captions`, `filter_decorative_pictures`, `level_source` (hybrid/typography/docling) |
| `EdgeWeightConfig` | `section`, `text`, `list_item`, `media`, `unreferenced_media`, `root` (edge weights by category), `relevancy: RelevancyWeightConfig` |
| `RelevancyWeightConfig` | `enabled`, `metric` (bm25/embedding/blend), `embedding_model`, `alpha`, `bm25_k1`, `bm25_b`, `bm25_scoring` (child_query/symmetric_mean/symmetric_max), `combination` (mean/multiply) |

## Running tests

```bash
pytest
```

Skip slow tests that download models:

```bash
pytest -m "not slow"
```
