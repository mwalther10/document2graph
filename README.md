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

Optional extras — none of them needed to build a graph, weight its edges structurally,
or run the rule-based need and supply measures:

| extra | for |
|---|---|
| `tokenizers` | counting a token budget with the retriever's own Hugging Face tokenizer |
| `surprisal` | the surprisal need estimator (one forward pass per edge) |
| `embeddings` | semantic similarity backends |
| `linguistic` | spaCy, an accuracy upgrade for the rule-based need estimators |
| `neo4j` | writing graphs to a Neo4j instance |
| `all` | everything above |

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

#### Edge weight metrics

Which metric produces the weight is `EdgeWeightConfig.metric`; how its result combines
with the structural weight above is `EdgeWeightConfig.combination` (`replace`, `mean` or
`multiply`).

| metric | weight of an edge |
| --- | --- |
| `structural` | the configured weight of the edge's relation (default; unchanged behaviour) |
| `uniform` | `1.0` everywhere — the baseline any weighting has to beat |
| `similarity` | parent/child content similarity in [0, 1] |
| `inverted_similarity` | `1 - similarity`; the parent is worth attaching when it does *not* repeat the child |
| `need_supply` | `need(child) * supply(child, parent)` |

`similarity` comes from `SimilarityConfig.backend`:

- `bm25` — lexical: Okapi BM25 over the document tree, clipped at zero and normalized by
  the largest score across the tree. `bm25_scoring` controls the query/document asymmetry:
  `child_query` (default), `symmetric_mean` or `symmetric_max`
- `embedding` — semantic: cosine similarity of sentence embeddings (requires the
  `embeddings` extra: `pip install "document2graph[embeddings]"`)
- `blend` — `alpha * embedding + (1 - alpha) * bm25`

Both polarities are available because both are defensible: a *similar* parent is
topically applicable context, while a *dissimilar* one supplies what the child does not
already carry.

```python
from document2graph.models import EdgeWeightConfig, SimilarityConfig

edge_weights = EdgeWeightConfig(
    metric="inverted_similarity",
    combination="mean",                      # combine with the structural weight
    similarity=SimilarityConfig(backend="blend", alpha=0.5),
)
```

#### Need and supply

`need_supply` splits the question in two. **Need** is a property of the child alone — can
this unit be read on its own? **Supply** is what distinguishes one candidate parent from
another — how much of the missing context does *this* parent actually deliver?

Three need estimators, selected by `NeedConfig.estimator`:

- `reference_density` — the share of the child's pointers it cannot resolve by itself.
  Pronouns with no antecedent inside the unit, demonstratives (`diese Dosis`), definite
  descriptions whose entity was never introduced (`die Therapie` against `eine Therapie`),
  and acronyms the unit never expands. Scored as a *ratio*, so a long unit does not look
  self-sufficient just by being long. Bullets score high without a rule that says they
  should. Watch for generic definites (`Die Niere filtert Blut` points nowhere but matches
  the rule) — tolerable noise that inflates prose relative to lists, so compare across
  unit types with care, or drop the class via `pointer_classes`.
- `syntactic` — grammatical evidence that the unit is not standalone: no finite verb, a
  fragment rather than a main clause, a leading discourse connective (`Jedoch`, `Zudem`),
  a lowercase start (unusually informative in German, where capitalization is otherwise
  rigid), and — a property of the *edge*, not the child — a parent that ends in a colon,
  which is close to decisive. This measures grammatical incompleteness, which is not the
  same as interpretive incompleteness: a well-formed sentence can still mean little
  without the heading naming which patients it applies to. A complement to
  `reference_density`, not a replacement.
- `surprisal` — `dNLL = NLL(child) - NLL(child | parent)` from a small causal LM, per
  token so length cancels. The most general of the three: bullet to paragraph, cell to
  column header, continuation to predecessor are all the same computation on one scale.
  Surprisal drops for two reasons that look identical in the number, though — the parent
  genuinely supplies context, or it merely repeats the child's vocabulary. The control is
  `counterfactual`: condition on a non-ancestor parent from the same document and
  subtract, which collapses a genuine dependency but leaves shared-domain vocabulary
  intact. That correction is on by default; `counterfactual="none"` reports the raw drop.

`estimator="blend"` takes a weighted mean of several. The rule-based estimators use regex
lexicons for German and English out of the box; installing the `linguistic` extra
(`pip install "document2graph[linguistic]"`, plus a spaCy model such as
`python -m spacy download de_core_news_sm`) swaps in POS tags and dependency parses, which
are markedly better at finite verbs and clause structure.

`SupplyConfig.metric` selects how supply is measured. The default is `composite`:

| supply metric | what it measures |
|---|---|
| `antecedent_coverage` | of the pointers the child leaves dangling, the share this parent answers |
| `complementarity` | IDF-weighted content the parent adds that the child does not already carry |
| `composite` | `max(coverage, complementarity)`, floored at the relation's structural prior |
| `uniform` | need alone |
| `structural`, `similarity`, `inverted_similarity`, `surprisal` | the earlier arms, kept for comparison |

The composite keeps everything absolutely scaled and in [0, 1], so `need x supply`
stays readable as the share of the child's context that is still missing after the
merge. The floor is what keeps a three-word bullet or a table cell -- where both
measured components return near zero by construction -- from reading as "this parent
supplies nothing".

The priors are configurable per relation and ship with values measured from 50
hand-labelled edges (`tests/.test-data/edge_labels.json`), not chosen by hand:

```python
from document2graph.models import StructuralPriorConfig, SupplyConfig

SupplyConfig(structural_prior=StructuralPriorConfig(
    priors={"heading_list": 0.9, "text_text": 0.3},   # replaces the table wholesale
    default=0.5,                                      # any relation absent from it
))
```

Supply relations are finer than the structural relations: `text` splits into
`heading_text` / `text_text` and `list_item` into `heading_list` / `text_list`,
because a heading and a paragraph do not deliver the same kind of context. The
split is derived from the nodes, so it also covers edges that never existed during
graph construction.

Complementarity should be given a corpus-wide IDF table, without which its scores
are not comparable across documents:

```python
from document2graph.edge_weights import MetricDeps, compute_edge_weights, idf_table

deps = MetricDeps(idf=idf_table(every_node_text_in_the_corpus))
weights = compute_edge_weights(ctx, config, deps)
```


```python
from document2graph.models import EdgeWeightConfig, NeedConfig, SupplyConfig, SimilarityConfig

edge_weights = EdgeWeightConfig(
    metric="need_supply",
    need=NeedConfig(
        estimator="blend",
        blend_weights={"reference_density": 0.5, "syntactic": 0.5},
    ),
    supply=SupplyConfig(
        metric="inverted_similarity",
        similarity=SimilarityConfig(backend="bm25"),
    ),
)
```

Registering another metric needs no change to the package:

```python
from document2graph.edge_weights import register_metric

@register_metric("depth_decay")
def depth_decay(ctx, config, deps):
    return {edge: 0.5 ** ctx.unit(edge[1]).level for edge in ctx.edges}
```

`RelevancyWeightConfig` was removed in 0.2.0. It is replaced by
`metric="inverted_similarity"` with the matching `SimilarityConfig` and `combination`.

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
        merge_line_fragments=True,       # rejoin the lines of a figure/sidebar/masthead block
        merge_table_continuations=True,  # rejoin a table continued on the next page
        assign_regions=True,             # body / front_matter / sidebar / figure
        recover_captions=True,           # find captions docling left unlinked to their figure
        filter_decorative_pictures=True, # drop logos and rules
        absorb_figure_text=True,         # fold a chart's labels into the picture node
        figure_label_max_tokens=8,       # above this a figure-region node is prose, and stays
        level_source="hybrid",           # "hybrid" | "typography" | "docling"
    ),
)
```

Two of the steps rejoin units the parser cut in half, and both were calibrated against
a hand-labelled review of 298 candidate splits across the test corpus:

- `merge_line_fragments` — docling clusters running text into paragraphs but reads a
  figure's body, a boxed sidebar and the masthead *line by line*, so one bullet of a
  flowchart becomes five snippets and five graph nodes. Lines that sit directly under
  one another in the same column, and read as one continuing text, are merged back into
  one snippet. Body text is deliberately excluded: there docling's clustering is right,
  and joining neighbouring bullets would destroy the list. Needs `assign_regions`.
- `absorb_figure_text` — a chart is read line by line, so its axis ticks and data points
  arrive as one node each: `"42"`, `"33"`, `"51"`. None of them is retrievable on its own
  and the picture that gives them meaning has no text at all, so the whole figure is
  invisible to a retriever. The labels become the picture's `figure_text`, in reading
  order, and stop being nodes. Across the twenty-document test corpus, 87% of
  figure-region nodes are five tokens or fewer — in a chart-heavy document that is 86–97%
  of *everything* — and folding them in takes the corpus from 12515 units to 8520, while
  132 pictures that had no text become retrievable. A figure-region node above
  `figure_label_max_tokens` is prose rather than a label — a flowchart step, a boxed
  note — and stays a node of its own, though it still appears in the digest so the figure
  reads whole. Captions are never absorbed, nor is a node that parents another. Needs
  `assign_regions`.
- `merge_table_continuations` — `docling_doc.tables` holds one item per layout cluster,
  so a table broken by a page break arrives as two, with half the rows each and the
  caption on the first only. The continuation's rows are appended to the head, whose
  `provenance` then reports both pages and whose `continuation_refs` names the item it
  absorbed. Only page breaks are merged: in a two-column guideline layout, two tables
  side by side are two tables, even when they repeat a header row.

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
graph.pages                # [(page_no, width, height), ...] — the box of every page
graph.page(3)              # the geometry of one page, or None if it was not recorded
```

Every stored `bbox` is a docling provenance box in PDF points, which says nothing about
where it sits on the page without the page it belongs to. `graph.pages` is what a
consumer normalizes against — scoring bbox overlap against layout gold, drawing nodes
over a page image — so a reloaded graph needs neither the PDF nor a second parse.

`DocumentGraph.from_snippet_graph(graph, document)` builds the same object from a
`DocumentGraphExtractor.run()` result in memory; `extractor.run(...)["document_graph"]`
returns it directly.

### Building a graph from a parsed document

`DocumentGraphExtractor` owns a corpus directory and writes intermediate files, which is
the wrong shape for a harness that has already parsed the document and wants the graph as
a value. `graph_from_docling` is the same construction without the filesystem, so an
ablation over `PipelineFlags` or `EdgeWeightConfig` reuses one parse:

```python
from docling_core.types.doc.document import DoclingDocument
from docling_parse.pdf_parser import DoclingPdfParser
from document2graph import PipelineFlags, graph_from_docling

docling_doc = DoclingDocument.load_from_json("data/raw_texts/my_document_docling_doc.json")
pdf_doc = DoclingPdfParser().load(path_or_stream="pdfs/my_document.pdf")

graph = graph_from_docling(
    docling_doc, pdf_doc, "my_document",
    flags=PipelineFlags(stitch_continuations=False),
)
```

Nothing is written unless `save_gexf_to` is passed. `build_snippet_graph` returns the
raw `(SnippetGraph, Document)` instead, for a caller that wants the node objects.

`pdf_doc` is still required, and is why a condition sweep needs the PDF next to the
cached parse: graph construction reads the line and character cells behind line heights
and font keys, and the drawn shapes behind page blocks and regions, none of which the
`DoclingDocument` carries.

### Merging graph nodes into retrieval units

A graph node on its own is often not readable: a bullet under a colon, a paragraph whose
subject is named only in its heading, a table cell. Merging ancestors into it fixes that
and costs tokens, and `MergePolicy` is how you ask *which* ancestors are worth the tokens.

```python
from document2graph import load_graph, merge_units
from document2graph.models import MergePolicy

graph = load_graph("data/output/graphs/my_document_graph.json")

bare       = merge_units(graph, MergePolicy(strategy="none"))
merged     = merge_units(graph, MergePolicy(strategy="weighted", min_weight=0.8))
everything = merge_units(graph, MergePolicy(strategy="all_ancestors"))
sections   = merge_units(graph, MergePolicy(strategy="subtree"))
```

| strategy | the unit is |
| --- | --- |
| `none` | the bare node |
| `all_ancestors` | the node plus every ancestor up to the root |
| `weighted` | the node plus ancestors while the edge weight clears `min_weight` (and the running product clears `min_cumulative`) |
| `subtree` | one unit per heading, holding its whole subtree |

`min_tokens` is a floor on the other end: a node whose own text falls below it is folded
into the nearest ancestor that *is* a unit, rather than standing as one. It is a floor and
not a filter — merging only ever walks upwards, so a node simply left out would take its
text out of the corpus with it — and a fragment with no eligible ancestor keeps its own
unit for the same reason. It is 0 by default, and it is the retrieval-time counterpart to
`PipelineFlags.absorb_figure_text`, which catches the same problem at its source where the
fragments sit inside a figure.

`all_ancestors` is the baseline the weighted policy has to beat, and it has to beat it **at
equal token budget rather than at equal recall**: withholding nothing can hardly lose
recall, so the claim can only be the same recall for fewer tokens. On the twenty-document
test corpus, merging everything costs **2.06×** the index tokens of merging nothing, and
the weighted dial spans 1.07× to 1.75× of it.

Each unit records what it was built from, so a condition can be audited rather than only
scored:

```python
unit.text            # ancestors first, the unit's own node last
unit.member_ids      # every contributing snippet, in that order
unit.merged_edges    # which edges were followed, with weight and running product
unit.breadcrumb      # heading ancestors' texts, outermost first — kept out of `text`
unit.provenance      # the locations of every member, so a merged unit still places
unit.token_count
```

The breadcrumb stays out of `text` on purpose: prepending the heading path is its own
retrieval condition, and one any segmentation can be given, so folding it in silently
would make "structure-aware" and "heading-prepending" impossible to tell apart.

#### Comparing two weight metrics fairly

`min_weight` is a threshold on whatever scale the configured metric produces, and those
scales do not agree — running two metrics at `0.5` compares how much each happens to merge
at `0.5`. What makes them comparable is fixing the *outcome*: calibrate each metric's
threshold to the same mean unit size, then compare retrieval at that matched budget.

```python
from document2graph import calibrate_min_weight, mean_unit_tokens

target = mean_unit_tokens(graphs, MergePolicy(strategy="all_ancestors"))
calibration = calibrate_min_weight(graphs, MergePolicy(strategy="weighted"), target)

calibration.policy                # ready to pass to merge_units
calibration.achieved_mean_tokens
calibration.relative_error
calibration.within(0.02)
```

The achieved mean comes back beside the policy because **the reachable sizes are a step
function** of the distinct edge weights, so a target between two steps is not reachable at
all. The shipped structural weights take five values, so a corpus merged under them can
only land on a handful of means. Check `relative_error` before reporting two conditions as
budget-matched.

Token counts default to a tokenizer-free estimate, which is fine for exploring but not for
a real budget comparison — that has to charge every condition with the same tokenizer the
retriever will embed with:

```python
from document2graph.retrieval import huggingface_token_counter

count = huggingface_token_counter("intfloat/multilingual-e5-large")
units = merge_units(graph, policy, count_tokens=count)
```

### Expanding a retrieved set

Merging happens at index time and decides what gets embedded. Expansion happens at query
time and decides what gets *returned* — a different trade, because index-time context is
paid for in every unit whether a query needed it or not, while expansion is paid for only
on the handful of units a query actually retrieved. The two compose: embed small and
precise, return complete.

`expand` is a pure graph operation over `(node_id, score)` pairs — no index, no embedding
model, no query — so a retrieval harness keeps its own ranking and asks only what the
graph can answer:

```python
from document2graph import expand, load_graph
from document2graph.models import ExpansionPolicy

graph = load_graph("data/output/graphs/my_document_graph.json")
hits = [(node_id, score), ...]              # whatever your retriever ranked

for result in expand(graph, hits, ExpansionPolicy(mode="auto_merge")):
    result.unit.text        # what to put in the context (a MergedUnit, as merge_units returns)
    result.score
    result.reason           # "hit", or the mode that grew it
    result.added_ids        # nodes pulled in that were not themselves retrieved
    result.absorbed_ids     # other hits folded into this unit, not returned separately
    result.added_tokens     # what this unit's growth charged to the budget
```

| mode | adds |
| --- | --- |
| `none` | nothing |
| `parent` | each hit's parent |
| `ancestors` | the whole chain above each hit |
| `siblings` | the hit's neighbours under the same parent |
| `subtree` | what hangs below the hit, for when a heading is retrieved |
| `auto_merge` | where `min_siblings` retrieved nodes share a parent, the parent replaces them |
| `reference_edges` | the figure or table a retrieved mention points at, and vice versa |

`auto_merge` is the one that needs the weights rather than just the shape: a group of
siblings is only worth collapsing if the parent actually holds them together, which is
what `min_edge_weight` asks. It collapses one level only — walking on up would carry a
section to the document root as soon as two of its paragraphs matched, which says
something about the tree rather than about the query.

Two ceilings, and they answer different questions: `max_tokens_per_unit` is how large one
returned unit may be, `max_added_tokens` is the whole result's expansion allowance. Hits
grow in rank order, so a tight allowance grows the top of the result and leaves the tail
bare rather than spreading thinly.

`dedupe` (on by default) drops a hit that has been absorbed into another returned unit.
Without it a parent and its children both come back and the shared text is charged twice,
which silently understates what expansion costs.

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
| `PageGeometry` | `page_no`, `width`, `height` (the page box every stored `bbox` is expressed against) |
| `MergePolicy` | `strategy` (none/all_ancestors/subtree/weighted), `min_weight`, `min_cumulative`, `min_tokens`, `max_tokens`, `separator`, `breadcrumb_separator` |
| `MergedUnit` | `unit_id`, `document_id`, `text`, `member_ids`, `breadcrumb`, `token_count`, `provenance`, `page_no`, `node_type`, `merged_edges` |
| `MergedEdge` | `parent_id`, `child_id`, `weight`, `cumulative` |
| `ExpansionPolicy` | `mode` (none/parent/ancestors/siblings/subtree/auto_merge/reference_edges), `min_siblings`, `min_edge_weight`, `max_added_tokens`, `max_tokens_per_unit`, `dedupe`, `separator` |
| `ExpandedHit` | `unit: MergedUnit`, `score`, `hit_id`, `reason`, `added_ids`, `absorbed_ids`, `added_tokens` |
| `Calibration` | `policy`, `target_mean_tokens`, `achieved_mean_tokens`, `relative_error`, `within(tolerance)` |
| `DocumentGraph` | `document_id`, `filename`, `title`, `root_id`, `snippets: list[GraphSnippet]`, `edges`, `reference_edges`, `pages: list[PageGeometry]`, `edge_relations` |
| `GraphSnippet` | everything on `Snippet` plus `local_ref`, `parent_local_ref`, `snippet_type`, `line_heights`, `font_key`, `level_height`, `extracted_at`. A media snippet's `text` is what the node is made of: a table is its caption above its markdown, a picture its caption above whatever is printed inside the figure |
| `Document` | `document_id`, `document_type`, `filename`, `title`, `metadata: DocumentMetadata` |
| `DocumentMetadata` | `version`, `authors`, `institutions`, `bibliography`, `correspondence` |
| `MetadataExtractionConfig` | `title_page`, `version`, `authors`, `institutions`, `bibliography`, `correspondence` (each a `MetadataFieldConfig`) |
| `MetadataFieldConfig` | `label` (search string), `pages` (inclusive 1-based page range, e.g. `(1, 3)`) |
| `ChunkerConfig` | `tokenizer` (HF tokenizer name), `max_tokens`, `merge_peers` |
| `PipelineFlags` | `repair_reading_order`, `stitch_continuations`, `merge_line_fragments`, `merge_table_continuations`, `assign_regions`, `recover_captions`, `filter_decorative_pictures`, `absorb_figure_text`, `figure_label_max_tokens`, `level_source` (hybrid/typography/docling) |
| `EdgeWeightConfig` | `section`, `text`, `list_item`, `media`, `unreferenced_media`, `reference`, `root` (structural weights by relation), `metric` (structural/uniform/similarity/inverted_similarity/need_supply), `combination` (replace/mean/multiply), `similarity: SimilarityConfig`, `need: NeedConfig`, `supply: SupplyConfig` |
| `SimilarityConfig` | `backend` (bm25/embedding/blend), `embedding_model`, `alpha`, `bm25_k1`, `bm25_b`, `bm25_scoring` (child_query/symmetric_mean/symmetric_max) |
| `NeedConfig` | `estimator` (uniform/reference_density/syntactic/surprisal/blend), `blend_weights`, `reference_density: ReferenceDensityConfig`, `syntactic: SyntacticConfig`, `surprisal: SurprisalConfig` |
| `SupplyConfig` | `metric` (uniform/structural/similarity/inverted_similarity/surprisal/antecedent_coverage/complementarity/**composite**), `combine` (max/mean), `floor` (prior/none), `antecedent: AntecedentCoverageConfig`, `complementarity: ComplementarityConfig`, `structural_prior: StructuralPriorConfig`, `similarity: SimilarityConfig`, `surprisal: SurprisalConfig` |
| `AntecedentCoverageConfig` | `pointers: ReferenceDensityConfig` (acronyms excluded by default), `idf_weighted`, `default_when_no_pointers` |
| `ComplementarityConfig` | `idf_source` (document/corpus), `top_k_novel_terms`, `stem_terms`, `min_idf` |
| `StructuralPriorConfig` | `priors` (per supply relation; defaults measured), `default`, `enabled` |
| `ReferenceDensityConfig` | `language` (de/en/auto), `pointer_classes` (anaphor/demonstrative/definite/acronym), `default_when_no_pointers`, `use_spacy`, `spacy_model_de`, `spacy_model_en` |
| `SyntacticConfig` | `feature_weights` (no_finite_verb/fragment/leading_connective/lowercase_start/parent_ends_in_colon), `language`, `use_spacy`, `spacy_model_de`, `spacy_model_en` |
| `SurprisalConfig` | `model`, `counterfactual` (none/sibling_parent), `max_context_tokens`, `max_child_tokens`, `normalize` (max/logistic), `logistic_scale`, `device` |

## Running tests

```bash
pytest
```

Skip slow tests that download models:

```bash
pytest -m "not slow"
```
