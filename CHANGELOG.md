# Changelog

## 0.2.0

The release the retrieval harness needs: a graph that can be read back and worked on
without the PDF, edge weights as a package rather than a module, and policies for
turning a graph into retrievable units.

### Breaking

- **`document2graph.utils.edge_weight_lib` is gone.** Its contents became the
  `document2graph.edge_weights` package, which adds the need/supply decomposition and
  its estimators. Import from `document2graph.edge_weights`.
- **`RelevancyWeightConfig` is removed**, along with the `EdgeWeightConfig.relevancy`
  field that translated it. It was a shim for pre-0.1 configs. Replace
  `relevancy=RelevancyWeightConfig(enabled=True, metric="bm25", combination="mean")`
  with `metric="inverted_similarity", combination="mean",
  similarity=SimilarityConfig(backend="bm25")`.
- **`SCHEMA_VERSION` 1 → 4.** A graph written by 0.1.x still loads, but `pages` and
  `edge_relations` come back empty, `DocumentGraph.page()` returns `None`, and
  `recompute_weights()` refuses it rather than scoring on guessed relations.
  **Re-extract a 0.1.x corpus** before reweighting or scoring it.
- **A media snippet's `text` now composes every part of the node.** A table carries its
  caption above its markdown and a picture carries the text printed inside it, where
  before a table's text was the markdown alone and a picture's was its caption alone.
  Unit counts, mean unit size and anything calibrated against them shift; re-extract
  before comparing a corpus built under 3 with one built now.
- **A figure's labels are folded into the picture that holds them** by default
  (`PipelineFlags.absorb_figure_text`). Across the 20-document test corpus this is
  12515 → 8520 units. Set the flag to `False` for the old node set.
- **`transformers` is no longer a declared runtime dependency.** It still arrives with
  docling, so nothing breaks today; declare `document2graph[tokenizers]` or
  `[surprisal]` if you rely on it. `pytest` is no longer a runtime dependency at all —
  installing the package used to install pytest into the environment.

### Added

- **`need_supply` composes by evidence, with no per-relation priors** (`NeedSupplyConfig`,
  `composition="evidence"`, the default). Every edge starts at a neutral 0.5 for existing;
  `need x measured supply` moves it toward 1 and `(1 - need) x n / (n + half_length)`
  moves it toward 0. The product form could only trim: 80.5% of edges were floor-bound,
  and a prior-floored supply scales a weight by at most 2.5x. On the 22-document test
  corpus the evidence weights take 3970 distinct values with none at zero (product: 1133,
  206 zeros), and calibrate to the midpoint budget exactly (product: -3.1%).
  `composition="product"` restores the earlier weights. `evidence_components()` reports
  every term per edge, and `scripts/audit_edge_weights.py --section evidence` breaks them
  down by relation.

- **`graph_from_docling(docling_doc, pdf_doc, filename, ...)`** and
  `build_snippet_graph(...)`: construction from an already-parsed document, with no
  corpus directory and nothing written to disk. An ablation over `PipelineFlags` or
  `EdgeWeightConfig` now reuses one parse.
- **`DocumentGraph.pages`** (`list[PageGeometry]`) and `DocumentGraph.page(n)`. Every
  stored bbox is a docling provenance box in PDF points and says nothing about where it
  sits without the page box; a reloaded graph now carries it. Verified across the test
  corpus: every boxed snippet is placeable and lies inside the page it names.
- **`DocumentGraph.edge_relations`** and `DocumentGraph.relation(parent, child)`. Three
  relations — `reference`, `unreferenced_media`, `root` — describe how construction
  created an edge rather than what its two nodes are, so nothing downstream could
  recover them. `unreferenced_media` carries the highest measured supply prior and would
  otherwise read as an ordinary media edge.
- **`document2graph.retrieval`**: `merge_units()` with `none` / `all_ancestors` /
  `subtree` / `weighted` strategies, `expand()` with seven modes including auto-merge
  and reference-edge following, and `calibrate_min_weight()`, which sets a metric's
  threshold so that two metrics on incomparable scales can be compared at equal mean
  unit size.
- **`edge_weights.recompute_weights(graph, config)`** and `edge_context_from_graph()`:
  rescore a saved graph without re-parsing the PDF.
- `MergePolicy`, `MergedUnit`, `MergedEdge`, `ExpansionPolicy`, `ExpandedHit`,
  `PageGeometry`, `Calibration`.
- **`PipelineFlags.absorb_figure_text`** and `figure_label_max_tokens`. Docling reads a
  chart line by line, so a bar chart arrives as dozens of one-token nodes — 87% of this
  corpus's figure-region nodes are 5 tokens or fewer, and in a chart-heavy document that
  is 86–97% of everything — while the picture holding them has no text and is therefore
  not retrievable at all. The labels now become the picture's `figure_text`, in reading
  order. Figure *prose* above `figure_label_max_tokens` — a flowchart step, a boxed
  note — stays a node of its own.
- **`MergePolicy.min_tokens`**: a floor on what is worth retrieving on its own. A node
  below it is folded into the nearest ancestor that is a unit rather than dropped, so no
  text leaves the corpus; a fragment with no such ancestor keeps its own unit. The
  retrieval-time counterpart to `absorb_figure_text`, and 0 by default.
- `ImageSnippetNode.figure_text`, `ImageSnippetNode.content_text()` and
  `TableSnippetNode.content_text()` — what a media node's text is made of, asked of the
  node rather than assembled by each consumer.
- `document2graph.utils.tokens.approximate_token_count`, so graph construction can count
  tokens without importing the retrieval layer. Still exported from
  `document2graph.retrieval`.
- `tokenizers` and `surprisal` extras, and an `all` extra.
- `scripts/audit_edge_weights.py` — what the weight metrics do to a whole corpus, read
  from saved graphs with no PDFs: relation mix by language, how often the structural
  prior rather than the measured evidence sets the supply score, rank correlation
  between metrics, and what each costs once it gates a merge.
- `scripts/build_prior_queue.py` — a stratified, blind labelling queue for re-measuring
  the supply priors.

### Fixed

- **Graph construction was ~8.5× slower than it needed to be.**
  `SegmentedPdfPage.get_cells_in_bbox` deep-copies every cell on a page before testing
  one for overlap, and construction called it once per snippet, twice over — 81M
  `deepcopy` calls on a 12-page document, 90 of the 95 seconds a build took. Page cells
  are now read and converted once into a small bounded cache. A 12-page guideline builds
  in 2.3s rather than 19.6s, and output is byte-identical.

- **A persisted table had no caption anywhere in its text.** `from_snippet_graph` mapped
  a table node to `markdown_serialization` alone, while the edge-weight view of the same
  node had always used caption + markdown. The caption is the only part of a table
  written as a sentence, so this was the part a query was most likely to match.
- **`scripts/build_prior_queue.py` flattened tables before showing them to the judge.**
  In a one-line github table the row delimiter is `| |`, which is also how an empty cell
  is written, so the rows were not recoverable. Table snippets keep their line breaks.

## 0.1.2

Initial PyPI release.
