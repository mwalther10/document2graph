# Changelog

## 0.2.0 — unreleased

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
- **`SCHEMA_VERSION` 1 → 3.** A graph written by 0.1.x still loads, but `pages` and
  `edge_relations` come back empty, `DocumentGraph.page()` returns `None`, and
  `recompute_weights()` refuses it rather than scoring on guessed relations.
  **Re-extract a 0.1.x corpus** before reweighting or scoring it.
- **`transformers` is no longer a declared runtime dependency.** It still arrives with
  docling, so nothing breaks today; declare `document2graph[tokenizers]` or
  `[surprisal]` if you rely on it. `pytest` is no longer a runtime dependency at all —
  installing the package used to install pytest into the environment.

### Added

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

### Notes

- `MEASURED_SUPPLY_PRIORS` still rests on 50 hand-labelled edges (5 per relation) from
  six German documents. `scripts/audit_edge_weights.py --section priors` shows that the
  prior, not the measured components, sets the supply score on ~86% of a 20-document
  corpus — so those ten numbers are effectively the supply metric. A re-measurement at
  n=30 per cell, in two languages, is in progress.

## 0.1.2

Initial PyPI release.
