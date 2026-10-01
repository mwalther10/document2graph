# Development scripts

Six scripts that sit around the extractor: they either re-run a stage of the
pipeline in isolation, or read back what a run wrote. None of them is part of the
package — they exist to answer questions about a corpus that the test suite
cannot, because a test asserts what should be true while these show what is.

All of them are run from the repository root, so the relative paths below resolve:

```bash
uv run python scripts/<script>.py [options]
```

| Script | Reads | Answers |
|---|---|---|
| [`inspect_graph.py`](#inspect_graphpy) | saved graphs | is the hierarchy right? |
| [`debug_snippet_graph.py`](#debug_snippet_graphpy) | one PDF (cached) | why did construction decide *that*? |
| [`audit_fragmentation.py`](#audit_fragmentationpy) | saved graphs + docling cache | what got split that should be whole? |
| [`build_review_bundle.py`](#build_review_bundlepy) | — (called by the audit) | turns candidates into a labelling queue |
| [`compare_edge_weights.py`](#compare_edge_weightspy) | one PDF (cached) | what do the weight metrics do to the same graph? |
| [`audit_edge_weights.py`](#audit_edge_weightspy) | saved graphs | do the metrics measure anything, and are the priors carrying them? |

## Shared inputs

Everything works off the directories the test suite fills, so the usual order is
**run the suite once, then read its output as often as needed**:

```
tests/.test-pdf/                   the corpus itself
tests/.test-data/raw_texts/        docling conversions, cached per PDF
tests/.test-data/graphs/           *_graph.json  — the DocumentGraph, lossless
tests/.test-data/nx_graphs/        *.gexf        — for Gephi and friends
```

Converting a PDF through docling is the expensive step (minutes per document; the
full suite takes about half an hour). The cache under `raw_texts/` is what makes
these scripts cheap: `debug_snippet_graph.py` and `compare_edge_weights.py` reuse
it and only re-convert a PDF they have never seen, or when passed `--refresh`.
`inspect_graph.py` and `audit_fragmentation.py` never touch docling at all.

The GEXF export drops bbox, page numbers, char spans, sequence numbers, parent
refs and table serializations, so for anything but visualization read the JSON —
`load_graph()` from `document2graph.graph_store` returns the `DocumentGraph` back
exactly as it was written.

---

## `inspect_graph.py`

Checks the hierarchy the saved graphs encode. Three views:

```bash
uv run python scripts/inspect_graph.py                      # summary of every document
uv run python scripts/inspect_graph.py 054                  # summary of one (substring match)
uv run python scripts/inspect_graph.py 054 --tree           # the full tree
uv run python scripts/inspect_graph.py 054 --tree --headings # the outline only
uv run python scripts/inspect_graph.py 054 --tree --region sidebar
uv run python scripts/inspect_graph.py --checks             # diagnostics, whole corpus
```

**The summary** is one line per document, which is what makes an outlier visible:

```
054_Therapie_des_Typ_1_Diabetes …  n=  289 pages=  12 depth_max= 8 root_children=   9 (  3%)
   head=  42 body=  226 unknown=    0 body_region=  226 figure=    0 sidebar=  32 front= 31
```

`root_children` is the one to watch. Nodes land on the root when no parent could be
resolved for them, so a high share means the hierarchy did not form. `body_region`
near zero means the region pass decided the document has no body text, and since
the outline is built from body text alone, nothing downstream can rank its
headings. `unknown` counts nodes the level classifier could not place at all.

**The tree** prints the hierarchy as the graph holds it, one node per line with its
type, level, page and region. `--headings` reduces it to the outline, which is the
view to hold against the PDF page by page — it is short enough to read end to end
and it is exactly what the retrieval layer will walk.

**The checks** look for shapes that are wrong on their face:

- `[root]` — content left on the root, i.e. orphans, listed so they can be looked up
- `[levels]` — a heading ranked at or above the heading that holds it
- `[headings]` — a "heading" the length of a paragraph, usually a mislabelled block
- `[locality]` — edges spanning more than two pages; a parent that far away is
  rarely the section the child sits in
- `[media]` — how many figures and tables resolved a caption reference, and how
  many were left on the root
- `[repeats]` — a text occurring three times or more, i.e. a running header or
  footer read as content

None of it is a verdict. A check that fires is a place to open the PDF; several of
them fire on layouts that are merely unusual. The thresholds are constants at the
top of the file.

## `debug_snippet_graph.py`

Runs `SnippetGraphConstructor` over a single PDF with full logging, then prints
what it decided: the extracted metadata, the region counts, every heading style it
found as (font, height) → level with example headings, the resulting heading tree,
the root it resolved, the nodes attached directly to the root, the reference edges,
and whether the tree came out connected.

```bash
uv run python scripts/debug_snippet_graph.py
uv run python scripts/debug_snippet_graph.py --pdf tests/.test-pdf/<file>.pdf
uv run python scripts/debug_snippet_graph.py --refresh            # re-convert the PDF
uv run python scripts/debug_snippet_graph.py --log-level INFO     # less noise
uv run python scripts/debug_snippet_graph.py --level-source docling
uv run python scripts/debug_snippet_graph.py --save-gexf out.gexf
```

Where `inspect_graph.py` shows the outcome, this shows the reasoning — reach for it
once a check has told you *which* document is wrong. The heading-style block is
usually where the answer is: if two levels share one (font, height) style, no
ranking can separate them.

`--level-source` switches how heading levels are derived, and is the ablation
behind the hybrid classifier: `hybrid` (the default: font identity and height
clusters, ordered by how sections nest in reading order), `typography` and
`docling` (whatever the layout model decided, no fonts and no nesting evidence).
The metadata config is hard-coded to the one the tests use for the German
guideline PDFs; a different corpus needs that block edited.

## `audit_fragmentation.py`

Finds content the pipeline left split apart — a table continued on the next page
that stayed two table nodes, a paragraph cut mid-column that never got stitched.
Both detectors over-generate on purpose and score their candidates, so the output
is a review queue, not a verdict.

```bash
uv run python scripts/audit_fragmentation.py                   # summary table
uv run python scripts/audit_fragmentation.py --detail          # every candidate
uv run python scripts/audit_fragmentation.py --doc 037 --detail
uv run python scripts/audit_fragmentation.py --min-score 0.5
uv run python scripts/audit_fragmentation.py --bundle out/     # + page renders
```

Its continuation tests deliberately duplicate the constructor's rather than
importing them: the audit has to see the fragments the pipeline decided *not* to
join, so it must not inherit any later tightening of those rules.

## `build_review_bundle.py`

Not run directly — `audit_fragmentation.py --bundle <dir>` calls it. It writes a
`review_data.json` plus a JPEG of every page a candidate touches, so a reviewer
judges a split from the layout (which column the fragment sits in, whether a rule
separates two tables) instead of from text alone. Only pages carrying a candidate
are rendered. Candidate ids are hashes of the document name and the two refs, so
labels survive a rebuild of the bundle. Needs `pypdfium2`.

## `compare_edge_weights.py`

Builds one graph and then scores its edges with every weight metric against the
same `EdgeContext`, so a sweep costs one parse plus a cheap pass per metric
instead of a full pipeline run each.

```bash
uv run python scripts/compare_edge_weights.py
uv run python scripts/compare_edge_weights.py --pdf tests/.test-pdf/<file>.pdf
uv run python scripts/compare_edge_weights.py --metrics uniform structural need_supply
uv run python scripts/compare_edge_weights.py --top 10          # highest-weighted edges per metric
uv run python scripts/compare_edge_weights.py --backend embedding --supply complementarity
uv run python scripts/compare_edge_weights.py --with-surprisal  # loads a causal LM
```

`--backend` and `--supply` override the content settings across all metrics, which
keeps the sweep a one-variable comparison. `--with-surprisal` is off by default
because it pulls a language model.

---

# Reading the graphs in Neo4j

The saved JSON is the source of truth; Neo4j is for querying a corpus and for
seeing a hierarchy drawn. Writing needs the `neo4j` extra:

```bash
uv pip install -e ".[neo4j]"
```

## Connecting

The driver and its credentials belong to the caller — nothing in the package
opens a connection. The test suite reads `NEO4J_URI`, `NEO4J_USER` and
`NEO4J_PASSWORD` from the environment (`.env`), defaulting to
`bolt://localhost:7687` and `neo4j`/`password`, and skips its Neo4j test when
nothing answers there. A local instance:

```bash
docker run --rm -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5
```

Browser on <http://localhost:7474>, bolt on `7687`.

## Loading

Already-extracted graphs go in without re-parsing a single PDF — read the JSON
back and write it:

```python
import glob, os
from neo4j import GraphDatabase
from document2graph.graph_store import load_graph, ensure_neo4j_constraints, write_graph_to_neo4j

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])
)
with driver:
    ensure_neo4j_constraints(driver)          # once per database
    for path in sorted(glob.glob("tests/.test-data/graphs/*_graph.json")):
        write_graph_to_neo4j(driver, load_graph(path))
```

`ensure_neo4j_constraints` creates the uniqueness constraints on
`Snippet.snippet_id` and `Document.document_id`; without them every `MERGE`
degrades to a full scan once a corpus grows past a few documents. The write is
idempotent — document ids and the snippet ids derived from them are
deterministic, so re-running updates in place rather than duplicating.

## What lands in the database

```
(:Document {document_id, filename, title, document_type, version, authors, institutions})
  -[:HAS_ROOT]->     (:Snippet)                       the title node, or a synthetic root
(:Snippet)-[:HAS_CHILD {weight}]->(:Snippet)          the hierarchy tree, one parent per node
(:Snippet)-[:REFERENCES {weight}]->(:Snippet)         a text node mentioning a figure or table
```

`Document.filename` is the cleaned stem the extractor derived, without the `.pdf`
(`054_Therapie_des_Typ_1_Diabetes_Kurzfassung_…`), which is also the name the files
under `tests/.test-data/graphs/` carry.

A `Snippet` carries `snippet_id`, `local_ref` (the docling ref, `#/texts/12`),
`document_id`, `sequence_no`, `label`, `snippet_type` (`text` | `image` | `table` |
`root`), `level`, `level_label`, `region`, `is_grouped`, `page_no`, `pages` (every
page it spans), `line_heights`, `font_key`, `level_height`, the flattened
`bbox_l/t/r/b` and `bbox_origin`, `charspan_start/end`, and **`text_preview`**.

Note the last one: Neo4j gets the first 100 characters only. Full text, full
provenance and table markdown stay in the JSON, so a query that needs the body of
a node uses the `snippet_id` to look it up there.

## Querying the hierarchy

The outline of one document, in reading order:

```cypher
MATCH (d:Document {filename: $filename})-[:HAS_ROOT]->(root)
MATCH path = (root)-[:HAS_CHILD*0..]->(s:Snippet)
WHERE s.level_label IN ['Title', 'Heading']
RETURN length(path) AS depth, s.level, s.page_no, s.region, s.text_preview
ORDER BY s.sequence_no
```

Everything under one section, which is the subtree a retrieval layer would pull:

```cypher
MATCH (h:Snippet {snippet_id: $snippet_id})-[:HAS_CHILD*]->(s:Snippet)
RETURN s.snippet_type, s.level_label, s.page_no, s.text_preview
ORDER BY s.sequence_no
```

The same checks `inspect_graph.py --checks` runs, across the whole corpus — what
is stranded on a root:

```cypher
MATCH (d:Document)-[:HAS_ROOT]->(root)-[:HAS_CHILD]->(s:Snippet)
WHERE NOT s.level_label IN ['Title', 'Heading']
RETURN d.filename, count(s) AS stranded
ORDER BY stranded DESC
```

…and where a child fails to rank below its parent:

```cypher
MATCH (p:Snippet)-[:HAS_CHILD]->(c:Snippet)
WHERE p.level_label IN ['Title', 'Heading'] AND c.level_label IN ['Title', 'Heading']
  AND c.level <= p.level
RETURN p.document_id, p.level, p.text_preview, c.level, c.text_preview
```

Which figures and tables a section mentions:

```cypher
MATCH (t:Snippet)-[:REFERENCES]->(m:Snippet)
WHERE t.document_id = $document_id
RETURN t.page_no, t.text_preview, m.snippet_type, m.page_no
ORDER BY t.sequence_no
```

To draw a hierarchy in the browser, return the paths rather than the rows — the
visualizer lays out what a query returns as nodes and relationships:

```cypher
MATCH (d:Document {filename: $filename})-[:HAS_ROOT]->(root)
MATCH path = (root)-[:HAS_CHILD*1..3]->(:Snippet)
RETURN path LIMIT 300
```

Start the caption at `text_preview` under **Browser → settings → node caption** to
read the tree rather than a wall of hashes. For anything larger than a few hundred
nodes, the GEXF export under `tests/.test-data/nx_graphs/` and Gephi will be more
comfortable than the browser.

Removing a corpus to load it again from scratch:

```cypher
MATCH (n) WHERE n:Snippet OR n:Document DETACH DELETE n
```


## `audit_edge_weights.py`

Where `compare_edge_weights.py` shows one document, this reads the whole saved corpus —
no PDFs, no docling — and asks whether a retrieval run is worth doing at all.

```bash
uv run python scripts/audit_edge_weights.py
uv run python scripts/audit_edge_weights.py --section priors
uv run python scripts/audit_edge_weights.py --section agreement merge
uv run python scripts/audit_edge_weights.py --language de     # priors were measured on German
```

Four sections:

- **corpus** — the relation mix, split by detected language. The supply priors were
  measured on German guidelines, and the mix is not the same in both halves of this
  corpus: list relations are 37% of German edges and 5% of English ones.
- **priors** — per supply relation, what the measured components say against what the
  prior says, and `floor-bound`: the share of edges where the prior, not the evidence,
  produced the score. A relation that is mostly floor-bound has no measurement behind it.
- **agreement** — rank correlation between metrics, plus each one's spread and how many
  edges it scores exactly zero. A pair near 1.00 is one condition run twice.
- **merge** — what each metric costs in index tokens once it actually gates a merge,
  calibrated against the merge-everything baseline it has to beat.

It needs graphs written under **schema version 3** or later, which carry the edge
relation table; it refuses to run on older ones rather than guessing the three relations
that describe how an edge was created.

**It cannot produce new priors.** The shipped ones come from hand labels — "does this
parent supply the child's missing context: fully, partly, not at all" — and the
components measured here are proxies for that judgement, not the judgement. Flooring a
proxy at a prior derived from the same proxy would be circular. What the audit shows is
whether the priors are *doing work* and whether they *disagree with the evidence*.
