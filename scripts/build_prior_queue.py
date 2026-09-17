"""Sample a stratified queue of edges to hand-label, for re-measuring the supply priors.

The shipped ``MEASURED_SUPPLY_PRIORS`` rest on 50 edges, five per relation, from six
German guidelines. ``audit_edge_weights.py --section priors`` shows why that is no
longer enough: the prior, not the measured evidence, sets the supply score on about
85% of this corpus's edges, so those ten numbers *are* the supply metric -- and the
corpus they were measured on is not the corpus they are now applied to, whose German
and English halves differ by 7x in how much of them is list relations.

What this writes:

    queue.jsonl     one edge per line, with empty `need` and `supply` fields to fill
    scores.jsonl    the measured components per edge -- NOT for the labeller
    README.md       the label definitions, and what each cell could be filled to

**The queue is blind on purpose.** It carries the two texts and the relation, and
nothing else: no component scores, no current prior, no existing weight. Labelling
against a number derived from the same proxy the label is meant to validate is how
the circularity the audit warned about gets back in. The scores are written to a
separate file so the two can be compared *after* labelling, never during.

**Two labels per edge, not one.** The original scale asked only whether the parent
supplies what the child is missing. But ``heading_text`` was already known to be a
split cell -- some children need nothing at all -- and need and supply are separate
halves of the metric, so conflating them loses exactly the distinction that matters.

Usage:
    uv run python scripts/build_prior_queue.py
    uv run python scripts/build_prior_queue.py --per-cell 30 --out data/prior_labels/
    uv run python scripts/build_prior_queue.py --language de
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document2graph import load_graph
from document2graph.edge_weights import (
    SUPPLY_RELATIONS,
    components,
    edge_context_from_graph,
    supply_relation,
)
from document2graph.edge_weights import linguistic as ling
from document2graph.models import MEASURED_SUPPLY_PRIORS, SimilarityConfig, SupplyConfig

GRAPHS = "tests/.test-data/graphs/*_graph.json"
OUT_DIR = "tests/.test-data/prior_labels/"
# enough that one edge moving between two grades shifts a cell mean by ~0.02
# rather than the 0.1 that five per cell allowed
PER_CELL = 30
# a parent or child longer than this is truncated for the labeller. A 7000-token
# table cannot be judged as a whole anyway, and the judgement is about whether the
# parent names what the child leaves dangling, which the opening carries.
MAX_CHARS = 1200


def edge_uid(document_id: str, parent_ref: str, child_ref: str) -> str:
    """Stable id, so labels survive a regeneration of the corpus."""
    return hashlib.sha256(f"{document_id}:{parent_ref}:{child_ref}".encode()).hexdigest()[:16]


def clip(text: str, keep_lines: bool = False) -> str:
    """Flatten a snippet to one line and cut it to MAX_CHARS, for the labeller.

    ``keep_lines`` spares the line breaks, which a table cannot be read without: in a
    one-line github table the row delimiter is ``| |``, which is also how an empty
    cell is written, so the rows stop being recoverable and the judge is guessing at
    what lines up with what.
    """
    text = re.sub(r"[ \t]+", " ", text).strip() if keep_lines else " ".join(text.split())
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + " […]"


def document_language(graph) -> str:
    return ling.resolve_language("auto", " ".join(s.text for s in graph.snippets[:200]))


def collect(graphs, languages, supply_config, regions):
    """Every labellable edge, bucketed by (language, relation), with its scores.

    ``regions`` restricts which children count. The default excludes ``figure`` and
    ``front_matter``, which together are 38% of this corpus's edges: inside a figure a
    node is one line of a flowchart, and the masthead is an author list. Both are real
    retrievable content, but "does this parent supply what the child is missing" means
    something different there than in running prose, and a prior meant to describe
    document hierarchy should not average the two together.

    ``sidebar`` is kept, and that is not a close call: in the German guidelines the
    sidebars are the recommendation boxes, which is where the actual clinical content
    sits. Dropping them also leaves ``text_text`` with 11 labellable German edges
    instead of 38, because German paragraph chains almost all sit in a box.
    """
    cells = defaultdict(list)
    scores = {}
    for graph in graphs:
        ctx = edge_context_from_graph(graph)
        local = {s.snippet_id: s for s in graph.snippets}
        parts = components(ctx, supply_config)
        for edge in ctx.edges:
            parent_id, child_id = edge
            parent, child = local.get(parent_id), local.get(child_id)
            if parent is None or child is None:
                continue
            if not parent.text.strip() or not child.text.strip():
                continue
            if regions and child.region not in regions:
                continue
            uid = edge_uid(graph.document_id, parent.local_ref, child.local_ref)
            relation = supply_relation(ctx, edge)
            cells[(languages[graph.document_id], relation)].append({
                "edge_uid": uid,
                "document": graph.filename,
                "language": languages[graph.document_id],
                "relation": relation,
                "region": child.region,
                "page_no": child.page_no,
                "parent_type": parent.snippet_type,
                "child_type": child.snippet_type,
                "parent_text": clip(parent.text, keep_lines=parent.snippet_type == "table"),
                "child_text": clip(child.text, keep_lines=child.snippet_type == "table"),
                # to be filled in by hand; see README.md for the scale
                "need": None,
                "supply": None,
                "note": "",
            })
            scores[uid] = {"edge_uid": uid, "relation": relation,
                           **{k: round(v, 4) for k, v in parts[edge].items()}}
    return cells, scores


def sample_cell(rows, per_cell: int, rng: random.Random) -> list:
    """Up to ``per_cell`` edges, spread across documents rather than taken from one.

    A cell filled from a single document measures that document's layout, which is
    how a prior ends up describing a house style instead of a relation.
    """
    by_document = defaultdict(list)
    for row in rows:
        by_document[row["document"]].append(row)
    for group in by_document.values():
        rng.shuffle(group)

    picked, documents = [], sorted(by_document)
    while len(picked) < per_cell and any(by_document[d] for d in documents):
        for document in documents:
            if by_document[document] and len(picked) < per_cell:
                picked.append(by_document[document].pop())
    return picked


README = """# Supply prior labelling queue

{n} edges to label, {cells} cells of (language x relation), sampled from the
`{regions}` region(s).

Each line of `queue.jsonl` is one parent/child pair. Fill in two fields and leave
everything else alone. `edge_uid` is stable across a regeneration of the corpus, so
a part-finished queue can be rebuilt without losing work.

## The two labels

**`need`** — read the *child* on its own, without the parent. Can it be understood?

| value | meaning |
|---|---|
| `0.0` | yes, it stands alone completely |
| `0.5` | mostly, but something is unclear or unscoped |
| `1.0` | no, it cannot be read without something above it |

**`supply`** — now read the parent. Of what the child was missing, how much does
*this* parent give you?

| value | meaning |
|---|---|
| `0.0` | nothing the child needed |
| `0.5` | some of it |
| `1.0` | everything the child was missing |

If `need` is `0.0`, set `supply` to `null` — a child that needs nothing cannot tell
you what a parent supplies, and averaging those in as zeros is what made
`heading_text` a split cell in the first measurement.

Use `note` for anything odd: a mis-parsed block, a caption attached to the wrong
figure, a parent that is clearly not the right one.

## Before you start

The queue deliberately shows you only the two texts and the relation. It does not
show the current prior, the computed weight, or the measured components — those are
in `scores.jsonl`, which is for comparing against *after* labelling. The whole point
of re-measuring is to get a judgement that is independent of the proxies, and a
number on the screen is not something you can un-see.

Texts over {max_chars} characters are truncated. If a truncated edge cannot be
judged, leave both fields `null` and say so in `note` rather than guessing.

## What the cells could be filled to

{coverage}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graphs", default=GRAPHS)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--per-cell", type=int, default=PER_CELL)
    parser.add_argument("--language", default=None, choices=["de", "en"])
    parser.add_argument("--seed", type=int, default=42, help="sampling seed (default: 42)")
    parser.add_argument("--backend", default="bm25", choices=["bm25", "embedding", "blend"])
    parser.add_argument("--regions", nargs="*", default=["body", "sidebar"],
                        choices=["body", "front_matter", "sidebar", "figure"],
                        help="which child regions to sample (default: body and sidebar; "
                             "pass nothing after the flag to take every region)")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.graphs))
    if not paths:
        parser.error(f"no graphs matched {args.graphs!r}")
    graphs = [load_graph(path) for path in paths]
    stale = [g.filename for g in graphs if g.edges and not g.edge_relations]
    if stale:
        parser.error(f"{len(stale)} graph(s) predate the edge relation table; re-extract first")

    languages = {g.document_id: document_language(g) for g in graphs}
    if args.language:
        graphs = [g for g in graphs if languages[g.document_id] == args.language]

    supply_config = SupplyConfig(similarity=SimilarityConfig(backend=args.backend))
    cells, scores = collect(graphs, languages, supply_config, set(args.regions))

    rng = random.Random(args.seed)
    queue, coverage = [], []
    seen_languages = sorted({lang for lang, _ in cells})
    for language in seen_languages:
        for relation in SUPPLY_RELATIONS:
            available = cells.get((language, relation), [])
            picked = sample_cell(available, args.per_cell, rng) if available else []
            queue.extend(picked)
            coverage.append((language, relation, len(available), len(picked),
                             len({row["document"] for row in picked})))

    queue.sort(key=lambda row: (row["language"], row["relation"], row["edge_uid"]))

    os.makedirs(args.out, exist_ok=True)
    queue_path = os.path.join(args.out, "queue.jsonl")
    with open(queue_path, "w", encoding="utf-8") as fh:
        for row in queue:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    scores_path = os.path.join(args.out, "scores.jsonl")
    kept = {row["edge_uid"] for row in queue}
    with open(scores_path, "w", encoding="utf-8") as fh:
        for uid in sorted(kept):
            fh.write(json.dumps(scores[uid], ensure_ascii=False) + "\n")

    width = max(len(r) for r in SUPPLY_RELATIONS)
    lines = [f"| {'language':<8} | {'relation':<{width}} | available | queued | documents | prior |",
             f"| {'-' * 8} | {'-' * width} | ---: | ---: | ---: | ---: |"]
    thin = []
    for language, relation, available, picked, documents in coverage:
        prior = MEASURED_SUPPLY_PRIORS.get(relation, 0.5)
        flag = "" if picked >= args.per_cell else "  **thin**"
        lines.append(f"| {language:<8} | {relation:<{width}} | {available} | {picked} | "
                     f"{documents} | {prior:.2f} |{flag}")
        if picked < args.per_cell:
            thin.append((language, relation, picked))
    coverage_md = "\n".join(lines)

    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README.format(n=len(queue), cells=len(coverage), max_chars=MAX_CHARS,
                               regions=", ".join(sorted(args.regions)), coverage=coverage_md))

    print(f"wrote {len(queue)} edges to {queue_path}")
    print(f"      {len(kept)} component rows to {scores_path} (do not read while labelling)")
    print(f"      label definitions and cell coverage to {os.path.join(args.out, 'README.md')}\n")
    print(coverage_md.replace("**thin**", "thin"))
    if thin:
        print(f"\n{len(thin)} cell(s) could not reach {args.per_cell}:")
        for language, relation, picked in thin:
            print(f"  {language}/{relation}: {picked}")
        print("  A cell the corpus cannot fill stays an estimate from few edges however it")
        print("  is labelled. Either widen the corpus for those relations, or report them")
        print("  as unmeasured and let the composite fall back to the default.")
    per_language = Counter(row["language"] for row in queue)
    print(f"\nby language: " + ", ".join(f"{n} {lang}" for lang, n in sorted(per_language.items())))


if __name__ == "__main__":
    main()
