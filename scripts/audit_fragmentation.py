"""Find media and paragraph fragments that graph construction leaves split apart.

Two failure modes are looked for, both of which come out of docling's layout
clustering and are never repaired downstream:

*   **Split tables.** ``compute_table_nodes`` takes ``docling_doc.tables`` as it
    finds them. A table continued into the next column or onto the next page is
    two ``TableItem``s there, so it becomes two unrelated table nodes with half
    the rows each and, usually, a caption on only the first.
*   **Split paragraphs.** ``stitch_continuations`` only joins a fragment that
    ends at the very bottom of its column with one that opens the very top of
    the next, and never touches text read out of a figure. Everything else
    docling cut -- a block broken mid-column, a figure's body set line by line --
    stays split.

Both detectors are deliberately generous: they over-generate candidates and
score them, so that the output is a review queue rather than a verdict. Run
``--bundle`` to turn that queue into the labelling artifact's input.

Usage:
    uv run python scripts/audit_fragmentation.py                  # summary table
    uv run python scripts/audit_fragmentation.py --detail         # every candidate
    uv run python scripts/audit_fragmentation.py --doc 037 --detail
    uv run python scripts/audit_fragmentation.py --bundle out/    # + page renders
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CACHE_DIR = "tests/.test-data/raw_texts"
GRAPH_DIR = "tests/.test-data/graphs"
PDF_DIR = "tests/.test-pdf"

# Mirrors of the constructor's continuation tests. Kept local on purpose: the
# audit has to see the fragments the pipeline decided *not* to join, so it must
# not inherit any later tightening of those rules by importing them.
TERMINAL_PUNCTUATION = re.compile(r'[.!?:;)\]"”’»]\s*$')
CONTINUATION_BREAKS = ",-­"
CAPTION_PATTERN = re.compile(r"^[\s▶▪●•·‣→>»|–—-]*(Tab|Abb|Table|Figure|Fig)\.?\s*\d+", re.IGNORECASE)
NUMBERED_ENTRY = re.compile(r"^\d+[\s.)]")
BIBLIOGRAPHY_ENTRY = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?\s+[A-Z]{1,3}[,.]")
# a comma or a trailing conjunction closing a bullet is how an enumeration is
# punctuated, not how a sentence breaks off
ENUMERATION_TAIL = re.compile(r"(,|\bund|\boder|\bbzw\.?|\bsowie|/oder)\s*$", re.IGNORECASE)
FLOW_LABELS = ("text", "list_item", "footnote")


def is_unfinished(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped or TERMINAL_PUNCTUATION.search(text):
        return False
    if stripped[-1] in CONTINUATION_BREAKS:
        return True
    return stripped.split()[-1][:1].islower()


def resumes(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped or CAPTION_PATTERN.match(stripped):
        return False
    stripped = stripped.lstrip("([")
    if not stripped or NUMBERED_ENTRY.match(stripped) or BIBLIOGRAPHY_ENTRY.match(stripped):
        return False
    return stripped[0].isalpha()


def to_bottom_left(bbox: dict, page_height: float) -> tuple[float, float, float, float]:
    """(l, r, b, t) of a bbox in bottom-left origin, whichever origin it carries."""
    if bbox["coord_origin"] == "BOTTOMLEFT":
        return bbox["l"], bbox["r"], bbox["b"], bbox["t"]
    return bbox["l"], bbox["r"], page_height - bbox["b"], page_height - bbox["t"]


class Doc:
    """A cached docling document paired with the graph that was built from it."""

    def __init__(self, name: str, docling: dict, graph: dict):
        self.name = name
        self.docling = docling
        self.graph = graph
        self.pages = {int(k): v["size"] for k, v in docling["pages"].items()}

    def height(self, page_no: int) -> float:
        return self.pages[page_no]["height"]

    def width(self, page_no: int) -> float:
        return self.pages[page_no]["width"]

    def column(self, page_no: int, left: float, right: float) -> int:
        return 0 if (left + right) / 2 < self.width(page_no) / 2 else 1


def load_docs(pattern: str | None = None) -> list[Doc]:
    docs = []
    for path in sorted(glob.glob(f"{CACHE_DIR}/*_docling_doc.json")):
        name = os.path.basename(path)[: -len("_docling_doc.json")]
        if name.endswith("_baseline"):
            continue  # the chunker's own cache, same PDF
        graph_path = f"{GRAPH_DIR}/{name}_graph.json"
        if not os.path.exists(graph_path):
            continue
        if pattern and pattern.lower() not in name.lower():
            continue
        with open(path) as handle:
            docling = json.load(handle)
        with open(graph_path) as handle:
            graph = json.load(handle)
        docs.append(Doc(name, docling, graph))
    return docs


# --------------------------------------------------------------------------- tables

def _row_text(table: dict, row: int) -> str:
    grid = table["data"]["grid"]
    if row >= len(grid):
        return ""
    return " | ".join(" ".join(cell["text"].split()) for cell in grid[row]).strip()


def _stub_column(table: dict) -> list[str]:
    """The table's stub: the text of its first column, one entry per row.

    A table continued on the next page repeats its row labels, which identifies the
    continuation even when the two halves re-parse to different column counts."""
    return [" ".join(row[0]["text"].split()) for row in table["data"]["grid"] if row]


def _stub_overlap(head: list[str], tail: list[str]) -> float:
    """Fraction of the shorter stub the two tables have in common."""
    head_set = {entry for entry in head if entry}
    tail_set = {entry for entry in tail if entry}
    if not head_set or not tail_set:
        return 0.0
    return len(head_set & tail_set) / min(len(head_set), len(tail_set))


def _table_geometry(doc: Doc, table: dict) -> dict | None:
    if not table.get("prov"):
        return None
    prov = table["prov"][0]
    page_no = prov["page_no"]
    left, right, bottom, top = to_bottom_left(prov["bbox"], doc.height(page_no))
    return dict(
        ref=table["self_ref"], page_no=page_no, l=left, r=right, b=bottom, t=top,
        rows=table["data"]["num_rows"], cols=table["data"]["num_cols"],
        header=_row_text(table, 0), captioned=bool(table.get("captions")),
        column=doc.column(page_no, left, right), stub=_stub_column(table),
    )


def table_candidates(doc: Doc) -> list[dict]:
    """Pairs of table items that look like one table cut in two."""
    geometries = [_table_geometry(doc, table) for table in doc.docling["tables"]]
    candidates = []
    for head, tail in zip(geometries, geometries[1:]):
        if head is None or tail is None:
            continue
        signals, score = [], 0.0

        if head["cols"] == tail["cols"]:
            signals.append("same column count")
            score += 0.2
        else:
            score -= 0.2
        if head["header"] and head["header"] == tail["header"]:
            # NOT evidence of a continuation, though it looks like it: the guideline
            # layout sets every recommendation in its own table headed "Empfehlungen".
            # All five labelled pairs that shared a header row were separate tables.
            signals.append("identical header row (weak: sibling boxes share one too)")
        stub_overlap = _stub_overlap(head["stub"], tail["stub"])
        shared = len({e for e in head["stub"] if e} & {e for e in tail["stub"] if e})
        if stub_overlap >= 0.6 and shared >= 2:
            # the same row labels on both halves: one table read twice, once per page
            signals.append(f"{stub_overlap:.0%} of the row labels repeat ({shared} of them)")
            score += 0.45
        if head["captioned"] and not tail["captioned"]:
            signals.append("caption only on the first half")
            score += 0.15

        overlap = min(head["r"], tail["r"]) - max(head["l"], tail["l"])
        widths = (head["r"] - head["l"], tail["r"] - tail["l"])
        aligned = overlap > 0.6 * min(widths) and abs(widths[0] - widths[1]) < 40

        if head["page_no"] == tail["page_no"] and head["column"] == tail["column"]:
            gap = head["b"] - tail["t"]
            if aligned and 0 <= gap < 60:
                signals.append(f"stacked {gap:.0f}pt apart in the same column")
                score += 0.4
                kind = "same-column"
            else:
                continue
        elif head["page_no"] == tail["page_no"] and tail["column"] > head["column"]:
            page_height = doc.height(head["page_no"])
            if head["b"] < 0.35 * page_height and tail["t"] > 0.6 * page_height:
                signals.append("ends the left column, resumes at the top of the right")
                score += 0.35
                kind = "column-break"
            else:
                continue
        elif tail["page_no"] == head["page_no"] + 1:
            if head["b"] < 0.35 * doc.height(head["page_no"]) and tail["t"] > 0.6 * doc.height(tail["page_no"]):
                signals.append("ends one page, resumes at the top of the next")
                score += 0.35
                kind = "page-break"
            else:
                continue
        else:
            continue

        if aligned:
            signals.append("same width and horizontal position")
            score += 0.2
        candidates.append(dict(
            type="table", kind=kind, score=round(min(max(score, 0.0), 1.0), 2),
            signals=signals, a=head, b=tail,
            a_preview=_row_text(doc.docling["tables"][geometries.index(head)], 0),
        ))
    return candidates


# ----------------------------------------------------------------------- paragraphs

def _line_pitch(node: dict) -> float | None:
    heights = node.get("line_heights") or []
    return max(heights) if heights else None


def paragraph_candidates(doc: Doc) -> list[dict]:
    """Pairs of neighbouring text nodes that read as one paragraph cut in two.

    Reads the *graph*, not the docling document, so anything ``stitch_continuations``
    already joined is invisible here and only what it left behind is reported."""
    nodes = sorted((s for s in doc.graph["snippets"] if s["snippet_type"] == "text"),
                   key=lambda s: s["sequence_no"])
    docling_parent = {t["self_ref"]: t.get("parent", {}).get("$ref") for t in doc.docling["texts"]}

    candidates = []
    for head, tail in zip(nodes, nodes[1:]):
        if head["label"] not in FLOW_LABELS or tail["label"] not in FLOW_LABELS:
            continue
        if not is_unfinished(head["text"]) or not resumes(tail["text"]):
            continue

        head_page, tail_page = head["page_no"], tail["page_no"]
        head_box = to_bottom_left(head["bbox"], doc.height(head_page))
        tail_box = to_bottom_left(tail["bbox"], doc.height(tail_page))
        head_column = doc.column(head_page, head_box[0], head_box[1])
        tail_column = doc.column(tail_page, tail_box[0], tail_box[1])

        gap = None
        if head_page == tail_page and head_column == tail_column:
            kind, gap = "same-column", head_box[2] - tail_box[3]
        elif head_page == tail_page and tail_column > head_column:
            kind = "column-break"
        elif tail_page == head_page + 1:
            kind = "page-break"
        else:
            kind = "out-of-order"

        signals, score = [], 0.3
        stripped = head["text"].rstrip()
        if stripped.endswith("­") or stripped.endswith("-"):
            signals.append("the first half ends on a hyphen")
            score += 0.35
        if tail["text"].lstrip()[:1].islower():
            signals.append("the second half opens in lower case")
            score += 0.3
        pitch = _line_pitch(head)
        if kind == "same-column" and gap is not None and pitch and -2 <= gap <= 1.6 * pitch:
            signals.append(f"consecutive lines of one block ({gap:.1f}pt apart)")
            score += 0.25
        if kind in ("column-break", "page-break"):
            signals.append("broken across a column or page break")
            score += 0.15
        if head["region"] == "figure":
            signals.append("text read out of a figure, which the stitcher skips")
        same_group = (head["is_grouped"] and tail["is_grouped"]
                      and docling_parent.get(head["local_ref"]) == docling_parent.get(tail["local_ref"]))
        if same_group and ENUMERATION_TAIL.search(head["text"]) and not tail["text"].lstrip()[:1].islower():
            # "... SIDD (schwerer insulindefizienter Diabetes)," / "SIRD (...)" --
            # two bullets of one enumeration, correctly kept apart
            signals.append("looks like two bullets of one enumeration")
            score -= 0.45
        if BIBLIOGRAPHY_ENTRY.match(tail["text"].lstrip()):
            signals.append("the second half opens a new bibliography entry")
            score -= 0.3

        candidates.append(dict(
            type="paragraph", kind=kind, score=round(min(max(score, 0.0), 1.0), 2),
            signals=signals, gap=None if gap is None else round(gap, 1),
            region=head["region"], a=head, b=tail,
        ))
    return candidates


# ---------------------------------------------------------------------------- output

def verdict(score: float) -> str:
    return "likely split" if score >= 0.6 else ("unsure" if score >= 0.35 else "probably fine")


def print_summary(docs: list[Doc]) -> None:
    print(f"{'document':50s} {'tables':>7s} {'split?':>7s} {'texts':>7s} {'split?':>7s}  paragraph candidates by region")
    for doc in docs:
        tables = table_candidates(doc)
        paragraphs = paragraph_candidates(doc)
        regions = Counter(c["region"] for c in paragraphs if c["score"] >= 0.35)
        print(f"{doc.name[:50]:50s} {len(doc.docling['tables']):7d} {len(tables):7d} "
              f"{len(doc.docling['texts']):7d} {len(paragraphs):7d}  {dict(regions)}")


def print_detail(docs: list[Doc], min_score: float) -> None:
    for doc in docs:
        candidates = [c for c in table_candidates(doc) + paragraph_candidates(doc) if c["score"] >= min_score]
        if not candidates:
            continue
        print(f"\n=== {doc.name}")
        for candidate in sorted(candidates, key=lambda c: -c["score"]):
            head, tail = candidate["a"], candidate["b"]
            print(f"\n  [{candidate['type']}/{candidate['kind']}] score {candidate['score']} "
                  f"({verdict(candidate['score'])})  p{head['page_no']}→p{tail['page_no']}")
            print(f"    signals: {'; '.join(candidate['signals']) or '—'}")
            if candidate["type"] == "table":
                print(f"    A {head['ref']}  {head['rows']}×{head['cols']}  header: {head['header'][:80]!r}")
                print(f"    B {tail['ref']}  {tail['rows']}×{tail['cols']}  header: {tail['header'][:80]!r}")
            else:
                print(f"    A {head['local_ref']} ({head['label']}, {head['region']}): …{head['text'][-90:]!r}")
                print(f"    B {tail['local_ref']} ({tail['label']}, {tail['region']}): {tail['text'][:90]!r}…")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc", help="substring of the document name to restrict to")
    parser.add_argument("--detail", action="store_true", help="print every candidate")
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--bundle", help="write the labelling bundle (JSON + page renders) here")
    args = parser.parse_args()

    docs = load_docs(args.doc)
    if not docs:
        print(f"No cached documents found under {CACHE_DIR}/ matching {args.doc!r}.", file=sys.stderr)
        raise SystemExit(1)

    if args.bundle:
        from build_review_bundle import write_bundle  # noqa: PLC0415  (optional dependency: pypdfium2)
        write_bundle(docs, args.bundle, args.min_score)
        return
    if args.detail:
        print_detail(docs, args.min_score)
    else:
        print_summary(docs)


if __name__ == "__main__":
    main()
