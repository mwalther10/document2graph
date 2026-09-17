"""Turn the fragmentation audit into the input of the labelling artifact.

Writes one ``review_data.json`` plus a JPEG render of every page a candidate
touches, so the reviewer can see the actual layout — which column a fragment
sits in, whether a rule separates two tables — instead of judging from text
alone. Only pages that carry a candidate are rendered.

Called through ``audit_fragmentation.py --bundle <dir>``; needs ``pypdfium2``.
"""

import hashlib
import json
import os

PDF_DIR = "tests/.test-pdf"
RENDER_SCALE = 1.6  # ~115 dpi: legible body text at full-page width
JPEG_QUALITY = 72


def candidate_id(doc_name: str, candidate: dict) -> str:
    """Stable id for a candidate, so labels survive a rebuild of the bundle."""
    head, tail = candidate["a"], candidate["b"]
    key = f"{doc_name}|{candidate['type']}|{head.get('ref') or head.get('local_ref')}|{tail.get('ref') or tail.get('local_ref')}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _find_pdf(name: str) -> str | None:
    for candidate in (f"{PDF_DIR}/{name}.pdf", f"{PDF_DIR}/{name}.PDF"):
        if os.path.exists(candidate):
            return candidate
    return None


def _side(candidate: dict, which: str, doc) -> dict:
    """The half of a candidate the UI draws and quotes."""
    item = candidate[which]
    if candidate["type"] == "table":
        page_no = item["page_no"]
        return dict(
            ref=item["ref"], page_no=page_no, label="table",
            text=f"{item['rows']}×{item['cols']} — " + " | ".join(item["stub"][:6]),
            bbox=[item["l"], item["r"], item["b"], item["t"]],
        )
    page_no = item["page_no"]
    from audit_fragmentation import to_bottom_left  # noqa: PLC0415
    left, right, bottom, top = to_bottom_left(item["bbox"], doc.height(page_no))
    return dict(
        ref=item["local_ref"], page_no=page_no, label=item["label"],
        region=item["region"], text=item["text"], bbox=[left, right, bottom, top],
    )


def write_bundle(docs, out_dir: str, min_score: float) -> None:
    import pypdfium2  # noqa: PLC0415

    from audit_fragmentation import paragraph_candidates, table_candidates, verdict

    pages_dir = os.path.join(out_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)

    bundle = {"documents": [], "candidates": []}
    for doc in docs:
        pdf_path = _find_pdf(doc.name)
        if pdf_path is None:
            print(f"  ! no PDF for {doc.name}, skipping")
            continue
        candidates = [c for c in table_candidates(doc) + paragraph_candidates(doc)
                      if c["score"] >= min_score]
        if not candidates:
            continue

        slug = _slug(doc.name)
        entries, needed = [], set()
        for candidate in candidates:
            head, tail = _side(candidate, "a", doc), _side(candidate, "b", doc)
            needed.update((head["page_no"], tail["page_no"]))
            entries.append(dict(
                id=candidate_id(doc.name, candidate), document=doc.name, doc_slug=slug,
                type=candidate["type"], kind=candidate["kind"], score=candidate["score"],
                verdict=verdict(candidate["score"]), signals=candidate["signals"],
                region=candidate.get("region", "—"), a=head, b=tail,
            ))

        pdf = pypdfium2.PdfDocument(pdf_path)
        page_meta = {}
        for page_no in sorted(needed):
            page = pdf[page_no - 1]
            image = page.render(scale=RENDER_SCALE).to_pil().convert("RGB")
            filename = f"pages/{slug}_p{page_no}.jpg"
            image.save(os.path.join(out_dir, filename), "JPEG", quality=JPEG_QUALITY, optimize=True)
            page_meta[str(page_no)] = dict(
                file=filename, px_width=image.width, px_height=image.height,
                pt_width=doc.width(page_no), pt_height=doc.height(page_no),
            )
        pdf.close()

        bundle["documents"].append(dict(
            name=doc.name, slug=slug, page_count=len(doc.pages),
            table_count=len(doc.docling["tables"]), text_count=len(doc.docling["texts"]),
            pages=page_meta,
        ))
        bundle["candidates"].extend(entries)
        print(f"  {doc.name[:55]:57s} {len(entries):4d} candidates, {len(page_meta):3d} pages rendered")

    with open(os.path.join(out_dir, "review_data.json"), "w") as handle:
        json.dump(bundle, handle, ensure_ascii=False)
    size = sum(os.path.getsize(os.path.join(root, f))
               for root, _, files in os.walk(out_dir) for f in files)
    print(f"\n{len(bundle['candidates'])} candidates across {len(bundle['documents'])} documents "
          f"-> {out_dir} ({size / 1e6:.1f} MB)")
