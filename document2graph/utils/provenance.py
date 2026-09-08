"""Adapters that turn the two docling provenance shapes into ``Provenance``.

The graph side reads provenance off ``DocItem.prov``; the baseline side only
gets the JSON dict of a chunk's ``DocMeta``. Both land on the same model so
that ``unit.provenance`` means the same thing everywhere.
"""

from typing import Any, Iterable, Mapping, Sequence

from docling_core.types.doc.base import BoundingBox

from ..models.Provenance import Provenance


def provenance_from_prov(prov: Iterable[Any] | None) -> list[Provenance]:
    """Map *every* entry of a docling ``prov`` list, not just ``prov[0]``.

    Stitched paragraphs and multi-page tables carry several provenance items;
    reading only the first one under-localizes them to their head fragment.
    """
    return [
        Provenance(
            page_no=item.page_no,
            bbox=item.bbox,
            charspan=tuple(item.charspan) if item.charspan is not None else None,
        )
        for item in (prov or [])
    ]


def provenance_from_chunk_meta(meta: Mapping[str, Any] | None) -> list[Provenance]:
    """Collect the provenance of a chunk from the JSON dict of its ``DocMeta``.

    ``DocMeta.export_json_dict()`` keeps ``doc_items[*].prov[*]`` verbatim, so a
    chunk merged from several document items reports each of their locations.
    """
    provenance: list[Provenance] = []
    for doc_item in (meta or {}).get("doc_items") or []:
        for entry in doc_item.get("prov") or []:
            bbox = entry.get("bbox")
            charspan = entry.get("charspan")
            provenance.append(
                Provenance(
                    page_no=entry["page_no"],
                    bbox=BoundingBox.model_validate(bbox) if bbox is not None else None,
                    charspan=tuple(charspan) if isinstance(charspan, Sequence) and len(charspan) == 2 else None,
                )
            )
    return provenance
