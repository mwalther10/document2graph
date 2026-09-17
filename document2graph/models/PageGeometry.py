"""The size of a page, kept so a stored bounding box can be placed on it."""

from pydantic import BaseModel


class PageGeometry(BaseModel):
    """The box of one page of the source PDF.

    Every ``bbox`` this package stores is a docling provenance box in PDF points,
    and a box in points says nothing about where it sits on the page without the
    page it belongs to. A consumer that has to normalize a node onto a rendered
    page -- a retrieval harness scoring bbox overlap against layout gold, an
    annotation tool drawing nodes over a page image -- needs both, so the page
    table travels with the graph rather than being recovered from the PDF, which
    a reloaded graph no longer has.

    These are docling's own page dimensions, taken from ``DoclingDocument.pages``
    rather than re-measured: they are the coordinate space the stored boxes are
    already expressed in. (They agree with what ``docling_parse`` reports for the
    same page, which is what graph construction measures typography against.)

    The origin is deliberately not recorded here. A page box is the same either
    way up, and each ``BoundingBox`` carries the ``coord_origin`` it was reported
    in; a second origin on the page would be a second source of truth for a
    question the box already answers.
    """

    page_no: int
    width: float
    height: float
