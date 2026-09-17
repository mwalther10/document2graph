from typing import Literal

from pydantic import BaseModel

# how hierarchy levels are derived from the parsed document
LevelSource = Literal["hybrid", "typography", "docling"]


class PipelineFlags(BaseModel):
    """Switches for the individual repair steps of graph construction.

    Every flag defaults to the full pipeline, so leaving this alone changes
    nothing. They exist to ablate one step at a time: each is applied *after*
    parsing, to the docling document, so a sweep over many conditions reads the
    docling cache instead of re-parsing the corpus.
    """

    # sort snippets by page block instead of trusting docling's reading order
    repair_reading_order: bool = True
    # merge a paragraph split across a column or page break back into one snippet
    stitch_continuations: bool = True
    # merge the line-by-line fragments docling emits for a figure, a boxed sidebar
    # or the masthead back into one snippet per text block. Needs assign_regions:
    # running text is never merged this way, only the regions that are set line by line
    merge_line_fragments: bool = True
    # merge a table continued on the next page back into one table item
    merge_table_continuations: bool = True
    # label snippets body / front_matter / sidebar / figure; off means all body,
    # so the masthead and boxed sidebars take part in the document outline
    assign_regions: bool = True
    # find the caption of a figure or table docling left unlinked
    recover_captions: bool = True
    # drop logos, rules and other pictures that carry no content
    filter_decorative_pictures: bool = True

    # how heading levels are ranked:
    # "hybrid"     - (font, height) styles ordered by how they nest in reading order
    # "typography" - font height alone, no nesting evidence
    # "docling"    - docling's own SectionHeaderItem.level
    level_source: LevelSource = "hybrid"
