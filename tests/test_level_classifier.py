from docling_core.types.doc.document import SectionHeaderItem, TextItem

from document2graph.document2graph_extractor.level_classifier import LevelClassifier
from document2graph.models.TextSnippet import TextSnippet


def header(idx: int, text: str, *line_heights: float) -> TextSnippet:
    item = SectionHeaderItem(self_ref=f"#/texts/{idx}", orig=text, text=text)
    return TextSnippet(text_item=item, line_heights=list(line_heights))


def body(idx: int, text: str, height: float) -> TextSnippet:
    item = TextItem(self_ref=f"#/texts/{idx}", label="text", orig=text, text=text)
    return TextSnippet(text_item=item, line_heights=[height])


def _bodies(height: float = 10.0) -> list[TextSnippet]:
    # enough body text that the body font height dominates the median
    return [body(i, f"body paragraph {i}", height) for i in range(10, 18)]


def test_title_gets_level_0_and_next_tier_gets_level_1_when_title_height_unique():
    """Title height larger than every section header (unique).

    The title takes level 0 and the largest real section headers must be level 1,
    not level 2 -- the title's own font height must not consume a header tier.
    """
    title = header(0, "My Title", 30.0)  # unique, taller than every header
    section_one = header(1, "Section One", 25.0)
    subsection = header(2, "Subsection", 20.0)

    lc = LevelClassifier([title, section_one, subsection, *_bodies()], title="My Title")

    assert lc.classify(title) == (0, "Title")
    assert lc.classify(section_one) == (1, "Heading")
    assert lc.classify(subsection) == (2, "Heading")


def test_title_gets_level_0_and_next_tier_gets_level_1_when_title_height_shared():
    """Title height shared with the largest section headers.

    The title still takes level 0, and a header at the title's own font height is
    the first tier below it (level 1).
    """
    title = header(0, "My Title", 30.0)
    section_one = header(1, "Section One", 30.0)  # same height as the title
    subsection = header(2, "Subsection", 20.0)

    lc = LevelClassifier([title, section_one, subsection, *_bodies()], title="My Title")

    assert lc.classify(title) == (0, "Title")
    assert lc.classify(section_one) == (1, "Heading")
    assert lc.classify(subsection) == (2, "Heading")


def test_no_gap_in_header_levels_when_title_wraps_to_multiple_line_heights():
    """The title's own font height must never open a header tier of its own.

    A wrapping title has several line heights whose median matches none of them;
    if those heights were ranked as headers they would create levels that
    classify() never assigns (the title always goes to level 0), leaving a gap.
    Real section headers must occupy a contiguous range starting at level 1.
    """
    title = header(0, "A Very Long Wrapping Document Title", 32.0, 30.0)
    section_a = header(1, "Section A", 24.0)
    section_b = header(2, "Section B", 20.0)
    section_c = header(3, "Section C", 16.0)
    headers = [title, section_a, section_b, section_c]
    # plenty of body lines so the header heights sit clearly above the body band
    bodies = [body(i, f"body paragraph {i}", 10.0) for i in range(10, 40)]

    lc = LevelClassifier([*headers, *bodies], title=title.text_item.text)

    realized = sorted({lc.classify(s)[0] for s in headers})
    assert realized == [0, 1, 2, 3]  # contiguous, no phantom levels
    assert lc.classify(title)[1] == "Title"
    assert lc.classify(section_a) == (1, "Heading")


def test_title_level_is_reported_as_a_header_level():
    title = header(0, "My Title", 30.0)
    section_one = header(1, "Section One", 25.0)

    lc = LevelClassifier([title, section_one, *_bodies()], title="My Title")

    assert 0 in lc.header_levels()
