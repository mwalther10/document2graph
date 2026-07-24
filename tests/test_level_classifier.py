from docling_core.types.doc.document import SectionHeaderItem, TextItem

from document2graph.document2graph_extractor.level_classifier import LevelClassifier
from document2graph.models.TextSnippet import TextSnippet


def header(idx: int, text: str, *line_heights: float, font: str | None = None,
           region: str = "body") -> TextSnippet:
    item = SectionHeaderItem(self_ref=f"#/texts/{idx}", orig=text, text=text)
    return TextSnippet(text_item=item, line_heights=list(line_heights), font_key=font, region=region)


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


# The guideline layout that motivated the style ranking: sections are set in a
# *smaller* face (F2/10.9) than the subsections they contain (F3/11.6), the opening
# sections carry no subsections at all, and the front matter shares the subsection
# font at a smaller size.
SECTION_FONT, SUB_FONT = "/F2", "/F3"


def _guideline_headers() -> dict[str, list[TextSnippet]]:
    idx = iter(range(100, 400))
    front = [header(next(idx), name, 9.5, font=SUB_FONT)
             for name in ("Autorinnen/Autoren", "Institute", "Bibliografie", "Korrespondenzadresse")]
    # the document opens with several sections that have no subsections whatsoever
    flat = [header(next(idx), name, 10.9, font=SECTION_FONT)
            for name in ("Vorbemerkung", "Definition", "Pathophysiologie", "Epidemiologie")]
    sections, subsections, subsubsections = [], [], []
    for name in ("Folgen für Mutter und Kind", "Screening und Diagnostik", "Therapie"):
        sections.append(header(next(idx), name, 10.9, font=SECTION_FONT))
        for i in range(3):
            subsections.append(header(next(idx), f"{name}: Unterabschnitt {i}", 11.6, font=SUB_FONT))
            subsubsections.append(header(next(idx), f"{name}: Detail {i}", 8.2, font=SECTION_FONT))
    ordered = [*front, *flat]
    for section, subs, details in zip(sections, _chunks(subsections, 3), _chunks(subsubsections, 3)):
        ordered.append(section)
        for sub, detail in zip(subs, details):
            ordered += [sub, detail]
    return {"front": front, "flat": flat, "sections": sections, "subsections": subsections,
            "subsubsections": subsubsections, "ordered": ordered}


def _chunks(items: list[TextSnippet], size: int) -> list[list[TextSnippet]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def test_sections_outrank_subsections_set_in_a_taller_font():
    """Enclosure beats size: the section font is smaller than the subsection font."""
    doc = _guideline_headers()
    lc = LevelClassifier([*doc["ordered"], *_bodies()])

    section_level = lc.classify(doc["sections"][0])[0]
    assert all(lc.classify(s)[0] == section_level for s in doc["sections"])
    assert all(lc.classify(s)[0] == section_level + 1 for s in doc["subsections"])
    assert all(lc.classify(s)[0] == section_level + 2 for s in doc["subsubsections"])


def test_opening_sections_without_subsections_do_not_sink_their_style():
    """The first sections of a document are often flat; that must not push the whole
    section style below the subsection style it encloses further down."""
    doc = _guideline_headers()
    lc = LevelClassifier([*doc["ordered"], *_bodies()])

    assert {lc.classify(s)[0] for s in doc["flat"]} == {lc.classify(s)[0] for s in doc["sections"]}


def test_front_matter_headings_never_become_parents_of_the_sections():
    """Front-matter headings run consecutively and enclose nothing, so they may not
    outrank the sections that follow them -- that made every section a child of
    "Korrespondenzadresse"."""
    doc = _guideline_headers()
    lc = LevelClassifier([*doc["ordered"], *_bodies()])

    section_level = lc.classify(doc["sections"][0])[0]
    assert all(lc.classify(s)[0] >= section_level for s in doc["front"])


def test_frequent_small_headings_stay_leaves():
    """A small recurring heading (a "recommendations" box) is interleaved with the
    subsections and so appears to enclose them, but its size keeps it a leaf."""
    idx = iter(range(200, 400))
    sections, boxes = [], []
    ordered = []
    for name in ("Erstes Kapitel", "Zweites Kapitel", "Drittes Kapitel"):
        section = header(next(idx), name, 10.9, font=SECTION_FONT)
        sections.append(section)
        ordered.append(section)
        for i in range(2):
            ordered.append(header(next(idx), f"{name}: Unterabschnitt {i}", 11.6, font=SUB_FONT))
            box = header(next(idx), "EMPFEHLUNGEN", 6.0, font="/F9")
            boxes.append(box)
            ordered.append(box)

    lc = LevelClassifier([*ordered, *_bodies()])

    assert all(lc.classify(b)[0] > lc.classify(sections[0])[0] + 1 for b in boxes)


def test_a_sidebar_heading_never_holds_a_section():
    """A boxed changelog whose two titles span the opening sections looks exactly like a
    chapter that contains them; its region keeps it out of the outline."""
    idx = iter(range(200, 400))
    changelog = header(next(idx), "INHALTLICHE NEUERUNGEN", 6.9, font="/F3", region="sidebar")
    sections = [header(next(idx), name, 10.9, font="/F4")
                for name in ("Definition", "Prävalenz", "Risikofaktoren")]
    second_box = header(next(idx), "RISIKOFAKTOREN", 6.9, font="/F3", region="sidebar")
    ordered = [changelog, *sections, second_box]

    lc = LevelClassifier([*ordered, *_bodies()])

    section_level = lc.classify(sections[0])[0]
    assert all(lc.classify(s)[0] == section_level for s in sections)
    assert all(lc.classify(box)[0] > section_level for box in (changelog, second_box))


def test_front_matter_headings_are_peers_of_the_sections():
    masthead = [header(i, name, 9.5, font="/F2", region="front_matter")
                for i, name in enumerate(("Autorinnen/Autoren", "Institute", "Bibliografie"))]
    sections = [header(10 + i, name, 10.9, font="/F4")
                for i, name in enumerate(("Vorbemerkung", "Definition", "Therapie"))]

    lc = LevelClassifier([*masthead, *sections, *_bodies()])

    assert {lc.classify(s)[0] for s in masthead} == {lc.classify(s)[0] for s in sections}


def test_a_style_that_starts_late_cannot_hold_the_sections():
    """A rare "Merke" note sees several sections between two of its own occurrences and
    so looks like their container. It starts after the sections do, which it could not
    if it held them."""
    idx = iter(range(300, 500))
    ordered, sections, notes = [], [], []
    for position in range(12):
        section = header(next(idx), f"Abschnitt {position}", 10.9, font="/F4")
        sections.append(section)
        ordered.append(section)
        if position in (3, 8):  # a note deep inside the document, spanning many sections
            note = header(next(idx), "Merke", 11.6, font="/F2")
            notes.append(note)
            ordered.append(note)

    lc = LevelClassifier([*ordered, *_bodies()])

    assert all(lc.classify(n)[0] > lc.classify(sections[0])[0] for n in notes)


def test_two_headings_cannot_overturn_a_size_step():
    """One wide-spanning pair of headings is a coincidence, not a chapter: without
    several sections of evidence the sizes decide."""
    idx = iter(range(400, 600))
    small_first = header(next(idx), "Kleiner Kasten", 7.0, font="/F3")
    sections = [header(next(idx), name, 11.0, font="/F2")
                for name in ("Definition", "Klassifikation", "Diagnostik")]
    small_last = header(next(idx), "Kleiner Kasten", 7.0, font="/F3")

    lc = LevelClassifier([small_first, *sections, small_last, *_bodies()])

    assert all(lc.classify(s)[0] < lc.classify(small_first)[0] for s in sections)


def test_title_level_is_reported_as_a_header_level():
    title = header(0, "My Title", 30.0)
    section_one = header(1, "Section One", 25.0)

    lc = LevelClassifier([title, section_one, *_bodies()], title="My Title")

    assert 0 in lc.header_levels()
