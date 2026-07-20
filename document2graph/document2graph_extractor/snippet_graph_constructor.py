from docling_core.types.doc.document import TextItem, PictureItem, TableItem, DoclingDocument # type: ignore
from docling_parse.pdf_parser import PdfDocument
from docling_core.types.doc.page import TextCellUnit
import regex as re
import networkx as nx
from collections import defaultdict
from ..models.TextSnippet import TextSnippet
from ..models.TextSnippetNode import TextSnippetNode
from ..models.ImageSnippetNode import ImageSnippetNode
from ..models.TableSnippetNode import TableSnippetNode
from ..models.DocumentMetadata import MetadataExtractionConfig
from ..models.EdgeWeightConfig import EdgeWeightConfig
from typing import Any, Callable, NamedTuple, TypeVar

from ..utils.edge_weight_lib import apply_relevancy_weights
from ..utils.log import Log
from .level_classifier import LevelClassifier
from .metadata_extractor import DocumentMetadataExtractor

T = TypeVar('T', bound=ImageSnippetNode)

ROOT_NODE_ID = "#/document-root"

# caption recovery: caption-like text ("Tab. 3", "Abb.1", "Figure 2", ...) adjacent
# to a media bbox is treated as its caption. Above/below captions may be up to
# MAX_CAPTION_DISTANCE away; margin captions (rotated, beside the media) must hug
# it within MAX_CAPTION_SIDE_GAP so the search never crosses a column gutter.
CAPTION_PATTERN = re.compile(r"^\s*(Tab|Abb|Table|Figure|Fig)\.?\s*\d+", re.IGNORECASE)
MAX_CAPTION_DISTANCE = 50.0
MAX_CAPTION_SIDE_GAP = 12.0

# decoration filter: pictures shorter than this (pt), or recurring at the same
# position on this many pages, are page decorations (logos, header art)
DECORATION_MAX_HEIGHT = 15.0
DECORATION_MIN_REPEATS = 3

SnippetNode = TextSnippetNode | ImageSnippetNode | TableSnippetNode
WeightedEdge = tuple[str, str, float]

class SnippetGraph(NamedTuple):
    """Result of get_graph: all snippet nodes, weighted edges, and the id of the root node.

    edges form the hierarchy tree (one parent per node); reference_edges link every
    text snippet that mentions a media item to that item and are not part of the tree.
    root_id is the snippet_id of the document title node, or the synthetic
    ROOT_NODE_ID (which is not part of the node lists) if no title node was found.
    """
    text_nodes: list[TextSnippetNode]
    image_nodes: list[ImageSnippetNode]
    table_nodes: list[TableSnippetNode]
    edges: list[WeightedEdge]
    reference_edges: list[WeightedEdge]
    root_id: str

class SnippetGraphConstructor():
    def __init__(self, pdf_doc: PdfDocument, docling_doc: DoclingDocument, filename: str, document_type: str, metadata_config: MetadataExtractionConfig | None = None, edge_weights: EdgeWeightConfig | None = None):
        self.pdf_doc = pdf_doc
        self.docling_doc = docling_doc
        self.edge_weights = edge_weights or EdgeWeightConfig()
        self.logger = Log("SnippetGraphConstructor").logger
        self.text_items = [TextSnippet(text_item=item, line_heights=self.add_line_heights(item)) for item in docling_doc.texts if item.label not in ("page_footer", "page_header")]
        self.table_items = docling_doc.tables
        self.image_items = self.filter_decorative_pictures(docling_doc.pictures)
        self._document_metadata = DocumentMetadataExtractor(self.text_items).extract(filename, document_type, metadata_config or MetadataExtractionConfig())
        self.levels = LevelClassifier(self.text_items, self._document_metadata.title)

    @property
    def document_metadata(self):
        return self._document_metadata

    def add_line_heights(self, snippet: TextItem) -> list[float]:
        lines = self.pdf_doc.get_page(snippet.prov[0].page_no).get_cells_in_bbox(cell_unit=TextCellUnit.LINE, bbox=snippet.prov[0].bbox)
        heights = []
        for line in lines:
            # check if line is in snippet bbox
            line_bbox = line.rect
            if line.text in snippet.text:
                heights.append(
                        ((line_bbox.r_y3 - line_bbox.r_y0) + (line_bbox.r_y2 - line_bbox.r_y1)) / 2
                    )
        return [round(h, 1) for h in heights]

    def filter_decorative_pictures(self, pictures: list[PictureItem]) -> list[PictureItem]:
        """Drop repeated page decorations (logos, header/footer art) that docling
        leaves in the body layer: tiny pictures, or pictures recurring at the
        same position on several pages. They carry no retrievable content and
        can never be caption-matched."""
        pages_by_position = defaultdict(set)
        for pic in pictures:
            pages_by_position[self._bbox_position_key(pic)].add(pic.prov[0].page_no)
        kept = []
        for pic in pictures:
            bbox = pic.prov[0].bbox
            is_tiny = abs(bbox.t - bbox.b) < DECORATION_MAX_HEIGHT
            is_repeated = len(pages_by_position[self._bbox_position_key(pic)]) >= DECORATION_MIN_REPEATS
            if is_tiny or is_repeated:
                self.logger.info(f"Dropping decorative picture {pic.self_ref} on page {pic.prov[0].page_no} (tiny={is_tiny}, repeated={is_repeated}).")
            else:
                kept.append(pic)
        return kept

    @staticmethod
    def _bbox_position_key(pic: PictureItem) -> tuple[int, int, int, int]:
        # ~5pt tolerance for position jitter of the same decoration between pages
        bbox = pic.prov[0].bbox
        return (round(bbox.l / 5), round(bbox.t / 5), round(bbox.r / 5), round(bbox.b / 5))

    def compute_parent_id(self, node: TextSnippetNode, nodes: list[TextSnippetNode]) -> str | None:
        if(node.is_grouped):
            # first case: if node is grouped, parent is the adjacent non-grouped node
            group = [n for n in nodes if n.docling_parent_ref.get_ref().cref == node.docling_parent_ref.get_ref().cref]
            first_bullet = min(group, key=lambda n: n.sequence_no)
            return nodes[first_bullet.sequence_no - 1].snippet_id if first_bullet.sequence_no > 0 else None

        elif(node.docling_parent_ref.get_ref().cref != "#/body"):
            # second case: if node has a meaningful parent_id from docling, use it
            parent_ref = node.docling_parent_ref.get_ref().cref
            return parent_ref
        # default case for text items with inaccurate parent_id from docling: find the closest preceding node with a lower level
        for prev_node in reversed(nodes[:node.sequence_no]):
            if prev_node.level < node.level:
                return prev_node.snippet_id
        # if that is not possible, connect to the document title node (level 0) if it exists
        return next((n.snippet_id for n in nodes if n.level == 0 and n.snippet_id != node.snippet_id), None)

    def compute_text_nodes(self, snippets: list[TextSnippet]) -> list[TextSnippetNode]:
        nodes = []
        for i, snippet in enumerate(snippets):
            if snippet.text_item.label in ("page_footer", "page_header"):
                self.logger.debug(f"Skipping text snippet {snippet.text_item.self_ref} with label {snippet.text_item.label}.")
                continue
            snippet_level, snippet_level_label = self.levels.classify(snippet)
            assert snippet.text_item.parent is not None, f"Text item {snippet.text_item.text} has no parent reference."
            node = TextSnippetNode(
                snippet_id=snippet.text_item.self_ref,
                document_id=self._document_metadata.document_id,
                label=snippet.text_item.label,
                sequence_no=i,
                level=snippet_level,
                level_label=snippet_level_label,
                parent_id=None,  # resolved below once all nodes exist
                docling_parent_ref=snippet.text_item.parent.get_ref(),
                docling_self_ref=snippet.text_item.get_ref(),
                is_grouped="group" in snippet.text_item.parent.get_ref().cref,
                text=snippet.text_item.text,
                bbox=snippet.text_item.prov[0].bbox,
                charspan=snippet.text_item.prov[0].charspan,
                page_no=snippet.text_item.prov[0].page_no
            )
            nodes.append(node)
        nodes = [node.model_copy(update={"parent_id": self.compute_parent_id(node, nodes)}) for node in nodes]
        for node in nodes:
            if node.parent_id is None:
                self.logger.debug(f"Node {node.text} has no parent_id assigned, it will attach to the document root.")
        return nodes

    def rb_image_parent_matching(self, images: list[T], text_nodes: list[TextSnippetNode]) -> list[T]:
        # find text node that contains reference to image or table using a rule-based approach
        ret_images = images.copy()
        for i, image in enumerate(ret_images):
            matched=False
            if image.caption_nodes is not None:
                # get the full caption text
                caption = " ".join([node.text for node in image.caption_nodes])
                caption_node_ids = [node.snippet_id for node in image.caption_nodes]
                rules = ["Figure", "Fig.", "Table", "Abbildung", "Tabelle", "Abb.", "Tab.", "Abb", "Tab", "Abb .", "Tab .", "Abb ", "Tab "]
                self.logger.debug(f"Image {image.docling_self_ref} caption: {caption}")
                for rule in rules:
                    if rule in caption:
                        # normalize the rule into a tolerant pattern (optional dot, flexible
                        # spacing); \b and (?!\d) keep it from matching inside words or
                        # matching "Tab.1" inside "Tab.10"
                        base = re.escape(rule.strip().rstrip(".").strip())
                        match = re.search(rf"\b{base}\s*\.?\s*(\d+)", caption, re.IGNORECASE)
                        if match:
                            mention_pattern = rf"\b{base}\s*\.?\s*{match.group(1)}(?!\d)"
                            # make sure we do not match caption nodes
                            mentions = [node for node in text_nodes if re.search(mention_pattern, node.text, re.IGNORECASE) and node.snippet_id not in caption_node_ids]
                            if mentions and mentions[0].parent_id is not None:
                                # the first mention determines parent and level (hierarchy tree);
                                # all mentions are kept for reference edges
                                self.logger.info(f"Image {image.snippet_id} matched to text node {mentions[0].snippet_id} using rule '{rule}' with pattern '{mention_pattern}' ({len(mentions)} mention(s)).")
                                ret_images[i] = image.model_copy(update={
                                    "level": mentions[0].level,
                                    "level_label": mentions[0].level_label,
                                    "parent_id": mentions[0].parent_id,
                                    "referencing_node_ids": [m.snippet_id for m in mentions],
                                })
                                matched=True
                                break
            if not matched:
                # no caption or no match: assign to closest preceding heading
                preceding_headers = [node for node in text_nodes if node.level in self.levels.header_levels() and node.sequence_no < image.sequence_no]
                if preceding_headers:
                    lowest_level_header = min(preceding_headers, key=lambda n: n.level)
                    ret_images[i] = image.model_copy(update={"level": lowest_level_header.level + 1, "level_label": "Unreferenced Image", "parent_id": lowest_level_header.snippet_id})

        return ret_images

    def _build_media_nodes(self, items: list[PictureItem] | list[TableItem], text_nodes: list[TextSnippetNode], node_cls: type[T], extra_fields: Callable[[Any], dict[str, Any]]) -> list[T]:
        """Shared builder for image and table nodes; parents and levels are
        resolved afterwards via rule-based caption matching."""
        nodes = []
        for i, item in enumerate(items):
            caption_nodes = self.get_caption_nodes(item, text_nodes)
            node = node_cls(
                snippet_id=item.self_ref,
                document_id=self._document_metadata.document_id,
                label=item.label,
                sequence_no=i,
                level=1000,  # placeholder, resolved by rb_image_parent_matching
                level_label="",
                parent_id=None,  # resolved by rb_image_parent_matching
                docling_parent_ref=item.parent.get_ref() if item.parent else None,
                docling_self_ref=item.get_ref(),
                is_grouped=item.parent is not None and "group" in item.parent.get_ref().cref,
                caption_nodes=caption_nodes,
                caption_text=" ".join([node.text for node in caption_nodes]) if len(caption_nodes) > 0 else "",
                bbox=item.prov[0].bbox,
                page_no=item.prov[0].page_no,
                **extra_fields(item),
            )
            nodes.append(node)
        # map parents and levels using rule-based matching
        return self.rb_image_parent_matching(nodes, text_nodes)

    def compute_image_nodes(self, images: list[PictureItem], text_nodes: list[TextSnippetNode]) -> list[ImageSnippetNode]:
        return self._build_media_nodes(images, text_nodes, ImageSnippetNode, lambda image: {})

    def compute_table_nodes(self, tables: list[TableItem], text_nodes: list[TextSnippetNode]) -> list[TableSnippetNode]:
        return self._build_media_nodes(tables, text_nodes, TableSnippetNode, lambda table: {
            "markdown_serialization": table.export_to_markdown(self.docling_doc),
            "html_serialization": table.export_to_html(self.docling_doc, add_caption=True),
        })

    def get_caption_nodes(self, image: PictureItem | TableItem, text_nodes: list[TextSnippetNode]) -> list[TextSnippetNode] | list:
        # find text node that contains reference to image or table
        caption_refs = image.captions if image.captions else []
        caption_nodes = []
        for caption_ref in caption_refs:
            for text_node in text_nodes:
                if caption_ref.get_ref().cref == text_node.docling_self_ref.get_ref().cref:
                    caption_nodes.append(text_node)
        if not caption_nodes:
            # docling often detects the caption but leaves it as a plain text/list
            # item without linking it to the media item: recover it spatially
            recovered = self.recover_caption_node(image, text_nodes)
            if recovered is not None:
                caption_nodes.append(recovered)
        return caption_nodes

    def recover_caption_node(self, image: PictureItem | TableItem, text_nodes: list[TextSnippetNode]) -> TextSnippetNode | None:
        """Find an unlinked caption for a media item: caption-like text on the same
        page, vertically adjacent and horizontally overlapping (same column)."""
        bbox = image.prov[0].bbox
        page_no = image.prov[0].page_no
        candidates = []
        for node in text_nodes:
            if node.page_no != page_no or not CAPTION_PATTERN.match(node.text):
                continue
            # gap per axis between the two bboxes, 0 if they overlap on that axis
            h_gap = max(bbox.l - node.bbox.r, node.bbox.l - bbox.r, 0)
            v_gap = max(bbox.b - node.bbox.t, node.bbox.b - bbox.t, 0)
            above_or_below = h_gap == 0 and v_gap <= MAX_CAPTION_DISTANCE
            in_margin = v_gap == 0 and h_gap <= MAX_CAPTION_SIDE_GAP
            if above_or_below or in_margin:
                candidates.append((h_gap + v_gap, node))
        if not candidates:
            return None
        distance, best = min(candidates, key=lambda c: c[0])
        self.logger.info(f"Recovered caption {best.snippet_id} ({best.text[:60]!r}) for {image.self_ref} at distance {distance:.1f}pt.")
        return best

    def write_nx_graph(self, filename: str, text_nodes: list[TextSnippetNode], image_nodes: list[ImageSnippetNode], table_nodes: list[TableSnippetNode], edges: list[WeightedEdge], root_id: str = ROOT_NODE_ID, reference_edges: list[WeightedEdge] | None = None):
        nx_graph = self.build_nx_graph(text_nodes, image_nodes, table_nodes, edges, root_id, reference_edges)
        nx.write_gexf(nx_graph, filename, version="1.3")
        return nx_graph

    def build_nx_graph(self, text_nodes: list[TextSnippetNode], image_nodes: list[ImageSnippetNode], table_nodes: list[TableSnippetNode], edges: list[WeightedEdge], root_id: str = ROOT_NODE_ID, reference_edges: list[WeightedEdge] | None = None) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        if root_id == ROOT_NODE_ID:
            # synthetic root is not part of the node lists, add it explicitly
            nx_graph.add_node(ROOT_NODE_ID, label="document_root", level=-1, level_label="Root", text=self._document_metadata.title or "")
        for node in text_nodes:
            nx_graph.add_node(node.snippet_id, label=node.label, level=node.level, level_label=node.level_label, text=node.text)
        for node in image_nodes + table_nodes:
            nx_graph.add_node(node.snippet_id, label=node.label, level=node.level, level_label=node.level_label, text=node.caption_text)
        for parent_id, child_id, weight in edges:
            nx_graph.add_edge(parent_id, child_id, weight=weight, edge_type="hierarchy")
        for source_id, target_id, weight in reference_edges or []:
            # DiGraph holds one edge per node pair: never overwrite a hierarchy edge
            if not nx_graph.has_edge(source_id, target_id):
                nx_graph.add_edge(source_id, target_id, weight=weight, edge_type="reference")
        return nx_graph

    def construct_snippet_nodes(self) -> tuple[list[TextSnippetNode], list[ImageSnippetNode], list[TableSnippetNode]]:
        text_nodes = self.compute_text_nodes(self.text_items)
        image_nodes = self.compute_image_nodes(self.image_items, text_nodes)
        table_nodes = self.compute_table_nodes(self.table_items, text_nodes) 
        return text_nodes, image_nodes, table_nodes

    def resolve_root_id(self, text_nodes: list[TextSnippetNode]) -> str:
        """Return the snippet_id of the document title node to use as graph root.

        Prefers the node whose text matches the extracted document title, then any
        node labeled "Title"; falls back to the synthetic ROOT_NODE_ID if neither exists.
        """
        title = (self._document_metadata.title or "").strip().lower()
        if title:
            for node in text_nodes:
                if node.text.strip().lower() == title:
                    return node.snippet_id
        for node in text_nodes:
            if node.level_label == "Title":
                return node.snippet_id
        return ROOT_NODE_ID

    def compute_edge_weight(self, parent: SnippetNode | None, child: SnippetNode) -> float:
        weights = self.edge_weights
        if isinstance(child, (ImageSnippetNode, TableSnippetNode)):
            if child.level_label == "Unreferenced Image":
                return weights.unreferenced_media
            return weights.media
        if child.is_grouped:
            return weights.list_item
        if child.level_label in ("Title", "Heading") and parent is not None and parent.level_label in ("Title", "Heading"):
            return weights.section
        return weights.text

    def construct_snippet_edges(self, nodes: list[SnippetNode], root_id: str) -> list[WeightedEdge]:
        """Build weighted parent->child edges; nodes without a resolvable parent attach to the document root."""
        node_by_id = {node.snippet_id: node for node in nodes}
        edges = []
        for node in nodes:
            if node.snippet_id == root_id:
                continue  # the root node has no parent
            parent = node_by_id.get(node.parent_id) if node.parent_id else None
            if parent is not None and parent.snippet_id != node.snippet_id:
                edges.append((parent.snippet_id, node.snippet_id, self.compute_edge_weight(parent, node)))
            else:
                # missing or dangling parent_id: attach to the document root
                edges.append((root_id, node.snippet_id, self.edge_weights.root))
        return edges

    def connect_components(self, nodes: list[SnippetNode], edges: list[WeightedEdge], root_id: str) -> list[WeightedEdge]:
        """Attach every weakly connected component that does not reach the document root to it.

        construct_snippet_edges already links orphan nodes to the root, but parent
        cycles can still leave a component detached; this guarantees a single
        connected graph per document.
        """
        graph = nx.DiGraph()
        graph.add_node(root_id)
        graph.add_nodes_from(node.snippet_id for node in nodes)
        graph.add_edges_from((p, c) for p, c, _ in edges)
        node_by_id = {node.snippet_id: node for node in nodes}
        extra_edges = []
        for component in nx.weakly_connected_components(graph):
            if root_id in component:
                continue
            component_nodes = [node_by_id[n] for n in component if n in node_by_id]
            if not component_nodes:
                continue
            # attach the structurally highest node of the component to the root
            top = min(component_nodes, key=lambda n: (n.level, n.sequence_no))
            extra_edges.append((root_id, top.snippet_id, self.edge_weights.root))
        return extra_edges

    def construct_reference_edges(self, media_nodes: list[ImageSnippetNode | TableSnippetNode]) -> list[WeightedEdge]:
        """One edge per mentioning text snippet -> media item.

        These complement the hierarchy tree (where only the first mention
        determines the parent) and are kept separate from it: they may form
        cycles with tree edges and must not affect connectivity handling.
        """
        edges = []
        for node in media_nodes:
            for referencing_id in node.referencing_node_ids:
                edges.append((referencing_id, node.snippet_id, self.edge_weights.reference))
        return edges

    def node_texts(self, nodes: list[SnippetNode], root_id: str) -> dict[str, str]:
        """Text content per node id, used for content-based relevancy weights."""
        texts = {}
        if root_id == ROOT_NODE_ID:
            # synthetic root is not part of the node lists
            texts[ROOT_NODE_ID] = self._document_metadata.title or ""
        for node in nodes:
            if isinstance(node, TableSnippetNode):
                texts[node.snippet_id] = f"{node.caption_text}\n{node.markdown_serialization}".strip()
            elif isinstance(node, ImageSnippetNode):
                texts[node.snippet_id] = node.caption_text
            else:
                texts[node.snippet_id] = node.text
        return texts

    def get_graph(self, save_to="") -> SnippetGraph:
        text_nodes, image_nodes, table_nodes = self.construct_snippet_nodes()
        all_nodes: list[SnippetNode] = text_nodes + image_nodes + table_nodes
        root_id = self.resolve_root_id(text_nodes)
        edges = self.construct_snippet_edges(all_nodes, root_id)
        edges += self.connect_components(all_nodes, edges, root_id)
        reference_edges = self.construct_reference_edges(image_nodes + table_nodes)
        if self.edge_weights.relevancy.enabled:
            # apply in one pass so both edge kinds share the same score normalization
            combined = apply_relevancy_weights(edges + reference_edges, self.node_texts(all_nodes, root_id), self.edge_weights.relevancy)
            edges, reference_edges = combined[:len(edges)], combined[len(edges):]
        if save_to != "":
            self.write_nx_graph(save_to, text_nodes, image_nodes, table_nodes, edges, root_id, reference_edges)
        return SnippetGraph(text_nodes, image_nodes, table_nodes, edges, reference_edges, root_id)