from docling_core.types.doc.document import TextItem, SectionHeaderItem, PictureItem, TableItem, RefItem, DoclingDocument
from docling_parse.pdf_parser import PdfDocument
from docling_core.types.doc.page import TextCellUnit
import numpy as np
from collections import defaultdict
import regex as re
import networkx as nx
from ..models.TextSnippet import TextSnippet
from ..models.TextSnippetNode import TextSnippetNode
from ..models.ImageSnippetNode import ImageSnippetNode
from ..models.TableSnippetNode import TableSnippetNode
from ..models.Document import Document
from ..models.DocumentMetadata import DocumentMetadata, MetadataExtractionConfig
from ..models.EdgeWeightConfig import EdgeWeightConfig
import uuid
from typing import TypeVar

from ..utils.doc_to_tree_lib import get_heights, get_height_hist, get_text_body_height, compute_snippet_level
from ..utils.edge_weight_lib import apply_relevancy_weights

T = TypeVar('T', bound=ImageSnippetNode)

ROOT_NODE_ID = "#/document-root"

SnippetNode = TextSnippetNode | ImageSnippetNode | TableSnippetNode
WeightedEdge = tuple[str, str, float]

class SnippetGraphConstructor():

    def __init__(self, pdf_doc: PdfDocument, docling_doc: DoclingDocument, filename: str, document_type: str, metadata_config: MetadataExtractionConfig | None = None, edge_weights: EdgeWeightConfig | None = None):
        self.pdf_doc = pdf_doc
        self.docling_doc = docling_doc
        self.edge_weights = edge_weights or EdgeWeightConfig()
        self.text_items = [TextSnippet(text_item=item, line_heights=self.add_line_heights(item)) for item in docling_doc.texts]
        self.table_items = docling_doc.tables
        self.image_items = docling_doc.pictures
        self.heights = get_heights(self.text_items, skip_first_page=True)
        self._document_metadata = self.extract_doc_metadata(filename, document_type, metadata_config or MetadataExtractionConfig())
        self.section_header_levels, self.section_level_to_label, self.text_body_levels, self.text_level_to_label = self.compute_doc_levels()  # compute document levels once and pass to methods that need it
    
    # document metadata extraction methods can be added here
    def __safe_check_substring(self, search_string: str, item: TextItem) -> bool:
        try:
            return search_string.lower() in item.text.lower()
        except Exception as e:
            print(f"Error checking item text: {e}")
            return False    

    def __match_metadata_by_section_header(self, section_header: SectionHeaderItem | TextItem, texts: list[TextItem]) -> RefItem | None:
        bottom = section_header.prov[0].bbox.b
        left = section_header.prov[0].bbox.l

        for text in texts:
            text_top = text.prov[0].bbox.t
            text_left = text.prov[0].bbox.l
            if (abs(bottom - text_top) < 6) and (abs(left - text_left) < 6) and text.parent is not None and section_header.parent is not None:
                if text.parent.get_ref() == section_header.parent.get_ref():
                    return text.get_ref()
                if text.parent.get_ref() != section_header.parent.get_ref():
                    return text.parent.get_ref()
        return None
    
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

    def compute_section_header_levels(self) -> tuple[dict[int, int], dict[int, str]]:
        all_heights = self.heights
        if not all_heights.size:
            return {}, {}
        section_headers = [s for s in self.text_items if isinstance(s.text_item, SectionHeaderItem)]
        section_header_height_hist = get_height_hist(get_heights(section_headers, skip_first_page=True))
        unique_section_header_heights = sorted(section_header_height_hist.keys(), reverse=True)
        text_body_height, iqr = get_text_body_height(all_heights)

        # group levels that are very close to each other on text body level (within iqr)
        level_dict = defaultdict(int)
        level_to_label = defaultdict(str)
        level = 0
        for h in unique_section_header_heights:
            if(h >= text_body_height - iqr and h <= text_body_height + iqr):
                level_dict[int(h)] = level  # body text headers, usually just bold printed text
                level_to_label[level] = "Heading"
            else:
                level_dict[int(h)] = level
                level_to_label[level] = "Heading"
                level += 1

        # check if largest height is unique -> title
        if len(unique_section_header_heights) > 0 and (section_header_height_hist[unique_section_header_heights[0]] == 1):
            level_to_label[0] = "Title"
        
        return level_dict, level_to_label

    def compute_text_body_levels(self, num_section_headers: int) -> tuple[dict[int, int], dict[int, str]]:
        all_heights = self.heights
        if not all_heights.size:
            return {}, {}
        
        texts = [s for s in self.text_items if not isinstance(s.text_item, SectionHeaderItem)]

        text_body_heights = get_heights(texts, skip_first_page=True)
        text_body_height_hist = get_height_hist(text_body_heights)
        unique_text_body_heights = sorted(text_body_height_hist.keys(), reverse=True)
        text_body_height, iqr = get_text_body_height(all_heights)
        subtext_heights = [h for h in all_heights if h < text_body_height - iqr]
        subtext_median = np.percentile(subtext_heights, 50) if subtext_heights else 0

        
        level_dict = defaultdict(int)
        level_to_label = defaultdict(str)
        level = num_section_headers  # continue after section header levels
        subtexts = [h for h in unique_text_body_heights if h < text_body_height - iqr]

        for h in unique_text_body_heights:
            if(h >= text_body_height - iqr):
                level_dict[int(h)] = level  # body text
                level_to_label[ level ] = "Body" 
            else:
                level += 1
                break
        
        for h in subtexts:
            if(h >= subtext_median - iqr and h <= subtext_median + iqr):
                level_dict[int(h)] = level  # subtext
                level_to_label[ level ] = "Subtext" 
            else:
                level += 1
                level_dict[int(h)] = level  # smaller subtext, footnotes, etc.
                level_to_label[ level ] = "Subsubtext"

        return level_dict, level_to_label

    def compute_doc_levels(self) -> tuple[dict[int, int], dict[int, str], dict[int, int], dict[int, str]]:
        # section headers
        section_header_levels, section_level_to_label = self.compute_section_header_levels()

        # text body levels
        text_body_levels, text_level_to_label = self.compute_text_body_levels(num_section_headers=max(section_header_levels.values())+1)

        return section_header_levels, section_level_to_label, text_body_levels, text_level_to_label
    
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
        return None

    def compute_text_nodes(self, snippets: list[TextSnippet]) -> list[TextSnippetNode]:
        nodes = []
        for i, snippet in enumerate(snippets):
            if isinstance(snippet.text_item, SectionHeaderItem):
                snippet_level = compute_snippet_level(snippet, self.section_header_levels)
                snippet_level_label = self.section_level_to_label.get(snippet_level, "unknown")
            else:
                snippet_level = compute_snippet_level(snippet, self.text_body_levels)
                snippet_level_label = self.text_level_to_label.get(snippet_level, "unkown")
            assert snippet.text_item.parent is not None, f"Text item {snippet.text_item.text} has no parent reference."
            node = TextSnippetNode(
                snippet_id=snippet.text_item.self_ref,
                document_id=self._document_metadata.document_id,
                label=snippet.text_item.label,
                sequence_no=i,
                level=snippet_level,
                level_label=snippet_level_label,
                parent_id="placeholder", # compute later
                docling_parent_ref=snippet.text_item.parent.get_ref(),
                docling_self_ref=snippet.text_item.get_ref(),
                is_grouped=True if "group" in snippet.text_item.parent.get_ref().cref else False,
                text=snippet.text_item.text,
                bbox=snippet.text_item.prov[0].bbox,
                charspan=snippet.text_item.prov[0].charspan,
                page_no=snippet.text_item.prov[0].page_no
            )
            nodes.append(node)
        nodes = [node.model_copy(update={"parent_id": self.compute_parent_id(node, nodes)}) for node in nodes]
        for node in nodes:
            if node.parent_id is None:
                print(f"Node {node.text} has no parent_id assigned.")
            elif node.parent_id == "placeholder":
                print(f"Node {node.text} has placeholder parent_id.")
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
                rules = ["Figure", "Fig.", "Table", "Abbildung", "Tabelle", "Abb.", "Tab."]
                for rule in rules:
                    if rule in caption:
                        regex_pattern = rf"{rule}\s*\d+"
                        match = re.search(regex_pattern, caption, re.IGNORECASE)
                        if match:
                            matched_pattern = rf"{match.group(0)}"
                            # make sure we do not match caption nodes
                            text_item = [node for node in text_nodes if re.search(matched_pattern, node.text, re.IGNORECASE) and node.snippet_id not in caption_node_ids]
                            if text_item and text_item[0].parent_id is not None:
                                # we assign the level and parent_id of the matched text item to the image
                                # the first matched text item is taken
                                ret_images[i] = image.model_copy(update={"level": text_item[0].level, "level_label": text_item[0].level_label, "parent_id": text_item[0].parent_id})  
                                matched=True
                                break
            if not matched:
                # no caption or no match: assign to closest preceding heading
                preceding_headers = [node for node in text_nodes if node.level in self.section_header_levels.values() and node.sequence_no < image.sequence_no]
                if preceding_headers:
                    lowest_level_header = min(preceding_headers, key=lambda n: n.level)
                    ret_images[i] = image.model_copy(update={"level": lowest_level_header.level + 1, "level_label": "Unreferenced Image", "parent_id": lowest_level_header.snippet_id})
                    
        return ret_images

    def compute_image_nodes(self, images: list[PictureItem], text_nodes: list[TextSnippetNode]) -> list[ImageSnippetNode]:
        image_nodes = []
        for i, image in enumerate(images):
            caption_nodes = self.get_caption_nodes(image, text_nodes)
            node = ImageSnippetNode(
                snippet_id=image.self_ref,
                document_id=self._document_metadata.document_id,
                label=image.label,
                sequence_no=i,
                level=1000,  # placeholder, compute later
                level_label="",
                parent_id="placeholder", # compute later
                docling_parent_ref=image.parent.get_ref() if image.parent else None,
                docling_self_ref=image.get_ref(),
                is_grouped=True if image.parent is not None and "group" in image.parent.get_ref().cref else False,
                caption_nodes=caption_nodes,
                caption_text=" ".join([node.text for node in caption_nodes]) if len(caption_nodes) > 0 else "",
                bbox=image.prov[0].bbox,
                page_no=image.prov[0].page_no
            )
            image_nodes.append(node)
        # map parents and levels using rule-based matching
        ret_images = self.rb_image_parent_matching(image_nodes, text_nodes)
        return ret_images
    
    def compute_table_nodes(self, tables: list[TableItem], text_nodes: list[TextSnippetNode]) -> list[TableSnippetNode]:
        # for now, treat tables similar to images in terms of parent matching
        table_nodes = []
        for i, table in enumerate(tables):
            caption_nodes = self.get_caption_nodes(table, text_nodes)
            node = TableSnippetNode(
                snippet_id=table.self_ref,
                document_id=self._document_metadata.document_id,
                label=table.label,
                sequence_no=i,
                level=1000,  # placeholder, compute later
                level_label="",
                parent_id="placeholder", # compute later
                docling_parent_ref=table.parent.get_ref() if table.parent else None,
                docling_self_ref=table.get_ref(),
                is_grouped=True if table.parent is not None and "group" in table.parent.get_ref().cref else False,
                caption_nodes=caption_nodes,
                caption_text=" ".join([node.text for node in caption_nodes]) if len(caption_nodes) > 0 else "",
                markdown_serialization=table.export_to_markdown(self.docling_doc),
                bbox=table.prov[0].bbox,
                page_no=table.prov[0].page_no
            )
            table_nodes.append(node)
        # map parents and levels using rule-based matching
        ret_tables = self.rb_image_parent_matching(table_nodes, text_nodes)
        return ret_tables
    
    def get_caption_nodes(self, image: PictureItem | TableItem, text_nodes: list[TextSnippetNode]) -> list[TextSnippetNode] | list:
        # find text node that contains reference to image or table
        caption_refs = image.captions if image.captions else []
        caption_nodes = []
        for caption_ref in caption_refs:
            for text_node in text_nodes:
                if caption_ref.get_ref().cref == text_node.docling_self_ref.get_ref().cref:
                    caption_nodes.append(text_node)
        return caption_nodes

    def get_node_by_id(self, node_id: str, nodes: list[TextSnippetNode]) -> TextSnippetNode | None:
        for node in nodes:
            if node.snippet_id == node_id:
                return node
        return None
    
    def write_nx_graph(self, filename: str, text_nodes: list[TextSnippetNode], image_nodes: list[ImageSnippetNode], table_nodes: list[TableSnippetNode], edges: list[WeightedEdge]):
        nx_graph = self.build_nx_graph(text_nodes, image_nodes, table_nodes, edges)
        nx.write_gexf(nx_graph, filename, version="1.3")
        return nx_graph

    def build_nx_graph(self, text_nodes: list[TextSnippetNode], image_nodes: list[ImageSnippetNode], table_nodes: list[TableSnippetNode], edges: list[WeightedEdge]) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        nx_graph.add_node(ROOT_NODE_ID, label="document_root", level=-1, level_label="Root", text=self._document_metadata.title)
        for node in text_nodes:
            nx_graph.add_node(node.snippet_id, label=node.label, level=node.level, level_label=node.level_label, text=node.text)
        for node in image_nodes + table_nodes:
            nx_graph.add_node(node.snippet_id, label=node.label, level=node.level, level_label=node.level_label, text=node.caption_text)
        for parent_id, child_id, weight in edges:
            nx_graph.add_edge(parent_id, child_id, weight=weight)
        return nx_graph

    def construct_snippet_nodes(self) -> tuple[list[TextSnippetNode], list[ImageSnippetNode], list[TableSnippetNode]]:
        # we ignore the first page
        text_nodes = self.compute_text_nodes(self.text_items)
        image_nodes = self.compute_image_nodes(self.image_items, text_nodes)
        table_nodes = self.compute_table_nodes(self.table_items, text_nodes)  # add markdownserialization to caption text for better matching
        return text_nodes, image_nodes, table_nodes

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

    def construct_snippet_edges(self, nodes: list[SnippetNode]) -> list[WeightedEdge]:
        """Build weighted parent->child edges; nodes without a resolvable parent attach to the document root."""
        node_by_id = {node.snippet_id: node for node in nodes}
        edges = []
        for node in nodes:
            parent = node_by_id.get(node.parent_id) if node.parent_id else None
            if parent is not None:
                edges.append((parent.snippet_id, node.snippet_id, self.compute_edge_weight(parent, node)))
            else:
                # missing or dangling parent_id: attach to the synthetic document root
                edges.append((ROOT_NODE_ID, node.snippet_id, self.edge_weights.root))
        return edges

    def connect_components(self, nodes: list[SnippetNode], edges: list[WeightedEdge]) -> list[WeightedEdge]:
        """Attach every weakly connected component that does not reach the document root to it.

        construct_snippet_edges already links orphan nodes to the root, but parent
        cycles can still leave a component detached; this guarantees a single
        connected graph per document.
        """
        graph = nx.DiGraph()
        graph.add_node(ROOT_NODE_ID)
        graph.add_nodes_from(node.snippet_id for node in nodes)
        graph.add_edges_from((p, c) for p, c, _ in edges)
        node_by_id = {node.snippet_id: node for node in nodes}
        extra_edges = []
        for component in nx.weakly_connected_components(graph):
            if ROOT_NODE_ID in component:
                continue
            component_nodes = [node_by_id[n] for n in component if n in node_by_id]
            if not component_nodes:
                continue
            # attach the structurally highest node of the component to the root
            top = min(component_nodes, key=lambda n: (n.level, n.sequence_no))
            extra_edges.append((ROOT_NODE_ID, top.snippet_id, self.edge_weights.root))
        return extra_edges

    def node_texts(self, nodes: list[SnippetNode]) -> dict[str, str]:
        """Text content per node id, used for content-based relevancy weights."""
        texts = {ROOT_NODE_ID: self._document_metadata.title or ""}
        for node in nodes:
            if isinstance(node, TableSnippetNode):
                texts[node.snippet_id] = f"{node.caption_text}\n{node.markdown_serialization}".strip()
            elif isinstance(node, ImageSnippetNode):
                texts[node.snippet_id] = node.caption_text
            else:
                texts[node.snippet_id] = node.text
        return texts

    def get_graph(self, save_to="") -> tuple[list[TextSnippetNode], list[ImageSnippetNode], list[TableSnippetNode], list[WeightedEdge]]:
        text_nodes, image_nodes, table_nodes = self.construct_snippet_nodes()
        all_nodes: list[SnippetNode] = text_nodes + image_nodes + table_nodes
        edges = self.construct_snippet_edges(all_nodes)
        edges += self.connect_components(all_nodes, edges)
        if self.edge_weights.relevancy.enabled:
            edges = apply_relevancy_weights(edges, self.node_texts(all_nodes), self.edge_weights.relevancy)
        if save_to != "":
            self.write_nx_graph(save_to, text_nodes, image_nodes, table_nodes, edges)
        return text_nodes, image_nodes, table_nodes, edges
    
    def extract_title(self, page_two_snippets: list[TextSnippet]) -> str:
        # condition: page 2 and largest font size and on top of page
        title = max(page_two_snippets, key=lambda s: s.line_heights[0] if s.line_heights else 0)
        for snippet in page_two_snippets:
            if snippet.line_heights and snippet.line_heights[0] >= title.line_heights[0] and snippet.text_item.prov[0].bbox.t < title.text_item.prov[0].bbox.t:
                title = snippet
        return title.text_item.text if title else ""

    def extract_doc_metadata(self, filename: str, document_type: str, config: MetadataExtractionConfig) -> Document:
        metadata_values: dict[str, str] = {}
        for field_name in ["version", "authors", "institutions", "bibliography", "correspondence"]:
            field_cfg = getattr(config, field_name)
            if field_cfg is None:
                metadata_values[field_name] = ""
                continue
            page_items = [item.text_item for item in self.text_items
                          if field_cfg.pages[0] <= item.text_item.prov[0].page_no <= field_cfg.pages[1]]
            matched_ref = None
            for text_item in page_items:
                if self.__safe_check_substring(field_cfg.label, text_item):
                    matched_ref = self.__match_metadata_by_section_header(text_item, page_items)
                    if matched_ref is None:
                        matched_ref = text_item.get_ref()
                    break
            metadata_values[field_name] = matched_ref.cref if matched_ref else ""

        title_page_snippets = [item for item in self.text_items
                               if item.text_item.prov[0].page_no == config.title_page]
        title = self.extract_title(title_page_snippets) if title_page_snippets else ""

        return Document(
            document_id=str(uuid.uuid4()),
            title=title,
            document_type=document_type,
            filename=filename,
            metadata=DocumentMetadata(**metadata_values),
        )
