from pydantic import BaseModel


class EdgeWeightConfig(BaseModel):
    """Weights assigned to graph edges by structural category.

    Higher values indicate a stronger structural connection between
    parent and child node. All weights are stored on the edge as the
    ``weight`` attribute (both in the returned edge list and in the
    exported GEXF graph).
    """

    section: float = 1.0            # heading -> subheading
    text: float = 0.8               # heading/body -> body text, subtext, footnotes
    list_item: float = 0.6          # anchor text -> grouped list item (bullets)
    media: float = 0.9              # referencing text -> image/table (matched via caption)
    unreferenced_media: float = 0.3 # fallback heading -> image/table without a caption match
    root: float = 0.1               # synthetic document root -> otherwise disconnected node
