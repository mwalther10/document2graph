from pydantic import BaseModel, Field


class ChunkerConfig(BaseModel):
    """Settings for the baseline ``HybridChunker`` used by ``BaselineExtractor``.

    The chunker splits a document into flat, token-bounded chunks that serve
    as a retrieval baseline against the hierarchical document graph. For that
    comparison to be meaningful, ``max_tokens`` has to reflect the context
    budget of the downstream retriever: a value large enough to hold a whole
    document turns the chunker into a no-op and yields document-sized chunks.

    ``tokenizer`` must name a Hugging Face tokenizer. It only determines how
    chunk lengths are measured, so it should match the tokenizer of the model
    that later embeds the chunks.
    """

    tokenizer: str = "intfloat/multilingual-e5-large"  # multilingual by default; the corpus this package targets is German
    max_tokens: int = Field(default=512, gt=0)
    merge_peers: bool = True  # merge adjacent chunks sharing a heading while they fit within max_tokens
