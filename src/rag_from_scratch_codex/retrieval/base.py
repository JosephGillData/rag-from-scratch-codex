"""Retriever abstractions."""

from rag_from_scratch_codex.chunking.base import Chunk


class Retriever:
    """Base interface for retrieving relevant chunks."""

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Retrieve chunks relevant to the input query."""
        _ = (query, top_k)
        return []
