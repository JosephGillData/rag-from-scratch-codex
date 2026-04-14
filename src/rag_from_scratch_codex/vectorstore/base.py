"""Vector store abstractions."""

from rag_from_scratch_codex.chunking.base import Chunk


class VectorStore:
    """Base interface for vector store implementations."""

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks and their embeddings."""
        _ = (chunks, embeddings)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Return the most relevant chunks for a query embedding."""
        _ = (query_embedding, top_k)
        return []
