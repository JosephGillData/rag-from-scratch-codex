"""Embedding abstractions.

The project keeps the embedding interface small on purpose so the provider can
be swapped later without changing the rest of the pipeline.
"""

from abc import ABC, abstractmethod

from rag_from_scratch_codex.chunking.base import Chunk


class EmbeddingModel(ABC):
    """Base interface for text embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return vector embeddings for a batch of texts."""

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Return embeddings for a list of chunks."""
        return self.embed_texts([chunk.text for chunk in chunks])

    def embed_query(self, query: str) -> list[float]:
        """Return an embedding for a single query string."""
        embeddings = self.embed_texts([query])
        return embeddings[0]
