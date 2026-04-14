"""Embedding abstractions."""


class EmbeddingModel:
    """Base interface for text embedding providers."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return vector embeddings for a batch of texts."""
        _ = texts
        return []
