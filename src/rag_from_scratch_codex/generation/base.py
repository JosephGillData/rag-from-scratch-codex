"""Answer generation abstractions."""

from rag_from_scratch_codex.chunking.base import Chunk


class Generator:
    """Base interface for answer generation."""

    def generate(self, query: str, context_chunks: list[Chunk]) -> str:
        """Generate an answer from a query and retrieved context."""
        _ = (query, context_chunks)
        return ""
