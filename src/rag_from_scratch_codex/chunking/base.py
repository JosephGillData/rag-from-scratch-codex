"""Chunking abstractions and a simple default chunker."""

from dataclasses import dataclass, field

from rag_from_scratch_codex.config.settings import AppConfig
from rag_from_scratch_codex.loaders.markdown import Document


@dataclass
class Chunk:
    """A chunk of document text plus metadata.

    Metadata includes the original document metadata so source information is
    preserved across the pipeline.
    """

    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class TextChunker:
    """Base interface for chunking strategies."""

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split a list of documents into chunks."""
        raise NotImplementedError


class SimpleTextChunker(TextChunker):
    """Split documents into overlapping character-based chunks.

    The implementation is intentionally direct and easy to inspect so it can
    serve as a baseline before experimenting with other chunking strategies.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Create a chunker with fixed size and overlap settings."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be zero or greater.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @classmethod
    def from_config(cls, config: AppConfig) -> "SimpleTextChunker":
        """Build a chunker from the application config."""
        return cls(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks while preserving source metadata."""
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split one document into overlapping chunks."""
        if not document.text:
            return []

        chunks: list[Chunk] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        chunk_index = 0

        while start < len(document.text):
            end = start + self.chunk_size
            chunk_text = document.text[start:end]
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": str(chunk_index),
                        "start_char": str(start),
                        "end_char": str(min(end, len(document.text))),
                    },
                )
            )
            start += step
            chunk_index += 1

        return chunks
