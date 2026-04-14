"""Shared trace data structures for pipeline observability.

These models are intentionally lightweight so they are easy to inspect in the
CLI, render in the UI, and serialize with standard dataclass tooling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag_from_scratch_codex.chunking.base import Chunk
from rag_from_scratch_codex.config.settings import AppConfig
from rag_from_scratch_codex.generation.base import GenerationResult
from rag_from_scratch_codex.loaders.markdown import Document
from rag_from_scratch_codex.vectorstore.base import SimilarChunk


@dataclass
class LoadedDocumentTrace:
    """A loaded source document captured in a trace."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_document(cls, document: Document) -> "LoadedDocumentTrace":
        """Create a trace model from a loaded document."""
        return cls(text=document.text, metadata=dict(document.metadata))


@dataclass
class ChunkTrace:
    """A chunk captured during ingestion or retrieval."""

    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkTrace":
        """Create a trace model from a chunk."""
        return cls(text=chunk.text, metadata=dict(chunk.metadata))


@dataclass
class ChunkSummaryTrace:
    """A lightweight summary of one chunk for inspection views."""

    chunk_id: str
    relative_path: str
    chunk_index: str
    start_char: int
    end_char: int
    char_count: int

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "ChunkSummaryTrace":
        """Create a summary trace from a chunk."""
        metadata = chunk.metadata
        relative_path = metadata.get("relative_path", "unknown")
        chunk_index = metadata.get("chunk_index", "unknown")
        start_char = int(metadata.get("start_char", "0"))
        end_char = int(metadata.get("end_char", str(len(chunk.text))))
        return cls(
            chunk_id=f"{relative_path}::chunk-{chunk_index}",
            relative_path=relative_path,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            char_count=len(chunk.text),
        )


@dataclass
class FileChunkCountTrace:
    """Chunk counts grouped by source file."""

    relative_path: str
    file_name: str
    chunk_count: int


@dataclass
class IndexingMetadataTrace:
    """Small metadata summary for one ingestion run."""

    docs_path: str
    chunks_path: str
    use_saved_chunks: bool
    vector_store_path: str
    embedding_model: str

    @classmethod
    def from_config(cls, config: AppConfig) -> "IndexingMetadataTrace":
        """Create indexing metadata from application config."""
        return cls(
            docs_path=config.docs_path,
            chunks_path=config.chunks_path,
            use_saved_chunks=config.use_saved_chunks,
            vector_store_path=config.vector_store_path,
            embedding_model=config.embedding_model,
        )


@dataclass
class RetrievalResultTrace:
    """One retrieved chunk plus ranking information."""

    id: str
    distance: float
    chunk: ChunkTrace

    @classmethod
    def from_similar_chunk(cls, item: SimilarChunk) -> "RetrievalResultTrace":
        """Create a trace model from a vector-store retrieval result."""
        return cls(
            id=item.id,
            distance=item.distance,
            chunk=ChunkTrace.from_chunk(item.chunk),
        )


@dataclass
class PromptPayloadTrace:
    """The prompt payload sent to the language model."""

    system_prompt: str
    user_prompt: str
    model: str


@dataclass
class SourceReferenceTrace:
    """A source reference used to ground the final answer."""

    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_metadata(cls, metadata: dict[str, str]) -> "SourceReferenceTrace":
        """Create a source reference trace from source metadata."""
        return cls(metadata=dict(metadata))


@dataclass
class FinalAnswerTrace:
    """The final answer plus the source references used."""

    answer: str
    sources: list[SourceReferenceTrace] = field(default_factory=list)

    @classmethod
    def from_generation_result(cls, result: GenerationResult) -> "FinalAnswerTrace":
        """Create a trace model from a generation result."""
        return cls(
            answer=result.answer,
            sources=[SourceReferenceTrace.from_metadata(source) for source in result.sources],
        )


@dataclass
class IngestionRunTrace:
    """Trace for one ingestion run."""

    documents: list[LoadedDocumentTrace] = field(default_factory=list)
    chunks: list[ChunkTrace] = field(default_factory=list)
    chunk_summaries: list[ChunkSummaryTrace] = field(default_factory=list)
    counts_per_file: list[FileChunkCountTrace] = field(default_factory=list)
    indexing_metadata: IndexingMetadataTrace | None = None
    embeddings_count: int = 0

    def to_dict(self) -> dict:
        """Return a plain dictionary representation."""
        return asdict(self)


@dataclass
class QueryRunTrace:
    """Trace for one query run."""

    query: str
    retrieval_results: list[RetrievalResultTrace] = field(default_factory=list)
    prompt_payload: PromptPayloadTrace | None = None
    final_answer: FinalAnswerTrace | None = None

    def to_dict(self) -> dict:
        """Return a plain dictionary representation."""
        return asdict(self)
