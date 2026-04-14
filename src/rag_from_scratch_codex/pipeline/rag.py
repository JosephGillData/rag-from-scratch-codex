"""High-level pipeline orchestration.

The goal of this module is to compose the individual building blocks without
moving their logic into the pipeline itself. Each pipeline reads top-to-bottom
so it is easy to understand how data flows through the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_from_scratch_codex.chunking.base import Chunk, TextChunker
from rag_from_scratch_codex.chunking.storage import get_or_create_chunks
from rag_from_scratch_codex.config.settings import AppConfig
from rag_from_scratch_codex.embeddings.base import EmbeddingModel
from rag_from_scratch_codex.generation.base import GenerationResult, Generator
from rag_from_scratch_codex.loaders.markdown import Document, MarkdownLoader
from rag_from_scratch_codex.retrieval.base import Retriever
from rag_from_scratch_codex.vectorstore.base import SimilarChunk, VectorStore


@dataclass
class IngestionResult:
    """Summary of an ingestion run."""

    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    embeddings_count: int = 0


@dataclass
class QueryPipelineResult:
    """Container for a query pipeline run."""

    answer: str
    retrieved_chunks: list[SimilarChunk] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)


class IngestionPipeline:
    """Load documents, chunk them, embed them, and store them."""

    def __init__(
        self,
        loader: MarkdownLoader,
        chunker: TextChunker,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        config: AppConfig,
    ) -> None:
        """Create an ingestion pipeline from modular components."""
        self.loader = loader
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.config = config

    def run(self) -> IngestionResult:
        """Execute the ingestion flow from source documents to vector store."""
        documents = self.loader.load_from_config(self.config)
        chunks = get_or_create_chunks(documents, self.chunker, self.config)
        embeddings = self.embedding_model.embed_chunks(chunks)
        self.vector_store.add(chunks, embeddings)
        return IngestionResult(
            documents=documents,
            chunks=chunks,
            embeddings_count=len(embeddings),
        )


class QueryPipeline:
    """Embed a query, retrieve relevant chunks, and generate an answer."""

    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        """Create a query pipeline from retrieval and generation components."""
        self.retriever = retriever
        self.generator = generator

    def run(self, query: str, top_k: int = 5) -> QueryPipelineResult:
        """Execute the query flow."""
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        generation_result: GenerationResult = self.generator.generate(query, retrieved_chunks)
        return QueryPipelineResult(
            answer=generation_result.answer,
            retrieved_chunks=retrieved_chunks,
            sources=generation_result.sources,
        )
