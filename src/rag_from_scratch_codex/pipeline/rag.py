"""High-level pipeline orchestration.

The goal of this module is to compose the individual building blocks without
moving their logic into the pipeline itself. Each pipeline reads top-to-bottom
so it is easy to understand how data flows through the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag_from_scratch_codex.chunking.base import Chunk, TextChunker
from rag_from_scratch_codex.chunking.storage import get_or_create_chunks
from rag_from_scratch_codex.config.settings import AppConfig
from rag_from_scratch_codex.embeddings.base import EmbeddingModel
from rag_from_scratch_codex.generation.base import GenerationResult, Generator
from rag_from_scratch_codex.loaders.markdown import Document, MarkdownLoader
from rag_from_scratch_codex.pipeline.trace import (
    ChunkSummaryTrace,
    ChunkTrace,
    FinalAnswerTrace,
    FileChunkCountTrace,
    IndexingMetadataTrace,
    IngestionRunTrace,
    LoadedDocumentTrace,
    PromptPayloadTrace,
    QueryRunTrace,
    RetrievalResultTrace,
)
from rag_from_scratch_codex.retrieval.base import Retriever
from rag_from_scratch_codex.vectorstore.base import SimilarChunk, VectorStore


@dataclass
class IngestionResult:
    """Summary of an ingestion run."""

    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    embeddings_count: int = 0
    trace: IngestionRunTrace | None = None


@dataclass
class QueryPipelineResult:
    """Container for a query pipeline run."""

    answer: str
    retrieved_chunks: list[SimilarChunk] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    trace: QueryRunTrace | None = None


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
        trace = self._build_trace(documents=documents, chunks=chunks, embeddings_count=len(embeddings))
        return IngestionResult(
            documents=documents,
            chunks=chunks,
            embeddings_count=len(embeddings),
            trace=trace,
        )

    def _build_trace(
        self,
        documents: list[Document],
        chunks: list[Chunk],
        embeddings_count: int,
    ) -> IngestionRunTrace:
        """Build a structured trace for one ingestion run."""
        counts_by_path: dict[str, FileChunkCountTrace] = {}
        for chunk in chunks:
            relative_path = chunk.metadata.get("relative_path", "unknown")
            file_name = chunk.metadata.get("file_name", Path(relative_path).name)
            if relative_path not in counts_by_path:
                counts_by_path[relative_path] = FileChunkCountTrace(
                    relative_path=relative_path,
                    file_name=file_name,
                    chunk_count=0,
                )
            counts_by_path[relative_path].chunk_count += 1

        return IngestionRunTrace(
            documents=[LoadedDocumentTrace.from_document(document) for document in documents],
            chunks=[ChunkTrace.from_chunk(chunk) for chunk in chunks],
            chunk_summaries=[ChunkSummaryTrace.from_chunk(chunk) for chunk in chunks],
            counts_per_file=sorted(
                counts_by_path.values(),
                key=lambda item: item.relative_path,
            ),
            indexing_metadata=IndexingMetadataTrace.from_config(self.config),
            embeddings_count=embeddings_count,
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
        trace = self._build_trace(
            query=query,
            retrieved_chunks=retrieved_chunks,
            generation_result=generation_result,
        )
        return QueryPipelineResult(
            answer=generation_result.answer,
            retrieved_chunks=retrieved_chunks,
            sources=generation_result.sources,
            trace=trace,
        )

    def _build_trace(
        self,
        query: str,
        retrieved_chunks: list[SimilarChunk],
        generation_result: GenerationResult,
    ) -> QueryRunTrace:
        """Build a structured trace for one query run."""
        prompt_payload = self._build_prompt_payload(query, retrieved_chunks)
        return QueryRunTrace(
            query=query,
            retrieval_results=[
                RetrievalResultTrace.from_similar_chunk(item) for item in retrieved_chunks
            ],
            prompt_payload=prompt_payload,
            final_answer=FinalAnswerTrace.from_generation_result(generation_result),
        )

    def _build_prompt_payload(
        self,
        query: str,
        retrieved_chunks: list[SimilarChunk],
    ) -> PromptPayloadTrace | None:
        """Build the prompt payload if the generator exposes prompt helpers."""
        build_system_prompt = getattr(self.generator, "build_system_prompt", None)
        build_user_prompt = getattr(self.generator, "build_user_prompt", None)
        model = getattr(self.generator, "model", None)

        if not callable(build_system_prompt) or not callable(build_user_prompt):
            return None

        return PromptPayloadTrace(
            system_prompt=build_system_prompt(),
            user_prompt=build_user_prompt(query, retrieved_chunks),
            model=str(model or ""),
        )
