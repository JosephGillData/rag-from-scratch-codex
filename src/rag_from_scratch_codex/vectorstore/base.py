"""Vector store abstractions and a ChromaDB implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from rag_from_scratch_codex.chunking.base import Chunk
from rag_from_scratch_codex.config.settings import AppConfig


@dataclass
class SimilarChunk:
    """A retrieved chunk plus similarity information."""

    chunk: Chunk
    id: str
    distance: float


class VectorStore(ABC):
    """Base interface for vector store implementations."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks and their embeddings."""

    @abstractmethod
    def query_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SimilarChunk]:
        """Return chunks similar to the query embedding."""


class ChromaVectorStore(VectorStore):
    """Persist chunk embeddings locally with ChromaDB."""

    def __init__(self, persist_directory: str, collection_name: str = "rag_chunks") -> None:
        """Create a vector store backed by a local Chroma collection."""
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError(
                "chromadb is not installed. Install dependencies with `pip install -r requirements.txt`."
            ) from error

        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    @classmethod
    def from_config(cls, config: AppConfig) -> "ChromaVectorStore":
        """Build a vector store from the application config."""
        return cls(persist_directory=config.vector_store_path)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunk text, embeddings, and metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks must match the number of embeddings.")
        if not chunks:
            return

        ids = [self._make_chunk_id(chunk, index) for index, chunk in enumerate(chunks)]
        documents = [chunk.text for chunk in chunks]
        metadatas = [dict(chunk.metadata) for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SimilarChunk]:
        """Return the most similar chunks for a query embedding."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        similar_chunks: list[SimilarChunk] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            similar_chunks.append(
                SimilarChunk(
                    id=str(chunk_id),
                    chunk=Chunk(
                        text=document or "",
                        metadata={str(key): str(value) for key, value in (metadata or {}).items()},
                    ),
                    distance=float(distance),
                )
            )

        return similar_chunks

    def _make_chunk_id(self, chunk: Chunk, index: int) -> str:
        """Create a stable chunk identifier for persistence."""
        relative_path = chunk.metadata.get("relative_path", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", str(index))
        return f"{relative_path}::chunk-{chunk_index}"
