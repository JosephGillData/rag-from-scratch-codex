"""Retriever abstractions and a simple vector-store-backed retriever."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_from_scratch_codex.config.settings import AppConfig
from rag_from_scratch_codex.embeddings.base import EmbeddingModel
from rag_from_scratch_codex.vectorstore.base import SimilarChunk, VectorStore


class Retriever(ABC):
    """Base interface for retrieving relevant chunks."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[SimilarChunk]:
        """Retrieve chunks relevant to the input query."""


class VectorStoreRetriever(Retriever):
    """Retrieve chunks by embedding the query and searching the vector store.

    This class keeps the retrieval step intentionally explicit:
    1. Embed the query string
    2. Query the vector store with that embedding
    """

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore) -> None:
        """Create a retriever from an embedding model and vector store."""
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> "VectorStoreRetriever":
        """Build a retriever from config and the required dependencies."""
        _ = config
        return cls(embedding_model=embedding_model, vector_store=vector_store)

    def retrieve(self, query: str, top_k: int = 5) -> list[SimilarChunk]:
        """Embed the query and return the top matching chunks."""
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedding_model.embed_query(query)
        return self.vector_store.query_similar(query_embedding, top_k=top_k)
