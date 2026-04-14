"""Embedding components."""

from rag_from_scratch_codex.embeddings.base import EmbeddingModel
from rag_from_scratch_codex.embeddings.openai_embeddings import OpenAIEmbeddingModel

__all__ = ["EmbeddingModel", "OpenAIEmbeddingModel"]
