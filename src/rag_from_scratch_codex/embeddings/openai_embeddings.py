"""OpenAI embedding implementation."""

from __future__ import annotations

import os

from rag_from_scratch_codex.config.settings import AppConfig
from rag_from_scratch_codex.embeddings.base import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    """Embed text with the OpenAI embeddings API."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        """Create an embedding client.

        The API key is read from the environment by default so local
        development works cleanly with a `.env` file.
        """
        try:
            from dotenv import load_dotenv
        except ImportError as error:
            raise RuntimeError(
                "python-dotenv is not installed. Install dependencies with `pip install -r requirements.txt`."
            ) from error

        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "openai is not installed. Install dependencies with `pip install -r requirements.txt`."
            ) from error

        load_dotenv()

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your environment or .env file."
            )

        self.model = model
        self.client = OpenAI(api_key=resolved_api_key)

    @classmethod
    def from_config(cls, config: AppConfig) -> "OpenAIEmbeddingModel":
        """Build an embedding model from the application config."""
        return cls(model=config.embedding_model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a batch of texts."""
        if not texts:
            return []

        try:
            from openai import OpenAIError

            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
        except OpenAIError as error:
            raise RuntimeError(f"Failed to create embeddings with OpenAI: {error}") from error

        return [item.embedding for item in response.data]
