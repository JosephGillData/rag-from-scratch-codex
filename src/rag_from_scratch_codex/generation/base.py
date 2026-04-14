"""Answer generation abstractions and a simple OpenAI-backed generator."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rag_from_scratch_codex.config.settings import AppConfig
from rag_from_scratch_codex.vectorstore.base import SimilarChunk


DEFAULT_GENERATION_MODEL = "gpt-4.1-mini"


@dataclass
class GenerationResult:
    """A generated answer plus the source metadata used to ground it."""

    answer: str
    sources: list[dict[str, str]] = field(default_factory=list)


class Generator(ABC):
    """Base interface for answer generation."""

    @abstractmethod
    def generate(self, query: str, retrieved_chunks: list[SimilarChunk]) -> GenerationResult:
        """Generate an answer from a query and retrieved chunks."""


class OpenAIAnswerGenerator(Generator):
    """Generate grounded answers from retrieved chunks with OpenAI."""

    def __init__(self, model: str = DEFAULT_GENERATION_MODEL, api_key: str | None = None) -> None:
        """Create an answer generator.

        The model name is kept as a small constructor argument so it is easy to
        inspect and change later without introducing a full prompt-management
        layer.
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
    def from_config(cls, config: AppConfig) -> "OpenAIAnswerGenerator":
        """Build a generator from app config.

        The current config does not expose generation model settings yet, so a
        small default is used for now.
        """
        _ = config
        return cls()

    def generate(self, query: str, retrieved_chunks: list[SimilarChunk]) -> GenerationResult:
        """Generate an answer grounded in the retrieved context."""
        if not query.strip():
            raise ValueError("query must not be empty.")
        if not retrieved_chunks:
            raise ValueError("retrieved_chunks must not be empty.")

        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(query, retrieved_chunks)

        try:
            from openai import OpenAIError

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except OpenAIError as error:
            raise RuntimeError(f"Failed to generate answer with OpenAI: {error}") from error

        answer = response.choices[0].message.content or ""
        return GenerationResult(
            answer=answer.strip(),
            sources=[dict(chunk.chunk.metadata) for chunk in retrieved_chunks],
        )

    def build_system_prompt(self) -> str:
        """Return the system instruction used for grounded answering."""
        return (
            "You answer questions using only the provided context chunks. "
            "If the context is insufficient, say so clearly. "
            "Prefer concise, grounded answers and do not invent facts."
        )

    def build_user_prompt(self, query: str, retrieved_chunks: list[SimilarChunk]) -> str:
        """Build the user prompt from the query and retrieved context."""
        context_sections: list[str] = []
        for index, item in enumerate(retrieved_chunks, start=1):
            metadata = item.chunk.metadata
            header = (
                f"Chunk {index}\n"
                f"Source: {metadata.get('relative_path', 'unknown')}\n"
                f"Chunk Index: {metadata.get('chunk_index', 'unknown')}\n"
                f"Distance: {item.distance}"
            )
            context_sections.append(f"{header}\nText:\n{item.chunk.text}")

        context_block = "\n\n---\n\n".join(context_sections)
        return (
            f"Question:\n{query}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            "Answer the question using the retrieved context."
        )
