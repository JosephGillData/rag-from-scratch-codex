"""Persistence helpers for saving and loading chunks."""

from __future__ import annotations

import json
from pathlib import Path

from rag_from_scratch_codex.chunking.base import Chunk, TextChunker
from rag_from_scratch_codex.config.settings import AppConfig
from rag_from_scratch_codex.loaders.markdown import Document


class ChunkStore:
    """Save and load chunks from a JSON file.

    The format is intentionally simple so users can inspect the saved chunks
    directly and understand what the system is operating on.
    """

    def __init__(self, path: Path) -> None:
        """Create a chunk store bound to a file path."""
        self.path = path

    def save(self, chunks: list[Chunk]) -> None:
        """Write chunks to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def load(self) -> list[Chunk]:
        """Load chunks from disk."""
        with self.path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, list):
            raise ValueError(f"Expected a list of chunks in {self.path}.")

        chunks: list[Chunk] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"Each saved chunk must be an object in {self.path}.")

            text = item.get("text")
            metadata = item.get("metadata")

            if not isinstance(text, str):
                raise ValueError(f"Saved chunk text must be a string in {self.path}.")
            if not isinstance(metadata, dict):
                raise ValueError(f"Saved chunk metadata must be an object in {self.path}.")

            chunks.append(
                Chunk(
                    text=text,
                    metadata={str(key): str(value) for key, value in metadata.items()},
                )
            )

        return chunks

    def exists(self) -> bool:
        """Return whether the chunk file exists."""
        return self.path.exists()


def get_or_create_chunks(
    documents: list[Document],
    chunker: TextChunker,
    config: AppConfig,
) -> list[Chunk]:
    """Load saved chunks or create and save a fresh set.

    If ``use_saved_chunks`` is enabled, the function reads the chunk file from
    ``chunks_path``. Otherwise it generates new chunks from the input documents,
    saves them, and returns the new result.
    """
    store = ChunkStore(Path(config.chunks_path))

    if config.use_saved_chunks:
        if not store.exists():
            raise FileNotFoundError(
                f"Configured to use saved chunks, but no chunk file was found at {store.path}."
            )
        return store.load()

    chunks = chunker.chunk_documents(documents)
    store.save(chunks)
    return chunks
