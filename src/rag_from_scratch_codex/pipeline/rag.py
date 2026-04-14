"""High-level RAG pipeline orchestration."""

from dataclasses import dataclass, field

from rag_from_scratch_codex.chunking.base import Chunk
from rag_from_scratch_codex.generation.base import Generator
from rag_from_scratch_codex.retrieval.base import Retriever


@dataclass
class QueryResult:
    """Container for a pipeline query result."""

    answer: str
    retrieved_chunks: list[Chunk] = field(default_factory=list)


class RagPipeline:
    """Coordinate retrieval and generation steps.

    The goal of this class is to keep pipeline orchestration explicit and easy to inspect.
    """

    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        """Create a pipeline from modular retrieval and generation components."""
        self.retriever = retriever
        self.generator = generator

    def query(self, question: str, top_k: int = 5) -> QueryResult:
        """Run a retrieval-then-generation query flow."""
        chunks = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.generate(question, chunks)
        return QueryResult(answer=answer, retrieved_chunks=chunks)
