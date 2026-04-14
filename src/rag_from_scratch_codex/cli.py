"""Typer-based CLI entry point for the project."""

from pathlib import Path

import typer

from rag_from_scratch_codex.chunking import SimpleTextChunker
from rag_from_scratch_codex.config.settings import AppConfig, load_config
from rag_from_scratch_codex.embeddings import OpenAIEmbeddingModel
from rag_from_scratch_codex.generation import OpenAIAnswerGenerator
from rag_from_scratch_codex.loaders.markdown import MarkdownLoader
from rag_from_scratch_codex.pipeline import IngestionPipeline, QueryPipeline
from rag_from_scratch_codex.retrieval import VectorStoreRetriever
from rag_from_scratch_codex.vectorstore import ChromaVectorStore

app = typer.Typer(
    help="A local-first, developer-focused RAG playground for Markdown knowledge bases."
)


@app.command()
def show_config(
    config_path: Path = typer.Option(Path("config.yaml"), help="Path to config file."),
) -> None:
    """Print the current configuration."""
    config: AppConfig = load_config(config_path)
    typer.echo(f"Docs path: {config.docs_path}")
    typer.echo(f"Chunk size: {config.chunk_size}")
    typer.echo(f"Chunk overlap: {config.chunk_overlap}")
    typer.echo(f"Use saved chunks: {config.use_saved_chunks}")
    typer.echo(f"Chunks path: {config.chunks_path}")
    typer.echo(f"Embedding model: {config.embedding_model}")
    typer.echo(f"Vector store path: {config.vector_store_path}")
    typer.echo(f"Top k: {config.top_k}")
    typer.echo(f"Show sources: {config.show_sources}")
    typer.echo(f"Show retrieved chunks: {config.show_retrieved_chunks}")


@app.command()
def ingest(
    config_path: Path = typer.Option(Path("config.yaml"), help="Path to config file."),
) -> None:
    """Load documents, create embeddings, and store them in the vector database."""
    config = load_config(config_path)
    pipeline = IngestionPipeline(
        loader=MarkdownLoader(),
        chunker=SimpleTextChunker.from_config(config),
        embedding_model=OpenAIEmbeddingModel.from_config(config),
        vector_store=ChromaVectorStore.from_config(config),
        config=config,
    )
    result = pipeline.run()

    typer.echo("Ingestion complete.")
    typer.echo(f"Documents: {len(result.documents)}")
    typer.echo(f"Chunks: {len(result.chunks)}")
    typer.echo(f"Embeddings stored: {result.embeddings_count}")
    typer.echo(f"Vector store path: {config.vector_store_path}")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask about the knowledge base."),
    config_path: Path = typer.Option(Path("config.yaml"), help="Path to config file."),
) -> None:
    """Query the indexed knowledge base and print a grounded answer."""
    config = load_config(config_path)
    embedding_model = OpenAIEmbeddingModel.from_config(config)
    vector_store = ChromaVectorStore.from_config(config)
    pipeline = QueryPipeline(
        retriever=VectorStoreRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
        ),
        generator=OpenAIAnswerGenerator.from_config(config),
    )
    result = pipeline.run(question, top_k=config.top_k)

    typer.echo("Answer")
    typer.echo(result.answer)

    if config.show_sources:
        typer.echo("")
        typer.echo("Sources")
        for source in result.sources:
            relative_path = source.get("relative_path", "unknown")
            chunk_index = source.get("chunk_index", "unknown")
            typer.echo(f"- {relative_path} (chunk {chunk_index})")

    if config.show_retrieved_chunks:
        typer.echo("")
        typer.echo("Retrieved Chunks")
        for index, item in enumerate(result.retrieved_chunks, start=1):
            typer.echo(f"[{index}] {item.id} | distance={item.distance}")
            typer.echo(_truncate_text(item.chunk.text))
            typer.echo("")


def _truncate_text(text: str, max_length: int = 300) -> str:
    """Keep retrieved chunk output compact in the CLI."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
