"""Typer-based CLI entry point for the project."""

from pathlib import Path

import typer

from rag_from_scratch_codex.config.settings import AppConfig, load_config

app = typer.Typer(
    help="A local-first, developer-focused RAG playground for Markdown knowledge bases."
)


@app.command()
def info(config_path: Path = typer.Option(Path("config.yaml"), help="Path to config file.")) -> None:
    """Show basic project configuration information."""
    config: AppConfig = load_config(config_path)
    typer.echo(f"Docs path: {config.docs_path}")
    typer.echo(f"Chunk size: {config.chunk_size}")
    typer.echo(f"Chunk overlap: {config.chunk_overlap}")
    typer.echo(f"Embedding model: {config.embedding_model}")
    typer.echo(f"Vector store path: {config.vector_store_path}")
    typer.echo(f"Top k: {config.top_k}")
    typer.echo(f"Show sources: {config.show_sources}")
    typer.echo(f"Show retrieved chunks: {config.show_retrieved_chunks}")


@app.command()
def index() -> None:
    """Placeholder command for future indexing logic."""
    typer.echo("Indexing is not implemented yet.")


@app.command()
def query(question: str) -> None:
    """Placeholder command for future query logic."""
    typer.echo(f"Querying is not implemented yet. Received question: {question}")
