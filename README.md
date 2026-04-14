# rag-from-scratch-codex

A local-first, developer-focused RAG playground for experimenting with Retrieval-Augmented Generation over Markdown knowledge bases.

## Goals

- Index and query a folder of Markdown files
- Keep the architecture modular and easy to modify
- Expose intermediate pipeline steps for learning and debugging
- Favor simplicity, readability, and incremental development

## Project Structure

This repository uses a small `src`-based Python package layout and is organized around separable RAG pipeline components:

- `config`
- `loaders`
- `chunking`
- `embeddings`
- `vectorstore`
- `retrieval`
- `generation`
- `pipeline`

Each module currently contains placeholders and docstrings so the project can be built incrementally.

## Quick Start

1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment variables:

```bash
cp .env.example .env
```

4. Run the CLI:

```bash
python -m rag_from_scratch_codex --help
```

## Configuration

Default settings live in `config.yaml`. The initial config is intentionally small and readable so it can evolve with the project.

## Markdown Loader Example

```python
from pathlib import Path

from rag_from_scratch_codex.loaders.markdown import MarkdownLoader

loader = MarkdownLoader()
documents = loader.load_directory(Path("./docs"))

first_document = documents[0]
print(first_document.text)
print(first_document.metadata)
```

## Chunking Example

```python
from rag_from_scratch_codex.chunking import SimpleTextChunker, get_or_create_chunks
from rag_from_scratch_codex.config.settings import load_config
from rag_from_scratch_codex.loaders.markdown import MarkdownLoader

config = load_config()
loader = MarkdownLoader()
documents = loader.load_from_config(config)

chunker = SimpleTextChunker.from_config(config)
chunks = get_or_create_chunks(documents, chunker, config)

print(chunks[0].text)
print(chunks[0].metadata)
```

## Status

This is the initial scaffold only. The full RAG logic is intentionally not implemented yet.
