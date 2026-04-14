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

## Run The UI

Launch the local Streamlit dashboard from the repository root:

```bash
streamlit run ui/app.py
```

The UI is intentionally thin and local-first. It is meant to become an
observability dashboard for the RAG pipeline rather than a generic chat app.

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

## Embeddings Example

```python
from rag_from_scratch_codex.chunking import SimpleTextChunker
from rag_from_scratch_codex.config.settings import load_config
from rag_from_scratch_codex.embeddings import OpenAIEmbeddingModel
from rag_from_scratch_codex.loaders.markdown import MarkdownLoader

config = load_config()
documents = MarkdownLoader().load_from_config(config)
chunks = SimpleTextChunker.from_config(config).chunk_documents(documents)

embedding_model = OpenAIEmbeddingModel.from_config(config)
chunk_embeddings = embedding_model.embed_chunks(chunks[:5])
query_embedding = embedding_model.embed_query("What is this vault about?")

print(len(chunk_embeddings))
print(len(query_embedding))
```

## Vector Store Example

```python
from rag_from_scratch_codex.chunking import SimpleTextChunker, get_or_create_chunks
from rag_from_scratch_codex.config.settings import load_config
from rag_from_scratch_codex.embeddings import OpenAIEmbeddingModel
from rag_from_scratch_codex.loaders.markdown import MarkdownLoader
from rag_from_scratch_codex.vectorstore import ChromaVectorStore

config = load_config()
documents = MarkdownLoader().load_from_config(config)
chunks = get_or_create_chunks(documents, SimpleTextChunker.from_config(config), config)

embedding_model = OpenAIEmbeddingModel.from_config(config)
chunk_embeddings = embedding_model.embed_chunks(chunks)

vector_store = ChromaVectorStore.from_config(config)
vector_store.add(chunks, chunk_embeddings)

query_embedding = embedding_model.embed_query("What is this knowledge base about?")
results = vector_store.query_similar(query_embedding, top_k=config.top_k)

print(results[0].chunk.metadata)
print(results[0].distance)
```

## Retrieval Example

```python
from rag_from_scratch_codex.config.settings import load_config
from rag_from_scratch_codex.embeddings import OpenAIEmbeddingModel
from rag_from_scratch_codex.retrieval import VectorStoreRetriever
from rag_from_scratch_codex.vectorstore import ChromaVectorStore

config = load_config()
embedding_model = OpenAIEmbeddingModel.from_config(config)
vector_store = ChromaVectorStore.from_config(config)
retriever = VectorStoreRetriever(embedding_model=embedding_model, vector_store=vector_store)

results = retriever.retrieve("What topics are covered in these notes?", top_k=config.top_k)

print(results[0].chunk.text)
print(results[0].chunk.metadata)
print(results[0].distance)
```

## Generation Example

```python
from rag_from_scratch_codex.config.settings import load_config
from rag_from_scratch_codex.embeddings import OpenAIEmbeddingModel
from rag_from_scratch_codex.generation import OpenAIAnswerGenerator
from rag_from_scratch_codex.retrieval import VectorStoreRetriever
from rag_from_scratch_codex.vectorstore import ChromaVectorStore

config = load_config()
embedding_model = OpenAIEmbeddingModel.from_config(config)
vector_store = ChromaVectorStore.from_config(config)
retriever = VectorStoreRetriever(embedding_model=embedding_model, vector_store=vector_store)
generator = OpenAIAnswerGenerator.from_config(config)

retrieved_chunks = retriever.retrieve("What topics are covered in these notes?", top_k=config.top_k)
result = generator.generate("What topics are covered in these notes?", retrieved_chunks)

print(result.answer)
print(result.sources)
```

## Pipeline Example

```python
from rag_from_scratch_codex.chunking import SimpleTextChunker
from rag_from_scratch_codex.config.settings import load_config
from rag_from_scratch_codex.embeddings import OpenAIEmbeddingModel
from rag_from_scratch_codex.generation import OpenAIAnswerGenerator
from rag_from_scratch_codex.loaders.markdown import MarkdownLoader
from rag_from_scratch_codex.pipeline import IngestionPipeline, QueryPipeline
from rag_from_scratch_codex.retrieval import VectorStoreRetriever
from rag_from_scratch_codex.vectorstore import ChromaVectorStore

config = load_config()

loader = MarkdownLoader()
chunker = SimpleTextChunker.from_config(config)
embedding_model = OpenAIEmbeddingModel.from_config(config)
vector_store = ChromaVectorStore.from_config(config)
generator = OpenAIAnswerGenerator.from_config(config)
retriever = VectorStoreRetriever(embedding_model=embedding_model, vector_store=vector_store)

ingestion_pipeline = IngestionPipeline(
    loader=loader,
    chunker=chunker,
    embedding_model=embedding_model,
    vector_store=vector_store,
    config=config,
)
ingestion_result = ingestion_pipeline.run()

query_pipeline = QueryPipeline(retriever=retriever, generator=generator)
query_result = query_pipeline.run("What topics are covered in these notes?", top_k=config.top_k)

print(len(ingestion_result.documents))
print(len(ingestion_result.chunks))
print(query_result.answer)
print(query_result.sources)
```

## Status

This is the initial scaffold only. The full RAG logic is intentionally not implemented yet.
