# rag-from-scratch-codex

A local-first RAG playground for learning, debugging, and experimenting with Retrieval-Augmented Generation over Markdown files.

This project is built to make the RAG pipeline visible. Instead of hiding everything behind a chat box, it exposes the major steps end to end:

- document loading
- chunking
- embedding creation
- vector storage
- retrieval
- prompt construction
- grounded answer generation

The repository includes both a CLI and a Streamlit UI so you can inspect the system from different angles.

## What This Project Does

Given a folder of Markdown files, the app can:

- load them into a simple document representation
- split them into chunks
- embed those chunks with OpenAI embeddings
- store embeddings locally in ChromaDB
- retrieve the most relevant chunks for a question
- build a grounded prompt from those chunks
- generate an answer with source references

The Streamlit dashboard is intentionally observability-first. It is meant to help you understand how the system behaves, not just produce answers.

## Current Features

- Local Markdown ingestion from `docs/`
- Simple chunking with configurable `chunk_size` and `chunk_overlap`
- OpenAI embedding support using `text-embedding-3-small`
- Local Chroma persistence for chunk embeddings
- Query answering with OpenAI chat completions
- A Streamlit dashboard for inspecting:
  - loaded documents
  - chunk boundaries
  - retrieval ranking
  - embedding vectors
  - prompt payloads
  - final answers and sources
- Small experiment controls in the UI for chunking and retrieval settings

## Project Structure

The code is organized around separable RAG components:

- `src/rag_from_scratch_codex/config`
  Loads and validates app configuration.
- `src/rag_from_scratch_codex/loaders`
  Reads Markdown documents from disk.
- `src/rag_from_scratch_codex/chunking`
  Splits documents into chunks and optionally persists them.
- `src/rag_from_scratch_codex/embeddings`
  Generates embeddings for chunks and queries.
- `src/rag_from_scratch_codex/vectorstore`
  Stores and retrieves embeddings with ChromaDB.
- `src/rag_from_scratch_codex/retrieval`
  Turns a query into a vector search.
- `src/rag_from_scratch_codex/generation`
  Builds prompts and generates grounded answers.
- `src/rag_from_scratch_codex/pipeline`
  Orchestrates ingestion and query execution and captures traces.
- `ui/app.py`
  Streamlit observability dashboard.

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment file:

```bash
cp .env.example .env
```

4. Add your API key to `.env`:

```env
OPENAI_API_KEY=your_key_here
```

5. Put your Markdown files in `docs/`, or point `docs_path` in `config.yaml` to another folder.

## Quick Start

Show the current config:

```bash
python -m rag_from_scratch_codex show-config
```

Run ingestion:

```bash
python -m rag_from_scratch_codex ingest
```

Ask a question:

```bash
python -m rag_from_scratch_codex query "What topics are covered in these notes?"
```

Launch the UI:

```bash
streamlit run ui/app.py
```

## CLI Commands

The Typer CLI currently exposes three commands:

- `show-config`
  Prints the effective settings from `config.yaml`.
- `ingest`
  Loads documents, chunks them, creates embeddings, and stores them in Chroma.
- `query "<question>"`
  Retrieves relevant chunks and prints a grounded answer.

You can inspect command help with:

```bash
python -m rag_from_scratch_codex --help
```

## UI Overview

The Streamlit app is designed as a local dashboard for understanding the pipeline. Depending on what you have run, it can show:

- `Corpus`
  Which files were loaded and how large they are.
- `Chunk Explorer`
  How files were split and what each chunk contains.
- `Embedding Explorer`
  The vector for a selected chunk, its value distribution, and similarity to the latest query.
- `Retrieval Inspector`
  Which chunks were retrieved and how they were ranked.
- `Prompt Viewer`
  The system prompt and user prompt sent to the model.
- `Answer and Sources`
  The final response plus supporting source references.

The sidebar also includes a few experiment controls so you can change chunk size, overlap, and `top_k`, then rerun ingestion or query to compare behavior.

## Configuration

Project settings live in [`config.yaml`](./config.yaml).

Current config fields:

- `docs_path`
  Folder containing Markdown files to index.
- `chunk_size`
  Number of characters per chunk.
- `chunk_overlap`
  Number of overlapping characters between adjacent chunks.
- `use_saved_chunks`
  Whether to reuse the chunk JSON file instead of regenerating chunks.
- `chunks_path`
  Path to the saved chunk JSON file.
- `embedding_model`
  OpenAI embedding model name.
- `vector_store_path`
  Directory for the local Chroma database.
- `top_k`
  Number of chunks to retrieve for each query.
- `show_sources`
  Whether the CLI prints source references.
- `show_retrieved_chunks`
  Whether the CLI prints retrieved chunk previews.

Example:

```yaml
docs_path: ./docs
chunk_size: 500
chunk_overlap: 50
use_saved_chunks: false
chunks_path: ./storage/chunks.json
embedding_model: text-embedding-3-small
vector_store_path: ./storage/chroma
top_k: 5
show_sources: true
show_retrieved_chunks: false
```

## How Data Flows

The high-level pipeline looks like this:

1. Load Markdown documents from `docs_path`.
2. Split documents into chunks.
3. Create embeddings for each chunk.
4. Store chunks and embeddings in Chroma.
5. Embed the user query.
6. Retrieve the most similar chunks.
7. Build a prompt from the query and retrieved context.
8. Generate a grounded answer.

This is intentionally straightforward so each stage is easy to inspect and modify.

## Limitations

- The prompt and answer quality have not been formally evaluated yet.
- The generation model is currently hardcoded in the generator layer rather than exposed in config.
- The experiment workflow is still lightweight and mostly UI-driven.
- This project is optimized for clarity and inspection, not production scale.

## TODO

- Verify that the RAG prompt actually produces grounded, reliable answers across a representative question set.
- Add a lightweight evaluation workflow for prompt and retrieval changes.
- Support more explicit experiments across chunking, retrieval, and prompt settings.
- Make experiments easier to compare side by side in the UI.
- Expand model and prompt configuration without making the project harder to understand.

## Status

This project is functional as a local learning and observability tool, but it is still intentionally small. The goal is to keep it easy to understand while gradually improving retrieval quality, prompt quality, and experimentation support.
