"""Main Streamlit app for the local RAG observability dashboard."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_from_scratch_codex.chunking import SimpleTextChunker
from rag_from_scratch_codex.config.settings import AppConfig, load_config
from rag_from_scratch_codex.embeddings import OpenAIEmbeddingModel
from rag_from_scratch_codex.generation import OpenAIAnswerGenerator
from rag_from_scratch_codex.loaders.markdown import MarkdownLoader
from rag_from_scratch_codex.pipeline import IngestionPipeline, QueryPipeline
from rag_from_scratch_codex.retrieval import VectorStoreRetriever
from rag_from_scratch_codex.vectorstore import ChromaVectorStore


st.set_page_config(page_title="raglab", page_icon=":mag:", layout="wide")


@st.cache_data(show_spinner=False)
def _load_config(config_path: str) -> AppConfig:
    """Load the app config for display and pipeline construction."""
    return load_config(Path(config_path))


def _run_ingestion(config: AppConfig):
    """Run the ingestion pipeline using the shared backend modules."""
    pipeline = IngestionPipeline(
        loader=MarkdownLoader(),
        chunker=SimpleTextChunker.from_config(config),
        embedding_model=OpenAIEmbeddingModel.from_config(config),
        vector_store=ChromaVectorStore.from_config(config),
        config=config,
    )
    return pipeline.run()


def _run_query(config: AppConfig, question: str):
    """Run the query pipeline using the shared backend modules."""
    embedding_model = OpenAIEmbeddingModel.from_config(config)
    vector_store = ChromaVectorStore.from_config(config)
    pipeline = QueryPipeline(
        retriever=VectorStoreRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
        ),
        generator=OpenAIAnswerGenerator.from_config(config),
    )
    return pipeline.run(question, top_k=config.top_k)


def _render_config_summary(config: AppConfig) -> None:
    """Render a compact configuration summary."""
    st.caption("Basic configuration")
    st.json(
        {
            "docs_path": config.docs_path,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "use_saved_chunks": config.use_saved_chunks,
            "chunks_path": config.chunks_path,
            "embedding_model": config.embedding_model,
            "vector_store_path": config.vector_store_path,
            "top_k": config.top_k,
            "show_sources": config.show_sources,
            "show_retrieved_chunks": config.show_retrieved_chunks,
        }
    )


def _render_experiment_controls(config: AppConfig) -> AppConfig:
    """Render a small set of experimental parameter controls.

    The returned config is derived from the loaded config object so the UI keeps
    using the same backend configuration pathway rather than maintaining a
    separate ad hoc settings model.
    """
    st.caption("Experimental parameters")

    chunk_size = st.number_input(
        "Chunk size",
        min_value=1,
        value=config.chunk_size,
        step=50,
        help="How many characters to place in each chunk.",
    )
    chunk_overlap = st.number_input(
        "Chunk overlap",
        min_value=0,
        max_value=max(chunk_size - 1, 0),
        value=min(config.chunk_overlap, max(chunk_size - 1, 0)),
        step=10,
        help="How many characters adjacent chunks share.",
    )
    top_k = st.number_input(
        "Top k",
        min_value=1,
        value=config.top_k,
        step=1,
        help="How many retrieved chunks to use for a query.",
    )

    return replace(
        config,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        top_k=int(top_k),
    )


def _config_snapshot(config: AppConfig) -> dict[str, int]:
    """Return the experimental settings that affect visible pipeline behavior."""
    return {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": config.top_k,
    }


def _format_snapshot(snapshot: dict[str, int]) -> str:
    """Render a compact settings summary for status messages."""
    return ", ".join(f"{key}={value}" for key, value in snapshot.items())


def _render_corpus_section() -> None:
    """Render the corpus inspection section."""
    st.subheader("Corpus")
    st.caption(
        "The corpus is the raw input to the RAG system: the files that were loaded before "
        "any chunking, embedding, or retrieval happens."
    )
    ingestion_result = st.session_state.get("ingestion_result")
    if not ingestion_result or not ingestion_result.trace:
        st.info("Run ingestion to inspect loaded files and indexing metadata.")
        return

    trace = ingestion_result.trace
    st.caption(
        "This panel shows what entered the pipeline before retrieval begins. "
        "It helps you verify which files were loaded, how large they are, and "
        "how aggressively each file was split into chunks."
    )

    chunk_counts_by_path = {
        item.relative_path: item.chunk_count for item in trace.counts_per_file
    }

    total_files_column, total_chunks_column = st.columns(2)
    with total_files_column:
        st.metric("Total files", len(trace.documents))
    with total_chunks_column:
        st.metric("Total chunks", len(trace.chunk_summaries))

    if trace.indexing_metadata:
        with st.expander("Indexing Metadata", expanded=False):
            st.json(
                {
                    "docs_path": trace.indexing_metadata.docs_path,
                    "chunks_path": trace.indexing_metadata.chunks_path,
                    "use_saved_chunks": trace.indexing_metadata.use_saved_chunks,
                    "vector_store_path": trace.indexing_metadata.vector_store_path,
                    "embedding_model": trace.indexing_metadata.embedding_model,
                }
            )

    st.caption("Loaded files")
    st.dataframe(
        [
            {
                "file_name": document.metadata.get("file_name", "unknown"),
                "relative_path": document.metadata.get("relative_path", "unknown"),
                "character_count": len(document.text),
                "chunk_count": chunk_counts_by_path.get(
                    document.metadata.get("relative_path", "unknown"),
                    0,
                ),
            }
            for document in trace.documents
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Loaded Documents", expanded=False):
        for document in trace.documents:
            st.markdown(f"**{document.metadata.get('relative_path', 'unknown')}**")
            st.caption(f"{len(document.text)} characters")


def _render_chunk_section() -> None:
    """Render the chunk explorer section."""
    st.subheader("Chunk Explorer")
    st.caption(
        "Chunking splits long documents into smaller units that can be embedded and retrieved. "
        "Chunk boundaries strongly affect what context the model can later find."
    )
    ingestion_result = st.session_state.get("ingestion_result")
    if not ingestion_result or not ingestion_result.trace:
        st.info("Run ingestion to inspect chunk summaries and chunk text.")
        return

    trace = ingestion_result.trace
    st.caption(
        "Chunking breaks each document into smaller pieces for embedding and retrieval. "
        "Good chunk boundaries preserve meaning; poor chunk boundaries can hide the right "
        "information or split it across multiple retrieval results."
    )

    available_files = [
        document.metadata.get("relative_path", "unknown") for document in trace.documents
    ]
    selected_file = st.selectbox(
        "Source file",
        options=available_files,
        key="chunk_explorer_selected_file",
    )

    selected_document = next(
        (
            document
            for document in trace.documents
            if document.metadata.get("relative_path", "unknown") == selected_file
        ),
        None,
    )
    selected_chunks = [
        chunk
        for chunk in trace.chunks
        if chunk.metadata.get("relative_path", "unknown") == selected_file
    ]

    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.metric("Chunks for file", len(selected_chunks))
    with summary_right:
        st.metric("Document characters", len(selected_document.text) if selected_document else 0)

    for chunk in sorted(
        selected_chunks,
        key=lambda item: int(item.metadata.get("chunk_index", "0")),
    ):
        chunk_index = chunk.metadata.get("chunk_index", "unknown")
        start_char = chunk.metadata.get("start_char", "unknown")
        end_char = chunk.metadata.get("end_char", "unknown")
        with st.expander(f"Chunk {chunk_index} ({start_char}-{end_char})", expanded=False):
            metadata_left, metadata_right = st.columns(2)
            with metadata_left:
                st.write(f"**Chunk length:** {len(chunk.text)} characters")
                st.write(f"**Relative path:** {chunk.metadata.get('relative_path', 'unknown')}")
            with metadata_right:
                st.write(f"**Start char:** {start_char}")
                st.write(f"**End char:** {end_char}")

            st.caption("Source metadata")
            st.json(chunk.metadata)
            st.caption("Chunk text")
            st.code(chunk.text)


def _render_retrieval_section() -> None:
    """Render the retrieval inspector section."""
    st.subheader("Retrieval Inspector")
    st.caption(
        "Retrieval ranks stored chunks against the query embedding. These results are the "
        "context the generator will rely on, so ranking quality matters a lot."
    )
    query_result = st.session_state.get("query_result")
    if not query_result or not query_result.trace:
        st.info("Run a query to inspect ranked retrieval results.")
        return

    trace = query_result.trace
    st.caption(
        "Retrieval quality strongly shapes final answer quality. If the wrong chunks are "
        "ranked highly here, the model will be grounded in weak or irrelevant context."
    )

    st.dataframe(
        [
            {
                "rank": index + 1,
                "source_file": item.chunk.metadata.get("file_name", "unknown"),
                "relative_path": item.chunk.metadata.get("relative_path", "unknown"),
                "chunk_index": item.chunk.metadata.get("chunk_index", "unknown"),
                "distance": item.distance,
            }
            for index, item in enumerate(trace.retrieval_results)
        ],
        use_container_width=True,
        hide_index=True,
    )

    for index, item in enumerate(trace.retrieval_results, start=1):
        source_file = item.chunk.metadata.get("file_name", "unknown")
        relative_path = item.chunk.metadata.get("relative_path", "unknown")
        chunk_index = item.chunk.metadata.get("chunk_index", "unknown")
        with st.expander(
            f"Rank {index} - {source_file} - chunk {chunk_index} - distance {item.distance}",
            expanded=False,
        ):
            left, right = st.columns(2)
            with left:
                st.write(f"**Rank:** {index}")
                st.write(f"**Source file:** {source_file}")
                st.write(f"**Relative path:** {relative_path}")
            with right:
                st.write(f"**Chunk index:** {chunk_index}")
                st.write(f"**Distance:** {item.distance}")
                st.write(f"**Chunk length:** {len(item.chunk.text)} characters")

            st.caption("Chunk text")
            st.code(item.chunk.text)


def _render_prompt_section() -> None:
    """Render the prompt viewer section."""
    st.subheader("Prompt Viewer")
    st.caption(
        "This panel shows what was actually sent to the model. In RAG, prompt construction "
        "is where the user question and retrieved context are combined."
    )
    query_result = st.session_state.get("query_result")
    if not query_result or not query_result.trace or not query_result.trace.prompt_payload:
        st.info("Run a query to inspect the prompt payload sent to the model.")
        return

    st.caption(
        "Prompt construction matters because retrieval does not help unless the right "
        "context is actually passed to the model in a clear, grounded form."
    )

    prompt_payload = query_result.trace.prompt_payload
    query_text, retrieved_context = _split_user_prompt(prompt_payload.user_prompt)

    st.caption(f"Model: {prompt_payload.model}")

    with st.expander("System Prompt", expanded=True):
        st.text_area(
            "System Prompt Text",
            value=prompt_payload.system_prompt,
            height=180,
            key="system_prompt_text",
        )

    with st.expander("User Query", expanded=True):
        st.text_area(
            "User Query Text",
            value=query_text,
            height=120,
            key="user_query_text",
        )

    with st.expander("Retrieved Context Passed To The Model", expanded=True):
        st.text_area(
            "Retrieved Context Text",
            value=retrieved_context,
            height=420,
            key="retrieved_context_text",
        )


def _split_user_prompt(user_prompt: str) -> tuple[str, str]:
    """Split the combined user prompt into query text and retrieved context."""
    question_marker = "Question:\n"
    context_marker = "\n\nRetrieved context:\n"
    answer_marker = "\n\nAnswer the question using the retrieved context."

    if context_marker not in user_prompt:
        return user_prompt, ""

    query_text = user_prompt
    if user_prompt.startswith(question_marker):
        query_text = user_prompt[len(question_marker) :]
    query_text, context_text = query_text.split(context_marker, maxsplit=1)

    if answer_marker in context_text:
        context_text = context_text.split(answer_marker, maxsplit=1)[0]

    return query_text.strip(), context_text.strip()


def _render_answer_section() -> None:
    """Render the answer and source references section."""
    st.subheader("Answer and Sources")
    st.caption(
        "The answer should be grounded in the retrieved evidence. Source references help you "
        "check whether the response is supported by the context that was actually retrieved."
    )
    query_result = st.session_state.get("query_result")
    if not query_result or not query_result.trace or not query_result.trace.final_answer:
        st.info("Run a query to inspect the grounded answer and sources.")
        return

    final_answer = query_result.trace.final_answer
    retrieval_results = (
        query_result.trace.retrieval_results if query_result.trace.retrieval_results else []
    )

    st.caption(
        "A grounded answer should be traceable back to retrieved source material. "
        "This panel shows the final answer alongside the file and chunk references "
        "that likely supported it."
    )

    st.markdown("### Final Answer")
    st.write(final_answer.answer)

    st.markdown("### Source References")
    if not final_answer.sources:
        st.info("No source references were returned for this answer.")
        return

    for index, source in enumerate(final_answer.sources, start=1):
        metadata = source.metadata
        relative_path = metadata.get("relative_path", "unknown")
        chunk_index = metadata.get("chunk_index", "unknown")
        matching_result = next(
            (
                item
                for item in retrieval_results
                if item.chunk.metadata.get("relative_path") == relative_path
                and item.chunk.metadata.get("chunk_index") == chunk_index
            ),
            None,
        )

        expander_label = f"Source {index} - {relative_path} - chunk {chunk_index}"
        with st.expander(expander_label, expanded=False):
            left, right = st.columns(2)
            with left:
                st.write(f"**File:** {metadata.get('file_name', 'unknown')}")
                st.write(f"**Relative path:** {relative_path}")
                st.write(f"**Chunk index:** {chunk_index}")
            with right:
                st.write(f"**Start char:** {metadata.get('start_char', 'unknown')}")
                st.write(f"**End char:** {metadata.get('end_char', 'unknown')}")
                if matching_result is not None:
                    st.write(f"**Distance:** {matching_result.distance}")

            st.caption("Source metadata")
            st.json(metadata)

            if matching_result is not None:
                st.caption("Supporting chunk text")
                st.code(matching_result.chunk.text)


def main() -> None:
    """Render the main Streamlit dashboard."""
    st.title("raglab")
    st.write(
        "A local-first RAG observability dashboard for inspecting how the pipeline "
        "works over your Markdown knowledge base."
    )

    config_path = "config.yaml"
    base_config = _load_config(config_path)

    with st.sidebar:
        st.header("Local Dashboard")
        st.write(
            "This is a local RAG observability dashboard for inspecting the pipeline, "
            "not a generic chat frontend."
        )
        st.text_input("Docs Path", value=base_config.docs_path, disabled=True)
        _render_config_summary(base_config)

        st.divider()
        st.subheader("Experiments")
        st.write("Adjust a few parameters to see how pipeline behavior changes.")
        config = _render_experiment_controls(base_config)
        current_snapshot = _config_snapshot(config)

        ingestion_snapshot = st.session_state.get("ingestion_settings")
        query_snapshot = st.session_state.get("query_settings")

        if ingestion_snapshot is not None:
            if ingestion_snapshot == current_snapshot:
                st.success(
                    f"Ingestion results reflect current settings: {_format_snapshot(ingestion_snapshot)}"
                )
            else:
                st.warning(
                    "Ingestion results are stale for the current experimental settings. "
                    "Run ingestion again to update the corpus and chunk traces."
                )

        if query_snapshot is not None:
            if query_snapshot == current_snapshot:
                st.success(
                    f"Query results reflect current settings: {_format_snapshot(query_snapshot)}"
                )
            else:
                st.warning(
                    "Query results are stale for the current experimental settings. "
                    "Run the query again to compare retrieval and answer changes."
                )

        st.divider()
        st.subheader("Ingestion")
        if st.button("Run Ingestion", use_container_width=True):
            with st.spinner("Running ingestion pipeline..."):
                st.session_state["ingestion_result"] = _run_ingestion(config)
                st.session_state["ingestion_settings"] = current_snapshot
            st.success("Ingestion complete.")

        st.divider()
        st.subheader("Query")
        question = st.text_input(
            "Question",
            value=st.session_state.get("question_input", ""),
            placeholder="Ask a question about your knowledge base...",
        )
        st.session_state["question_input"] = question
        if st.button("Run Query", use_container_width=True):
            if not question.strip():
                st.warning("Enter a question before running the query pipeline.")
            else:
                with st.spinner("Running query pipeline..."):
                    st.session_state["query_result"] = _run_query(config, question)
                    st.session_state["query_settings"] = current_snapshot
                    st.session_state["last_query_text"] = question
                st.success("Query complete.")

    if st.session_state.get("ingestion_settings") and st.session_state["ingestion_settings"] != _config_snapshot(config):
        st.info(
            "Visible ingestion artifacts were produced with older chunking settings. "
            "Re-run ingestion to refresh the corpus, chunk explorer, and downstream indexing view."
        )

    if st.session_state.get("query_settings") and st.session_state["query_settings"] != _config_snapshot(config):
        st.info(
            "Visible query artifacts were produced with older retrieval settings. "
            "Re-run the query to refresh retrieval, prompt, and answer panels."
        )

    with st.container():
        _render_corpus_section()

    with st.container():
        _render_chunk_section()

    with st.container():
        _render_retrieval_section()

    prompt_column, answer_column = st.columns(2)
    with prompt_column:
        _render_prompt_section()
    with answer_column:
        _render_answer_section()


if __name__ == "__main__":
    main()
