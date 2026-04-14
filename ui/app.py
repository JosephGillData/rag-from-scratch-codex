"""Minimal Streamlit entrypoint for the local RAG observability dashboard."""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="raglab",
    page_icon=":mag:",
    layout="wide",
)

st.title("raglab")
st.write(
    "A local-first RAG observability dashboard for inspecting how the pipeline works "
    "over your Markdown knowledge base."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This UI is a local, single-user inspection dashboard for the RAG pipeline."
    )
    st.write(
        "It is designed to make intermediate steps visible, including loaded files, "
        "chunking, retrieval, prompt construction, and grounded answers."
    )
    st.caption("The backend pipeline remains the source of truth.")

st.subheader("UI Scaffold")
st.write(
    "This is the initial Streamlit scaffold. Future views will expose pipeline "
    "artifacts without reimplementing backend logic."
)

left_column, right_column = st.columns(2)

with left_column:
    st.info("Planned: corpus and chunk inspection panels")

with right_column:
    st.info("Planned: retrieval, prompt, and answer inspection panels")
