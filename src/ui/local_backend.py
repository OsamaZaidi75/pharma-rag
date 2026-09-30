"""Standalone (in-process) backend for the Streamlit UI.

When API_URL is not set — e.g. on Streamlit Community Cloud, where there is
no separate FastAPI server — the UI runs the RAG pipeline directly instead of
calling HTTP endpoints. Same code paths as the API (rewrite -> on-demand
index -> retrieve -> generate), so answers are identical.

Heavy models (embeddings, reranker) load lazily on first use and are cached
across Streamlit reruns. The ChromaDB index starts empty on a fresh deploy;
the first question about a drug triggers on-demand indexing from DailyMed.
"""
import logging
import os
import sys

# Make `src.*` importable when Streamlit runs app.py as a script
# (streamlit run src/ui/app.py puts src/ui/ on sys.path, not the repo root).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st

log = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading models (first run downloads them, takes a few minutes)...")
def _get_retriever():
    from src.config import settings
    from src.embeddings.embedder import Embedder
    from src.retrieval.retriever import Retriever
    from src.store import vectorstore

    vectorstore.init_db()
    return Retriever(embedder=Embedder(settings.embedding_model))


def get_health() -> dict:
    from src.store import vectorstore

    _get_retriever()  # ensures init_db() ran
    return {"status": "ok", "indexed_chunks": vectorstore.count_chunks()}


def get_drugs() -> dict:
    from src.store import vectorstore

    _get_retriever()
    return {"drugs": vectorstore.list_drugs()}


def ask(
    question: str,
    history: list[dict] | None = None,
    top_k: int = 20,
    top_n: int = 5,
    rerank: bool = True,
) -> dict:
    from src.generation.answer import generate_answer
    from src.generation.rewrite import rewrite_question
    from src.ingest.ondemand import ensure_drug_indexed

    retriever = _get_retriever()
    retriever.use_reranker = rerank
    history = history or []
    rewritten = rewrite_question(question, history)
    new_drug = ensure_drug_indexed(rewritten, retriever.embedder)
    chunks = retriever.retrieve(rewritten, top_k=top_k, top_n=top_n)
    answer = generate_answer(rewritten, chunks, history=history)
    return {
        "answer": answer.text,
        "sources": answer.sources,
        "model": answer.model,
        "indexed_drug": new_drug,
        "rewritten_question": rewritten if rewritten != question else None,
    }


def ask_agent(question: str, max_iterations: int = 8) -> dict:
    from src.agent import react

    retriever = _get_retriever()
    try:
        result = react.run_agent(
            question, retriever=retriever, max_iterations=max_iterations
        )
    except Exception as e:
        # Most likely cause on hosted Streamlit: no LLM reachable
        # (Ollama doesn't run on Streamlit Cloud).
        raise RuntimeError(
            "Agent mode needs a chat LLM. On Streamlit Cloud, add "
            "OPENAI_API_KEY (and LLM_PROVIDER=openai) to the app's Secrets. "
            f"Details: {e}"
        )
    return {
        "answer": result.answer,
        "sources": result.sources,
        "model": result.model,
        "trace": result.trace,
        "iterations": result.iterations,
        "indexed_drugs": result.indexed_drugs,
    }
