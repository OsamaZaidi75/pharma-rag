"""Tool registry for the ReAct agent.

Each tool is a plain Python function wrapping an existing module:
- search_labels     -> retrieval.retriever (hybrid search + rerank)
- fetch_drug_label  -> ingest.ondemand (download + index a missing label)
- list_indexed_drugs -> store.vectorstore

The agent only ever sees the TOOLS registry (name + description + args);
the implementations stay ordinary code. That separation is the whole idea:
tools are *capabilities*, the model supplies the *judgement* about when to
use them. Adding a new capability = writing a function + one registry entry.
"""
import logging
from dataclasses import dataclass
from typing import Any

from ..ingest.ondemand import ensure_drug_indexed
from ..retrieval.retriever import RetrievedChunk, Retriever
from ..store import vectorstore

log = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Shared handles the tools need. Created once per agent run and passed in,
    so tools stay pure functions (easy to test, no hidden globals)."""

    retriever: Retriever


def search_labels(ctx: ToolContext, question: str, top_n: int = 5) -> dict[str, Any]:
    """Search the indexed FDA drug labels for passages relevant to `question`.

    Returns the full RetrievedChunk objects (kept for the final grounded
    answer) plus a truncated summary for the agent's observation window.
    """
    chunks: list[RetrievedChunk] = ctx.retriever.retrieve(question, top_n=top_n)
    return {
        "chunks": chunks,
        "summary": [
            {
                "drug": c.drug_name,
                "section": c.section_title,
                "preview": c.content[:600],
            }
            for c in chunks
        ],
    }


def fetch_drug_label(ctx: ToolContext, drug_name: str) -> dict[str, Any]:
    """Download the official FDA label for `drug_name` from DailyMed and index
    it, when it isn't already indexed. Afterwards search_labels can find it."""
    indexed = ensure_drug_indexed(drug_name, ctx.retriever.embedder)
    if indexed:
        return {
            "indexed_drug": indexed,
            "note": (
                f"Downloaded and indexed the FDA label for '{indexed}'. "
                "You can now use search_labels to find passages in it."
            ),
        }
    return {
        "indexed_drug": None,
        "note": (
            f"No FDA label found for '{drug_name}' (it may already be indexed "
            "— check with list_indexed_drugs)."
        ),
    }


def list_indexed_drugs(ctx: ToolContext) -> dict[str, Any]:
    """List the drug names currently in the index."""
    return {"drugs": vectorstore.list_drugs()}


TOOLS: dict[str, dict[str, Any]] = {
    "search_labels": {
        "description": (
            'Search the indexed FDA drug labels for passages relevant to a question. '
            'Arguments: {"question": string, "top_n": number of passages (default 5)}.'
        ),
        "function": search_labels,
    },
    "fetch_drug_label": {
        "description": (
            "Download the official FDA label for a drug from DailyMed and index it, "
            'when the drug you need is not indexed yet. Arguments: {"drug_name": string}. '
            "After this succeeds, use search_labels on that drug."
        ),
        "function": fetch_drug_label,
    },
    "list_indexed_drugs": {
        "description": (
            "List the drug names currently indexed, so you know whether to search "
            "directly or fetch a label first. Takes no arguments: {}."
        ),
        "function": list_indexed_drugs,
    },
}


def describe_tools() -> str:
    """Render the registry as the tool section of the agent's system prompt.

    Generated from TOOLS so the prompt and the code can never drift apart.
    """
    return "\n".join(f"- {name}: {spec['description']}" for name, spec in TOOLS.items())
