"""Conversational query reformulation.

Turns a follow-up question into a standalone question using the conversation
history, so the retriever (which is stateless) gets something searchable.

Example:
    history:  "What are the contraindications of atorvastatin?"
    question: "what about its dosage?"
    -> "What is the dosage of atorvastatin?"
"""
import logging

from .answer import get_llm_client

log = logging.getLogger(__name__)

REWRITE_SYSTEM = """You rewrite a follow-up question into a standalone question, using the conversation history for context.

Rules:
- Output ONLY the rewritten question. No preamble, no quotation marks, no explanation.
- Resolve pronouns (it, they, this drug) and any omitted drug names using the history.
- If the question is already standalone and needs no history to understand, return it unchanged.
- Never answer the question. Only rewrite it."""


def rewrite_question(question: str, history: list[dict] | None) -> str:
    """Return a standalone version of `question`.

    Skips the LLM call entirely when there is no history, and falls back to
    the original question if the rewrite call fails.
    """
    history = [
        m for m in (history or [])
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
    ]
    if not history:
        return question

    try:
        client, model = get_llm_client()
        messages = [{"role": "system", "content": REWRITE_SYSTEM}]
        messages.extend(history[-10:])
        messages.append(
            {
                "role": "user",
                "content": f"Follow-up question: {question}\nRewritten standalone question:",
            }
        )
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=150
        )
        rewritten = resp.choices[0].message.content.strip().strip('"').strip()
        return rewritten or question
    except Exception as e:
        log.warning("query rewrite failed (%s); using original question", e)
        return question
