"""Answer generation with strict grounding and citations.

Works with any OpenAI-compatible chat endpoint:
- Ollama (default, local, free): http://localhost:11434/v1
- OpenAI: set LLM_PROVIDER=openai and OPENAI_API_KEY
- Gemini: set LLM_PROVIDER=gemini and GEMINI_API_KEY
  (uses Google's OpenAI-compatible endpoint; free tier is fine for demos)
- Graceful extractive fallback when the LLM is unreachable.

Conversational mode: pass `history` (prior user/assistant turns) and the model
can resolve follow-ups ("what about its dosage?"). History is only ever used
to *understand* the question — every factual claim must still come from the
retrieved excerpts.
"""
from dataclasses import dataclass
import logging

from ..config import settings
from ..retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a pharmaceutical information assistant. Answer the user's question using ONLY the context excerpts below, which come from official FDA drug labels (DailyMed).

Rules:
- Every factual claim in your answer must be supported by the excerpts. Cite each claim with the excerpt number like [1], [2].
- If the excerpts do not contain the answer, say so explicitly and do not guess.
- The conversation history (if any) exists only to help you understand follow-up questions — e.g. what "it" or "this drug" refers to. Do not treat anything said in the history as a fact; facts come from the excerpts alone.
- Be concise and use plain language, but keep drug names and medical terms exact.
- End your answer with: "This is not medical advice. Consult a healthcare professional."
"""

DISCLAIMER = "This is not medical advice. Consult a healthcare professional."


@dataclass
class Answer:
    text: str
    sources: list[dict]
    model: str


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> list[dict]:
    """System prompt + optional conversation history + grounded question."""
    context = "\n\n".join(
        f"[{i + 1}] Drug: {c.drug_name} | Section: {c.section_title}\n{c.content}"
        for i, c in enumerate(chunks)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # History is bounded and sanitized: only user/assistant turns, no system.
    for m in (history or [])[-10:]:
        if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip():
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"Context excerpts:\n{context}\n\nQuestion: {question}",
        }
    )
    return messages


def get_llm_client():
    """Return (client, model_name) for the configured provider."""
    from openai import OpenAI

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAI(api_key=settings.openai_api_key), settings.openai_model
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set")
        # Gemini exposes an OpenAI-compatible endpoint, so the same client works.
        return (
            OpenAI(
                api_key=settings.gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            settings.gemini_model,
        )
    # Ollama exposes an OpenAI-compatible API; no real key needed.
    return (
        OpenAI(base_url=settings.ollama_base_url, api_key="ollama"),
        settings.ollama_model,
    )


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> Answer:
    if not chunks:
        return Answer(
            text=(
                "I couldn't find relevant information in the drug labels I have "
                f"indexed. {DISCLAIMER}"
            ),
            sources=[],
            model="none",
        )

    sources = [
        {
            "ref": i + 1,
            "drug_name": c.drug_name,
            "section_title": c.section_title,
            "score": round(c.score, 4),
            "preview": c.content[:300],
        }
        for i, c in enumerate(chunks)
    ]

    try:
        client, model = get_llm_client()
        messages = build_prompt(question, chunks, history=history)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.1, max_tokens=800
        )
        text = resp.choices[0].message.content.strip()
        used_model = model
    except Exception as e:
        logger.warning("LLM call failed (%s); using extractive fallback. Error: %s", settings.llm_provider, e)
        # Resilient extractive synthesis directly from top retrieved chunks
        lines = [
            f"According to the official FDA drug label for **{chunks[0].drug_name.capitalize()}** (*{chunks[0].section_title}*):",
            "",
        ]
        for i, c in enumerate(chunks[:3]):
            body = c.content.split("\n", 1)[-1].strip()
            sentences = [s.strip() for s in body.split(". ") if len(s.strip()) > 10]
            excerpt = ". ".join(sentences[:2]).strip()
            if excerpt and not excerpt.endswith("."):
                excerpt += "."
            if excerpt:
                lines.append(f"• [{i + 1}] {excerpt}")
        text = "\n".join(lines)
        used_model = f"extractive-fallback ({settings.llm_provider} offline)"

    if DISCLAIMER not in text:
        text = text.rstrip() + f"\n\n{DISCLAIMER}"

    return Answer(text=text, sources=sources, model=used_model)
