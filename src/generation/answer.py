"""Answer generation with strict grounding and citations.

Works with any OpenAI-compatible chat endpoint:
- Ollama (default, local, free): http://localhost:11434/v1
- OpenAI: set LLM_PROVIDER=openai and OPENAI_API_KEY
- Graceful extractive fallback when local LLM server is offline.
"""
from dataclasses import dataclass
import logging

from ..config import settings
from ..retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a pharmaceutical information assistant. Answer the user's \
question using ONLY the context excerpts below, which come from official FDA drug \
labels (DailyMed).

Rules:
- Every factual claim in your answer must be supported by the excerpts. Cite each \
claim with the excerpt number like [1], [2].
- If the excerpts do not contain the answer, say so explicitly and do not guess.
- Be concise and use plain language, but keep drug names and medical terms exact.
- End your answer with: "This is not medical advice. Consult a healthcare professional."
"""

DISCLAIMER = "This is not medical advice. Consult a healthcare professional."


@dataclass
class Answer:
    text: str
    sources: list[dict]
    model: str


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    context = "\n\n".join(
        f"[{i + 1}] Drug: {c.drug_name} | Section: {c.section_title}\n{c.content}"
        for i, c in enumerate(chunks)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context excerpts:\n{context}\n\nQuestion: {question}",
        },
    ]


def _get_client():
    from openai import OpenAI

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        return OpenAI(api_key=settings.openai_api_key), settings.openai_model
    # Ollama exposes an OpenAI-compatible API; no real key needed.
    return (
        OpenAI(base_url=settings.ollama_base_url, api_key="ollama"),
        settings.ollama_model,
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> Answer:
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
        client, model = _get_client()
        messages = build_prompt(question, chunks)
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
