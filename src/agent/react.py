"""ReAct loop (Reason + Act).

On each iteration the model either calls a tool::

    ACTION: <tool_name>
    ARGS: {"question": "..."}

or finishes::

    ANSWER: <one-sentence summary of what it found>

Tool results come back as observations. When the model answers (or the
iteration budget runs out), the accumulated chunks go through the standard
grounded generation pipeline, so citations and the medical disclaimer are
identical to /ask. In other words: the agent plans the *information
gathering*; the final answer still comes from the proven path.

Malformed turns don't crash the loop — the model gets a format error as its
observation and a chance to self-correct, which is itself very ReAct.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..embeddings.embedder import Embedder
from ..generation.answer import generate_answer, get_llm_client
from ..retrieval.retriever import RetrievedChunk, Retriever
from . import tools

log = logging.getLogger(__name__)

_ACTION_RE = re.compile(
    r"ACTION:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\nARGS:\s*(\{.*\})", re.DOTALL
)
_ANSWER_RE = re.compile(r"^\s*ANSWER:\s*(.*)$", re.DOTALL)


def _system_prompt() -> str:
    return f"""You are a pharmaceutical research agent. You answer questions about drugs by using tools — never guess from memory.

Available tools:
{tools.describe_tools()}

On each turn, reply with EXACTLY one of:

ACTION: <tool_name>
ARGS: {{"arg": "value"}}

or, when you have gathered enough label evidence to answer:

ANSWER: <one-sentence summary of what you found>

Rules:
- ARGS must be a single line of valid JSON.
- Prefer search_labels first; use fetch_drug_label when a drug you need is not indexed.
- You may call tools several times (e.g. fetch one label per drug when comparing two drugs).
- Do not repeat a tool call with identical arguments.
- Base everything on label evidence. When you have enough, output ANSWER."""


@dataclass
class ParsedStep:
    kind: str  # "answer" | "action"
    thought: str = ""
    summary: str = ""
    tool_name: str = ""
    args: dict = field(default_factory=dict)


def parse_step(text: str) -> ParsedStep:
    """Parse one model turn.

    Returns a ParsedStep; raises ValueError when the turn matches neither the
    ACTION nor the ANSWER format (caller feeds the error back as an
    observation so the model can self-correct).
    """
    m = _ANSWER_RE.match(text.strip())
    if m:
        return ParsedStep(kind="answer", summary=m.group(1).strip())

    m = _ACTION_RE.search(text)
    if m:
        name, raw_args = m.group(1), m.group(2)
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            raise ValueError(f"ARGS is not valid JSON: {e}")
        if not isinstance(args, dict):
            raise ValueError("ARGS must be a JSON object, e.g. {\"question\": \"...\"}")
        return ParsedStep(
            kind="action",
            thought=text[: m.start()].strip(),
            tool_name=name,
            args=args,
        )

    raise ValueError(
        "Reply with either 'ACTION: <tool_name>' on one line and 'ARGS: {...}' "
        "on the next, or 'ANSWER: <summary>'."
    )


@dataclass
class AgentResult:
    answer: str
    sources: list[dict]
    model: str
    trace: list[dict]  # each: iteration, thought, action, args, observation
    iterations: int
    indexed_drugs: list[str]


def _summarize_observation(tool_name: str, result: dict[str, Any]) -> str:
    """Compact the tool result for the agent's observation window.

    The full chunks are kept separately for final generation; the model only
    needs enough to plan its next step.
    """
    if tool_name == "search_labels":
        parts = [
            f"[{s['drug']} | {s['section']}] {s['preview']}"
            for s in result.get("summary", [])
        ]
        return "Found passages:\n" + "\n".join(parts) if parts else "No passages found."
    if tool_name == "fetch_drug_label":
        return result.get("note", "")
    if tool_name == "list_indexed_drugs":
        drugs = result.get("drugs", [])
        return "Indexed drugs: " + (", ".join(drugs) if drugs else "(none)")
    return str(result)[:2000]


def run_agent(
    question: str,
    retriever: Retriever | None = None,
    max_iterations: int = 8,
) -> AgentResult:
    """Run the ReAct loop for `question` and return the grounded result."""
    retriever = retriever or Retriever(embedder=Embedder(settings.embedding_model))
    ctx = tools.ToolContext(retriever=retriever)
    client, model = get_llm_client()

    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    gathered: dict[Any, RetrievedChunk] = {}

    for i in range(1, max_iterations + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=500
        )
        text = resp.choices[0].message.content.strip()

        try:
            step = parse_step(text)
        except ValueError as e:
            obs = f"Format error: {e} Try again."
            trace.append(
                {
                    "iteration": i,
                    "thought": text[:500],
                    "action": None,
                    "args": {},
                    "observation": obs,
                }
            )
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": obs})
            continue

        if step.kind == "answer":
            trace.append(
                {
                    "iteration": i,
                    "thought": step.thought,
                    "action": None,
                    "args": {},
                    "observation": "",
                }
            )
            break

        spec = tools.TOOLS.get(step.tool_name)
        if spec is None:
            obs = (
                f"Unknown tool '{step.tool_name}'. "
                f"Available: {', '.join(tools.TOOLS)}."
            )
        else:
            try:
                result = spec["function"](ctx, **step.args)
            except TypeError as e:
                obs = f"Bad arguments for '{step.tool_name}': {e}."
            except Exception as e:
                log.warning("tool %s failed: %s", step.tool_name, e)
                obs = f"Tool '{step.tool_name}' failed: {e}."
            else:
                for c in result.get("chunks", []):
                    gathered[c.id] = c
                obs = _summarize_observation(step.tool_name, result)

        trace.append(
            {
                "iteration": i,
                "thought": step.thought,
                "action": step.tool_name,
                "args": step.args,
                "observation": obs,
            }
        )
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Observation: {obs}"})
    else:
        log.info("agent hit max_iterations=%d without ANSWER", max_iterations)

    # Final answer comes from the proven grounded pipeline, over everything
    # the agent gathered — citations and disclaimer identical to /ask.
    chunks = sorted(gathered.values(), key=lambda c: c.score, reverse=True)
    chunks = chunks[: settings.rerank_top_n]
    answer = generate_answer(question, chunks)
    return AgentResult(
        answer=answer.text,
        sources=answer.sources,
        model=f"agent({answer.model})",
        trace=trace,
        iterations=len(trace),
        indexed_drugs=sorted({c.drug_name for c in gathered.values()}),
    )
