"""FastAPI service.

- POST /ask       -> grounded answer with citations.
                   Single question, or conversational: pass `history` (prior
                   turns) and follow-ups like "what about its dosage?" get
                   rewritten into standalone questions before retrieval.
- POST /ask-agent -> ReAct agent: the model plans its own tool calls
                   (search/fetch labels), then answers from what it gathered.
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..agent import react as agent_react
from ..config import settings
from ..embeddings.embedder import Embedder
from ..generation.answer import generate_answer
from ..generation.rewrite import rewrite_question
from ..ingest.ondemand import ensure_drug_indexed
from ..retrieval.retriever import Retriever
from ..store import vectorstore

_retriever: Retriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    # Models load lazily on first use; initialize ChromaDB collection here.
    vectorstore.init_db()
    yield


app = FastAPI(title="Pharma RAG", version="1.0.0", lifespan=lifespan)


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(embedder=Embedder(settings.embedding_model))
    return _retriever


class ChatMessage(BaseModel):
    """One turn of conversation history (no system messages allowed)."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    history: list[ChatMessage] | None = None
    top_k: int = Field(default=20, ge=1, le=100)
    top_n: int = Field(default=5, ge=1, le=20)
    rerank: bool = True


class Source(BaseModel):
    ref: int
    drug_name: str
    section_title: str
    score: float
    preview: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    indexed_drug: str | None = None
    rewritten_question: str | None = None


@app.get("/health")
def health():
    try:
        n = vectorstore.count_chunks()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unreachable: {e}")
    return {"status": "ok", "indexed_chunks": n}


@app.get("/drugs")
def drugs():
    try:
        return {"drugs": vectorstore.list_drugs()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unreachable: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    retriever = get_retriever()
    retriever.use_reranker = req.rerank
    history = [m.model_dump() for m in (req.history or [])]
    # Follow-ups ("what about its dosage?") become standalone questions
    # ("What is the dosage of atorvastatin?") before retrieval.
    rewritten = rewrite_question(req.question, history)
    try:
        new_drug = ensure_drug_indexed(rewritten, retriever.embedder)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Drug lookup failed: {e}")
    try:
        chunks = retriever.retrieve(rewritten, top_k=req.top_k, top_n=req.top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
    try:
        answer = generate_answer(rewritten, chunks, history=history)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return AskResponse(
        answer=answer.text,
        sources=[Source(**s) for s in answer.sources],
        model=answer.model,
        indexed_drug=new_drug,
        rewritten_question=rewritten if rewritten != req.question else None,
    )


class AgentAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    max_iterations: int = Field(default=8, ge=1, le=15)


class TraceStep(BaseModel):
    iteration: int
    thought: str = ""
    action: str | None = None
    args: dict = {}
    observation: str = ""


class AgentAskResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    trace: list[TraceStep]
    iterations: int
    indexed_drugs: list[str]


@app.post("/ask-agent", response_model=AgentAskResponse)
def ask_agent(req: AgentAskRequest):
    """ReAct agent endpoint.

    The model plans its own tool calls (search_labels, fetch_drug_label,
    list_indexed_drugs) instead of following the fixed pipeline. The trace
    shows every thought/action/observation — useful for demos and debugging.
    """
    try:
        result = agent_react.run_agent(
            req.question,
            retriever=get_retriever(),
            max_iterations=req.max_iterations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}")
    return AgentAskResponse(
        answer=result.answer,
        sources=[Source(**s) for s in result.sources],
        model=result.model,
        trace=[TraceStep(**t) for t in result.trace],
        iterations=result.iterations,
        indexed_drugs=result.indexed_drugs,
    )
