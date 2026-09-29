"""FastAPI service: POST /ask -> grounded answer with citations."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import settings
from ..embeddings.embedder import Embedder
from ..generation.answer import generate_answer
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


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
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
    try:
        chunks = retriever.retrieve(req.question, top_k=req.top_k, top_n=req.top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")
    try:
        answer = generate_answer(req.question, chunks)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return AskResponse(
        answer=answer.text,
        sources=[Source(**s) for s in answer.sources],
        model=answer.model,
    )
