# 💊 Pharma RAG — Grounded Q&A over FDA Drug Labels

Ask questions about prescription drugs and get answers grounded in official FDA
labels (DailyMed SPL documents), with citations to the exact label sections.
Built as a portfolio project for AI/LLM engineer roles.

```
Question
   │
   ▼
┌─────────────┐   ┌──────────────────┐   ┌───────────────┐   ┌────────────┐
│ Embed query │──▶│ Hybrid retrieval │──▶│ Cross-encoder │──▶│ LLM answer │
│ (bge-small) │   │ vector + keyword │   │ rerank        │   │ + citations│
└─────────────┘   │ fused with RRF   │   └───────────────┘   └────────────┘
                  └──────────────────┘
                           │
                    ChromaDB + BM25
                    (SPL sections, chunked)
```

## Why this design

| Decision | Reason |
|---|---|
| DailyMed SPL XML, not web scraping | Official FDA labels, structured sections, stable API, no auth |
| Chunk by label section | Keeps contraindications/dosage/warnings topically pure; better than fixed windows |
| Hybrid search (ChromaDB cosine + BM25 keyword, RRF fusion) | Drug names are exact-match heavy ("atorvastatin" vs "a statin") — keyword search catches what vectors miss *(Tradeoff: BM25 runs over the full corpus in Python, fine at this scale)* |
| Cross-encoder rerank | Cheap precision boost on the top-20 before generation |
| Local embeddings (bge-small) + Ollama | Whole pipeline runs at **₹0** without any Docker dependency |
| Strict grounded prompting | Every claim cited `[1]`; "not in the labels" instead of hallucinating; medical disclaimer appended |
| Hand-rolled evals (hit@k, MRR) | Retrieval quality measured against a golden set, not vibes |

## Quickstart

**1. Install & configure**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**2. Download labels**
```bash
python scripts/download_labels.py --drugs atorvastatin metformin lisinopril
```

**3. Build the index** (downloads bge-small ~130MB once, CPU fine)
```bash
python scripts/build_index.py
```

**4. Run the LLM** (Ollama, local and free)
```bash
ollama pull llama3.1 && ollama serve
# ...or set LLM_PROVIDER=openai + OPENAI_API_KEY in .env
```

**5. Ask questions**
```bash
uvicorn src.api.main:app --reload          # API on :8000
streamlit run src/ui/app.py               # UI (standalone; add API_URL env to use the API instead)
```
Try: *"What are the contraindications of atorvastatin?"*

The UI has three modes (sidebar): **Ask** (single question), **Chat** (remembers the
conversation — try a follow-up like *"what about its dosage?"*), and **Agent** (ReAct:
the model plans its own tool calls, with a visible trace).

## Deploying the demo (Streamlit Community Cloud)

The Streamlit app runs **standalone** — no FastAPI server needed:

1. On [share.streamlit.io](https://share.streamlit.io), choose **Deploy an app**,
   repository `OsamaZaidi75/pharma-rag`, branch `main`,
   **Main file path: `src/ui/app.py`** (forward slashes — this is Linux).
2. Hit **Deploy**. First load takes a few minutes (embedding + reranker models download once).
3. The index starts empty; the first question about any drug triggers on-demand
   indexing from DailyMed (~30–60s), then it's instant.

Notes:
- Without an LLM, answers use the built-in extractive fallback (still cited).
  For full LLM answers on the cloud, add secrets in the app's **Secrets** page
  (Streamlit exposes secrets as env vars, which the settings pick up) — either:
  - Gemini (free tier): `LLM_PROVIDER=gemini` + `GEMINI_API_KEY` (get one at
    [Google AI Studio](https://aistudio.google.com/apikey)), or
  - OpenAI: `LLM_PROVIDER=openai` + `OPENAI_API_KEY`.
- Agent mode needs a chat LLM, so on the cloud it requires one of the secrets above.
- To run the UI against your own API server instead, set the `API_URL` secret/env
  (e.g. `http://localhost:8000`) — the UI switches to API mode automatically.

**6. Run evals**
```bash
pytest tests/                             # unit tests
python -m src.evals.run_evals             # retrieval hit@k / MRR on golden set
python -m src.evals.run_evals --no-rerank # ablation: reranker off
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status and indexed chunk count |
| `GET` | `/drugs` | Drugs currently indexed |
| `POST` | `/ask` | Grounded answer with citations; supports chat history and follow-ups |
| `POST` | `/ask-agent` | ReAct agent: plans its own tool calls, returns answer + full trace |

- `POST /ask` → `{answer, sources[{ref, drug_name, section_title, score, preview}], model}`
  - accepts optional `history: [{role: "user"|"assistant", content}]` for chat mode;
    follow-ups are rewritten into standalone questions before retrieval, and the
    response includes `rewritten_question` when a rewrite happened.
- `POST /ask-agent` → `{answer, sources, model, trace[{iteration, thought, action, args, observation}], iterations, indexed_drugs}`
  - ReAct agent: the model plans its own tool calls (`search_labels`,
    `fetch_drug_label`, `list_indexed_drugs`) instead of following the fixed
    pipeline. The trace shows every step — great for demos and debugging.

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "Which drugs interact with atorvastatin?"}'

# Chat mode: follow-up resolved against history
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' -d '{
  "question": "what about its dosage?",
  "history": [{"role": "user", "content": "What are the contraindications of atorvastatin?"},
              {"role": "assistant", "content": "Atorvastatin is contraindicated in ..."}]
}'

# Agent mode: compare two drugs (the agent fetches + searches on its own)
curl -X POST localhost:8000/ask-agent -H 'Content-Type: application/json' \
  -d '{"question": "Compare the side effects of atorvastatin and rosuvastatin."}'

# Terminal demo of the agent loop
python scripts/agent_chat.py "Compare the side effects of atorvastatin and rosuvastatin"
```

## Modes: Ask, Chat, Agent

The project now demonstrates all three patterns from the chatbot → agent ladder:

| | Ask (RAG pipeline) | Chat (chatbot) | Agent (ReAct) |
|---|---|---|---|
| Control flow | Fixed: retrieve → rerank → generate | Fixed pipeline + memory | Model decides: think → act → observe → … |
| Memory | None | Conversation history (`st.session_state`); follow-ups rewritten via `generation/rewrite.py` | Tool-call trace; gathers its own evidence |
| New code | — | `src/generation/rewrite.py`, `history` in `/ask` + UI chat mode | `src/agent/` (`tools.py` registry + `react.py` loop), `/ask-agent` |
| Good for | One-shot questions | Follow-ups: "what about its dosage?" | Multi-step: "compare X and Y", unindexed drugs |

Grounding is identical in all three: the agent's gathered chunks go through the same
`generate_answer` pipeline, so citations and the medical disclaimer never change.

## Project layout

```
src/
  ingest/      dailymed.py (API client) · spl_parser.py (SPL XML → sections) · chunk.py
  embeddings/  embedder.py (sentence-transformers, lazy-loaded)
  store/       vectorstore.py (ChromaDB + BM25 keyword search, RRF hybrid search)
  retrieval/   retriever.py (hybrid → cross-encoder rerank)
  generation/  answer.py (grounded prompting, Ollama/OpenAI) · rewrite.py (follow-up → standalone question)
  agent/       tools.py (tool registry: search/fetch/list) · react.py (ReAct think→act→observe loop)
  api/         main.py (FastAPI: /health, /drugs, /ask, /ask-agent)
  ui/          app.py (Streamlit demo: Ask / Chat / Agent modes)
  evals/       golden.json · run_evals.py
scripts/       download_labels.py · build_index.py · agent_chat.py (terminal agent demo)
tests/         unit tests (parser, chunker, RRF, API, generation, ReAct parser)
```

## Known limitations (honest)

- Answers only cover indexed labels; add more drugs via `download_labels.py`.
- Brand ↔ generic name resolution is naive (DailyMed search handles most cases).
- No dosage *calculations* — the model quotes label text; it must never compute a dose.
- Eval set is small (12 questions); grow it as you index more drugs.
- This is a demo, not a medical device. The disclaimer is load-bearing.

## Cost

₹0 with Ollama + local embeddings. With OpenAI (`gpt-4o-mini`), roughly ₹2–5 per 100 questions.
