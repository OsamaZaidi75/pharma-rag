# 💊 Pharma RAG — Grounded Clinical Q&A over FDA Drug Labels

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange.svg)](https://www.trychroma.com/)
[![Tests](https://img.shields.io/badge/Tests-24%20Passed-brightgreen.svg)]()

Ask complex clinical and pharmacological questions about prescription drugs and receive **hallucination-resistant answers** strictly grounded in official FDA drug labels (DailyMed Structured Product Label XMLs), complete with exact citations and section breadcrumbs.

---

## 📐 Architecture & Workflow

```
User Question
      │
      ▼
┌─────────────────────────────────────────┐
│ Query Processing & Embedding Generation │  (BAAI/bge-small-en-v1.5, 384-dim)
└────────────────────┬────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Hybrid Retrieval (Candidate Generation)                                │
│  ├── Dense Vector Search (ChromaDB Cosine Space)                       │
│  └── Sparse Keyword Search (BM25Okapi over Tokenized Corpus)           │
│                                                                        │
│ ➡️ Reciprocal Rank Fusion (RRF, k=60):                                  │
│    RRF_Score(d) = Σ [ 1 / (60 + rank_i(d)) ]                           │
└────────────────────┬───────────────────────────────────────────────────┘
                     │ Top-K Candidates (Default: 20)
                     ▼
┌─────────────────────────────────────────┐
│ Cross-Encoder Reranking                 │  (ms-marco-MiniLM-L-6-v2)
│ Re-scores query-document pairs          │
└────────────────────┬────────────────────┘
                     │ Top-N Reranked Excerpts (Default: 5)
                     ▼
┌─────────────────────────────────────────┐
│ Grounded LLM Generation & Verification  │  (Ollama llama3.1 / OpenAI GPT-4o-mini)
│ Strict grounding + bracketed citations  │  (with extractive synthesis fallback)
└────────────────────┬────────────────────┘
                     │
                     ▼
  Grounded Answer + Section Citations + Medical Disclaimer
```

---

## 💡 Key Design Decisions

| Component | Technical Choice | Rationale & Tradeoffs |
|---|---|---|
| **Data Source** | DailyMed SPL (HL7 v3 XML) | Official FDA-approved package inserts; structured section LOINC codes and hierarchy; no scraping required. |
| **Chunking Strategy** | Section-aware with context header | Preserves clinical section integrity (Contraindications, Adverse Reactions, Dosage) while maintaining breadcrumbs (`Drug: X \| Section: Y`). |
| **Retrieval Engine** | Hybrid (Dense Vector + Sparse BM25) | Dense vectors capture semantic intent, while BM25 guarantees precision for exact pharmaceutical terms and brand names. |
| **Fusion Mechanism** | Reciprocal Rank Fusion (RRF) | Combines non-calibrated vector similarity and BM25 scores without requiring complex score normalization. |
| **Precision Layer** | Cross-Encoder Reranker | Full cross-attention over query-document pairs re-ranks top 20 candidates down to the 5 most relevant excerpts. |
| **Generation Guardrails** | Strict Grounded Prompting + Fallback | Demands explicit citation tags (`[1]`, `[2]`), refuses out-of-context facts, appends medical disclaimers, and gracefully degrades to extractive synthesis when LLM servers are offline. |
| **Evaluation Suite** | Hit@K & Mean Reciprocal Rank (MRR) | Automated golden set evaluation (12 clinical queries) comparing hybrid vs. reranked retrieval pipelines. |

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment

```bash
# Clone repository
git clone <YOUR_GITHUB_REPO_URL>
cd pharma-rag

# Create virtual environment
# On Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy template
# On Linux/macOS:
cp .env.example .env

# On Windows (PowerShell / CMD):
copy .env.example .env
```

*(Optional: Edit `.env` to configure OpenAI API key, Ollama endpoint, or chunking/retrieval parameters.)*

### 4. Ingest FDA Drug Labels

Download official DailyMed SPL XML files for your target drugs:

```bash
python scripts/download_labels.py --drugs atorvastatin metformin lisinopril amlodipine omeprazole
```

### 5. Build the Vector & Keyword Index

Parse XMLs, generate section-aware chunks, calculate embeddings with `bge-small-en-v1.5`, and populate ChromaDB:

```bash
python scripts/build_index.py
```

To drop and rebuild the index from scratch:
```bash
python scripts/build_index.py --rebuild
```

### 6. Run LLM Backend (Ollama or OpenAI)

**Option A: Local & Free with Ollama (Default)**
```bash
ollama pull llama3.1
ollama serve
```

**Option B: OpenAI API**
Update your `.env`:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

---

## 🖥️ Running the Application

### Start the FastAPI Backend
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Start the Streamlit User Interface
```bash
streamlit run src/ui/app.py
```
Open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🔌 API Reference

### 1. Health Check
`GET /health`
```json
{
  "status": "ok",
  "indexed_chunks": 184
}
```

### 2. List Indexed Drugs
`GET /drugs`
```json
{
  "drugs": ["amlodipine", "atorvastatin", "lisinopril", "metformin", "omeprazole"]
}
```

### 3. Ask Question
`POST /ask`

**Request Body:**
```json
{
  "question": "What are the contraindications for atorvastatin?",
  "top_k": 20,
  "top_n": 5,
  "rerank": true
}
```

**Response:**
```json
{
  "answer": "Atorvastatin is contraindicated in patients with hypersensitivity to any component of this medication [1]. It is also contraindicated in patients with active liver disease, including unexplained persistent elevations in hepatic transaminase levels [2].\n\nThis is not medical advice. Consult a healthcare professional.",
  "sources": [
    {
      "ref": 1,
      "drug_name": "atorvastatin",
      "section_title": "Contraindications",
      "score": 0.9412,
      "preview": "Drug: atorvastatin | Section: Contraindications\nAtorvastatin is contraindicated in patients with hypersensitivity to any component of this product..."
    },
    {
      "ref": 2,
      "drug_name": "atorvastatin",
      "section_title": "Warnings and Precautions > Liver Dysfunction",
      "score": 0.8875,
      "preview": "Drug: atorvastatin | Section: Warnings and Precautions > Liver Dysfunction\nIncreases in serum transaminases have been reported..."
    }
  ],
  "model": "llama3.1"
}
```

---

## 🧪 Testing & Retrieval Evaluations

### Run Unit Tests
```bash
python -m pytest tests/
```
Output:
```
============================= 24 passed in 20.42s =============================
```

### Run Retrieval Benchmark against Golden Dataset
The project includes a curated golden evaluation set (`src/evals/golden.json`) measuring **Hit@K** and **Mean Reciprocal Rank (MRR)**:

```bash
# Evaluate full pipeline with Cross-Encoder reranker
python -m src.evals.run_evals

# Ablation study: evaluate pure Hybrid Search (reranker disabled)
python -m src.evals.run_evals --no-rerank
```

---

## 📂 Project Structure

```
pharma-rag/
├── data/
│   ├── sample/
│   │   └── sample_spl.xml          # Mock SPL XML for isolated unit tests
│   └── labels/                     # Downloaded DailyMed XMLs (gitignored)
├── scripts/
│   ├── download_labels.py          # DailyMed REST API ingest tool
│   └── build_index.py              # Parsing, chunking, embedding, indexing
├── src/
│   ├── __init__.py
│   ├── config.py                   # Pydantic BaseSettings & env configuration
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                 # FastAPI service endpoints
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py             # SentenceTransformers BGE wrapper
│   ├── evals/
│   │   ├── __init__.py
│   │   ├── golden.json             # Golden Q&A evaluation benchmark
│   │   └── run_evals.py            # Hit@K and MRR benchmark runner
│   ├── generation/
│   │   ├── __init__.py
│   │   └── answer.py               # Grounded answer synthesis & prompt engineering
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── chunk.py                # Smart section-aware recursive chunker
│   │   ├── dailymed.py             # DailyMed API client
│   │   └── spl_parser.py           # HL7 v3 XML parser & section tree walker
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py            # Hybrid retrieval + Cross-Encoder reranker
│   ├── store/
│   │   ├── __init__.py
│   │   └── vectorstore.py          # ChromaDB + BM25 Okapi + RRF engine
│   └── ui/
│       ├── __init__.py
│       └── app.py                  # Streamlit interactive web dashboard
├── tests/
│   ├── __init__.py
│   ├── test_api.py                 # FastAPI integration tests
│   ├── test_chunk.py               # XML parsing & chunking tests
│   ├── test_embedder_reranker.py   # Embedding dimension & reranking tests
│   ├── test_evals.py               # Evaluation metric tests
│   ├── test_generation.py          # LLM prompt & fallback tests
│   └── test_vectorstore.py         # ChromaDB, BM25 & RRF fusion tests
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore specifications
├── requirements.txt                # Python package dependencies
└── README.md                       # Documentation and architecture guide
```

---

## ⚙️ Configuration Options (`.env`)

| Variable | Default | Description |
|---|---|---|
| `CHROMA_DIR` | `./data/chroma` | Persistent ChromaDB storage path |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Hugging Face embedding model identifier |
| `EMBEDDING_DIM` | `384` | Embedding vector dimensionality |
| `LLM_PROVIDER` | `ollama` | Generation provider (`ollama` or `openai`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `llama3.1` | Local Ollama model tag |
| `OPENAI_API_KEY` | `None` | OpenAI API key (when `LLM_PROVIDER=openai`) |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model name |
| `RETRIEVAL_TOP_K` | `20` | Number of candidates fetched via hybrid search |
| `RERANK_TOP_N` | `5` | Number of candidates retained after reranking |
| `USE_RERANKER` | `true` | Enable/disable Cross-Encoder reranking |
| `CHUNK_MAX_CHARS` | `1200` | Maximum character length per chunk |
| `CHUNK_OVERLAP_CHARS`| `200` | Sentence overlap character count |
| `API_URL` | `http://localhost:8000` | Streamlit backend URL |

---

## ⚠️ Medical & Clinical Disclaimer

> **IMPORTANT:** This application is designed solely for informational, research, and portfolio demonstration purposes. It does not provide medical advice, diagnosis, or treatment plans. All extracted information reflects indexed label documents at the time of retrieval. Always consult a licensed healthcare professional or pharmacist for medical advice.

---

## 📄 License

MIT License — free for educational, research, and portfolio use.
