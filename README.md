# CiteFirst RAG-pipeline — Python Corrective RAG Backend

A citation-grounded corrective Retrieval-Augmented Generation (CRAG) backend in Python. CiteFirst ingests academic and technical PDFs, decomposes them into page-aware parent and child chunks, runs dense and sparse hybrid retrieval with Reciprocal Rank Fusion (RRF), cross-encoder reranking, tri-state relevance grading, ambiguous evidence rescue, query rewriting, pre-flight combined relevance verification, and OpenRouter-backed cited answer generation.

This repository is a **standalone, backend-only** service providing both a FastAPI HTTP/SSE server with Swagger UI and standalone CLI entry points.

---

## System Architecture

```mermaid
flowchart TD
    PDF[PDF Documents in papers/] --> Ingest[PyMuPDF Page Parser]
    Ingest --> Chunk[Parent-Child Chunking]
    Chunk -->|350-word context| Parents[(ChromaDB: ieee_papers_parent)]
    Chunk -->|100-word passages| DenseEmbed[bge-small-en-v1.5]
    DenseEmbed --> Children[(ChromaDB: ieee_papers)]
    Chunk --> SparseIdx[BM25 Index & Cache]

    UserQuery[User Query / API / CLI] --> Search[Hybrid Retrieval: Dense + BM25 + RRF]
    Children -.-> Search
    SparseIdx -.-> Search
    Search --> ParentExpand[Expand to Parent Chunks]
    Parents -.-> ParentExpand
    ParentExpand --> Rerank[Cross-Encoder Reranker: bge-reranker-base]
    Rerank --> Grade{Tri-State Grading}

    Grade -->|Score >= 0.55| Correct[CORRECT]
    Grade -->|0.25 <= Score < 0.55| Ambiguous[AMBIGUOUS]
    Grade -->|Score < 0.25| Incorrect[INCORRECT]

    Ambiguous --> Rescue[Supplemental Search & Anchor Expansion]
    Rescue --> Preflight
    Correct --> Preflight{Pre-Flight Combined Relevance Check}

    Preflight -->|PASS| Gen[LLM Generation via OpenRouter]
    Preflight -->|FAIL| Rewrite[Adaptive Query Rewrite]
    Incorrect --> Rewrite
    Rewrite --> Search2[Secondary Hybrid Search & Verification]
    Search2 -->|PASS| Gen
    Search2 -->|FAIL| Bendy{Bendy Mode: Score >= 0.10}

    Bendy -->|PASS| PartialGen[Partial Labeled Answer]
    Bendy -->|FAIL| Abstain[Honest Abstention / No Info Found]

    Gen --> CitedResponse[Cited Answer + Evidence Snippets + 1-Based Page Numbers]
    PartialGen --> CitedResponse
    Abstain --> CitedResponse
```

### Key Components

1. **Page-Aware Ingestion & Numbering**: Extracts text from PDFs using PyMuPDF while recording zero-based internal page indices (`page_index`). Displays one-based physical page numbers (`page_number = page_index + 1`) across all evidence snippets, citations, and API outputs.
2. **Hierarchical Parent-Child Chunking**:
   - **Parent Chunks**: ~300–350 words, capturing surrounding narrative context without truncation.
   - **Child Chunks**: ~100 words, embedded via `BAAI/bge-small-en-v1.5` for high-precision vector search.
3. **Dense + Sparse Hybrid Search**: Combines cosine distance from ChromaDB with BM25 Okapi term scores using Reciprocal Rank Fusion (RRF).
4. **Tri-State Relevance Grading**: `BAAI/bge-reranker-base` cross-encoder grades retrieved chunks into `CORRECT` (>= 0.55), `AMBIGUOUS` (0.25–0.55), or `INCORRECT` (< 0.25).
5. **Pre-Flight Combined Relevance Verification**: Checks candidate passages collectively before passing to the generator, preventing false-positive hallucinations.
6. **Adaptive Fallback & Query Rewrite**: Rewrites ambiguous queries and executes secondary search passes before falling back to bendy mode (partial matches) or honest abstention.
7. **Safe Upload & Deletion**: PDF uploads reject duplicate filenames with HTTP 409 Conflict to protect existing documents and indices from destructive overwrite. To replace a document, delete it first via `DELETE /api/vault/delete/{filename}` and re-upload. Partial index writes on failed new uploads roll back cleanly, and physical files are preserved if index deletion fails.

---

## Quickstart (Windows PowerShell)

### 1. Prerequisites
- **Python**: Python 3.12+ (tested on Python 3.14 on Windows 64-bit).
- **PowerShell**: Windows PowerShell 5.1 or PowerShell 7+.

### 2. Environment Setup

Open PowerShell and navigate to the project directory:

```powershell
Set-Location 'C:\PROJECTS\CiteFirst RAG-pipeline'
```

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install production and test dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
```

### 3. Configure Credentials

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Edit `.env` to supply your OpenRouter API key:

```ini
OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
CITEFIRST_VECTOR_DIR=./vector_vault
```

> **Note**: An OpenRouter key is required only for live generation (`POST /api/query`, `/api/query/stream`, or direct CLI answer generation). Offline tests and retrieval evaluation run model-free without an API key.

---

## Running the Backend

### Run the FastAPI Server

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

- **Interactive Swagger Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **OpenAPI Schema**: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)
- **Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Direct CLI Ingestion

Place your PDF files into the `papers/` folder, then run batch ingestion directly:

```powershell
python database.py
```

Or upload papers via HTTP using `upload_papers.py`:

```powershell
python upload_papers.py
```

### Direct CLI Query

Execute queries directly against the vault without starting the HTTP server:

```powershell
python logic_engine.py "What is the criterion for turbulent flow?"
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health & pipeline readiness (`status: "ok"` or `"degraded"`). |
| `POST` | `/api/query` | Synchronous RAG query. Returns answer, facts, citations, and evidence. Returns `503` if pipeline dependencies are unavailable. |
| `POST` | `/api/query/stream` | Server-Sent Events (SSE) stream reporting real stage progress (`fetching`, `reranking`, `resolving`, `generating`, `done`). Returns `503` if pipeline dependencies are unavailable. |
| `POST` | `/api/vault/upload` | Upload a PDF. Rejects duplicate filenames with HTTP 409 Conflict to protect existing documents. Cleans up physical file on indexing failure. |
| `GET` | `/api/vault/list` | List all indexed and available vault papers. |
| `GET` | `/api/vault/document/{filename}` | Download or view an ingested PDF document. |
| `DELETE` | `/api/vault/delete/{filename}` | Remove a PDF from ChromaDB collections, BM25 cache, and disk. Protects physical file if index delete fails. |
| `GET` | `/api/settings` | Retrieve active pipeline thresholds and limits. |
| `POST` | `/api/settings` | Dynamically update pipeline hyperparameters. |
| `POST` | `/api/evaluate` | Start an asynchronous evaluation task against `reviewed_dataset.json`. |
| `GET` | `/api/evaluate/{task_id}` | Poll evaluation task status and view results. |

---

## Configuration (`settings.json`)

Pipeline hyperparameters are loaded from `settings.json` with fallback defaults in `settings_config.py`:

```json
{
  "FETCH_K": 30,
  "PROCESS_K": 10,
  "CHUNK_WORD_LIMIT": 300,
  "CORRECT_THRESHOLD": 0.55,
  "AMBIGUOUS_THRESHOLD": 0.25,
  "BENDY_THRESHOLD": 0.10,
  "MAX_VERIFIED_CHUNKS": 2,
  "SOFTMAX_MASS_CUTOFF": 0.90,
  "DISTANCE_THRESHOLD": 0.70,
  "TOTAL_DOCS": 60
}
```

- `FETCH_K`: Candidate chunk count fetched from vector & sparse search.
- `PROCESS_K`: Top passages evaluated by the cross-encoder reranker.
- `CORRECT_THRESHOLD`: Cross-encoder score above which a passage is deemed relevant.
- `AMBIGUOUS_THRESHOLD`: Passages between this threshold and `CORRECT_THRESHOLD` trigger ambiguous rescue.
- `BENDY_THRESHOLD`: Minimum relevance score for partial matches when all primary searches fail.
- `MAX_VERIFIED_CHUNKS`: Maximum facts passed into final LLM synthesis to prevent distraction.

---

## Offline Testing vs. Live Validation

### 1. Offline Regression Suite (Model-Free)
The test suite in `tests/` verifies API routing, HTTP status codes, security boundaries, error propagation, path traversal defense, and deterministic retrieval logic without downloading models or calling external APIs:

```powershell
python -m pytest -q tests
```

- **104 passing tests**, 3 skipped (requiring elevated symlink permissions on Windows).
- Fast execution (~20 seconds).
- Validates failure modes: pipeline down returns honest `503 Service Unavailable`, not false `200 [STUB]`.

### 2. Bytecode Verification
```powershell
python -m compileall -q api.py database.py logic_engine.py providers.py settings_config.py evaluator.py phase4_diagnostics.py upload_papers.py evaluation scripts tests
```

### 3. Live Pipeline Smoke Test
Ingests a real multi-page PDF, computes embeddings with SentenceTransformers, stores chunks in ChromaDB, queries through hybrid retrieval, reranks with cross-encoder, and generates cited answers with OpenRouter:

```powershell
python logic_engine.py "Sample test query"
```

---

## Evaluation Dataset & Historical Benchmarks

### Standalone dataset audit

The bundled `reviewed_dataset.json` contains 18 schema-valid questions. Run its audit with:

```powershell
python evaluator.py --audit-only
```

The [methodology](docs/evaluation/methodology.md) reports **zero records eligible for the default scored run** in this checked-in dataset. This audit checks the data and does not reproduce the historical benchmark.

### Historical CiteFirst evaluation (copied; not rerun here)

The figures below are copied from the [original CiteFirst evaluation section](https://github.com/Qusai7-04/CiteFirst/blob/main/README.md#evaluation-benchmark--verified-results). They were obtained with the original local corpus and evaluation protocol. The 66 PDFs and their index are not tracked in this repository, so a fresh clone cannot reproduce these scores. The standalone backend's synthetic two-page smoke test verifies operation, not benchmark accuracy.

CiteFirst was evaluated against its local benchmark corpus of **66 academic research papers** (7,816 chunks).

#### 1. Chunk-Level Relevance Evaluation

Evaluated using an LLM Grader (`meta-llama/llama-3.1-8b-instruct`) on the 66-paper corpus across **593 chunk-level relevance classifications**:

- **Accuracy**: **0.9848**
- **Precision**: **0.7000**
- **Recall**: **0.8235**
- **F1 Score**: **0.7568**

##### Confusion Matrix (593 Chunk Decisions)

| | Predicted Irrelevant | Predicted Relevant | Total Decisions |
| :--- | :---: | :---: | :---: |
| **Actual Irrelevant** | **570** (True Negatives) | **6** (False Positives) | 576 |
| **Actual Relevant** | **3** (False Negatives) | **14** (True Positives) | 17 |
| **Total Predictions** | 573 | 20 | **593** |

> [!IMPORTANT]
> **Understanding Chunk Relevance vs. Answer Correctness**:
> The high accuracy (98.48%) is driven primarily by the large true-negative denominator (570 correct rejections of irrelevant passages out of 593 total decisions; 570 + 6 + 3 + 14 = 593). This measures the chunk filter's ability to reject distractor passages. **It does NOT represent the accuracy of generated answers.**

#### 2. Answer-Accuracy Evaluation (Human Review)

Answer correctness was evaluated across **26 human-reviewed questions** with signed evidence sign-offs and structured verdict forms, partitioned into 22 development questions and a disclosed 4-question pilot:

| Partition | Questions | ACCEPT | PARTIAL | REJECT | Claim Coverage (Covered / Partial / Omitted) | Approved Pages in Context |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Development (Q001–Q022)** | 22 | **2 (9.1%)** | **9 (40.9%)** | **11 (50.0%)** | 14 / 21 / 39 (N=74) | 15 / 22 |
| **Disclosed Pilot (Q023–Q026)** | 4 | **4 (100.0%)** | **0 (0.0%)** | **0 (0.0%)** | 9 / 0 / 0 (N=9) | 4 / 4 |
| **Overall Benchmark Total** | **26** | **6 (23.1%)** | **9 (34.6%)** | **11 (42.3%)** | **23 / 21 / 39 (N=83)** | **19 / 26** |

##### Key Evaluation Findings
- **Context Presence vs. Claim Synthesis**: In 19 of 26 questions (73.1%), approved source pages successfully reached the generation context. However, multi-part synthesis requirements caused the generator to miss secondary claims or output cautious abstentions.
- **Pilot Generalizability Caveat**: The pilot questions (Q023–Q026) were inspected during prompt tuning; their 4/4 acceptance rate is disclosed as an internal pilot and must not be interpreted as an unbiased general holdout rate.

#### 3. Historical Ablation Study

The table below reflects the progressive ablation findings from the initial research study:

| Configuration | Accuracy | Precision | Recall | F1 Score | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dense-only baseline** (no logic) | 0.938 | 0.390 | 0.760 | 0.514 | High noise, low precision |
| **Hybrid dense + BM25 / RRF** | 0.885 | 0.380 | 0.860 | 0.525 | Higher recall, broader pool |
| **Hybrid + Cross-Encoder** (static thresholds) | 0.910 | 0.480 | 0.720 | 0.576 | Reranker sharpens top ranks |
| **Full Logic + Chunk-level verification** | 0.974 | 0.500 | 0.670 | 0.572 | Early gate reduced true positives |
| **Full Logic + Pre-flight combined check** | **0.985** | **0.700** | **0.820** | **0.760** | Final architecture; cuts false positives |


For detailed protocols, cryptographic manifests, and signed verdict sheets, see the [Pipeline Evaluation & Benchmark Report](docs/evaluation/evaluation_report.md) and [`docs/evaluation/phase4_questions.json`](docs/evaluation/phase4_questions.json).

---

## Troubleshooting

- **Cold Start Delay**: The first query after startup downloads or initializes `bge-small-en-v1.5` and `bge-reranker-base` into memory. Subsequent queries execute significantly faster.
- **Missing OpenRouter Key**: If generation fails with `EnvironmentError: OPENROUTER_API_KEY not found`, ensure your key is set in `.env` or process environment.
- **Windows Paths with Spaces**: Project paths use `Path(__file__).resolve().parent` anchoring, safely handling spaces and non-standard drive roots. Always quote paths when passing CLI arguments in PowerShell.

---

## License & Attribution

See `NOTICE.md` for historical CRAG-Pipeline co-author attribution and usage permissions.
