"""
FastAPI backend service for the CiteFirst citation-grounded research RAG pipeline.
Run locally with: uvicorn api:app --reload --host 127.0.0.1 --port 8000
"""
import json
import os
import uuid
import time
import asyncio
import threading
import re
import tempfile
import logging
from functools import partial
from email.message import Message
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from settings_config import DEFAULTS, load_settings, save_settings

log = logging.getLogger(__name__)

app = FastAPI(title="CiteFirst — Research Assistant with Citations API")

cors_origins_env = os.getenv("CITEFIRST_CORS_ORIGINS")
if cors_origins_env:
    ALLOWED_ORIGINS = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    # Default local development origins
    ALLOWED_ORIGINS = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


@app.get("/static/{filename}", include_in_schema=False)
def evaluation_chart(filename: str):
    """Serve only evaluator chart PNGs from the artifacts directory."""
    if not re.fullmatch(r"(?:metrics_chart|confusion_matrix)_[a-zA-Z0-9_.-]+\.png", filename):
        raise HTTPException(status_code=404, detail="Chart not found")
    entry = ARTIFACTS_DIR / filename
    chart = entry.resolve()
    if entry.is_symlink() or chart.parent != ARTIFACTS_DIR.resolve() or not chart.is_file():
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(chart, media_type="image/png")

# ── Settings file path ────────────────────────────────────────────────────────
def _load_settings() -> dict:
    return load_settings()


def _save_settings(data: dict):
    save_settings(data)


# ── Lazy-load CiteFirst pipeline modules ─────────────────────────────────────
try:
    from logic_engine import run_logic_engine, reload_config  # type: ignore
    from database import process_and_save_pdf, delete_document, list_documents, document_exists  # type: ignore
    PIPELINE_AVAILABLE = True
except Exception as e:
    PIPELINE_AVAILABLE = False
    log.warning("CiteFirst pipeline modules unavailable; endpoints will return 503 Service Unavailable: %s", e)

# Backwards-compatible alias for existing consumers
CRAG_AVAILABLE = PIPELINE_AVAILABLE
crag_available = PIPELINE_AVAILABLE
pipeline_available = PIPELINE_AVAILABLE

from evaluator import run_ml_evaluation


# ── Pydantic models ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    file_filter: Optional[str] = None


class SettingsPayload(BaseModel):
    FETCH_K: int = DEFAULTS["FETCH_K"]
    PROCESS_K: int = DEFAULTS["PROCESS_K"]
    CHUNK_WORD_LIMIT: int = DEFAULTS["CHUNK_WORD_LIMIT"]
    CORRECT_THRESHOLD: float = DEFAULTS["CORRECT_THRESHOLD"]
    AMBIGUOUS_THRESHOLD: float = DEFAULTS["AMBIGUOUS_THRESHOLD"]
    BENDY_THRESHOLD: float = DEFAULTS["BENDY_THRESHOLD"]
    MAX_VERIFIED_CHUNKS: int = DEFAULTS["MAX_VERIFIED_CHUNKS"]
    SOFTMAX_MASS_CUTOFF: float = DEFAULTS["SOFTMAX_MASS_CUTOFF"]
    DISTANCE_THRESHOLD: float = DEFAULTS["DISTANCE_THRESHOLD"]
    TOTAL_DOCS: int = DEFAULTS["TOTAL_DOCS"]


# ── In-memory evaluation task store ──────────────────────────────────────────
_eval_tasks: dict[str, dict] = {}
_eval_lock = threading.Lock()
EVAL_RESULT_TTL_SECONDS = 3600
MAX_COMPLETED_EVAL_TASKS = 100


def _prune_eval_tasks():
    """Retain active tasks and the most recent bounded set of completed results."""
    with _eval_lock:
        now = time.monotonic()
        completed = sorted(
            ((task_id, task["_finished_at"]) for task_id, task in _eval_tasks.items()
             if "_finished_at" in task), key=lambda pair: pair[1], reverse=True)
        for task_id, finished_at in completed:
            if now - finished_at >= EVAL_RESULT_TTL_SECONDS:
                _eval_tasks.pop(task_id, None)
        for task_id, _ in completed[MAX_COMPLETED_EVAL_TASKS:]:
            _eval_tasks.pop(task_id, None)


# ── Health & Readiness ───────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    ready = bool(PIPELINE_AVAILABLE and CRAG_AVAILABLE)
    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "pipeline_available": ready,
        "crag_available": ready,  # Deprecated compatibility alias
        "detail": "Pipeline operational" if ready else "Pipeline modules unavailable; install dependencies",
    }


# ── Query (non-streaming) ───────────────────────────────────────────────────
@app.post("/api/query")
def query(req: QueryRequest):
    if not PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Pipeline modules unavailable. Verify model and database dependencies are installed."
        )
    try:
        return run_logic_engine(req.question, file_filter=req.file_filter)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Query (SSE streaming) ───────────────────────────────────────────────────
@app.post("/api/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    """
    Server-Sent Events endpoint.
    Emits real pipeline stage transitions, then the result.
    """
    if not PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Pipeline modules unavailable. Verify model and database dependencies are installed."
        )

    async def event_generator():
        try:
            loop = asyncio.get_event_loop()
            events = asyncio.Queue()

            def on_stage(step, detail):
                loop.call_soon_threadsafe(events.put_nowait, {"step": step, "detail": detail})

            work = loop.run_in_executor(None, partial(run_logic_engine, req.question,
                                                     file_filter=req.file_filter, on_stage=on_stage))
            while not work.done() or not events.empty():
                if await request.is_disconnected():
                    work.cancel()
                    return
                try:
                    event = await asyncio.wait_for(events.get(), timeout=0.25)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    continue
            result = await work
        except Exception as e:
            result = {
                "status": "failed",
                "answer": f"Pipeline error: {str(e)}",
                "facts": [], "sources": [], "evidence": [], "query_used": req.question,
            }

        done_event = {"step": "done", "result": result}
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Evaluation (async task) ──────────────────────────────────────────────────
@app.post("/api/evaluate")
def start_evaluation():
    _prune_eval_tasks()
    task_id = str(uuid.uuid4())
    _eval_tasks[task_id] = {"status": "pending", "result": None}

    def run_eval():
        _eval_tasks[task_id]["status"] = "running"
        try:
            _eval_tasks[task_id]["result"] = run_ml_evaluation()
            _eval_tasks[task_id]["status"] = "completed"
        except Exception as e:
            _eval_tasks[task_id]["status"] = "failed"
            _eval_tasks[task_id]["result"] = {"error": str(e)}
        finally:
            _eval_tasks[task_id]["_finished_at"] = time.monotonic()
            _prune_eval_tasks()

    thread = threading.Thread(target=run_eval, daemon=True)
    thread.start()

    return {"task_id": task_id}


@app.get("/api/evaluate/{task_id}")
def get_evaluation_status(task_id: str):
    _prune_eval_tasks()
    task = _eval_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": task["status"], "result": task["result"]}


# ── Vault ────────────────────────────────────────────────────────────────────
PAPERS_DIR = (Path(__file__).resolve().parent / "papers").resolve()


def _paper_path(filename: str) -> tuple[Path, Path]:
    """Reject path syntax before resolving a document inside the paper directory."""
    if (not filename or filename in {".", ".."} or "/" in filename or "\\" in filename
            or ":" in filename or any(ord(c) < 32 for c in filename)
            or not filename.lower().endswith(".pdf")):
        raise HTTPException(status_code=400, detail="Invalid PDF filename")
    if PAPERS_DIR.is_symlink():
        raise HTTPException(status_code=400, detail="Paper directory must not be a symlink")
    root = PAPERS_DIR.resolve()
    path = root / filename
    if path.resolve().parent != root or path.is_symlink():
        raise HTTPException(status_code=400, detail="Paper path must stay inside the paper directory")
    return root, path


@app.get("/api/vault/list")
def vault_list():
    docs = []

    # Always scan papers folder
    papers_dir = PAPERS_DIR
    pdf_files = set()
    if papers_dir.exists():
        for f in papers_dir.glob("*.pdf"):
            if f.is_symlink() or f.resolve().parent != papers_dir.resolve():
                continue
            pdf_files.add(f.name)
            docs.append({
                "id": f.name,
                "filename": f.name,
                "chunks": 0,
                "indexedAt": __import__("datetime").datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "fileType": "pdf",
                "ingested": False
            })

    if CRAG_AVAILABLE:
        try:
            chroma_docs = list_documents()
            # Mark the ones that are in chroma as ingested
            ingested_sources = {d["filename"] for d in chroma_docs if "filename" in d}
            for d in docs:
                if d["filename"] in ingested_sources:
                    d["ingested"] = True
                    # Let's also update the chunks count from Chroma DB
                    for cd in chroma_docs:
                        if cd["filename"] == d["filename"]:
                            d["chunks"] = cd["chunks"]

            # Add any extra docs from chroma that might not be in the folder
            for d in chroma_docs:
                if d.get("filename") and d["filename"] not in pdf_files:
                    d["ingested"] = True
                    docs.append(d)
        except Exception as e:
            pass # fallback to just the folder list

    return {"documents": docs}


@app.get("/api/vault/document/{filename}")
def vault_document(filename: str):
    _, path = _paper_path(filename)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Paper not found")
    return FileResponse(path, media_type="application/pdf", filename=filename,
                        content_disposition_type="inline")


@app.post("/api/vault/upload")
async def vault_upload(file: UploadFile = File(...)):
    # Multipart parsers can silently strip a Windows drive path from filename.
    disposition = Message()
    disposition["content-disposition"] = file.headers.get("content-disposition", "")
    supplied_name = disposition.get_param("filename", header="content-disposition")
    if not isinstance(supplied_name, str) or supplied_name != file.filename:
        raise HTTPException(status_code=400, detail="Invalid PDF filename")
    root, dest = _paper_path(file.filename)
    root.mkdir(exist_ok=True)

    # Invariant: Reject duplicate filenames to prevent destructive overwriting of existing documents
    doc_in_fs = dest.exists()
    doc_in_index = False
    if PIPELINE_AVAILABLE and CRAG_AVAILABLE:
        try:
            doc_in_index = document_exists(file.filename)
        except Exception:
            pass

    if doc_in_fs or doc_in_index:
        raise HTTPException(
            status_code=409,
            detail=f"Document '{file.filename}' already exists. To update, delete the existing document first."
        )

    content = await file.read()
    # Replace atomically: even an existing symlink is never followed while writing.
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        if dest.is_symlink():
            raise HTTPException(status_code=400, detail="Paper path must not be a symlink")
        temporary_path.replace(dest)
    finally:
        temporary_path.unlink(missing_ok=True)

    # Index in ChromaDB (offloaded to thread to avoid blocking FastAPI event loop)
    pipeline_ready = bool(PIPELINE_AVAILABLE and CRAG_AVAILABLE)
    if pipeline_ready:
        try:
            await asyncio.to_thread(process_and_save_pdf, str(dest), replace=False)
        except Exception as e:
            dest.unlink(missing_ok=True)
            try:
                delete_document(file.filename)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return {"filename": file.filename, "size": len(content), "indexed": pipeline_ready}


@app.delete("/api/vault/delete/{filename}")
def vault_delete(filename: str):
    root, f = _paper_path(filename)
    deleted_chunks = 0
    file_existed = f.exists()

    if PIPELINE_AVAILABLE and CRAG_AVAILABLE:
        try:
            deleted_chunks = delete_document(filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ChromaDB delete failed: {e}")

    # Delete physical file
    # A directory handle confines unlink even if its entry changes meanwhile.
    if root.exists():
        if (hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW")
                and os.open in os.supports_dir_fd and os.unlink in os.supports_dir_fd):
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                if f.is_symlink():
                    raise HTTPException(status_code=400, detail="Paper path must not be a symlink")
                try:
                    os.unlink(filename, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
                except OSError as err:
                    raise HTTPException(status_code=500, detail=f"Physical file deletion failed: {err}")
            finally:
                os.close(directory_fd)
        else:  # Windows does not provide directory-relative unlink.
            if f.is_symlink():
                raise HTTPException(status_code=400, detail="Paper path must not be a symlink")
            try:
                f.unlink(missing_ok=True)
            except OSError as err:
                raise HTTPException(status_code=500, detail=f"Physical file deletion failed: {err}")

    if not file_existed and deleted_chunks == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")

    return {"deleted": filename, "chunks_removed": deleted_chunks}


# ── Settings ─────────────────────────────────────────────────────────────────
@app.get("/api/settings")
def get_settings():
    return _load_settings()


@app.put("/api/settings")
def put_settings(payload: SettingsPayload):
    data = payload.model_dump()
    _save_settings(data)

    # Hot-reload the config in logic_engine
    if CRAG_AVAILABLE:
        try:
            reload_config()
        except Exception:
            pass

    return {"status": "saved", "settings": data}
