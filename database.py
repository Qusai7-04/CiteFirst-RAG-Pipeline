import fitz
import chromadb
import os
import re
import json
import logging
import tempfile
import hashlib
import time
from pathlib import Path
from threading import Lock
import numpy as np
from sentence_transformers import SentenceTransformer
from settings_config import load_settings

log = logging.getLogger(__name__)

# ── Optional BM25 ──────────────────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    log.warning("rank_bm25 is unavailable; falling back to dense-only search")

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ── Setup ChromaDB ─────────────────────────────────────────────────────────────
_default_vector_dir = ROOT / "vector_vault"
_env_vector_dir = os.getenv("CITEFIRST_VECTOR_DIR")
if _env_vector_dir:
    _p = Path(_env_vector_dir)
    VECTOR_DIR = _p.resolve() if _p.is_absolute() else (ROOT / _p).resolve()
else:
    VECTOR_DIR = _default_vector_dir

# Safety guard: prevent accidentally opening or modifying original CiteFirst vault
_orig_project = Path(os.environ.get("ORIGINAL_CITEFIRST_DIR", "C:/PROJECTS/CiteFirst")).resolve()
if _orig_project.exists():
    _orig_vault = (_orig_project / "vector_vault").resolve()
    if VECTOR_DIR == _orig_vault or _orig_vault in VECTOR_DIR.parents or VECTOR_DIR == _orig_project:
        raise ValueError(
            f"Safety guard: CITEFIRST_VECTOR_DIR ({VECTOR_DIR}) points inside original project '{_orig_project}'. "
            "Use a directory inside the standalone backend project."
        )

chroma_client = chromadb.PersistentClient(path=str(VECTOR_DIR))

# Child collection  — small 100-word chunks, embedded for precise matching
collection = chroma_client.get_or_create_collection(name="ieee_papers")

# Parent collection — large 350-word chunks, fetched by ID for rich LLM context
parent_collection = chroma_client.get_or_create_collection(name="ieee_papers_parent")

_embedding_model = None
_embedding_lock = Lock()


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                _embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return _embedding_model

# ── BM25 cache path ────────────────────────────────────────────────────────────
BM25_CACHE_PATH = VECTOR_DIR / "bm25_corpus.json"
BM25_CACHE_VERSION = 1
_bm25_memory_cache = None
_bm25_lock = Lock()

# ── Domain taxonomy ────────────────────────────────────────────────────────────
DOMAIN_KEYWORDS = {
    "computer_networks": ["tcp", "ip", "routing", "protocol", "network", "bandwidth",
                          "latency", "packet", "congestion", "wireless", "subnet"],
    "machine_learning":  ["neural", "gradient", "backpropagation", "training", "loss",
                          "accuracy", "classification", "regression", "overfitting",
                          "transformer", "attention", "embedding"],
    "quantum":           ["qubit", "quantum", "superposition", "entanglement",
                          "decoherence", "hamiltonian", "bloch"],
    "signal_processing": ["fourier", "filter", "frequency", "signal", "sampling",
                          "nyquist", "convolution", "wavelet", "spectral"],
    "security":          ["encryption", "cipher", "cryptography", "attack",
                          "vulnerability", "authentication", "firewall", "malware"],
    "database_systems":  ["sql", "transaction", "acid", "index", "query", "relation",
                          "schema", "join", "normalization"],
    "computer_vision":   ["image", "pixel", "convolution", "cnn", "segmentation",
                          "detection", "recognition", "bounding box", "feature map"],
    "nlp":               ["language model", "tokenize", "parse", "syntax", "semantic",
                          "bert", "gpt", "corpus", "sentiment"],
}


def detect_domain(text: str) -> str:
    text_lower = text.lower()
    hit_counts = {
        domain: sum(1 for kw in keywords if kw in text_lower)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best = max(hit_counts, key=hit_counts.get)
    return best if hit_counts[best] > 0 else "general"


# ── Vault schema detection ─────────────────────────────────────────────────────

def check_vault_schema():
    """
    Checks whether the current vault was ingested with the new child/parent
    pipeline.  Prints a prominent warning if old-schema chunks are detected.

    The new schema stores every chunk with a 'parent_id' metadata field.
    If 'parent_id' is absent from all chunks, the vault is legacy (350-word
    single chunks with no parent linkage).

    The logic engine handles both schemas. For page provenance, build a
    separate index with CITEFIRST_VECTOR_DIR; keep the old vault for rollback.
    """
    try:
        sample = collection.get(limit=5, include=["metadatas"])
        metas  = sample.get("metadatas", [])
        if metas and not any(m.get("parent_id") for m in metas):
            log.warning("Legacy vault schema detected (chunks lack parent_id); preserve the vault before any re-indexing")
            return "legacy"
        return "new"
    except Exception:
        return "unknown"


# ── BM25 helpers ───────────────────────────────────────────────────────────────

def _build_bm25_index():
    all_data = collection.get(include=["documents", "metadatas"])
    docs     = all_data.get("documents", [])
    if not docs:
        return None, [], []
    tokenized = [d.lower().split() for d in docs]
    bm25      = BM25Okapi(tokenized)
    return bm25, docs, all_data.get("metadatas", [])


def _load_or_build_bm25():
    global _bm25_memory_cache
    if BM25_CACHE_PATH.exists() and not BM25_CACHE_PATH.is_symlink():
        try:
            stat = BM25_CACHE_PATH.stat()
            with _bm25_lock:
                if (_bm25_memory_cache is not None and
                        _bm25_memory_cache[0] == str(BM25_CACHE_PATH) and
                        _bm25_memory_cache[1] == stat.st_mtime and
                        _bm25_memory_cache[2] == stat.st_size):
                    return _bm25_memory_cache[3]

            with BM25_CACHE_PATH.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            docs = cached["documents"]
            metas = cached["metadatas"]
            tokens = cached["tokens"]
            if (cached["version"] != BM25_CACHE_VERSION or not isinstance(docs, list)
                    or not docs or not isinstance(metas, list) or not isinstance(tokens, list)
                    or len(docs) != len(metas) or len(docs) != len(tokens)
                    or any(not isinstance(doc, str) for doc in docs)
                    or any(not isinstance(meta, dict) or not all(isinstance(k, str) for k in meta)
                           for meta in metas)
                    or any(not isinstance(row, list) or not all(isinstance(t, str) for t in row)
                           for row in tokens)
                    or any(row != doc.lower().split() for row, doc in zip(tokens, docs))):
                raise ValueError("Invalid BM25 cache schema")
            loaded = (BM25Okapi(tokens), docs, metas)
            with _bm25_lock:
                _bm25_memory_cache = (str(BM25_CACHE_PATH), stat.st_mtime, stat.st_size, loaded)
            return loaded
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass  # Rebuild from the vector collection; never read the legacy pickle.
    bm25, corpus, metadatas = _build_bm25_index()
    if bm25 is not None:
        BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=BM25_CACHE_PATH.parent,
                                         delete=False) as f:
            temporary_path = Path(f.name)
            try:
                json.dump({"version": BM25_CACHE_VERSION, "documents": corpus,
                           "metadatas": metadatas,
                           "tokens": [doc.lower().split() for doc in corpus]}, f)
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise
        temporary_path.replace(BM25_CACHE_PATH)
        try:
            stat = BM25_CACHE_PATH.stat()
            with _bm25_lock:
                _bm25_memory_cache = (str(BM25_CACHE_PATH), stat.st_mtime, stat.st_size, (bm25, corpus, metadatas))
        except OSError:
            pass
    return bm25, corpus, metadatas


def invalidate_bm25_cache():
    global _bm25_memory_cache
    with _bm25_lock:
        _bm25_memory_cache = None
    if BM25_CACHE_PATH.exists():
        BM25_CACHE_PATH.unlink()
        log.info("[BM25] Cache invalidated. Will rebuild on next search.")


def _reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    rrf_scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            rrf_scores[item] = rrf_scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)


# ── Chunking ───────────────────────────────────────────────────────────────────

def _chunk_sentences(sentences: list, max_words: int, overlap_words: int) -> list:
    chunks        = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if current_words + words_in_sentence > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_chunk = []
            overlap_count = 0
            for prev_sentence in reversed(current_chunk):
                prev_words = len(prev_sentence.split())
                if overlap_count + prev_words <= overlap_words:
                    overlap_chunk.insert(0, prev_sentence)
                    overlap_count += prev_words
                else:
                    break
            current_chunk = overlap_chunk + [sentence]
            current_words = overlap_count + words_in_sentence
        else:
            current_chunk.append(sentence)
            current_words += words_in_sentence

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def _prepare_sentences(text: str) -> list:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\s+', ' ', text)
    return re.split(r'(?<=[.!?]) +', text.strip())


def split_into_sentence_chunks(text, max_words=100, overlap_words=20):
    """100-word child chunks — embedded for precise retrieval."""
    return _chunk_sentences(_prepare_sentences(text), max_words, overlap_words)


def split_into_semantic_chunks(text, max_words=350, overlap_words=75):
    """350-word parent chunks — stored by ID for LLM context."""
    return _chunk_sentences(_prepare_sentences(text), max_words, overlap_words)


# ── Parent expansion ───────────────────────────────────────────────────────────

def _expand_to_parent(child_doc: str, child_meta: dict) -> tuple:
    """
    Returns the parent chunk for LLM context, with child_text stored in metadata
    so logic_engine can score with the focused child text (not the diluted parent).

    If the vault is legacy (no parent_id), returns the chunk unchanged.
    """
    parent_id = child_meta.get("parent_id")
    if not parent_id:
        return child_doc, child_meta

    try:
        result = parent_collection.get(ids=[parent_id], include=["documents", "metadatas"])
        if result["documents"]:
            parent_doc  = result["documents"][0]
            parent_meta = dict(result["metadatas"][0])
            parent_meta["child_text"] = child_doc   # ← key: score with this, not parent
            parent_meta.update({key: child_meta[key] for key in
                                ("source", "document_id", "page_index", "chunk_id") if key in child_meta})
            return parent_doc, parent_meta
    except Exception:
        pass

    return child_doc, child_meta


# ── Ingestion ──────────────────────────────────────────────────────────────────

def _page_text(page) -> str:
    """Extract a page with a readable Markdown copy of detected tables."""
    text = page.get_text(sort=True)
    if hasattr(page, "find_tables"):
        for table in page.find_tables().tables:
            rows = table.extract()
            if not rows:
                continue
            formatted = [[str(cell or "").replace("|", "\\|") for cell in row] for row in rows]
            header = "| " + " | ".join(formatted[0]) + " |"
            divider = "| " + " | ".join("---" for _ in formatted[0]) + " |"
            body = ["| " + " | ".join(row) + " |" for row in formatted[1:]]
            text += "\n\n" + "\n".join([header, divider, *body])
    return text


def _page_chunks(pages: list[str]) -> tuple[list, list]:
    """Keep parent and child chunks within one PDF page for honest provenance."""
    child, parent = [], []
    for page_index, text in enumerate(pages):
        if not text.strip():
            continue
        parents = split_into_semantic_chunks(text)
        children = split_into_sentence_chunks(text)
        if not parents:
            continue
        first_parent = len(parent)
        parent.extend((chunk, page_index) for chunk in parents)
        ratio = max(1, len(children) // len(parents))
        child.extend((chunk, page_index, first_parent + min(i // ratio, len(parents) - 1))
                     for i, chunk in enumerate(children))
    return child, parent

def process_and_save_pdf(pdf_path, replace=False):
    file_name = os.path.basename(pdf_path)

    existing = collection.get(where={"source": file_name})
    if len(existing.get('ids', [])) > 0:
        if not replace:
            log.info(f" Skipping {file_name}: Already in the vault.")
            return
        # shortcut: atomic same-name replacement is not guaranteed across ChromaDB and disk;
        # clients must delete before uploading to replace.
        raise FileExistsError(
            f"Document '{file_name}' already exists in index. Delete it first before re-indexing."
        )

    with fitz.open(pdf_path) as doc:
        pages = [_page_text(page) for page in doc]
    full_text = "\n".join(pages)
    document_id = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:16]

    domain = detect_domain(full_text)
    log.info(f" Detected domain for {file_name}: [{domain}]")

    child_entries, parent_entries = _page_chunks(pages)
    child_chunks = [entry[0] for entry in child_entries]
    parent_chunks = [entry[0] for entry in parent_entries]
    if not parent_chunks or not child_chunks:
        raise ValueError(f"No extractable text in {file_name}")

    try:
        # Store parent chunks (no embedding — fetched by ID only)
        dummy_dim        = 384
        dummy_vecs       = [[0.0] * dummy_dim for _ in parent_chunks]
        parent_ids       = [f"{document_id}_p{page}_parent_{i}"
                            for i, (_, page) in enumerate(parent_entries)]
        parent_metadatas = [{"source": file_name, "document_id": document_id,
                             "page_index": page, "chunk_id": parent_ids[i], "domain": domain}
                            for i, (_, page) in enumerate(parent_entries)]

        parent_collection.add(
            embeddings=dummy_vecs,
            documents=parent_chunks,
            metadatas=parent_metadatas,
            ids=parent_ids,
        )

        # Store child chunks (embedded for retrieval)
        child_ids       = []
        child_metadatas = []

        for i, (_, page_index, parent_idx) in enumerate(child_entries):
            child_ids.append(f"{document_id}_p{page_index}_child_{i}")
            child_metadatas.append({
                "source":    file_name,
                "domain":    domain,
                "document_id": document_id,
                "page_index": page_index,
                "chunk_id": child_ids[-1],
                "parent_id": parent_ids[parent_idx],
            })

        log.info(f"Translating {len(child_chunks)} child chunks from {file_name}...")
        prefixed = [
            f"Represent this sentence for searching relevant passages: {c}"
            for c in child_chunks
        ]
        vectors = _get_embedding_model().encode(prefixed, show_progress_bar=False).tolist()

        collection.add(
            embeddings=vectors,
            documents=child_chunks,
            metadatas=child_metadatas,
            ids=child_ids,
        )
        log.info(f" Saved {file_name} ({len(child_chunks)} child + {len(parent_chunks)} parent chunks)")
        invalidate_bm25_cache()
    except Exception:
        delete_document(file_name)
        raise


# ── Retrieval ──────────────────────────────────────────────────────────────────

def search_vault(query, n_results=10, domain_filter=None, file_filter=None,
                 trace=None, candidate_source="hybrid"):
    """
    Hybrid Search (Dense + BM25) + RRF + Parent expansion.
    Returns parent chunks in documents[], child_text in metadatas[] for scoring.
    Restricts retrieval to file_filter when supplied by a request.
    """
    if candidate_source not in ("hybrid", "dense-only", "bm25-only"):
        raise ValueError("Unknown candidate source")
    if candidate_source == "bm25-only" and not BM25_AVAILABLE:
        raise RuntimeError("BM25 is unavailable")
    started = time.perf_counter()
    fetch_n = n_results * 3
    distance_threshold = load_settings()["DISTANCE_THRESHOLD"]

    search_query = f"Represent this sentence for searching relevant passages: {query}"
    query_vector = _get_embedding_model().encode([search_query]).tolist()

    # Build ChromaDB where clause — respect both domain filter and active file filter
    active_src = file_filter
    if active_src and domain_filter:
        where_filter = {"$and": [{"source": active_src}, {"domain": domain_filter}]}
    elif active_src:
        where_filter = {"source": active_src}
    elif domain_filter:
        where_filter = {"domain": domain_filter}
    else:
        where_filter = None

    results = collection.query(
        query_embeddings=query_vector,
        n_results=min(fetch_n, 100),
        include=["documents", "metadatas", "distances"],
        where=where_filter,
    )

    raw_docs      = results["documents"][0]
    raw_metas     = results["metadatas"][0]
    raw_distances = results["distances"][0]
    dense_seconds = time.perf_counter() - started

    # Content can repeat across pages. Keep each chunk's identity with its text.
    def key_for(doc, meta):
        return (doc, meta.get("source"), meta.get("page_index"), meta.get("chunk_id"))

    doc_info: dict = {}
    dense_ranking = []
    for doc, meta, dist in zip(raw_docs, raw_metas, raw_distances):
        key = key_for(doc, meta)
        dense_ranking.append(key)
        doc_info[key] = (meta, dist)

    # BM25 sparse retrieval
    bm25_ranking = []
    bm25_error = None
    if BM25_AVAILABLE and candidate_source != "dense-only":
        try:
            bm25, corpus, bm25_metas = _load_or_build_bm25()
            if bm25 and corpus:
                tokenized_query = query.lower().split()
                scores          = bm25.get_scores(tokenized_query)
                top_n           = min(fetch_n, len(corpus))
                top_indices     = np.argsort(scores)[::-1][:top_n]

                for idx in top_indices:
                    doc  = corpus[idx]
                    meta = bm25_metas[idx] if bm25_metas else {}
                    if domain_filter and meta.get("domain") != domain_filter:
                        continue
                    if active_src and meta.get("source") != active_src:
                        continue
                    key = key_for(doc, meta)
                    bm25_ranking.append(key)
                    if key not in doc_info:
                        doc_info[key] = (meta, distance_threshold)
        except Exception as e:
            if candidate_source == "bm25-only":
                raise
            bm25_error = type(e).__name__
            log.warning("BM25 search failed: %s", e)

    if candidate_source == "bm25-only" and not bm25_ranking:
        raise RuntimeError("BM25 returned no candidates")
    bm25_seconds = time.perf_counter() - started - dense_seconds

    all_rankings = [r for r in ([dense_ranking] if candidate_source == "dense-only"
                                 else [bm25_ranking] if candidate_source == "bm25-only"
                                 else [dense_ranking, bm25_ranking]) if r]
    if len(all_rankings) > 1:
        merged_ranking = _reciprocal_rank_fusion(all_rankings)
        log.info(f"  [Hybrid] Dense={len(dense_ranking)} | BM25={len(bm25_ranking)} → merged={len(merged_ranking)}")
    else:
        merged_ranking = all_rankings[0] if all_rankings else []
        log.info(f"  [Dense-only] Retrieved {len(dense_ranking)} chunks")
    fusion_seconds = time.perf_counter() - started - dense_seconds - bm25_seconds

    filtered_docs  = []
    filtered_metas = []

    if trace is not None:
        def identity(key):
            meta = doc_info[key][0]
            return {"source_filename": meta.get("source"), "page": (
                meta["page_index"] + 1 if type(meta.get("page_index")) is int
                and meta["page_index"] >= 0 else None), "chunk_id": meta.get("chunk_id"),
                "parent_id": meta.get("parent_id")}

        trace.update(candidate_source=candidate_source, fetch_n=fetch_n,
                     bm25_error=bm25_error,
                     dense=[identity(k) for k in dense_ranking],
                     bm25=[identity(k) for k in bm25_ranking],
                     fused=[identity(k) for k in merged_ranking],
                     distance_filter=[], expanded=[],
                     timing_seconds={"dense": dense_seconds, "bm25": bm25_seconds,
                                     "fusion": fusion_seconds})

    filter_started = time.perf_counter()
    for doc_key in merged_ranking[:fetch_n]:
        doc = doc_key[0]
        meta, dist = doc_info.get(doc_key, ({}, distance_threshold))
        if dist <= distance_threshold:
            if trace is not None:
                trace["distance_filter"].append(identity(doc_key))
            expanded_doc, expanded_meta = _expand_to_parent(doc, meta)
            filtered_docs.append(expanded_doc)
            filtered_metas.append(expanded_meta)
            if trace is not None:
                trace["expanded"].append({"source_filename": expanded_meta.get("source"),
                    "page": (expanded_meta["page_index"] + 1 if
                             type(expanded_meta.get("page_index")) is int and
                             expanded_meta["page_index"] >= 0 else None),
                    "chunk_id": expanded_meta.get("chunk_id"),
                    "parent_id": expanded_meta.get("parent_id")})
        else:
            src = meta.get("source", "Unknown")
            log.info(f"  [pre-filter] Dropped '{src}' — distance={dist:.3f} > {distance_threshold}")

    log.info(f"  [DB] {len(merged_ranking)} merged → {len(filtered_docs)} passed pre-filter")
    if trace is not None:
        trace["timing_seconds"]["filter_expand"] = time.perf_counter() - filter_started

    results["documents"][0] = filtered_docs
    results["metadatas"][0] = filtered_metas
    return results


# ── API helpers (used by api.py, not core pipeline) ────────────────────────────

def list_documents() -> list:
    """Return a summary of all unique source documents in the vault."""
    try:
        all_data = collection.get(include=["metadatas"])
        metas = all_data.get("metadatas", [])
        ids = all_data.get("ids", [])
        source_counts: dict = {}
        for meta in metas:
            src = meta.get("source", "Unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        return [
            {"id": src, "filename": src, "chunks": count,
             "indexedAt": "", "fileType": "pdf"}
            for src, count in source_counts.items()
        ]
    except Exception:
        return []


def delete_document(filename: str) -> int:
    """Delete all chunks (child + parent) for a given source filename."""
    deleted = 0
    # Delete child chunks
    child_data = collection.get(where={"source": filename})
    child_ids = child_data.get("ids", [])
    if child_ids:
        collection.delete(ids=child_ids)
        deleted += len(child_ids)

    # Delete parent chunks
    parent_data = parent_collection.get(where={"source": filename})
    parent_ids = parent_data.get("ids", [])
    if parent_ids:
        parent_collection.delete(ids=parent_ids)
        deleted += len(parent_ids)

    invalidate_bm25_cache()
    return deleted


def document_exists(filename: str) -> bool:
    """Return True if filename has any indexed chunks in ChromaDB."""
    try:
        existing = collection.get(where={"source": filename})
        return bool(existing and existing.get("ids"))
    except Exception:
        return False


# ── CLI / batch ingestion ──────────────────────────────────────────────────────
if __name__ == "__main__":
    paper_folder = ROOT / "papers"
    if not os.path.exists(paper_folder):
        os.makedirs(paper_folder)
        print(f"Please put your PDFs in the '{paper_folder}' folder.")

    pdf_files = [f for f in os.listdir(paper_folder) if f.endswith('.pdf')]
    print(f" Found {len(pdf_files)} papers. Starting upload...")

    for filename in pdf_files:
        path = os.path.join(paper_folder, filename)
        try:
            process_and_save_pdf(path)
        except Exception as e:
            print(f" Error with {filename}: {e}")

    print("\n DATABASE READY.")
