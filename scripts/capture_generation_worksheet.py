"""
Opt-in capture and evaluation interface for generation worksheets.
Populates docs/evaluation/phase4_generation_worksheet.json with exact
retrieved context, prompt metadata, grading, and answer capture (or NOT RUN).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.dataset import audit_records, load_dataset
from settings_config import load_settings

INDEX_DIR = ROOT / "vector_vault_eval"
os.environ["CITEFIRST_VECTOR_DIR"] = str(INDEX_DIR.resolve())

from database import search_vault
from logic_engine import (
    _get_reranker, _trim, _adaptive_select,
    PROCESS_K, CORRECT_THRESHOLD, AMBIGUOUS_THRESHOLD,
    MAX_VERIFIED_CHUNKS
)

WORKSHEET_PATH = ROOT / "evaluation_results/phase4_generation_worksheet_populated.json"
TRACE_PATH = ROOT / "evaluation_results/normal_production_path_trace.json"

def capture_generation_worksheet(limit=None):
    settings = load_settings()
    dataset_path = ROOT / "reviewed_dataset.json"
    records, dataset_digest = load_dataset(dataset_path)
    audited = audit_records(records)

    provider = "openrouter"
    o_key = os.environ.get("OPENROUTER_API_KEY", "")
    live_provider_configured = bool(o_key and not o_key.startswith("PASTE_YOUR_") and not o_key.startswith("your_"))

    print(f"Generation Worksheet Capture:")
    print(f"  Configured Provider: {provider} (Active: {live_provider_configured})")
    print(f"  Dataset: {dataset_path.name} ({len(records)} questions)")

    # Load trace cache if available
    trace_by_id = {}
    if TRACE_PATH.exists():
        try:
            trace_data = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
            for tr in trace_data.get("records", []):
                trace_by_id[tr["id"]] = tr
            print(f"  Loaded {len(trace_by_id)} pre-computed traces from {TRACE_PATH.name}")
        except Exception as e:
            print(f"  Could not load trace cache: {e}")

    items = []
    subset = list(zip(records, audited))[:limit] if limit else list(zip(records, audited))

    for rec, aud in subset:
        qid = aud["id"]
        qtext = rec["question"]
        gold_paper = aud["gold_papers"][0]
        ref_answer = rec.get("reference_answer", "")
        gold_claims = rec.get("answer_claims", [])

        if qid in trace_by_id:
            tr = trace_by_id[qid]
            raw_count = tr["actual_returned_count"]
            ver_count = tr["verified_chunks_count"]
            fallback = tr["fallback_triggered"]
            generation_context = [
                {
                    "source_filename": c.get("source"),
                    "page": c.get("page"),
                    "chunk_id": c.get("chunk_id"),
                    "exact_text_passed_to_generation": c.get("text")
                }
                for c in tr.get("selected_generation_context", [])
            ]
        else:
            # Live retrieval fallback if not traced
            raw = search_vault(qtext, n_results=30, candidate_source="hybrid")
            raw_chunks = raw["documents"][0]
            raw_metas = raw["metadatas"][0]
            raw_count = len(raw_chunks)

            has_child = any(m.get("child_text") for m in raw_metas if m)
            pairs = []
            item_data = []
            for doc_chunk, meta in zip(raw_chunks, raw_metas):
                source = meta.get("source", "Unknown") if meta else "Unknown"
                score_text = meta.get("child_text", doc_chunk) if (has_child and meta) else doc_chunk
                pairs.append([qtext, _trim(score_text)])
                item_data.append((doc_chunk, source, score_text, meta))

            raw_logits = _get_reranker().predict(pairs, batch_size=32)
            scored = []
            for logit, (doc_chunk, source, score_text, meta) in zip(raw_logits, item_data):
                scored.append((float(logit), doc_chunk, source, score_text, meta))
            scored.sort(key=lambda x: x[0], reverse=True)

            top_candidates = scored[:PROCESS_K]
            softmax_input = [(r, c, s, m) for r, c, s, _, m in top_candidates]
            adaptive_pool = _adaptive_select(softmax_input)

            verified_chunks = []
            verified_sources = []
            verified_metas = []
            for sigmoid_score, chunk, source, meta in adaptive_pool:
                if sigmoid_score >= CORRECT_THRESHOLD:
                    verified_chunks.append(chunk)
                    verified_sources.append(source)
                    verified_metas.append(meta)

            if len(verified_chunks) > MAX_VERIFIED_CHUNKS:
                verified_chunks = verified_chunks[:MAX_VERIFIED_CHUNKS]
                verified_sources = verified_sources[:MAX_VERIFIED_CHUNKS]
                verified_metas = verified_metas[:MAX_VERIFIED_CHUNKS]

            ver_count = len(verified_chunks)
            fallback = len(verified_chunks) == 0

            generation_context = [
                {
                    "source_filename": src,
                    "page": m.get("page_index") + 1 if isinstance(m.get("page_index"), int) and m.get("page_index") >= 0 else None,
                    "chunk_id": m.get("chunk_id"),
                    "exact_text_passed_to_generation": chk
                }
                for chk, src, m in zip(verified_chunks, verified_sources, verified_metas)
            ]

        # Generation execution or NOT RUN
        if live_provider_configured:
            gen_status = "PENDING_LIVE_GENERATION"
            generated_answer = None
            citations = []
            tokens = None
            cost = None
        else:
            gen_status = "NOT RUN — requires a configured live provider"
            generated_answer = None
            citations = []
            tokens = None
            cost = None

        # Human review fields for claims
        claim_verdicts = [
            {
                "claim": c if isinstance(c, str) else c.get("claim", str(c)),
                "correctness": None,
                "faithful_to_context": None,
                "notes": None
            }
            for c in gold_claims
        ]

        item = {
            "question_id": qid,
            "question": qtext,
            "gold_paper": gold_paper,
            "reference_answer": ref_answer,
            "run_configuration": {
                "dataset_digest": dataset_digest,
                "llm_provider": provider,
                "model": settings.get("MODEL_NAME", "default"),
                "temperature": settings.get("TEMPERATURE", 0.0),
                "distance_threshold": settings.get("DISTANCE_THRESHOLD", 0.70),
                "correct_threshold": CORRECT_THRESHOLD,
                "ambiguous_threshold": AMBIGUOUS_THRESHOLD
            },
            "retrieval_and_selection": {
                "total_returned_chunks": raw_count,
                "verified_chunks_count": ver_count,
                "fallback_triggered": fallback,
                "selected_context": generation_context
            },
            "generation": {
                "status": gen_status,
                "generated_answer": generated_answer,
                "citations": citations,
                "tokens": tokens,
                "estimated_cost_usd": cost
            },
            "human_evaluation": {
                "claim_verdicts": claim_verdicts,
                "citation_support_verdict": None,
                "overall_outcome": None,
                "reviewer": None,
                "reviewed_at": None
            }
        }
        items.append(item)
        print(f"Captured {qid}: {len(generation_context)} passages ({gen_status})")

    executed_count = sum(1 for item in items if item["generation"]["status"] == "EXECUTED")
    if executed_count == len(items) and len(items) > 0:
        overall_gen_status = "EXECUTED"
    elif executed_count > 0:
        overall_gen_status = f"PARTIALLY_EXECUTED ({executed_count}/{len(items)})"
    else:
        overall_gen_status = "NOT RUN — no live generation executed"

    worksheet = {
        "title": "CiteFirst Phase 4 End-to-End Generation & Evaluation Worksheet",
        "description": "Records retrieved passages passed to generation and reserves human review fields for claim correctness, context faithfulness, and citation validity.",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_status": overall_gen_status,
        "human_review_status": "PENDING_HUMAN_REVIEW",
        "records": items
    }

    WORKSHEET_PATH.write_text(json.dumps(worksheet, indent=2), encoding="utf-8")
    print(f"\nSaved generation worksheet to {WORKSHEET_PATH}")

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    capture_generation_worksheet(limit=limit)
