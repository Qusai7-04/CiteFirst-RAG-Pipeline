"""
Real end-to-end evaluation runner for CiteFirst Phase 4.
Executes the live pipeline across approved development (Q001-Q022) and disclosed pilot (Q023-Q026) sets.
Captures exact context, routing stages, live generation outputs (or exact provider errors),
and unfilled human-review fields for claim correctness, faithfulness, and citation support.
Outputs machine-readable artifacts under ignored evaluation_results/.
"""
import copy
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INDEX_DIR = ROOT / "vector_vault_eval"
os.environ["CITEFIRST_VECTOR_DIR"] = str(INDEX_DIR.resolve())

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from evaluation.dataset import audit_records, load_dataset
from settings_config import load_settings
import database
import logic_engine
from logic_engine import (
    run_logic_engine, CORRECT_THRESHOLD, AMBIGUOUS_THRESHOLD, MAX_VERIFIED_CHUNKS,
    ANSWER_MODEL, LIGHT_TEXT_MODEL
)
import providers

log = logging.getLogger(__name__)


def sanitize_message(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    for key_name in ("OPENROUTER_API_KEY",):
        val = os.environ.get(key_name)
        if val and len(val) > 4 and val in text:
            text = text.replace(val, f"[REDACTED_{key_name}]")
    return text


class InvocationTrace:
    """Explicit per-invocation trace callback for pipeline events."""
    def __init__(self):
        self.searches = []
        self.reranks = []
        self.combined_checks = []
        self.generation_boundaries = []
        self.provider_calls = []

    def __call__(self, event_type: str, data: dict):
        if event_type == "search":
            self.searches.append(data)
        elif event_type == "rerank":
            self.reranks.append(data)
        elif event_type == "combined_check":
            self.combined_checks.append(data)
        elif event_type == "generation_boundary":
            self.generation_boundaries.append(data)
        elif event_type == "provider_call":
            sanitized_data = {}
            for k, v in data.items():
                if isinstance(v, str):
                    sanitized_data[k] = sanitize_message(v)
                else:
                    sanitized_data[k] = v
            self.provider_calls.append(sanitized_data)


def load_all_question_data():
    """
    Loads canonical 26-question dataset (Q001-Q026) from docs/evaluation/phase4_questions.json.
    Builds verified approved source/page pairs without inferring unverified claim-page mappings for Q001-Q018.
    """
    canonical_path = ROOT / "docs/evaluation/phase4_questions.json"
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical Phase 4 question dataset not found at {canonical_path}.")

    records = json.loads(canonical_path.read_text(encoding="utf-8"))
    q_dict = {}
    for rec in records:
        qid = rec["id"]
        claims = rec.get("required_claims", [])
        cs = rec.get("claim_supports", [])
        if cs:
            claims_with_pages = cs
        else:
            claims_with_pages = [{"claim": c, "approved_pages": []} for c in claims]
        q_dict[qid] = {
            "id": qid,
            "question": rec["question"],
            "gold_papers": rec["gold_papers"],
            "reference_answer": rec.get("reference_answer", ""),
            "claims": claims,
            "claims_with_pages": claims_with_pages,
            "approved_pages": rec.get("approved_pages", []),
            "split": rec.get("partition", "development")
        }
    return q_dict

    return q_dict


def evaluate_single_question(q_data: dict, provider_name: str, model_name: str, trace_obj: InvocationTrace | None = None):
    """
    Executes a single question through the real pipeline invocation using explicit trace callback.
    Captures exact final context and source metadata at the generation boundary.
    No separate redundant retrieval or duplicate preflight call.
    """
    qid = q_data["id"]
    qtext = q_data["question"]
    gold_paper = q_data["gold_papers"][0]
    approved_pages_set = {
        (p["source_filename"], p["page"])
        for p in q_data["approved_pages"]
        if p.get("page") is not None
    }

    t0 = time.perf_counter()

    stage_events = []
    def on_stage(step, detail):
        stage_events.append({"step": step, "detail": detail, "timestamp": round(time.perf_counter() - t0, 3)})

    trace = trace_obj if trace_obj is not None else InvocationTrace()

    pipeline_result = None
    pipeline_error = None
    gen_status = "NOT RUN"
    generated_answer = None
    citations = []

    try:
        res = run_logic_engine(qtext, on_stage=on_stage, on_trace=trace)
        pipeline_result = res
        status = res.get("status", "unknown")
        if status == "success":
            gen_status = "EXECUTED_SUCCESS"
        elif status == "fallback_success":
            gen_status = "EXECUTED_FALLBACK_SUCCESS"
        elif status == "partial_match":
            gen_status = "EXECUTED_PARTIAL_MATCH"
        elif status == "failed":
            gen_status = "ABSTENTION_NO_RETRIEVAL"
        else:
            gen_status = f"EXECUTED_{status.upper()}"

        generated_answer = res.get("answer")
        citations = res.get("sources", [])
    except Exception as e:
        err_msg = sanitize_message(str(e))
        pipeline_error = {
            "type": type(e).__name__,
            "message": err_msg
        }
        gen_status = f"FAILED_PROVIDER_{type(e).__name__.upper()}"
        print(f"    [{qid}] Pipeline execution raised {type(e).__name__}: {err_msg[:120]}")

    t_total = time.perf_counter() - t0

    # 1. Retrieval diagnostics from the initial search
    first_search = trace.searches[0] if trace.searches else {"documents": [], "metadatas": []}
    raw_chunks = first_search.get("documents", [])
    if raw_chunks and isinstance(raw_chunks[0], list):
        raw_chunks = raw_chunks[0]
    raw_metas = first_search.get("metadatas", [])
    if raw_metas and isinstance(raw_metas[0], list):
        raw_metas = raw_metas[0]
    raw_papers = [m.get("source") for m in raw_metas if isinstance(m, dict)]
    distinct_raw_papers = list(dict.fromkeys(raw_papers))
    raw_paper_rank = distinct_raw_papers.index(gold_paper) + 1 if gold_paper in distinct_raw_papers else None
    known_source_retrieved = raw_paper_rank is not None

    # 2. Exact final context and evidence at generation boundary
    final_evidence = pipeline_result.get("evidence", []) if pipeline_result else []
    selected_context = [
        {
            "source_filename": ev.get("source"),
            "page": ev.get("page_number"),
            "chunk_id": ev.get("id"),
            "exact_text_passed_to_generation": ev.get("text")
        }
        for ev in final_evidence
    ]

    # Exact generation prompt
    exact_generation_prompt = None
    if trace.generation_boundaries:
        exact_generation_prompt = trace.generation_boundaries[-1].get("prompt")
    elif trace.provider_calls:
        for call in reversed(trace.provider_calls):
            p_text = call.get("prompt", "")
            if "Verified Context:" in p_text or "Related Context:" in p_text:
                exact_generation_prompt = p_text
                break

    # Approved page presence in final context
    context_pages = {
        (c["source_filename"], c["page"])
        for c in selected_context
        if c.get("page") is not None
    }
    approved_passage_in_final_context = bool(context_pages.intersection(approved_pages_set))

    # Claim coverage in final context
    claim_verdicts = []
    for c_info in q_data["claims_with_pages"]:
        claim_text = c_info["claim"]
        target_pages = {
            (p["source_filename"], p["page"])
            for p in c_info.get("approved_pages", [])
            if p.get("page") is not None
        }
        covered = bool(context_pages.intersection(target_pages)) if target_pages else None
        claim_verdicts.append({
            "claim": claim_text,
            "target_pages": list(target_pages),
            "covered_in_final_context": covered,
            "correctness": None,
            "faithful_to_context": None,
            "notes": None
        })

    # Citations validity check
    citations_all_in_retrieved = all(
        c_src in [c["source_filename"] for c in selected_context]
        for c_src in citations
    ) if citations and selected_context else False
    gold_paper_cited = gold_paper in citations

    return {
        "question_id": qid,
        "question": qtext,
        "gold_paper": gold_paper,
        "approved_pages": q_data["approved_pages"],
        "reference_answer": q_data["reference_answer"],
        "split": q_data["split"],
        "retrieval_diagnostics": {
            "returned_candidates_count": len(raw_chunks),
            "distinct_papers_count": len(distinct_raw_papers),
            "raw_paper_rank": raw_paper_rank,
            "known_source_retrieved": known_source_retrieved,
            "searches_count": len(trace.searches)
        },
        "generation_boundary": {
            "verified_context_count": len(selected_context),
            "approved_passage_in_final_context": approved_passage_in_final_context,
            "exact_context_passed_to_generation": selected_context,
            "exact_prompt_passed_to_generation": exact_generation_prompt
        },
        "pipeline_execution": {
            "provider": provider_name,
            "model_configured": model_name,
            "generation_status": gen_status,
            "route": pipeline_result.get("status") if pipeline_result else None,
            "query_used": pipeline_result.get("query_used", qtext) if pipeline_result else qtext,
            "fallback_triggered": pipeline_result.get("status") in ("fallback_success", "partial_match", "failed") if pipeline_result else False,
            "generated_answer": generated_answer,
            "citations": citations,
            "citations_validity": {
                "citations_returned": citations,
                "all_citations_in_evidence": citations_all_in_retrieved,
                "gold_paper_cited": gold_paper_cited
            },
            "provider_calls_count": len(trace.provider_calls),
            "provider_calls": trace.provider_calls,
            "stage_events": stage_events,
            "error": pipeline_error,
            "latency_seconds": round(t_total, 3)
        },
        "human_review_pending": {
            "claim_verdicts": claim_verdicts,
            "citation_support_verdict": None,
            "overall_outcome": None,
            "reviewer": None,
            "reviewed_at": None
        }
    }


def generate_worksheet(all_questions: dict, results_map: dict, provider_name: str, model_name: str, output_path: Path):
    """
    Generates the human review worksheet exclusively from real run records.
    Unrun questions are explicitly marked NOT_RUN with empty context (never copied historical context).
    """
    records = []
    all_qids = [f"Q{i:03d}" for i in range(1, 27)]
    for qid in all_qids:
        q_data = all_questions[qid]
        if qid in results_map:
            r = results_map[qid]
            records.append({
                "question_id": r["question_id"],
                "split": r["split"],
                "question": r["question"],
                "gold_paper": r["gold_paper"],
                "approved_pages": r["approved_pages"],
                "route": r["pipeline_execution"]["route"],
                "generation_status": r["pipeline_execution"]["generation_status"],
                "generated_answer": r["pipeline_execution"]["generated_answer"],
                "citations": r["pipeline_execution"]["citations"],
                "exact_context_passed_to_generation": r["generation_boundary"]["exact_context_passed_to_generation"],
                "exact_prompt_passed_to_generation": r["generation_boundary"]["exact_prompt_passed_to_generation"],
                "human_evaluation": r["human_review_pending"]
            })
        else:
            records.append({
                "question_id": qid,
                "split": q_data["split"],
                "question": q_data["question"],
                "gold_paper": q_data["gold_papers"][0],
                "approved_pages": q_data["approved_pages"],
                "route": None,
                "generation_status": "NOT_RUN",
                "generated_answer": None,
                "citations": [],
                "exact_context_passed_to_generation": [],
                "exact_prompt_passed_to_generation": None,
                "human_evaluation": {
                    "claim_verdicts": [
                        {
                            "claim": c["claim"],
                            "target_pages": [(p["source_filename"], p["page"]) for p in c.get("approved_pages", []) if p.get("page") is not None],
                            "covered_in_final_context": None,
                            "correctness": None,
                            "faithful_to_context": None,
                            "notes": None
                        }
                        for c in q_data["claims_with_pages"]
                    ],
                    "citation_support_verdict": None,
                    "overall_outcome": None,
                    "reviewer": None,
                    "reviewed_at": None
                }
            })

    worksheet_payload = {
        "title": "CiteFirst Phase 4 End-to-End Generation & Human Evaluation Worksheet",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": provider_name,
        "model": model_name,
        "instructions": (
            "Review each claim against the exact context passed to generation. "
            "Record True/False for claim correctness and faithfulness. "
            "Check whether citations accurately identify the source paper and 1-based physical page."
        ),
        "status": "PENDING_HUMAN_REVIEW",
        "question_count": len(records),
        "records": records
    }
    output_path.write_text(json.dumps(worksheet_payload, indent=2), encoding="utf-8")
    print(f"Wrote human review worksheet ({len(records)} questions, {len(results_map)} executed) to {output_path}")


def run_evaluation():
    settings = load_settings()
    provider_name = "openrouter"
    o_key = os.environ.get("OPENROUTER_API_KEY", "")
    key_configured = bool(o_key and not o_key.startswith("PASTE_YOUR_") and not o_key.startswith("your_"))
    model_name = os.getenv("OPENROUTER_MODEL", providers.OPENROUTER_DEFAULT_MODEL)

    print(f"=== CiteFirst Pipeline Evaluation ===")
    print(f"Provider: {provider_name} | Key configured: {key_configured}")
    print(f"Model Pinned: {model_name}")

    if not key_configured:
        print(f"ERROR: OPENROUTER_API_KEY is not configured. Halting execution.")
        return

    all_q = load_all_question_data()

    dev_qids = [f"Q{i:03d}" for i in range(1, 23)]
    pilot_qids = [f"Q{i:03d}" for i in range(23, 27)]

    dev_out = ROOT / "evaluation_results/live_dev_generation_results.json"
    pilot_out = ROOT / "evaluation_results/live_pilot_generation_results.json"
    worksheet_out = ROOT / "evaluation_results/live_generation_worksheet.json"

    # Check for safe resume in development
    dev_results = []
    if dev_out.exists():
        try:
            cached = json.loads(dev_out.read_text(encoding="utf-8"))
            if cached.get("run_configuration", {}).get("provider") == provider_name and \
               cached.get("run_configuration", {}).get("model") == model_name:
                for rec in cached.get("records", []):
                    if "FAILED_PROVIDER" not in rec.get("pipeline_execution", {}).get("generation_status", ""):
                        dev_results.append(rec)
        except Exception:
            dev_results = []

    # ── 1. Development Evaluation (Q001-Q022) ──────────────────────────────────
    print(f"\n--- Running Development Set ({len(dev_qids)} questions across 19 papers) ---")
    dev_done_ids = {r["question_id"] for r in dev_results}
    if dev_done_ids:
        print(f"Resuming: {len(dev_done_ids)} questions already completed for {provider_name}/{model_name}")

    for qid in dev_qids:
        if qid in dev_done_ids:
            continue
        q_data = all_q[qid]
        print(f"Evaluating {qid} ({q_data['gold_papers'][0]})...", end=" ", flush=True)
        res = evaluate_single_question(q_data, provider_name, model_name)
        dev_results.append(res)
        status = res["pipeline_execution"]["generation_status"]
        lat = res["pipeline_execution"]["latency_seconds"]
        appr = "APPR_IN_CTX" if res["generation_boundary"]["approved_passage_in_final_context"] else "NO_APPR"
        print(f"{status} | {appr} | {lat}s")

        dev_out.parent.mkdir(parents=True, exist_ok=True)
        dev_payload = {
            "title": "CiteFirst Phase 4 Development Real Pipeline Execution Results",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "partition": "development",
            "question_count": len(dev_results),
            "source_papers_count": 19,
            "run_configuration": {
                "provider": provider_name,
                "model": model_name,
                "correct_threshold": CORRECT_THRESHOLD,
                "ambiguous_threshold": AMBIGUOUS_THRESHOLD,
                "max_verified_chunks": MAX_VERIFIED_CHUNKS,
                "distance_threshold": load_settings().get("DISTANCE_THRESHOLD", 0.70)
            },
            "records": dev_results
        }
        dev_out.write_text(json.dumps(dev_payload, indent=2), encoding="utf-8")

        # Update worksheet checkpoint
        results_map = {r["question_id"]: r for r in dev_results}
        generate_worksheet(all_q, results_map, provider_name, model_name, worksheet_out)

        # Fail fast if provider error encountered
        if "FAILED_PROVIDER" in status:
            print(f"ABORTING BATCH: Provider error encountered on {qid}. Stopping execution.")
            break

    print(f"\nFinalized development results to {dev_out}")

    # Check if dev evaluation succeeded without provider error before proceeding to pilot
    dev_has_provider_failure = any("FAILED_PROVIDER" in r["pipeline_execution"]["generation_status"] for r in dev_results)
    if dev_has_provider_failure or len(dev_results) < len(dev_qids):
        print(f"\nStopping before pilot set: development set has {len(dev_results)}/{len(dev_qids)} completed.")
        return

    # ── 2. Disclosed Previously Exposed Pilot Evaluation (Q023-Q026) ───────────
    print(f"\n--- Running Disclosed Pilot Set ({len(pilot_qids)} questions across 4 papers) ---")
    pilot_results = []
    if pilot_out.exists():
        try:
            cached_pilot = json.loads(pilot_out.read_text(encoding="utf-8"))
            if cached_pilot.get("run_configuration", {}).get("provider") == provider_name and \
               cached_pilot.get("run_configuration", {}).get("model") == model_name:
                for rec in cached_pilot.get("records", []):
                    if "FAILED_PROVIDER" not in rec.get("pipeline_execution", {}).get("generation_status", ""):
                        pilot_results.append(rec)
        except Exception:
            pilot_results = []

    pilot_done_ids = {r["question_id"] for r in pilot_results}
    for qid in pilot_qids:
        if qid in pilot_done_ids:
            continue
        q_data = all_q[qid]
        print(f"Evaluating pilot {qid} ({q_data['gold_papers'][0]})...", end=" ", flush=True)
        res = evaluate_single_question(q_data, provider_name, model_name)
        pilot_results.append(res)
        status = res["pipeline_execution"]["generation_status"]
        lat = res["pipeline_execution"]["latency_seconds"]
        appr = "APPR_IN_CTX" if res["generation_boundary"]["approved_passage_in_final_context"] else "NO_APPR"
        print(f"{status} | {appr} | {lat}s")

        pilot_payload = {
            "title": "CiteFirst Phase 4 Disclosed Previously Exposed Pilot Results",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "partition": "disclosed_previously_exposed_pilot",
            "question_count": len(pilot_results),
            "source_papers_count": 4,
            "disclosure": "Questions Q023-Q026 previously had retrieval outcomes inspected; results represent a disclosed pilot, not an untouched holdout.",
            "run_configuration": {
                "provider": provider_name,
                "model": model_name,
                "correct_threshold": CORRECT_THRESHOLD,
                "ambiguous_threshold": AMBIGUOUS_THRESHOLD,
                "max_verified_chunks": MAX_VERIFIED_CHUNKS,
                "distance_threshold": load_settings().get("DISTANCE_THRESHOLD", 0.70)
            },
            "records": pilot_results
        }
        pilot_out.write_text(json.dumps(pilot_payload, indent=2), encoding="utf-8")

        # Update worksheet with both dev and pilot results
        results_map = {r["question_id"]: r for r in dev_results + pilot_results}
        generate_worksheet(all_q, results_map, provider_name, model_name, worksheet_out)

        # Fail fast if provider error encountered
        if "FAILED_PROVIDER" in status:
            print(f"ABORTING PILOT BATCH: Provider error encountered on {qid}. Stopping execution.")
            break

    print(f"\nFinalized pilot results to {pilot_out}")


if __name__ == "__main__":
    run_evaluation()
