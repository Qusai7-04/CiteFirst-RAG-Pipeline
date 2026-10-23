"""Evidence-gated document retrieval evaluation before corrective reranking/generation."""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from evaluation.dataset import audit_records, load_dataset
from evaluation.metrics import KS, macro_average, score_ranking
from settings_config import load_settings

ROOT = Path(__file__).resolve().parent
CHUNK_CANDIDATE_LIMIT = 10
DEFAULT_POOL_SIZE = 30


def evaluate_dataset(records, dataset_hash, retriever=None, limit=None, audit_only=False,
                     index_tag="unrecorded", code_commit="unknown", config_sha256="unrecorded",
                     protocol="within_first_ten_chunks", candidate_pool_size=DEFAULT_POOL_SIZE):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("--limit must be a positive integer")
    if protocol not in ("within_first_ten_chunks", "document_pool_top_k"):
        raise ValueError(f"Unknown protocol: {protocol}")
    if candidate_pool_size is not None and (type(candidate_pool_size) is not int or candidate_pool_size < 1):
        raise ValueError("--candidate-pool must be a positive integer")
    audit = audit_records(records)
    output, attempted = [], 0
    fetch_n = CHUNK_CANDIDATE_LIMIT if protocol == "within_first_ten_chunks" else candidate_pool_size
    for record, row in zip(records, audit):
        result = {key: value for key, value in row.items()
                  if key not in ("evidence", "judged_nonrelevant_papers")}
        result.update(dataset_sha256=dataset_hash, index_tag=index_tag, code_commit=code_commit,
                      config_sha256=config_sha256, protocol=protocol,
                      retrieved_papers=[], retrieved_chunks=[], metrics={}, error=None,
                      chunk_candidates_returned=0, distinct_papers_returned=0,
                      cutoff_coverage={},
                      gold_evidence_refs=[
                          {"source_filename": item.get("source_filename"), "page": item.get("page")}
                          for item in row["evidence"] if isinstance(item, dict)])
        if row["status"] == "eligible_abstention":
            result["status"] = "skipped_unanswerable"
        elif row["status"] == "eligible_retrieval":
            if audit_only:
                result["status"] = "skipped_audit_only"
            elif limit is not None and attempted >= limit:
                result["status"] = "skipped_limit"
            else:
                attempted += 1
                try:
                    if retriever is None:
                        from database import search_vault
                        retriever = search_vault
                    raw = retriever(record["question"], n_results=fetch_n)
                    # shortcut: in within_first_ten_chunks mode, score only the first ten returned chunks;
                    # document_pool_top_k uses the pre-specified candidate pool before deduplication.
                    raw_metas = raw["metadatas"][0] if raw and "metadatas" in raw and raw["metadatas"] else []
                    if protocol == "within_first_ten_chunks":
                        metas = raw_metas[:CHUNK_CANDIDATE_LIMIT]
                    else:
                        metas = raw_metas[:candidate_pool_size]
                    if not all(isinstance(meta, dict) and meta.get("source") for meta in metas):
                        raise ValueError("missing_source_metadata")
                    result["retrieved_chunks"] = [
                        {"rank": rank, "source_filename": meta["source"],
                         "chunk_id": meta.get("chunk_id"),
                         "page": (meta["page_index"] + 1
                                  if type(meta.get("page_index")) is int else None),
                         "score": None}
                        for rank, meta in enumerate(metas, 1)]
                    ranked = list(dict.fromkeys(meta["source"] for meta in metas))
                    if protocol == "within_first_ten_chunks":
                        ranked = ranked[:CHUNK_CANDIDATE_LIMIT]
                    result["chunk_candidates_returned"] = len(metas)
                    result["distinct_papers_returned"] = len(ranked)
                    result["cutoff_coverage"] = {f"k_{k}": len(ranked) >= k for k in KS}
                    result["retrieved_papers"] = [
                        {"rank": rank, "source_filename": name, "score": None}
                        for rank, name in enumerate(ranked, 1)]
                    grades = None
                    if all(item.get("relevance_grade") is not None for item in row["evidence"]):
                        grades = {}
                        for item in row["evidence"]:
                            name = item["source_filename"]
                            grades[name] = max(grades.get(name, 0), item["relevance_grade"])
                    judged = set(row["gold_papers"]) | set(row["judged_nonrelevant_papers"])
                    if protocol == "within_first_ten_chunks":
                        result["metrics"] = score_ranking(
                            ranked, row["gold_papers"], grades=grades, judged_ids=judged)
                        result["unavailable_cutoffs"] = {}
                    else:
                        available_ks = tuple(k for k in KS if len(ranked) >= k)
                        unavailable_ks = [k for k in KS if len(ranked) < k]
                        if available_ks:
                            result["metrics"] = score_ranking(
                                ranked, row["gold_papers"], k_values=available_ks,
                                grades=grades, judged_ids=judged)
                        else:
                            first = next((i for i, item in enumerate(ranked, 1) if item in set(row["gold_papers"])), None)
                            result["metrics"] = {"mrr": 1 / first if first else 0.0}
                        result["unavailable_cutoffs"] = {
                            f"k_{k}": f"pool_cap_insufficient_distinct_papers ({len(ranked)} < {k})"
                            for k in unavailable_ks
                        }
                    result["status"] = "scored"
                except Exception as exc:
                    result["status"] = "failed"
                    result["error"] = f"{type(exc).__name__}: {exc}"
        output.append(result)
    counts = Counter(row["status"] for row in output)
    scored_rows = [row for row in output if row["status"] == "scored"]
    summary = {
        "dataset_sha256": dataset_hash, "index_tag": index_tag, "code_commit": code_commit,
        "config_sha256": config_sha256,
        "config": {"retrieval": "database.search_vault",
                   "protocol": protocol,
                   "candidate_pool_size": fetch_n,
                   "chunk_candidate_limit": CHUNK_CANDIDATE_LIMIT if protocol == "within_first_ten_chunks" else fetch_n,
                   "paper_ranking": ("first occurrence among top ten returned chunks"
                                     if protocol == "within_first_ten_chunks"
                                     else f"first occurrence within candidate pool of {fetch_n} chunks"),
                   "k": list(KS), "limit": limit, "level": "source_filename",
                   "mode": "hybrid_pre_crag_no_llm"},
        "counts": {"present": len(records), "valid": sum(r["status"] != "invalid" for r in audit),
                   "eligible_retrieval": sum(r["status"] == "eligible_retrieval" for r in audit),
                   "eligible_abstention": sum(r["status"] == "eligible_abstention" for r in audit),
                   "excluded": sum(r["status"] in ("invalid", "needs_human_review") for r in audit),
                   "attempted": attempted, "scored": counts["scored"],
                   "failed": counts["failed"], "skipped": counts["skipped_limit"] +
                   counts["skipped_unanswerable"] + counts["skipped_audit_only"],
                   "cutoff_coverage": {
                       f"k_{k}": sum(r["cutoff_coverage"].get(f"k_{k}", False) for r in scored_rows)
                       for k in KS
                   },
                   "unavailable_cutoffs": {
                       f"k_{k}": sum(f"k_{k}" in r.get("unavailable_cutoffs", {}) for r in scored_rows)
                       for k in KS
                   }},
        "metrics": macro_average(row["metrics"] for row in scored_rows),
        "baseline_status": ("not_run_audit_only" if audit_only else
                            "blocked_no_verified_labels" if not any(
                                r["status"] == "eligible_retrieval" for r in audit) else
                            "retrieval_failed" if not counts["scored"] else
                            "partial_results" if counts["failed"] else "pilot_only"),
    }
    return audit, output, summary


def write_reports(output_dir, audit, results, summary):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "annotation-gaps.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "position", "status", "reasons",
                                                  "evidence_status", "possible_correction",
                                                  "needs_human_review"])
        writer.writeheader()
        for row in audit:
            writer.writerow({"id": row["id"], "position": row["position"],
                             "status": row["status"], "reasons": ";".join(row["reasons"]),
                             "evidence_status": "verified_recorded" if row["status"] == "eligible_retrieval"
                             else "not_verified",
                             "possible_correction": "Check original PDF; annotate source, page, quote and answer",
                             "needs_human_review": row["status"] != "eligible_retrieval"})
    with (directory / "per-question.jsonl").open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (directory / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Retrieval evaluation", "",
        f"Status: **{summary['baseline_status']}**; document-level hybrid search before corrective reranking.",
        f"Protocol: **{summary['config']['protocol']}**; candidate chunk limit: {summary['config']['chunk_candidate_limit']}.",
        f"Dataset SHA-256: {summary['dataset_sha256']}; index tag: {summary['index_tag']}; "
        f"code commit: {summary['code_commit']}.", "",
        "Counts: " + ", ".join(f"{key}={value}" for key, value in summary["counts"].items()
                               if key not in ("cutoff_coverage", "unavailable_cutoffs")) + ".",
        f"Cutoff coverage (queries reaching >= K distinct papers): {json.dumps(summary['counts']['cutoff_coverage'])}.",
        f"Unavailable cutoffs due to pool cap: {json.dumps(summary['counts']['unavailable_cutoffs'])}.",
        f"Config: {json.dumps(summary['config'], sort_keys=True)}", "",
        ("Paper ranks are first occurrences within the top ten returned chunks. "
         "Fewer than K distinct papers are padded as misses; this is not a search "
         "over the top K unique papers." if summary['config']['protocol'] == "within_first_ten_chunks"
         else f"Paper ranks are first occurrences within candidate pool of {summary['config']['candidate_pool_size']} chunks. "
              "For cutoffs K where fewer than K distinct papers were retrieved in the pool, document@K metrics are reported as unavailable."), "",
        "Macro means across scored answerable queries (missing ranks count as non-relevant):", "",
    ]

    if summary["metrics"]:
        lines += ["| Metric | Mean | N |", "|---|---:|---:|"]
        lines += [f"| {key} | {value['mean']:.4f} | {value['count']} |"
                  for key, value in summary["metrics"].items()]
    else:
        lines += ["No verified relevance labels; no retrieval quality score is reportable."]
    lines += ["", "Unanswerable questions and failed queries are excluded from means. "
              "Answer, citation, faithfulness and abstention require separate human review.", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_ml_evaluation(dataset=ROOT / "reviewed_dataset.json",
                      output_dir=ROOT / "evaluation_results", limit=None,
                      audit_only=False, index_tag=None, code_commit=None,
                      protocol="within_first_ten_chunks", candidate_pool_size=DEFAULT_POOL_SIZE):
    records, digest = load_dataset(dataset)
    if (not audit_only and any(row["status"] == "eligible_retrieval"
                           for row in audit_records(records)) and
            (not isinstance(index_tag, str) or not index_tag.strip()
             or index_tag.strip().lower() in {"unrecorded", "unknown"})):
        raise ValueError("A fixed --index-tag is required for a scored retrieval run")
    if code_commit is None:
        process = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True, check=False)
        code_commit = process.stdout.strip() if process.returncode == 0 else "unknown"
    settings = load_settings()
    config_sha256 = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    audit, rows, summary = evaluate_dataset(
        records, digest, limit=limit, audit_only=audit_only,
        index_tag=index_tag, code_commit=code_commit, config_sha256=config_sha256,
        protocol=protocol, candidate_pool_size=candidate_pool_size)
    write_reports(output_dir, audit, rows, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "reviewed_dataset.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_results")
    parser.add_argument("--limit", type=int, help="Explicit cap on eligible answerable queries")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--index-tag", help="Required for a scored retrieval run; identify the fixed local index")
    parser.add_argument("--code-commit")
    parser.add_argument("--protocol", choices=["within_first_ten_chunks", "document_pool_top_k"],
                        default="within_first_ten_chunks",
                        help="Retrieval evaluation protocol")
    parser.add_argument("--candidate-pool", dest="candidate_pool_size", type=int,
                        default=DEFAULT_POOL_SIZE,
                        help="Candidate chunk pool size for document_pool_top_k protocol")
    args = parser.parse_args()
    print(json.dumps(run_ml_evaluation(**vars(args)), indent=2))
