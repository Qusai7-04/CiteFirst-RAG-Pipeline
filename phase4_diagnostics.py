"""Opt-in Phase 4 retrieval trace; never used by the API or answer pipeline."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from evaluation.dataset import audit_records, load_dataset
from evaluation.metrics import KS
from settings_config import load_settings

ROOT = Path(__file__).resolve().parent
REVIEWED_SHA256 = "11b467995f423495e53207539c20764cf8d531b71a3162de7b5c8a3d57b6417b"
# Windows CRLF working tree has matching parsed JSON but CRLF line-ending hash
REVIEWED_SHA256_CRLF = "395b11567576562e5528c52801f0c8da79f86cbd66fc8e7ec0e1dad8ea847487"


def first_rank(items, source):
    return next((i for i, item in enumerate(items, 1)
                 if item.get("source_filename") == source), None)


def describe_pool(items, source):
    papers = list(dict.fromkeys(item.get("source_filename") for item in items
                                if item.get("source_filename")))
    ids = [item.get("chunk_id") for item in items if item.get("chunk_id")]
    parent_ids = [item.get("parent_id") for item in items if item.get("parent_id")]
    per_source = Counter(item.get("source_filename") for item in items
                         if item.get("source_filename"))
    return {"chunks": len(items), "distinct_papers": len(papers),
            "paper_rank": papers.index(source) + 1 if source in papers else None,
            "cutoff_coverage": {f"k_{k}": len(papers) >= k for k in KS},
            "raw_child_ids": ids, "unique_child_ids": list(dict.fromkeys(ids)),
            "raw_parent_ids": parent_ids, "unique_parent_ids": list(dict.fromkeys(parent_ids)),
            "chunks_per_source": dict(per_source)}


def rerank_fixed_candidates(query, raw, candidate_pool=None):
    """Score the retrieved set only; no grading, supplemental search, or LLM."""
    from logic_engine import _adaptive_select, _get_reranker, _trim, PROCESS_K

    docs, metas = raw["documents"][0], raw["metadatas"][0]
    if candidate_pool is not None:
        docs = docs[:candidate_pool]
        metas = metas[:candidate_pool]
    if len(docs) != len(metas):
        raise ValueError("Document and metadata counts differ")
    started = time.perf_counter()
    scored = []
    has_child = any(m.get("child_text") for m in metas if m)
    for doc, meta in zip(docs, metas):
        text = meta.get("child_text", doc) if has_child else doc
        scored.append((float(_get_reranker().predict([query, _trim(text)])), doc,
                       meta.get("source"), meta))
    scored.sort(key=lambda row: row[0], reverse=True)
    top = scored[:PROCESS_K]
    adaptive = _adaptive_select([(s, d, src, m) for s, d, src, m in top])

    def ref(meta):
        return {"source_filename": meta.get("source"), "chunk_id": meta.get("chunk_id"),
                "page": meta["page_index"] + 1 if type(meta.get("page_index")) is int
                and meta["page_index"] >= 0 else None}

    return {"ranked": [ref(row[3]) for row in scored],
            "process_k": [ref(row[3]) for row in top],
            "adaptive": [ref(row[3]) for row in adaptive],
            "timing_seconds": time.perf_counter() - started,
            "boundary": "LLM grading, supplemental retrieval and generation NOT RUN"}


def trace_one(record, audit, retriever, candidate_pool, source, rerank=False):
    known = audit["gold_papers"]
    row = {"id": audit["id"], "question": record["question"],
           "known_sources": known,
           "supporting_pages": [{"source_filename": e["source_filename"], "page": e["page"]}
                                for e in audit["evidence"]],
           "candidate_source": source, "candidate_pool": candidate_pool,
           "status": "pending", "stage": "retrieval", "error": None,
           "reranker": {"status": "NOT RUN — opt-in model required"}}
    trace = {}
    started = time.perf_counter()
    try:
        raw = retriever(record["question"], n_results=candidate_pool,
                        trace=trace, candidate_source=source)
        row["trace"] = trace
        # The Phase 3 document-pool protocol slices to the requested count.
        expanded = trace["expanded"][:candidate_pool]
        if len(raw["metadatas"][0]) < len(expanded):
            raise ValueError("Trace and retrieval result disagree")
        row["stages"] = {name: {known_source: first_rank(trace[name], known_source)
                                 for known_source in known}
                         for name in ("dense", "bm25", "fused", "distance_filter", "expanded")}
        row["pool"] = describe_pool(expanded, known[0])
        row["known_source_rank"] = row["pool"]["paper_rank"]
        row["known_source_ranks"] = {s: describe_pool(expanded, s)["paper_rank"] for s in known}
        if rerank:
            prerank_ids = [c.get("chunk_id") for c in expanded]
            reranker_ids = [m.get("chunk_id") for m in raw["metadatas"][0][:candidate_pool] if m]
            if prerank_ids != reranker_ids:
                raise ValueError("Prerank candidate IDs and reranker candidate IDs do not match")

            row["stage"] = "reranker"
            rerank_res = rerank_fixed_candidates(record["question"], raw, candidate_pool=candidate_pool)
            row["reranker"] = rerank_res
            row["reranker"]["status"] = "scored_without_llm"
            row["reranker"]["candidate_pool_evaluated"] = len(raw["documents"][0][:candidate_pool])
            row["reranker"]["raw_candidate_count"] = len(raw["documents"][0])
            row["reranker"]["candidate_ids"] = reranker_ids
            row["reranker"]["candidate_ids_match_prerank"] = True

            ranked_items = rerank_res["ranked"]
            distinct_papers = list(dict.fromkeys(item["source_filename"] for item in ranked_items if item.get("source_filename")))
            row["reranker"]["first_chunk_rank"] = first_rank(ranked_items, known[0])
            row["reranker"]["first_distinct_paper_rank"] = (distinct_papers.index(known[0]) + 1
                                                            if known[0] in distinct_papers else None)

            approved_pages = {(e["source_filename"], e["page"]) for e in audit["evidence"] if e.get("page") is not None}
            first_approved = next((i for i, item in enumerate(ranked_items, 1)
                                   if (item.get("source_filename"), item.get("page")) in approved_pages), None)
            row["reranker"]["first_approved_page_rank"] = first_approved
            row["reranker"]["survive_process_k"] = any((item.get("source_filename"), item.get("page")) in approved_pages
                                                       for item in rerank_res["process_k"])
            row["reranker"]["survive_adaptive"] = any((item.get("source_filename"), item.get("page")) in approved_pages
                                                      for item in rerank_res["adaptive"])
            row["reranker"]["missing_page_metadata"] = [item.get("chunk_id") for item in ranked_items if item.get("page") is None]
            row["reranker"]["known_source_ranks"] = {
                stage: {s: first_rank(row["reranker"][stage], s) for s in known}
                for stage in ("ranked", "process_k", "adaptive")}
        row["status"] = "scored"
        row["stage"] = "complete"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = type(exc).__name__  # Do not include provider secrets or local paths.
        row["trace"] = trace
    row["timing_seconds"] = time.perf_counter() - started
    return row


def aggregate(rows):
    good = [r for r in rows if r["status"] == "scored"]
    out = {"eligible": len(rows), "scored": len(good), "failed": len(rows) - len(good),
           "mrr_known_positive": {"mean": sum(1 / r["known_source_rank"]
                                             if r["known_source_rank"] else 0 for r in good) /
                                  len(good) if good else None, "n": len(good)},
           "cutoffs": {}}
    for k in KS:
        available = [r for r in good if r["pool"]["cutoff_coverage"][f"k_{k}"]]
        out["cutoffs"][f"k_{k}"] = {
            "coverage": len(available), "unavailable": len(good) - len(available),
            "n": len(available),
            "known_source_hit_mean": (sum(r["known_source_rank"] is not None and
                                          r["known_source_rank"] <= k for r in available) /
                                      len(available) if available else None)}
    reranked = [r for r in good if r.get("reranker", {}).get("status") == "scored_without_llm"]
    if reranked:
        out["reranker_aggregate"] = {
            "n": len(reranked),
            "mrr_known_positive_post_rerank": sum(1 / r["reranker"]["first_distinct_paper_rank"]
                                                  if r["reranker"]["first_distinct_paper_rank"] else 0
                                                  for r in reranked) / len(reranked),
            "first_distinct_paper_hit_1_count": sum(r["reranker"]["first_distinct_paper_rank"] == 1
                                                    for r in reranked),
            "first_chunk_hit_1_count": sum(r["reranker"]["first_chunk_rank"] == 1
                                           for r in reranked),
            "survive_process_k_count": sum(r["reranker"]["survive_process_k"] for r in reranked),
            "survive_adaptive_count": sum(r["reranker"]["survive_adaptive"] for r in reranked),
            "mean_timing_seconds": sum(r["reranker"]["timing_seconds"] for r in reranked) / len(reranked)
        }
    return out


def run(args, retriever=None):
    records, digest = load_dataset(args.dataset)
    if digest != REVIEWED_SHA256 and digest != REVIEWED_SHA256_CRLF:
        raise ValueError("Reviewed dataset differs from pinned main; inspect before scoring")
    audited = audit_records(records)
    if args.audit_only:
        return {"status": "audit_only", "dataset_sha256": digest,
                "counts": dict(Counter(row["status"] for row in audited))}
    if any(row["status"] != "eligible_retrieval" for row in audited):
        raise ValueError("Only fully eligible reviewed retrieval records are supported")
    if args.candidate_pool < 1:
        raise ValueError("Candidate pool must be positive")
    if not args.index_tag or not args.index_tag.strip():
        raise ValueError("--index-tag is required")
    index = Path(args.index)
    if not index.is_dir():
        raise ValueError("Existing evaluation index directory required")
    if retriever is None:
        os.environ["CITEFIRST_VECTOR_DIR"] = str(index.resolve())
        from database import collection, parent_collection, search_vault
        if collection.count() != 7816 or parent_collection.count() != 2418:
            raise ValueError("Evaluation vault counts differ from Phase 3 control")
        retriever = search_vault
    settings = load_settings()
    config_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    provenance = {"dataset_sha256": digest, "code_commit": commit.stdout.strip()
                  if commit.returncode == 0 else "unknown", "index_tag": args.index_tag,
                  "index_path": str(index), "settings_sha256": config_hash,
                  "code_files_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                        for name in ("database.py", "phase4_diagnostics.py")},
                  "candidate_pool": args.candidate_pool, "rerank": args.rerank}
    variants = ["hybrid"] if args.variant == "hybrid" else ["hybrid", args.variant]
    results = {variant: [trace_one(rec, audit, retriever, args.candidate_pool,
                                   variant, args.rerank)
                         for rec, audit in zip(records, audited)] for variant in variants}
    summary = {variant: aggregate(rows) for variant, rows in results.items()}
    paired = []
    if len(variants) == 2:
        for control, candidate in zip(results["hybrid"], results[args.variant]):
            paired.append({"id": control["id"], "control_rank": control.get("known_source_rank"),
                           "variant_rank": candidate.get("known_source_rank"),
                           "control_status": control["status"], "variant_status": candidate["status"],
                           "control_distinct_papers": control.get("pool", {}).get("distinct_papers"),
                           "variant_distinct_papers": candidate.get("pool", {}).get("distinct_papers"),
                           "control_seconds": control["timing_seconds"],
                           "variant_seconds": candidate["timing_seconds"]})
        both = [row for row in paired if row["control_status"] == row["variant_status"] == "scored"]
        summary["paired_deltas"] = {
            "n": len(both), "excluded_failed_pairs": len(paired) - len(both),
            "mrr_known_positive_delta": (sum((1 / row["variant_rank"] if row["variant_rank"] else 0) -
                                             (1 / row["control_rank"] if row["control_rank"] else 0)
                                             for row in both) / len(both) if both else None),
            "distinct_papers_delta": (sum(row["variant_distinct_papers"] -
                                          row["control_distinct_papers"] for row in both) /
                                      len(both) if both else None),
            "elapsed_seconds_delta": (sum(row["variant_seconds"] - row["control_seconds"]
                                          for row in both) / len(both) if both else None)}
    result = {"status": "local_measurement" if all(summary[v]["failed"] == 0 for v in variants)
              else "partial_results", "provenance": provenance, "aggregate": summary,
              "paired": paired, "per_question": results,
              "metric_scope": "known-positive development diagnostics; no complete recall/precision/NDCG"}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"status": result["status"], "output": str(output), "aggregate": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "reviewed_dataset.json")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--index", type=Path, default=ROOT / "vector_vault_eval")
    parser.add_argument("--index-tag")
    parser.add_argument("--candidate-pool", type=int, default=30)
    parser.add_argument("--variant", choices=("hybrid", "dense-only", "bm25-only"), default="hybrid")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_results/phase4-trace.json")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
