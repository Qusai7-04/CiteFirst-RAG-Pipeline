import json

from evaluator import evaluate_dataset, run_ml_evaluation
from test_phase3_dataset import annotated


def test_all_eligible_processed_by_default_and_limit_explicit():
    rows = [annotated(id=f"Q{i:03d}", question=f"Question {i}?")
            for i in range(12)]
    calls = []

    def retrieve(question, n_results):
        calls.append(question)
        assert n_results == 10
        return {"metadatas": [[{"source": "paper.pdf", "page_index": 1,
                                "chunk_id": "child-1"}]]}

    audit, results, summary = evaluate_dataset(rows, "hash", retriever=retrieve)
    assert len(audit) == len(results) == len(calls) == 12
    assert summary["counts"]["scored"] == 12
    assert summary["metrics"]["r@1"]["count"] == 12
    assert results[0]["retrieved_chunks"][0]["page"] == 2
    assert "tn" not in json.dumps(summary).lower()
    calls.clear()
    _, limited, summary = evaluate_dataset(rows, "hash", retriever=retrieve, limit=10)
    assert len(calls) == 10
    assert [r["status"] for r in limited[-2:]] == ["skipped_limit"] * 2
    assert summary["config"]["limit"] == 10


def test_failures_and_exclusions_not_scored():
    def fail(*args, **kwargs):
        raise RuntimeError("retrieval failed")

    rows = [annotated(), {"question": "Candidate only"}]
    _, results, summary = evaluate_dataset(rows, "hash", retriever=fail)
    assert results[0]["status"] == "failed"
    assert results[1]["status"] == "needs_human_review"
    assert summary["counts"]["failed"] == 1
    assert summary["counts"]["excluded"] == 1
    assert summary["metrics"] == {}
    assert summary["baseline_status"] == "retrieval_failed"


def test_baseline_status_distinguishes_no_labels_failure_and_partial():
    _, _, blocked = evaluate_dataset([{"question": "Candidate only"}], "hash")
    assert blocked["baseline_status"] == "blocked_no_verified_labels"
    rows = [annotated(id="Q001", question="One?"),
            annotated(id="Q002", question="Two?")]

    def sometimes_fails(question, n_results):
        if question == "Two?":
            raise RuntimeError("offline vault")
        return {"metadatas": [[{"source": "paper.pdf", "chunk_id": "c1"}]]}

    _, results, partial = evaluate_dataset(rows, "hash", retriever=sometimes_fails)
    assert [row["status"] for row in results] == ["scored", "failed"]
    assert partial["baseline_status"] == "partial_results"
    assert partial["counts"]["scored"] == partial["counts"]["failed"] == 1


def test_chunk_window_is_not_misreported_as_ten_unique_papers():
    rows = [annotated()]

    def repeated_chunks(question, n_results):
        return {"metadatas": [[{"source": "paper.pdf", "chunk_id": f"c{i}"}
                                for i in range(12)]]}

    _, results, summary = evaluate_dataset(rows, "hash", retriever=repeated_chunks)
    assert results[0]["chunk_candidates_returned"] == 10
    assert results[0]["distinct_papers_returned"] == 1
    assert len(results[0]["retrieved_papers"]) == 1
    assert summary["config"]["chunk_candidate_limit"] == 10


def test_scored_cli_requires_fixed_index_tag(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([annotated()]))
    for tag in (None, "unrecorded", " unknown "):
        try:
            run_ml_evaluation(path, tmp_path / "results", index_tag=tag)
        except ValueError as exc:
            assert "--index-tag" in str(exc)
        else:
            raise AssertionError("A scored run accepted an unrecorded index")
    # The checked-in unverified candidates still permit a no-model audit summary.
    path.write_text(json.dumps([{"question": "Draft only"}]))
    summary = run_ml_evaluation(path, tmp_path / "results")
    assert summary["baseline_status"] == "blocked_no_verified_labels"


def test_audit_only_never_imports_or_calls_retrieval(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([annotated()]))
    summary = run_ml_evaluation(path, tmp_path / "results", audit_only=True)
    assert summary["counts"]["attempted"] == 0
    assert summary["baseline_status"] == "not_run_audit_only"
    assert (tmp_path / "results" / "annotation-gaps.csv").exists()
    assert (tmp_path / "results" / "per-question.jsonl").exists()
    assert "dataset_sha256" in json.loads(
        (tmp_path / "results" / "aggregate.json").read_text())
    per_query = json.loads((tmp_path / "results" / "per-question.jsonl").read_text())
    assert per_query["status"] == "skipped_audit_only"
    assert per_query["config_sha256"] == summary["config_sha256"]
    assert summary["counts"]["present"] == 1
    assert summary["config"]["limit"] is None


def test_seven_chunks_same_paper_and_cutoff_coverage():
    rows = [annotated()]

    def seven_and_three(question, n_results):
        metas = ([{"source": "dominant.pdf", "chunk_id": f"c{i}"} for i in range(7)] +
                 [{"source": f"other_{i}.pdf", "chunk_id": f"o{i}"} for i in range(3)])
        return {"metadatas": [metas]}

    _, results, summary = evaluate_dataset(rows, "hash", retriever=seven_and_three)
    assert results[0]["chunk_candidates_returned"] == 10
    assert results[0]["distinct_papers_returned"] == 4
    assert [p["source_filename"] for p in results[0]["retrieved_papers"]] == [
        "dominant.pdf", "other_0.pdf", "other_1.pdf", "other_2.pdf"
    ]
    assert results[0]["cutoff_coverage"] == {
        "k_1": True, "k_3": True, "k_5": False, "k_10": False
    }
    assert summary["counts"]["cutoff_coverage"] == {
        "k_1": 1, "k_3": 1, "k_5": 0, "k_10": 0
    }


def test_document_pool_protocol_differs_from_ten_chunks():
    rows = [annotated()]
    calls = []

    def mock_large_vault(question, n_results):
        calls.append(n_results)
        # Simulate search_vault returning n_results * 3 chunks (e.g. 90 chunks for n_results=30)
        # where only 4 distinct papers appear across the chunks
        raw_count = n_results * 3
        metas = [{"source": f"paper_{i % 4}.pdf", "chunk_id": f"c{i}"} for i in range(raw_count)]
        return {"metadatas": [metas]}

    # 1. Default ten-chunk protocol: requests 10, slices to 10 chunks, scores all cutoffs (padding missing as misses)
    _, results_default, summary_default = evaluate_dataset(
        rows, "hash", retriever=mock_large_vault, protocol="within_first_ten_chunks")
    assert calls[-1] == 10
    assert summary_default["config"]["protocol"] == "within_first_ten_chunks"
    assert summary_default["config"]["chunk_candidate_limit"] == 10
    assert results_default[0]["chunk_candidates_returned"] == 10
    assert results_default[0]["distinct_papers_returned"] == 4
    # In ten-chunk mode, missing ranks are padded as misses, so p@10 and r@10 are present
    assert "r@10" in results_default[0]["metrics"]
    assert results_default[0]["unavailable_cutoffs"] == {}

    # 2. Document-pool protocol: requests 30, strictly slices 90 chunks down to stated 30-chunk pool
    _, results_pool, summary_pool = evaluate_dataset(
        rows, "hash", retriever=mock_large_vault, protocol="document_pool_top_k", candidate_pool_size=30)
    assert calls[-1] == 30
    assert summary_pool["config"]["protocol"] == "document_pool_top_k"
    assert summary_pool["config"]["candidate_pool_size"] == 30
    # Must enforce stated 30-chunk pool, NOT consume all 90 chunks
    assert results_pool[0]["chunk_candidates_returned"] == 30
    assert results_pool[0]["distinct_papers_returned"] == 4
    # Because only 4 distinct papers were retrieved, K=5 and K=10 are unavailable
    assert "r@1" in results_pool[0]["metrics"]
    assert "r@3" in results_pool[0]["metrics"]
    assert "r@5" not in results_pool[0]["metrics"]
    assert "r@10" not in results_pool[0]["metrics"]
    assert "p@10" not in results_pool[0]["metrics"]
    assert "k_5" in results_pool[0]["unavailable_cutoffs"]
    assert "k_10" in results_pool[0]["unavailable_cutoffs"]
    assert summary_pool["counts"]["unavailable_cutoffs"] == {
        "k_1": 0, "k_3": 0, "k_5": 1, "k_10": 1
    }



def test_evaluator_handles_explicit_negatives_and_macro_denominator():
    rows = [
        annotated(id="Q001", judged_nonrelevant_papers=["neg.pdf"]),
        annotated(id="Q002", question="Different question?",
                  relevant_evidence=[{"source_filename": "paper.pdf", "page": 1,
                                      "supporting_quote": "q", "relevance_grade": 2}])
    ]

    def mock_retriever(question, n_results):
        if question == rows[0]["question"]:
            metas = [{"source": "neg.pdf", "chunk_id": "1"},
                     {"source": "paper.pdf", "chunk_id": "2"},
                     {"source": "unjudged.pdf", "chunk_id": "3"}]
        else:
            metas = [{"source": "unjudged.pdf", "chunk_id": "1"},
                     {"source": "paper.pdf", "chunk_id": "2"}]
        return {"metadatas": [metas]}

    _, results, summary = evaluate_dataset(rows, "hash", retriever=mock_retriever)
    assert results[0]["status"] == "scored"
    assert results[1]["status"] == "scored"
    assert results[0]["metrics"]["p@1"] == 0
    assert "p@1" not in results[1]["metrics"]
    assert summary["metrics"]["p@1"] == {"mean": 0.0, "count": 1}
    assert summary["metrics"]["mrr"]["count"] == 2
