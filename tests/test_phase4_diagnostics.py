import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from phase4_diagnostics import aggregate, describe_pool, trace_one, run


def test_source_ranks_duplicate_ids_unavailable_cutoffs_and_missing_metadata():
    items = [{"source_filename": "other.pdf", "chunk_id": "c", "parent_id": "p"},
             {"source_filename": "other.pdf", "chunk_id": "c", "parent_id": "p"},
             {"source_filename": "gold.pdf", "chunk_id": None, "page": None}]
    pool = describe_pool(items, "gold.pdf")
    assert pool["paper_rank"] == 2
    assert pool["chunks"] == 3 and pool["distinct_papers"] == 2
    assert pool["raw_child_ids"] == ["c", "c"] and pool["unique_child_ids"] == ["c"]
    assert pool["raw_parent_ids"] == ["p", "p"] and pool["unique_parent_ids"] == ["p"]
    assert pool["cutoff_coverage"] == {"k_1": True, "k_3": False,
                                       "k_5": False, "k_10": False}
    assert describe_pool([{"chunk_id": "unknown"}], "gold.pdf")["paper_rank"] is None


def test_trace_preserves_query_and_errors_with_stage():
    record = {"question": "Unmodified question?"}
    audit = {"id": "Q015", "gold_papers": ["gold.pdf"],
             "evidence": [{"source_filename": "gold.pdf", "page": 4}]}

    def retrieve(question, n_results, trace, candidate_source):
        assert question == record["question"] and n_results == 3
        trace.update({name: [{"source_filename": "other.pdf", "chunk_id": "c"},
                             {"source_filename": "gold.pdf", "chunk_id": "g", "page": 4}]
                      for name in ("dense", "bm25", "fused", "distance_filter", "expanded")})
        return {"metadatas": [[{"source": "other.pdf"}, {"source": "gold.pdf"}]]}

    row = trace_one(record, audit, retrieve, 3, "hybrid")
    assert row["status"] == "scored" and row["known_source_rank"] == 2
    assert row["supporting_pages"] == [{"source_filename": "gold.pdf", "page": 4}]
    assert row["question"] == record["question"]
    assert row["stages"]["dense"]["gold.pdf"] == 2
    summary = aggregate([row])
    assert summary["cutoffs"]["k_3"]["n"] == 0
    assert summary["cutoffs"]["k_3"]["known_source_hit_mean"] is None
    def fail(*args, **kwargs):
        raise RuntimeError("private local path")
    failed = trace_one(record, audit, fail, 3, "hybrid")
    assert failed["stage"] == "retrieval" and failed["error"] == "RuntimeError"
    assert "private local path" not in json.dumps(failed)
    assert aggregate([failed])["mrr_known_positive"]["n"] == 0


def test_imported_retrieval_trace_and_opt_in_sources(tmp_path, monkeypatch):
    class Collection:
        def query(self, **kwargs):
            return {"documents": [["same", "same"]],
                    "metadatas": [[{"source": "other.pdf", "page_index": 0, "chunk_id": "a"},
                                   {"source": "gold.pdf", "chunk_id": "b"}]],
                    "distances": [[0.1, 0.2]]}

    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(
        PersistentClient=lambda **kw: SimpleNamespace(get_or_create_collection=lambda **kw: Collection())))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=lambda *a: SimpleNamespace(
            encode=lambda x: SimpleNamespace(tolist=lambda: [[0.0]]))))
    spec = importlib.util.spec_from_file_location("database_phase4_test", "database.py")
    db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db)
    db.BM25_AVAILABLE = True
    db._load_or_build_bm25 = lambda: (SimpleNamespace(get_scores=lambda q: [2, 1]),
                                     ["same", "same"],
                                     [{"source": "gold.pdf", "chunk_id": "b"},
                                      {"source": "other.pdf", "chunk_id": "a"}])
    control, trace = db.search_vault("q", n_results=2), {}
    observed = db.search_vault("q", n_results=2, trace=trace)
    assert observed == control
    assert trace["dense"][0]["source_filename"] == "other.pdf"
    assert trace["bm25"][0]["source_filename"] == "gold.pdf"
    assert any(item["page"] is None for item in trace["expanded"])
    assert len(trace["expanded"]) == len(observed["metadatas"][0])
    assert db.search_vault("q", n_results=2, candidate_source="dense-only")["metadatas"][0][0]["source"] == "other.pdf"
    with pytest.raises(ValueError, match="candidate source"):
        db.search_vault("q", candidate_source="invalid")
    db.BM25_AVAILABLE = False
    with pytest.raises(RuntimeError, match="unavailable"):
        db.search_vault("q", candidate_source="bm25-only")


def test_audit_only_does_not_import_database(tmp_path, monkeypatch):
    args = SimpleNamespace(dataset=tmp_path / "data.json", audit_only=True)
    args.dataset.write_text("[]")
    import phase4_diagnostics as phase4
    monkeypatch.setattr(phase4, "REVIEWED_SHA256", phase4.hashlib.sha256(b"[]").hexdigest())
    database_before = sys.modules.get("database")
    assert run(args)["status"] == "audit_only"
    assert sys.modules.get("database") is database_before


def test_paired_runner_keeps_control_inputs_and_reports_delta(tmp_path, monkeypatch):
    import phase4_diagnostics as phase4
    records = [{"id": "Q001", "question": "What?", "answerable": True,
                "reference_answer": "A", "question_type": "result", "split": "development",
                "human_verified": True, "relevant_evidence": [{"source_filename": "paper.pdf",
                "page": 2, "supporting_quote": "Evidence"}]}]
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps(records))
    monkeypatch.setattr(phase4, "REVIEWED_SHA256", phase4.hashlib.sha256(dataset.read_bytes()).hexdigest())
    index = tmp_path / "existing-index"
    index.mkdir()
    calls = []

    def retrieve(query, n_results, trace, candidate_source):
        calls.append((query, n_results, candidate_source))
        sources = (["other.pdf", "paper.pdf"] if candidate_source == "hybrid"
                   else ["paper.pdf", "other.pdf"])
        items = [{"source_filename": name, "chunk_id": str(i)}
                 for i, name in enumerate(sources)]
        trace.update({stage: items for stage in
                      ("dense", "bm25", "fused", "distance_filter", "expanded")})
        return {"metadatas": [[{"source": name} for name in sources]]}

    args = SimpleNamespace(dataset=dataset, audit_only=False, candidate_pool=30,
                           index_tag="fixed", index=index, variant="dense-only", rerank=False,
                           output=tmp_path / "result.json")
    run(args, retriever=retrieve)
    result = json.loads(args.output.read_text())
    assert calls == [("What?", 30, "hybrid"), ("What?", 30, "dense-only")]
    assert result["aggregate"]["paired_deltas"]["n"] == 1
    assert result["aggregate"]["paired_deltas"]["mrr_known_positive_delta"] == 0.5
    assert result["per_question"]["hybrid"][0]["known_source_rank"] == 2


def test_reviewed_dataset_crlf_and_lf_hashes_accepted(tmp_path, monkeypatch):
    import phase4_diagnostics as phase4
    dataset = tmp_path / "data.json"
    dataset.write_text("[]")
    digest = phase4.hashlib.sha256(dataset.read_bytes()).hexdigest()
    monkeypatch.setattr(phase4, "REVIEWED_SHA256", "something_else")
    monkeypatch.setattr(phase4, "REVIEWED_SHA256_CRLF", digest)
    args = SimpleNamespace(dataset=dataset, audit_only=True)
    assert phase4.run(args)["status"] == "audit_only"
    monkeypatch.setattr(phase4, "REVIEWED_SHA256", digest)
    monkeypatch.setattr(phase4, "REVIEWED_SHA256_CRLF", "something_else")
    assert phase4.run(args)["status"] == "audit_only"
    monkeypatch.setattr(phase4, "REVIEWED_SHA256", "mismatch1")
    monkeypatch.setattr(phase4, "REVIEWED_SHA256_CRLF", "mismatch2")
    with pytest.raises(ValueError, match="Reviewed dataset differs"):
        phase4.run(args)


def test_rerank_slices_to_candidate_pool(monkeypatch):
    import phase4_diagnostics as phase4
    mock_reranker = SimpleNamespace(predict=lambda pair: 0.9 if "gold" in pair[1] else 0.1)
    monkeypatch.setattr("logic_engine._get_reranker", lambda: mock_reranker)
    raw = {
        "documents": [["doc1", "gold_doc2", "doc3", "doc4"]],
        "metadatas": [[{"source": "other1.pdf", "chunk_id": "c1"},
                       {"source": "gold.pdf", "chunk_id": "c2"},
                       {"source": "other2.pdf", "chunk_id": "c3"},
                       {"source": "other3.pdf", "chunk_id": "c4"}]]
    }
    res = phase4.rerank_fixed_candidates("query", raw, candidate_pool=2)
    assert len(res["ranked"]) == 2
    assert [r["source_filename"] for r in res["ranked"]] == ["gold.pdf", "other1.pdf"]


def test_trace_one_rerank_fields_and_candidate_match(monkeypatch):
    import phase4_diagnostics as phase4
    mock_reranker = SimpleNamespace(predict=lambda pair: 0.95 if "gold" in pair[1] else 0.1)
    monkeypatch.setattr("logic_engine._get_reranker", lambda: mock_reranker)

    record = {"question": "What?"}
    audit = {"id": "Q001", "gold_papers": ["gold.pdf"],
             "evidence": [{"source_filename": "gold.pdf", "page": 2}]}

    def retriever(query, n_results, trace, candidate_source):
        items = [
            {"source_filename": "other.pdf", "chunk_id": "c1", "page": 1},
            {"source_filename": "gold.pdf", "chunk_id": "c2", "page": 2}
        ]
        trace.update({name: items for name in ("dense", "bm25", "fused", "distance_filter", "expanded")})
        return {
            "documents": [["other doc", "gold doc"]],
            "metadatas": [[{"source": "other.pdf", "chunk_id": "c1", "page_index": 0},
                           {"source": "gold.pdf", "chunk_id": "c2", "page_index": 1}]]
        }

    row = phase4.trace_one(record, audit, retriever, candidate_pool=2, source="hybrid", rerank=True)
    assert row["status"] == "scored"
    assert row["reranker"]["candidate_ids_match_prerank"] is True
    assert row["reranker"]["candidate_ids"] == ["c1", "c2"]
    assert row["reranker"]["first_chunk_rank"] == 1
    assert row["reranker"]["first_distinct_paper_rank"] == 1
    assert row["reranker"]["first_approved_page_rank"] == 1
    assert row["reranker"]["survive_process_k"] is True
    assert row["reranker"]["survive_adaptive"] is True

    summary = phase4.aggregate([row])
    assert summary["reranker_aggregate"]["first_distinct_paper_hit_1_count"] == 1
    assert summary["reranker_aggregate"]["survive_process_k_count"] == 1


def test_manifest_source_mapping_consistency():
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    with open(repo_root / "reviewed_dataset.json", "r", encoding="utf-8") as f:
        reviewed = json.load(f)
    with open(repo_root / "docs" / "evaluation" / "phase4_split_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_map = {}
    for part in [
        manifest["development_partition"]["existing_phase3_development"]["questions"],
        manifest["development_partition"]["new_phase4_development"]["questions"],
        manifest["reserved_held_out_partition"]["questions"],
    ]:
        for item in part:
            manifest_map[item["id"]] = item["source_paper"]

    reviewed_map = {
        q["id"]: q["relevant_evidence"][0]["source_filename"]
        for q in reviewed
        if q.get("relevant_evidence")
    }

    for qid, reviewed_source in reviewed_map.items():
        assert qid in manifest_map, f"QID {qid} missing from split manifest"
        assert manifest_map[qid] == reviewed_source, (
            f"Mapping mismatch for {qid}: manifest={manifest_map[qid]} vs reviewed={reviewed_source}"
        )


def test_relevance_schema_and_signed_human_verification():
    """
    Verifies that docs/evaluation/phase4_relevance.json conforms to its JSON Schema,
    that all 16 rows reflect the signed human decisions by reviewer Qusai (8 relevant, 8 nonrelevant),
    that Q023 and Q025 passage fidelity is preserved, that all four corrective fixtures
    have human_verified=True, and that CORR-003 is verified on physical page 5.
    """
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parent.parent

    relevance_path = repo_root / "docs" / "evaluation" / "phase4_relevance.json"
    schema_path = repo_root / "docs" / "evaluation" / "phase4_relevance.schema.json"
    fixtures_path = repo_root / "docs" / "evaluation" / "corrective_decision_fixtures.json"

    with open(relevance_path, "r", encoding="utf-8") as f:
        relevance_rows = json.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # 1. Schema validation
    try:
        import jsonschema
        jsonschema.validate(instance=relevance_rows, schema=schema)
    except ImportError:
        pass
    assert len(relevance_rows) == 16

    # 2. Verified state invariant: 16 rows signed by reviewer Qusai
    relevant_count = 0
    nonrelevant_count = 0
    for row in relevance_rows:
        assert row["judgment"] in ("relevant", "nonrelevant"), f"Row {row['query_id']} has unexpected judgment: {row['judgment']}"
        assert row["reviewer"] == "Qusai", f"Row {row['query_id']} has unexpected reviewer: {row['reviewer']}"
        assert row.get("proposed_judgment") in ("relevant", "nonrelevant")
        if row["judgment"] == "relevant":
            relevant_count += 1
        else:
            nonrelevant_count += 1

    assert relevant_count == 8, f"Expected 8 relevant rows, found {relevant_count}"
    assert nonrelevant_count == 8, f"Expected 8 nonrelevant rows, found {nonrelevant_count}"

    # 3. Passage fidelity checks
    by_key = {(r["query_id"], r["source_filename"], r["page"]): r for r in relevance_rows}

    # Q023 nonrelevant Dynamo title check
    q023_dynamo = by_key[("Q023", "1294261.1294281.pdf", 1)]
    assert q023_dynamo["passage"] == "Dynamo: Amazon's Highly Available Key-Value Store", (
        f"Q023 Dynamo passage error: {q023_dynamo['passage']}"
    )
    assert "The Tail at Scale" not in q023_dynamo["passage"]

    # Q023 relevant GFS full passage check
    q023_gfs = by_key[("Q023", "945445.945450.pdf", 4)]
    assert "Chunk size is 64 MB" in q023_gfs["passage"]
    assert "perhaps the greatest objection against such a large chunk size." in q023_gfs["passage"]

    # Q025 relevant AMA exact Figure 2 caption check
    q025_ama = by_key[("Q025", "AMA_AdaptiveMemoryviaMultiAgentCollaboration_compressed.pdf", 3)]
    assert q025_ama["passage"].startswith("Figure 2: Overview of the AMA framework.")
    assert "Finally, the Constructor synthesizes the validated context into structured memory entries." in q025_ama["passage"]
    assert "discrepancies" not in q025_ama["passage"], "Q025 passage contains unverified paraphrase"

    # 4. Corrective decision fixtures verified state
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)
    assert len(fixtures) == 4

    fix_map = {fix["id"]: fix for fix in fixtures}
    for fix_id, expected_decision in [
        ("CORR-001", "relevant"),
        ("CORR-002", "partially_relevant"),
        ("CORR-003", "irrelevant"),
        ("CORR-004", "insufficient_evidence"),
    ]:
        fix = fix_map[fix_id]
        assert fix["human_verified"] is True, f"Fixture {fix_id} human_verified is not True"
        assert fix["proposal_status"] == "human_verified", f"Fixture {fix_id} proposal_status != human_verified"
        assert fix["category"] == expected_decision
        assert fix["gold_decision"] == expected_decision
        assert "Qusai" in fix["reviewer"]

    # 5. CORR-003 verified on physical page 5
    corr_003 = fix_map["CORR-003"]
    assert corr_003["passages"][0]["page"] == 5, f"CORR-003 page is not 5: {corr_003['passages'][0]['page']}"
    assert "MultiHead(Q, K, V)" in corr_003["passages"][0]["text"]
    assert "physical page 5" in corr_003["rationale"]

    # 6. CORR-004 verified on physical page 4
    corr_004 = fix_map["CORR-004"]
    assert corr_004["passages"][0]["page"] == 4, f"CORR-004 page is not 4: {corr_004['passages'][0]['page']}"
    assert corr_004["passages"][0]["source_filename"] == "1512.03385v1.pdf"
    assert "Our implementation for ImageNet" in corr_004["passages"][0]["text"]

    # 7. CORR-002 working rationale distinguishes supplied passage from other Adam content
    corr_002 = fix_map["CORR-002"]
    assert corr_002["passages"][0]["page"] == 2
    assert "alpha = 0.001" in corr_002["passages"][0]["text"]
    assert "only Adam's recommended default hyperparameter values" in corr_002["rationale"]
    assert "bias-corrected update formulas" in corr_002["rationale"]



def test_target_citation_consistency():
    """
    Ensures that for all 26 questions, gold_paper in phase4_generation_worksheet.json
    and phase4_human_answer_verdicts.json match, and that citations validity is
    consistent across records.
    Prevents regressions where gold_paper cited boolean disagrees with emitted citations.
    """
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    worksheet_path = repo_root / "docs" / "evaluation" / "phase4_generation_worksheet.json"
    verdicts_path = repo_root / "docs" / "evaluation" / "phase4_human_answer_verdicts.json"

    with open(worksheet_path, "r", encoding="utf-8") as f:
        ws_records = {r["question_id"]: r for r in json.load(f)["records"]}

    with open(verdicts_path, "r", encoding="utf-8") as f:
        vd_records = {r["question_id"]: r for r in json.load(f)["records"]}

    assert len(ws_records) == 26
    assert len(vd_records) == 26

    for qid, ws_rec in ws_records.items():
        vd_rec = vd_records[qid]
        gold_paper = ws_rec["gold_paper"]
        assert vd_rec.get("target_paper", vd_rec.get("gold_paper")) == gold_paper, f"Gold paper mismatch for {qid}"

        emitted_citations = ws_rec.get("citations", [])
        expected_cited = gold_paper in emitted_citations
        actual_cited = ws_rec.get("gold_paper_cited")
        if actual_cited is not None:
            assert actual_cited == expected_cited, (
                f"Worksheet cited boolean mismatch for {qid}: {actual_cited} vs expected {expected_cited}"
            )


def test_human_answer_verdicts_schema_and_counts():
    """
    Validates that phase4_human_answer_verdicts.json conforms to its schema,
    verifies SHA-256 of the authoritative signed form, checks exact partition
    counts (Dev: 2 ACCEPT, 9 PARTIAL, 11 REJECT; Pilot: 4 ACCEPT), and checks
    nuanced signed judgments (e.g. Q013, Q020).
    """
    import pathlib
    import hashlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    verdicts_path = repo_root / "docs" / "evaluation" / "phase4_human_answer_verdicts.json"
    schema_path = repo_root / "docs" / "evaluation" / "phase4_human_answer_verdicts.schema.json"
    signed_form_path = repo_root / "phase4_human_answer_verdict_form.md"

    # 1. Authoritative signed form hash
    with open(verdicts_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_signed_sha = "172bb3a46973534430aa2c2e909cac76a76d0801c2618dbd22ca80a152ceeb08"
    assert data["metadata"]["authoritative_signed_form_sha256"] == expected_signed_sha
    signed_sha = expected_signed_sha
    if signed_form_path.exists():
        actual_signed_sha = hashlib.sha256(signed_form_path.read_bytes()).hexdigest()
        assert actual_signed_sha == expected_signed_sha
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
    except ImportError:
        pass

    # 3. Completeness & partition checks
    records = data["records"]
    assert len(records) == 26
    by_id = {r["question_id"]: r for r in records}
    assert sorted(list(by_id.keys())) == [f"Q{i:03d}" for i in range(1, 27)]

    dev_counts = {"ACCEPT": 0, "PARTIAL": 0, "REJECT": 0}
    pilot_counts = {"ACCEPT": 0, "PARTIAL": 0, "REJECT": 0}
    for r in records:
        assert r["reviewer"] == "Qusai"
        assert r["sign_off_date"] == "2026-10-21"
        assert r["signed_form_sha256"] == signed_sha
        assert isinstance(r["approved_pages"], list) and len(r["approved_pages"]) >= 1
        for ap in r["approved_pages"]:
            assert isinstance(ap["source_filename"], str) and len(ap["source_filename"]) > 0
            assert isinstance(ap["page"], int) and ap["page"] >= 1
            assert "**" not in ap["source_filename"]
        assert isinstance(r["pages_checked"], str) and len(r["pages_checked"]) > 0
        assert "**" not in r["pages_checked"]
        dec = r["overall_decision"]
        if int(r["question_id"][1:]) <= 22:
            assert r["partition"] == "development"
            dev_counts[dec] += 1
        else:
            assert r["partition"] == "disclosed_pilot"
            pilot_counts[dec] += 1

    assert dev_counts == {"ACCEPT": 2, "PARTIAL": 9, "REJECT": 11}
    assert pilot_counts == {"ACCEPT": 4, "PARTIAL": 0, "REJECT": 0}

    # 4. Nuanced grading checks
    # Q013: All claims covered, but refusal framing leads to PARTIAL overall
    q013 = by_id["Q013"]
    assert q013["overall_decision"] == "PARTIAL"
    assert all(c["coverage_choice"] == "Covered" for c in q013["claim_verdicts"])

    # Q020: Adam default error causes Incorrect factual, but bias explanation leads to PARTIAL overall
    q020 = by_id["Q020"]
    assert q020["overall_decision"] == "PARTIAL"
    assert q020["factual_correctness"] == "Incorrect"

    # Q021: Clean ACCEPT
    q021 = by_id["Q021"]
    assert q021["overall_decision"] == "ACCEPT"
    assert q021["factual_correctness"] == "Correct"


def test_no_unsupported_inferred_claim_page_mappings():
    """
    Regression test ensuring that Q001-Q018 do not contain inferred claim-level page mappings
    that were not individually verified by human review, while preserving verified question-level evidence.
    Also ensures Q019-Q026 retain their signed claim-level supports.
    """
    repo_root = Path(__file__).resolve().parent.parent
    p4_file = repo_root / "docs" / "evaluation" / "phase4_questions.json"
    rev_file = repo_root / "reviewed_dataset.json"
    new_file = repo_root / "docs" / "evaluation" / "phase4_new_questions.json"

    questions = json.loads(p4_file.read_text(encoding="utf-8"))
    reviewed = json.loads(rev_file.read_text(encoding="utf-8"))
    rev_map = {r["id"]: r for r in reviewed}
    new_map = None
    if new_file.exists():
        new_q = json.loads(new_file.read_text(encoding="utf-8"))
        new_map = {n["id"]: n for n in new_q}

    dev_count = 0
    pilot_count = 0

    for q in questions:
        qid = q["id"]
        if q["partition"] == "development":
            dev_count += 1
        elif q["partition"] == "disclosed_pilot":
            pilot_count += 1

        if qid in rev_map:
            # Q001-Q018: historical Phase 3 questions
            r = rev_map[qid]
            assert q["question"] == r["question"]
            assert q["reference_answer"] == r["reference_answer"]
            assert q["required_claims"] == r["required_claims"]
            assert len(q["approved_pages"]) > 0
            assert len(q["relevant_evidence"]) > 0

            # Must NOT contain inferred claim-level page mappings
            cs = q.get("claim_supports", [])
            assert cs == [], f"{qid} should not have inferred claim_supports; got: {cs}"

        elif qid in [f"Q{i:03d}" for i in range(19, 27)]:
            # Q019-Q026: Phase 4 new questions with human-signed claim supports
            cs = q.get("claim_supports", [])
            assert len(cs) > 0, f"{qid} must have claim_supports"
            for c_actual in cs:
                assert "claim" in c_actual and len(c_actual["claim"]) > 0
                assert isinstance(c_actual["page"], int) and c_actual["page"] >= 1
                assert c_actual["source_filename"].endswith(".pdf")
                assert len(c_actual.get("approved_pages", [])) >= 1

            if new_map and qid in new_map:
                n = new_map[qid]
                assert q["question"] == n["question"]
                assert q["reference_answer"] == n["proposed_answer"]
                assert len(cs) == len(n["claim_supports"]), f"{qid} claim_supports length mismatch"
                for c_actual, c_expected in zip(cs, n["claim_supports"]):
                    assert c_actual["claim"] == c_expected["claim"]
                    assert c_actual["page"] == c_expected["page"]
                    assert c_actual["source_filename"] == (c_expected.get("source_filename") or n["source_filename"])
                    assert len(c_actual.get("approved_pages", [])) == 1
                    assert c_actual["approved_pages"][0]["page"] == c_expected["page"]

    assert dev_count == 22
    assert pilot_count == 4
