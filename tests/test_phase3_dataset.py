import json

import pytest

from evaluation.dataset import audit_records, load_dataset


def annotated(**overrides):
    record = {"id": "Q001", "question": "What did the paper find?",
              "answerable": True, "reference_answer": "An approved answer.",
              "question_type": "result", "split": "development",
              "human_verified": True,
              "relevant_evidence": [{"source_filename": "paper.pdf", "page": 2,
                                     "supporting_quote": "Verified excerpt.",
                                     "relevance_grade": 2}]}
    return record | overrides


def test_malformed_json_and_missing_core_fields(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{broken")
    with pytest.raises(ValueError, match="Invalid dataset JSON"):
        load_dataset(path)
    path.write_text('{"not":"a list"}')
    with pytest.raises(ValueError, match="JSON list"):
        load_dataset(path)
    rows = audit_records([{}, "bad"])
    assert rows[0]["status"] == "invalid"
    assert "missing_question" in rows[0]["reasons"]
    assert "missing_id" in rows[0]["reasons"]
    assert rows[1]["status"] == "invalid"


def test_duplicates_and_unverified_legacy():
    rows = audit_records([annotated(), annotated(id="Q001"),
                          {"question": "candidate", "ground_truth": "generated",
                           "contexts": ["excerpt"]}])
    assert rows[0]["status"] == "eligible_retrieval"
    assert {"duplicate_id", "duplicate_question"} <= set(rows[1]["reasons"])
    assert rows[2]["status"] == "needs_human_review"
    assert rows[2]["id"] == "legacy-003"


@pytest.mark.parametrize("changes,reason", [
    ({"answerable": "yes"}, "invalid_answerable"),
    ({"split": "training"}, "invalid_split"),
    ({"question_type": "other"}, "invalid_question_type"),
    ({"answerable": False, "question_type": "factual",
      "relevant_evidence": []}, "inconsistent_answerability_type"),
    ({"answerable": True, "question_type": "unanswerable"},
     "inconsistent_answerability_type"),
    ({"relevant_evidence": [{"source_filename": "paper.pdf", "page": 0,
                             "supporting_quote": "text"}]}, "invalid_page"),
    ({"relevant_evidence": [{"source_filename": "paper.pdf", "page": 1,
                             "supporting_quote": "text", "relevance_grade": 4}]},
     "invalid_relevance_grade"),
])
def test_invalid_annotations(changes, reason):
    row = audit_records([annotated(**changes)])[0]
    assert row["status"] == "invalid"
    assert reason in row["reasons"]


def test_missing_evidence_and_verification_are_excluded():
    row = audit_records([annotated(human_verified=False,
                                   relevant_evidence=[{"source_filename": "paper.pdf",
                                                       "page": 1}])])[0]
    assert row["status"] == "needs_human_review"
    assert {"needs_human_verification", "missing_supporting_quote"} <= set(row["reasons"])
    assert "missing_relevant_evidence" in audit_records([
        annotated(relevant_evidence=[])])[0]["reasons"]


def test_unanswerable_needs_full_corpus_review():
    row = audit_records([annotated(answerable=False, question_type="unanswerable",
                                   relevant_evidence=[], reference_answer="")])[0]
    assert row["status"] == "needs_human_review"
    assert "missing_full_corpus_review" in row["reasons"]
    approved = annotated(answerable=False, question_type="unanswerable",
                         relevant_evidence=[], corpus_reviewed=True)
    assert audit_records([approved])[0]["status"] == "eligible_abstention"
