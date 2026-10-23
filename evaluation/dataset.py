"""Validate candidate questions without treating generated answers as gold."""
import hashlib
import json
from pathlib import Path

QUESTION_TYPES = {
    "factual", "method", "result", "limitation", "comparison",
    "multi_hop", "unanswerable",
}
SPLITS = {"development", "test"}


def load_dataset(path):
    data = Path(path).read_bytes()
    try:
        records = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset JSON: {exc}") from exc
    if not isinstance(records, list):
        raise ValueError("Dataset must be a JSON list")
    return records, hashlib.sha256(data).hexdigest()


def audit_records(records):
    """Return one row per input record, including invalid and duplicate rows."""
    seen_ids, seen_questions = set(), set()
    rows = []
    for position, record in enumerate(records, 1):
        errors, gaps = [], []
        if not isinstance(record, dict):
            rows.append({"id": f"legacy-{position:03d}", "position": position,
                         "status": "invalid", "reasons": ["record_not_object"],
                         "gold_papers": [], "evidence": [], "split": None})
            continue
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            identifier = f"legacy-{position:03d}"
            gaps.append("missing_id")
        else:
            identifier = identifier.strip()
        if identifier in seen_ids:
            errors.append("duplicate_id")
        seen_ids.add(identifier)
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            errors.append("missing_question")
        else:
            normalized = " ".join(question.casefold().split())
            if normalized in seen_questions:
                errors.append("duplicate_question")
            seen_questions.add(normalized)
        answerable = record.get("answerable")
        if answerable is None:
            gaps.append("missing_answerable")
        elif type(answerable) is not bool:
            errors.append("invalid_answerable")
        qtype = record.get("question_type")
        if qtype is None:
            gaps.append("missing_question_type")
        elif not isinstance(qtype, str) or qtype not in QUESTION_TYPES:
            errors.append("invalid_question_type")
        if type(answerable) is bool and qtype in QUESTION_TYPES:
            if (answerable is False) != (qtype == "unanswerable"):
                errors.append("inconsistent_answerability_type")
        split = record.get("split")
        if split is None:
            gaps.append("missing_split")
        elif not isinstance(split, str) or split not in SPLITS:
            errors.append("invalid_split")
        verified = record.get("human_verified")
        if verified is None or verified is False:
            gaps.append("needs_human_verification")
        elif verified is not True:
            errors.append("invalid_human_verified")
        if answerable is not False:
            reference = record.get("reference_answer")
            if not isinstance(reference, str) or not reference.strip():
                gaps.append("missing_verified_reference_answer")
        if answerable is False and record.get("corpus_reviewed") is not True:
            gaps.append("missing_full_corpus_review")
        evidence = record.get("relevant_evidence")
        if evidence is None or evidence == []:
            if answerable is not False:
                gaps.append("missing_relevant_evidence")
            evidence = []
        elif not isinstance(evidence, list):
            errors.append("invalid_relevant_evidence")
            evidence = []
        if answerable is False and evidence:
            errors.append("unanswerable_has_evidence")
        papers = set()
        for item in evidence:
            if not isinstance(item, dict):
                errors.append("invalid_evidence_item")
                continue
            name = item.get("source_filename")
            if (not isinstance(name, str) or not name.strip() or
                    Path(name).name != name or "\\" in name or not name.lower().endswith(".pdf")):
                errors.append("invalid_source_filename")
            else:
                papers.add(name)
            if type(item.get("page")) is not int or item["page"] < 1:
                errors.append("invalid_page")
            quote = item.get("supporting_quote")
            if not isinstance(quote, str) or not quote.strip():
                gaps.append("missing_supporting_quote")
            grade = item.get("relevance_grade")
            if grade is not None and (type(grade) is not int or grade not in (1, 2, 3)):
                errors.append("invalid_relevance_grade")
        negatives = record.get("judged_nonrelevant_papers", [])
        if not isinstance(negatives, list):
            errors.append("invalid_judged_nonrelevant_papers")
            negatives = []
        elif (any(not isinstance(name, str) or not name.lower().endswith(".pdf") or
                  Path(name).name != name or "\\" in name or name in papers
                  for name in negatives) or
              len([name for name in negatives if isinstance(name, str)]) !=
              len(set(name for name in negatives if isinstance(name, str)))):
            errors.append("invalid_judged_nonrelevant_papers")
        reasons = sorted(set(errors + gaps))
        status = ("invalid" if errors else "needs_human_review" if gaps else
                  "eligible_retrieval" if answerable else "eligible_abstention")
        rows.append({"id": identifier, "position": position, "status": status,
                     "reasons": reasons, "gold_papers": sorted(papers),
                     "evidence": evidence, "split": split,
                     "judged_nonrelevant_papers": negatives})
    return rows
