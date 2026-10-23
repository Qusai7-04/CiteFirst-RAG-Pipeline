"""
Tests for Phase 4 live capture interface:
- Status accuracy, context capture, error handling, secret leak prevention
- Chosen-model propagation to all LLM stages
- Multi-page evidence handling (Q019-Q021)
- Exact final-context capture at generation boundary without duplicate retrieval
- Fail-fast on provider error
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import providers
import logic_engine
from scripts.capture_generation_worksheet import capture_generation_worksheet
from scripts.run_real_pipeline_evaluation import (
    load_all_question_data, evaluate_single_question, sanitize_message
)


def test_generation_status_not_run_when_provider_not_called(tmp_path, monkeypatch):
    """
    Ensures that a configured key alone never marks generation_status as EXECUTED
    when no live generation actually executed.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-dummy_test_key_not_real")

    target_out = tmp_path / "test_worksheet.json"
    dummy_trace = tmp_path / "test_trace.json"
    dummy_trace.write_text(json.dumps({
        "records": [
            {
                "id": "Q001",
                "actual_returned_count": 1,
                "verified_chunks_count": 1,
                "fallback_triggered": False,
                "selected_generation_context": [
                    {
                        "source": "rsta.1895.0004.pdf",
                        "page": 1,
                        "chunk_id": "c1",
                        "text": "Exact text from page 1"
                    }
                ]
            }
        ]
    }), encoding="utf-8")

    import scripts.capture_generation_worksheet as cgw
    monkeypatch.setattr(cgw, "WORKSHEET_PATH", target_out)
    monkeypatch.setattr(cgw, "TRACE_PATH", dummy_trace)

    cgw.capture_generation_worksheet(limit=1)

    assert target_out.exists()
    data = json.loads(target_out.read_text(encoding="utf-8"))

    assert data["generation_status"] == "NOT RUN — no live generation executed"
    assert data["records"][0]["generation"]["status"] != "EXECUTED"
    assert ("NOT RUN" in data["records"][0]["generation"]["status"] or
            data["records"][0]["generation"]["status"] == "PENDING_LIVE_GENERATION")
    assert data["records"][0]["generation"]["generated_answer"] is None


def test_generation_context_page_and_metadata_integrity(tmp_path, monkeypatch):
    """
    Verifies that selected generation context records 1-based physical PDF pages,
    source filenames, and exact passage strings without inventing metadata.
    """
    target_out = tmp_path / "test_worksheet.json"
    dummy_trace = tmp_path / "test_trace.json"
    dummy_trace.write_text(json.dumps({
        "records": [
            {
                "id": "Q001",
                "actual_returned_count": 1,
                "verified_chunks_count": 1,
                "fallback_triggered": False,
                "selected_generation_context": [
                    {
                        "source": "rsta.1895.0004.pdf",
                        "page": 1,
                        "chunk_id": "c1",
                        "text": "Exact text passed to generation for testing."
                    }
                ]
            }
        ]
    }), encoding="utf-8")

    import scripts.capture_generation_worksheet as cgw
    monkeypatch.setattr(cgw, "WORKSHEET_PATH", target_out)
    monkeypatch.setattr(cgw, "TRACE_PATH", dummy_trace)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-dummy_key")

    cgw.capture_generation_worksheet(limit=1)
    data = json.loads(target_out.read_text(encoding="utf-8"))

    rec = data["records"][0]
    ctx = rec["retrieval_and_selection"]["selected_context"]
    assert len(ctx) > 0
    for chunk in ctx:
        assert "source_filename" in chunk
        assert "exact_text_passed_to_generation" in chunk
        assert isinstance(chunk["page"], int) or chunk["page"] is None
        if isinstance(chunk["page"], int):
            assert chunk["page"] >= 1


def test_error_handling_and_prevention_of_secret_logging():
    """
    Verifies that error recording sanitizes any potential secret keys or authorization headers
    for OpenRouter.
    """
    secret_or = "sk-or-v1-abcdef1234567890dummysecret"
    os.environ["OPENROUTER_API_KEY"] = secret_or
    try:
        error_msg = f"API request failed with Bearer {secret_or}: 401 Unauthorized"
        sanitized = sanitize_message(error_msg)
        assert secret_or not in sanitized
        assert "[REDACTED_OPENROUTER_API_KEY]" in sanitized
    finally:
        del os.environ["OPENROUTER_API_KEY"]


def test_multipage_approved_pages_extraction():
    """
    Verifies that load_all_question_data builds approved source/page pairs
    from every claim_supports item, including second pages for Q019–Q021.
    """
    all_q = load_all_question_data()

    # Q019 should have pages 4 and 5 of 1406.2661v1.pdf
    assert "Q019" in all_q
    q19_pages = {p["page"] for p in all_q["Q019"]["approved_pages"] if p["source_filename"] == "1406.2661v1.pdf"}
    assert 4 in q19_pages
    assert 5 in q19_pages

    # Q020 should have pages 2 and 3 of 1412.6980v9.pdf
    assert "Q020" in all_q
    q20_pages = {p["page"] for p in all_q["Q020"]["approved_pages"] if p["source_filename"] == "1412.6980v9.pdf"}
    assert 2 in q20_pages
    assert 3 in q20_pages

    # Q021 should have pages 2 and 3 of 1512.03385v1.pdf
    assert "Q021" in all_q
    q21_pages = {p["page"] for p in all_q["Q021"]["approved_pages"] if p["source_filename"] == "1512.03385v1.pdf"}
    assert 2 in q21_pages
    assert 3 in q21_pages


def test_model_propagation_to_all_logic_engine_stages(monkeypatch):
    """
    Verifies that logic_engine passes OPENROUTER_MODEL to all generation call sites.
    """
    monkeypatch.setenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    monkeypatch.setattr(logic_engine, "ANSWER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    monkeypatch.setattr(logic_engine, "LIGHT_TEXT_MODEL", "meta-llama/llama-3.3-70b-instruct")
    routed_models = []

    def mock_chat(prompt, system, model=None, max_tokens=1024, retries=2, on_trace=None):
        routed_models.append(model)
        if "Does the 'Retrieved Context'" in prompt:
            return "YES"
        return "mock answer"

    monkeypatch.setattr(providers, "chat", mock_chat)

    # 1. Combined relevance check
    logic_engine._combined_relevance_check("test query", ["chunk text"])
    assert routed_models[-1] == "meta-llama/llama-3.3-70b-instruct"

    # 2. Extract best sentence
    logic_engine.extract_best_sentence("test query", "some chunk text")
    assert routed_models[-1] == "meta-llama/llama-3.3-70b-instruct"

    # 3. Supplemental search
    monkeypatch.setattr(logic_engine, "search_vault", lambda *a, **k: {"documents": [[]], "metadatas": [[]]})
    logic_engine.supplemental_search("query", "anchor")
    assert routed_models[-1] == "meta-llama/llama-3.3-70b-instruct"

    # 4. Rewrite query
    logic_engine.rewrite_query("failed query")
    assert routed_models[-1] == "meta-llama/llama-3.3-70b-instruct"

    # 5. Final answer
    logic_engine.generate_final_answer("query", ["chunk"], ["source.pdf"])
    assert routed_models[-1] == "meta-llama/llama-3.3-70b-instruct"


def test_exact_context_capture_without_duplicate_retrieval(monkeypatch):
    """
    Verifies evaluate_single_question executes run_logic_engine once,
    capturing exact context and prompt at the generation boundary.
    """
    search_calls = []

    def mock_run_logic_engine(query, on_stage=None, on_trace=None, **kwargs):
        search_calls.append(query)
        if on_stage:
            on_stage("generating", "Generating answer")
        if on_trace:
            on_trace("search", {
                "stage": "initial",
                "query": query,
                "documents": ["Verbatim passage text from paper."],
                "metadatas": [{"source": "1406.2661v1.pdf", "page_index": 3, "chunk_id": "c1"}]
            })
            on_trace("generation_boundary", {
                "type": "final_answer",
                "prompt": "Verified Context: Verbatim passage text from paper.",
                "system": "System",
                "model": "meta-llama/llama-3.3-70b-instruct",
                "context": "Context",
                "chunks": ["Verbatim passage text from paper."],
                "sources": ["1406.2661v1.pdf"]
            })
        return {
            "status": "success",
            "answer": "Generated cited answer.",
            "facts": ["Verbatim passage text from paper."],
            "evidence": [{
                "id": "c1",
                "document_id": "d1",
                "source": "1406.2661v1.pdf",
                "page_index": 3,
                "page_number": 4,
                "text": "Verbatim passage text from paper."
            }],
            "sources": ["1406.2661v1.pdf"],
            "query_used": query
        }

    monkeypatch.setattr("scripts.run_real_pipeline_evaluation.run_logic_engine", mock_run_logic_engine)

    q_data = {
        "id": "Q019",
        "question": "What is the optimal discriminator?",
        "gold_papers": ["1406.2661v1.pdf"],
        "reference_answer": "D*",
        "claims": ["Claim 1"],
        "claims_with_pages": [{"claim": "Claim 1", "approved_pages": [{"source_filename": "1406.2661v1.pdf", "page": 4}]}],
        "approved_pages": [{"source_filename": "1406.2661v1.pdf", "page": 4}],
        "split": "development_new"
    }

    res = evaluate_single_question(q_data, provider_name="openrouter", model_name="meta-llama/llama-3.3-70b-instruct")

    # Verify only ONE pipeline call was made
    assert len(search_calls) == 1
    # Verify exact context captured
    ctx = res["generation_boundary"]["exact_context_passed_to_generation"]
    assert len(ctx) == 1
    assert ctx[0]["source_filename"] == "1406.2661v1.pdf"
    assert ctx[0]["page"] == 4
    assert ctx[0]["exact_text_passed_to_generation"] == "Verbatim passage text from paper."
    assert res["generation_boundary"]["approved_passage_in_final_context"] is True
    assert res["pipeline_execution"]["generation_status"] == "EXECUTED_SUCCESS"
    assert res["pipeline_execution"]["citations"] == ["1406.2661v1.pdf"]
    assert res["pipeline_execution"]["citations_validity"]["gold_paper_cited"] is True


def test_provider_error_propagation_and_classification(monkeypatch):
    """
    Verifies that a ProviderError during run_logic_engine is recorded
    with status FAILED_PROVIDER_... and does not masquerade as an abstention.
    """
    def mock_fail(query, *args, **kwargs):
        raise providers.ProviderError("OpenRouter API error (403): API key invalid")

    monkeypatch.setattr("scripts.run_real_pipeline_evaluation.run_logic_engine", mock_fail)

    q_data = {
        "id": "Q001",
        "question": "What is turbulent flow?",
        "gold_papers": ["rsta.1895.0004.pdf"],
        "reference_answer": "Ref",
        "claims": ["Claim"],
        "claims_with_pages": [{"claim": "Claim", "approved_pages": [{"source_filename": "rsta.1895.0004.pdf", "page": 17}]}],
        "approved_pages": [{"source_filename": "rsta.1895.0004.pdf", "page": 17}],
        "split": "development_existing"
    }

    res = evaluate_single_question(q_data, provider_name="openrouter", model_name="meta-llama/llama-3.3-70b-instruct")

    assert res["pipeline_execution"]["generation_status"] == "FAILED_PROVIDER_PROVIDERERROR"
    assert "ProviderError" in res["pipeline_execution"]["error"]["type"]
    assert res["pipeline_execution"]["generated_answer"] is None


def test_invocation_trace_captures_pipeline_searches(monkeypatch):
    """
    Verifies that the InvocationTrace callback properly captures search events
    at the real call site in logic_engine, avoiding the monkey-patching failure mode
    where searches_count was 0 despite pipeline activity.
    """
    mock_docs = ["This is a test chunk from rsta.1895.0004.pdf"]
    mock_metas = [{"source": "rsta.1895.0004.pdf", "page_index": 16, "chunk_id": "chunk_1"}]

    def mock_search_vault(query, *args, **kwargs):
        return {"documents": [mock_docs], "metadatas": [mock_metas]}

    monkeypatch.setattr(logic_engine, "search_vault", mock_search_vault)
    monkeypatch.setattr(logic_engine, "_combined_relevance_check", lambda q, c, **k: True)
    monkeypatch.setattr(providers, "chat", lambda p, s, **k: "Generated answer from paper.")

    q_data = {
        "id": "Q001",
        "question": "What is turbulent flow?",
        "gold_papers": ["rsta.1895.0004.pdf"],
        "reference_answer": "Reference answer",
        "claims": ["Claim 1"],
        "claims_with_pages": [{"claim": "Claim 1", "approved_pages": [{"source_filename": "rsta.1895.0004.pdf", "page": 17}]}],
        "approved_pages": [{"source_filename": "rsta.1895.0004.pdf", "page": 17}],
        "split": "development_existing"
    }

    res = evaluate_single_question(q_data, provider_name="openrouter", model_name="meta-llama/llama-3.3-70b-instruct")

    # Critical requirement: searches_count MUST be > 0 when the pipeline did search
    searches_count = res["retrieval_diagnostics"]["searches_count"]
    assert searches_count > 0, f"Expected searches_count > 0, got {searches_count}. InvocationTrace failed to capture search event."
    assert res["retrieval_diagnostics"]["returned_candidates_count"] == 1
    assert res["retrieval_diagnostics"]["known_source_retrieved"] is True
    assert res["retrieval_diagnostics"]["raw_paper_rank"] == 1
    # Check that generation boundary and exact context were captured
    assert res["generation_boundary"]["verified_context_count"] >= 1
    assert res["pipeline_execution"]["generation_status"] in ("EXECUTED_SUCCESS", "EXECUTED_FALLBACK_SUCCESS", "EXECUTED_PARTIAL_MATCH")


def test_extract_best_sentence_exact_match(monkeypatch):
    chunk = "This is irrelevant intro. Lazy space allocation avoids wasting space due to internal fragmentation. Extra text."
    monkeypatch.setattr(logic_engine, "call_llm", lambda *a, **k: "Lazy space allocation avoids wasting space due to internal fragmentation.")
    result = logic_engine.extract_best_sentence("query", chunk)
    assert result == "Lazy space allocation avoids wasting space due to internal fragmentation."


def test_extract_best_sentence_with_quotes(monkeypatch):
    chunk = "Intro. Chunk size is 64 MB, which is much larger than typical block sizes. Outro."
    monkeypatch.setattr(logic_engine, "call_llm", lambda *a, **k: '"Chunk size is 64 MB, which is much larger than typical block sizes."')
    result = logic_engine.extract_best_sentence("query", chunk)
    assert result == "Chunk size is 64 MB, which is much larger than typical block sizes."


def test_extract_best_sentence_truncation_recovery(monkeypatch):
    chunk = "Preliminary notes. The geometrical basis of the kinetic theory consists of the assumption that velocity is abstracted into a mean component. Concluding remark."
    # Simulate LLM truncation mid-sentence
    monkeypatch.setattr(logic_engine, "call_llm", lambda *a, **k: "The geometrical basis of the kinetic theory consists of the assumption that velocity")
    result = logic_engine.extract_best_sentence("query", chunk)
    assert result == "The geometrical basis of the kinetic theory consists of the assumption that velocity is abstracted into a mean component."


def test_extract_best_sentence_divergent_suffix(monkeypatch):
    """
    Verifies that a candidate whose prefix matches the chunk but whose suffix
    diverges (hallucination) is strictly rejected, guaranteeing no hallucinated
    or altered sentences are accepted.
    """
    chunk = "We evaluated the model on the ImageNet validation dataset for 90 epochs with standard augmentation. Concluding remarks."
    # First 45 characters match chunk exactly, but remainder diverges completely
    divergent_candidate = "We evaluated the model on the ImageNet valida and found that accuracy decreased drastically after 10 epochs."
    monkeypatch.setattr(logic_engine, "call_llm", lambda *a, **k: divergent_candidate)
    result = logic_engine.extract_best_sentence("query", chunk)
    assert result == "", "Divergent suffix MUST NOT be accepted via prefix matching"


def test_extract_best_sentence_abbreviation_and_decimal(monkeypatch):
    """
    Verifies that decimals (e.g. 0.001) and scientific abbreviations (e.g. et al., Fig.)
    do not trigger premature sentence boundary truncation.
    """
    chunk = "Introductory context. Good default settings are alpha = 0.001, beta1 = 0.9, as demonstrated by Kingma et al. in their benchmark experiments. Subsequent text."
    full_sentence = "Good default settings are alpha = 0.001, beta1 = 0.9, as demonstrated by Kingma et al. in their benchmark experiments."

    # 1. Full candidate provided
    monkeypatch.setattr(logic_engine, "call_llm", lambda *a, **k: full_sentence)
    result = logic_engine.extract_best_sentence("query", chunk)
    assert result == full_sentence

    # 2. Candidate truncated right after decimal or abbreviation
    truncated_candidate = "Good default settings are alpha = 0.001, beta1 = 0.9, as demonstrated by Kingma et al."
    monkeypatch.setattr(logic_engine, "call_llm", lambda *a, **k: truncated_candidate)
    result = logic_engine.extract_best_sentence("query", chunk)
    assert result == full_sentence, "Sentence must not truncate at '0.001' or 'et al.'"


def test_extract_best_sentence_mid_sentence_prefix(monkeypatch):
    """
    Verifies that candidate matching mid-sentence in chunk (lacking sentence-boundary start)
    is rejected rather than returning a sentence fragment.
    """
    chunk = "In preliminary experiments, residual learning helps train very deep networks effectively. Other remarks."
    # Candidate matches verbatim in chunk, but starts mid-sentence after comma
    mid_sentence_candidate = "residual learning helps train very deep networks effectively."
    monkeypatch.setattr(logic_engine, "call_llm", lambda *a, **k: mid_sentence_candidate)
    result = logic_engine.extract_best_sentence("query", chunk)
    assert result == "", "Mid-sentence fragment must not be accepted as a complete sentence"


def test_extract_best_sentence_length_truncated_provider_response(monkeypatch):
    """
    Simulates a real provider call that stops due to max_tokens (finish_reason='length'),
    leaving the candidate sentence incomplete. Verifies safe recovery to the true sentence end.
    """
    chunk = "We train the model on the ImageNet dataset with a batch size of 256 and learning rate of 0.1 for 90 epochs. Follow-up evaluation."
    # Truncated response from provider at max_tokens
    truncated_resp = "We train the model on the ImageNet dataset with a batch size of 256 and learning rate"

    def mock_chat(*args, **kwargs):
        on_trace = kwargs.get("on_trace")
        if on_trace:
            on_trace("provider_call", {
                "finish_reason": "length",
                "output": truncated_resp
            })
        return truncated_resp

    monkeypatch.setattr(providers, "chat", mock_chat)
    result = logic_engine.extract_best_sentence("query", chunk)
    assert result == "We train the model on the ImageNet dataset with a batch size of 256 and learning rate of 0.1 for 90 epochs."


def test_extract_best_sentence_invalid_fallback(monkeypatch):
    chunk = "Real chunk content discussing Dynamo storage engine architecture and caching layers."
    # Simulate LLM hallucination / repetition like Q010
    monkeypatch.setattr(logic_engine, "call_llm", lambda *a, **k: "To do this each node actively gossips the full routing table is not the correct answer.")
    result = logic_engine.extract_best_sentence("query", chunk)
    assert result == ""


def test_extract_best_sentence_no_useful_sentence(monkeypatch):
    chunk = "Completely unhelpful content."
    monkeypatch.setattr(logic_engine, "call_llm", lambda *a, **k: "NO_USEFUL_SENTENCE")
    result = logic_engine.extract_best_sentence("query", chunk)
    assert result == ""
