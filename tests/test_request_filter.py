import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

from fastapi.testclient import TestClient

import api


def test_concurrent_filters_propagate_through_pipeline(monkeypatch):
    barrier = Barrier(2)
    observed = []

    def search(query, *, file_filter=None, **kwargs):
        observed.append((query, file_filter))
        barrier.wait(timeout=5)
        return {"documents": [["evidence"]], "metadatas": [[{"source": file_filter}]]}

    fake_db = SimpleNamespace(search_vault=search, DOMAIN_KEYWORDS={},
                              check_vault_schema=lambda: "new")
    monkeypatch.setitem(sys.modules, "database", fake_db)
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(CrossEncoder=lambda *args: object()))
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")
    spec = importlib.util.spec_from_file_location("logic_engine_filter_test", "logic_engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    monkeypatch.setattr(engine, "detect_query_domain", lambda query: None)
    monkeypatch.setattr(engine, "evaluate_chunks", lambda query, chunks, metas, file_filter=None, on_stage=None:
                        (chunks, [metas[0]["source"]], metas))
    monkeypatch.setattr(engine, "_combined_relevance_check", lambda *args: True)
    monkeypatch.setattr(engine, "generate_final_answer", lambda *args: "answer")
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)
    monkeypatch.setattr(api, "run_logic_engine", engine.run_logic_engine, raising=False)

    def request(source):
        return TestClient(api.app).post("/api/query", json={"question": source, "file_filter": source})

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(request, ("first.pdf", "second.pdf")))
    assert all(response.status_code == 200 for response in responses)
    assert [response.json()["sources"] for response in responses] == [["first.pdf"], ["second.pdf"]]
    assert set(observed) == {("first.pdf", "first.pdf"), ("second.pdf", "second.pdf")}


def test_stream_passes_filter_to_pipeline(monkeypatch):
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)
    monkeypatch.setattr(api, "run_logic_engine", lambda question, file_filter=None, on_stage=None:
                        {"status": "success", "answer": file_filter, "facts": [], "sources": []}, raising=False)
    response = TestClient(api.app).post("/api/query/stream", json={"question": "q", "file_filter": "paper.pdf"})
    assert response.status_code == 200
    assert '"answer": "paper.pdf"' in response.text
