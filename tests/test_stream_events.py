import json

from fastapi.testclient import TestClient

import api


def test_stream_reports_actual_stage_callbacks_and_result(monkeypatch):
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)

    def pipeline(question, file_filter=None, on_stage=None):
        on_stage("fetching", "Searching")
        on_stage("reranking", "Reranking")
        on_stage("grading", "Grading")
        return {"status": "success", "answer": "ok", "facts": [], "sources": [],
                "evidence": [], "query_used": question}

    monkeypatch.setattr(api, "run_logic_engine", pipeline, raising=False)
    response = TestClient(api.app).post("/api/query/stream", json={"question": "test"})
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert [event["step"] for event in events] == ["fetching", "reranking", "grading", "done"]
    assert events[-1]["result"]["status"] == "success"
    assert all("progress" not in event for event in events)
