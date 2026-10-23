import time

from fastapi.testclient import TestClient

import api


def test_completed_eval_tasks_are_bounded_and_active_tasks_survive(monkeypatch):
    now = time.monotonic()
    monkeypatch.setattr(api, "_eval_tasks", {"active": {"status": "running", "result": None}})
    for n in range(110):
        api._eval_tasks[str(n)] = {"status": "completed", "result": n, "_finished_at": now + n / 1000}
    api._prune_eval_tasks()
    assert "active" in api._eval_tasks
    assert "0" not in api._eval_tasks
    assert "109" in api._eval_tasks
    assert len(api._eval_tasks) == 101
    assert api.get_evaluation_status("109") == {"status": "completed", "result": 109}


def test_eval_completion_and_expiration(monkeypatch):
    monkeypatch.setattr(api, "_eval_tasks", {})
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)
    monkeypatch.setattr(api, "run_ml_evaluation", lambda: {}, raising=False)
    client = TestClient(api.app)
    task_id = client.post("/api/evaluate").json()["task_id"]
    for _ in range(100):
        result = client.get(f"/api/evaluate/{task_id}")
        if result.json()["status"] == "completed":
            break
        time.sleep(0.01)
    assert result.status_code == 200
    assert result.json() == {"status": "completed", "result": {}}
    api._eval_tasks[task_id]["_finished_at"] -= api.EVAL_RESULT_TTL_SECONDS + 1
    assert client.get(f"/api/evaluate/{task_id}").status_code == 404


def test_eval_without_pipeline_returns_gap_summary_not_stub_scores(monkeypatch):
    monkeypatch.setattr(api, "_eval_tasks", {})
    monkeypatch.setattr(api, "CRAG_AVAILABLE", False)
    monkeypatch.setattr(api, "run_ml_evaluation",
                        lambda: {"baseline_status": "blocked_no_verified_labels",
                                 "counts": {"present": 18, "scored": 0}, "metrics": {}})
    client = TestClient(api.app)
    task_id = client.post("/api/evaluate").json()["task_id"]
    for _ in range(100):
        result = client.get(f"/api/evaluate/{task_id}").json()
        if result["status"] == "completed":
            break
        time.sleep(0.01)
    assert result["result"]["metrics"] == {}
    assert "accuracy" not in result["result"]
