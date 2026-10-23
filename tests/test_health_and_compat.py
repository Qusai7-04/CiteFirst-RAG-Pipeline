from fastapi.testclient import TestClient
import api


def test_api_title_and_brand_identity():
    assert api.app.title == "CiteFirst — Research Assistant with Citations API"
    assert hasattr(api, "PIPELINE_AVAILABLE")
    assert hasattr(api, "crag_available")
    assert api.crag_available == api.PIPELINE_AVAILABLE


def test_health_endpoint_truthful_when_pipeline_available(monkeypatch):
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", True)
    monkeypatch.setattr(api, "crag_available", True)
    client = TestClient(api.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["pipeline_available"] is True
    assert data["crag_available"] is True
    assert data["ready"] is True
    assert "operational" in data["detail"].lower() or "ready" in data["detail"].lower()


def test_health_endpoint_degraded_when_pipeline_unavailable(monkeypatch):
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", False)
    monkeypatch.setattr(api, "crag_available", False)
    client = TestClient(api.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["pipeline_available"] is False
    assert data["crag_available"] is False
    assert data["ready"] is False
    assert "unavailable" in data["detail"].lower() or "stub" in data["detail"].lower()


def test_query_honest_unavailable_when_pipeline_down(monkeypatch):
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", False)
    client = TestClient(api.app)
    response = client.post("/api/query", json={"question": "What is turbulent flow?"})
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_stream_honest_unavailable_when_pipeline_down(monkeypatch):
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", False)
    client = TestClient(api.app)
    response = client.post("/api/query/stream", json={"question": "What is turbulent flow?"})
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_cors_default_disallows_cross_origin_requests():
    client = TestClient(api.app)
    response = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    assert api.ALLOWED_ORIGINS == []


def test_cors_configured_origin_allows_explicit_client():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    cors_env = "http://trusted.local, http://client.example.com"
    configured = [o.strip() for o in cors_env.split(",") if o.strip()]
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured,
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/test")
    def _test():
        return {"status": "ok"}

    client = TestClient(app)
    res_trusted = client.get("/api/test", headers={"Origin": "http://trusted.local"})
    assert res_trusted.headers.get("access-control-allow-origin") == "http://trusted.local"

    res_untrusted = client.get("/api/test", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" not in res_untrusted.headers

