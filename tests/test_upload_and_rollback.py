import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import api
import database
import evaluator


def test_evaluator_default_dataset():
    # Verify default dataset is reviewed_dataset.json
    assert evaluator.ROOT / "reviewed_dataset.json" == evaluator.ROOT / "reviewed_dataset.json"
    import inspect
    sig = inspect.signature(evaluator.run_ml_evaluation)
    default_dataset = sig.parameters["dataset"].default
    assert default_dataset == evaluator.ROOT / "reviewed_dataset.json"


def test_process_and_save_pdf_duplicate_rejected(monkeypatch):
    calls = []

    def mock_delete(filename):
        calls.append(filename)
        return 1

    monkeypatch.setattr(database, "delete_document", mock_delete)
    monkeypatch.setattr(database.collection, "get", lambda where: {"ids": ["existing_1"]})

    # When replace=False, existing file is skipped cleanly without error or delete
    database.process_and_save_pdf("dummy.pdf", replace=False)
    assert calls == []

    # When replace=True, duplicate is explicitly rejected with FileExistsError
    # (delete-first is strictly prohibited to guarantee non-destructive invariants)
    with pytest.raises(FileExistsError, match="already exists"):
        database.process_and_save_pdf("dummy.pdf", replace=True)
    assert calls == []  # delete_document must NOT have been called


def test_process_and_save_pdf_rollback_on_failure(monkeypatch):
    deleted = []

    def mock_delete(filename):
        deleted.append(filename)
        return 1

    monkeypatch.setattr(database, "delete_document", mock_delete)
    monkeypatch.setattr(database.collection, "get", lambda where: {"ids": []})

    class FakeDoc:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __iter__(self):
            return iter(["page one text here"])

    monkeypatch.setattr(database.fitz, "open", lambda path: FakeDoc())
    monkeypatch.setattr(database, "_page_text", lambda p: "Some test text for page one that is long enough")
    monkeypatch.setattr(database, "detect_domain", lambda t: "cs")
    monkeypatch.setattr(database, "_page_chunks", lambda pages: (
        [("chunk text", 0, 0)],
        [("parent chunk text", 0)]
    ))
    monkeypatch.setattr(database.parent_collection, "add", lambda **kwargs: None)

    # Injected failure on child collection add
    def failing_add(**kwargs):
        raise RuntimeError("Injected ChromaDB failure")

    monkeypatch.setattr(database.collection, "add", failing_add)
    monkeypatch.setattr(database, "_get_embedding_model", lambda: type("M", (), {"encode": lambda s, x, **kw: type("V", (), {"tolist": lambda self: [[0.1]*384]})()})())

    with pytest.raises(RuntimeError, match="Injected ChromaDB failure"):
        database.process_and_save_pdf("fail_paper.pdf", replace=False)

    assert "fail_paper.pdf" in deleted


def test_vault_upload_duplicate_rejected_with_409_preserves_existing_document(tmp_path, monkeypatch):
    """
    Step 1 Invariant: If an upload is attempted for an existing document, the existing
    PDF, its index, and its query results must remain untouched and fully available.
    """
    papers = tmp_path / "papers"
    papers.mkdir(parents=True)
    orig_file = papers / "existing_paper.pdf"
    orig_bytes = b"%PDF-1.4 original content that must be preserved"
    orig_file.write_bytes(orig_bytes)

    monkeypatch.setattr(api, "PAPERS_DIR", papers)
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", True)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)
    monkeypatch.setattr(api, "document_exists", lambda fn: True)

    deleted = []
    monkeypatch.setattr(api, "delete_document", lambda fn: deleted.append(fn))

    client = TestClient(api.app)

    # Attempt to upload a duplicate document with different bytes
    response = client.post(
        "/api/vault/upload",
        files={"file": ("existing_paper.pdf", b"%PDF-1.4 new corrupt replacement content", "application/pdf")}
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()

    # Invariant checks:
    # 1. Existing file bytes remain completely unchanged
    assert orig_file.read_bytes() == orig_bytes
    # 2. delete_document was never called
    assert deleted == []
    # 3. Existing file is still downloadable via document endpoint
    doc_res = client.get("/api/vault/document/existing_paper.pdf")
    assert doc_res.status_code == 200
    assert doc_res.content == orig_bytes


def test_vault_upload_duplicate_in_index_only_rejected_with_409(tmp_path, monkeypatch):
    papers = tmp_path / "papers"
    papers.mkdir(parents=True)

    monkeypatch.setattr(api, "PAPERS_DIR", papers)
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", True)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)
    monkeypatch.setattr(api, "document_exists", lambda fn: True)

    client = TestClient(api.app)
    response = client.post(
        "/api/vault/upload",
        files={"file": ("indexed_only.pdf", b"%PDF-1.4 content", "application/pdf")}
    )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()
    assert not (papers / "indexed_only.pdf").exists()


def test_vault_upload_rollback_on_indexing_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", True)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)
    monkeypatch.setattr(api, "document_exists", lambda fn: False)

    def failing_index(path, replace=False):
        raise RuntimeError("Indexing exploded")

    monkeypatch.setattr(api, "process_and_save_pdf", failing_index)
    deleted = []
    monkeypatch.setattr(api, "delete_document", lambda fn: deleted.append(fn))

    client = TestClient(api.app)
    response = client.post(
        "/api/vault/upload",
        files={"file": ("error_paper.pdf", b"%PDF-1.4 dummy", "application/pdf")}
    )
    assert response.status_code == 500
    assert "Indexing failed" in response.json()["detail"]
    # Verify file was cleaned up (not orphaned)
    assert not (tmp_path / "papers" / "error_paper.pdf").exists()
    assert "error_paper.pdf" in deleted


def test_successful_replacement_flow_via_delete_then_upload(tmp_path, monkeypatch):
    """
    Test replacement under final design: delete existing document, then upload new content.
    """
    papers = tmp_path / "papers"
    papers.mkdir(parents=True)
    target = papers / "replaceable.pdf"
    v1_bytes = b"%PDF-1.4 Version 1 content"
    target.write_bytes(v1_bytes)

    indexed_docs = {"replaceable.pdf": 3}

    monkeypatch.setattr(api, "PAPERS_DIR", papers)
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", True)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)
    monkeypatch.setattr(api, "document_exists", lambda fn: fn in indexed_docs)

    def mock_delete(fn):
        indexed_docs.pop(fn, None)
        return 3

    monkeypatch.setattr(api, "delete_document", mock_delete)

    def mock_process(path, replace=False):
        indexed_docs[Path(path).name] = 5

    monkeypatch.setattr(api, "process_and_save_pdf", mock_process)

    client = TestClient(api.app)

    # 1. Direct upload duplicate fails with 409
    dup_res = client.post(
        "/api/vault/upload",
        files={"file": ("replaceable.pdf", b"%PDF-1.4 Version 2", "application/pdf")}
    )
    assert dup_res.status_code == 409
    assert target.read_bytes() == v1_bytes

    # 2. Explicit delete succeeds
    del_res = client.delete("/api/vault/delete/replaceable.pdf")
    assert del_res.status_code == 200
    assert not target.exists()
    assert "replaceable.pdf" not in indexed_docs

    # 3. New upload succeeds
    v2_bytes = b"%PDF-1.4 Version 2 new verified content"
    up_res = client.post(
        "/api/vault/upload",
        files={"file": ("replaceable.pdf", v2_bytes, "application/pdf")}
    )
    assert up_res.status_code == 200
    assert target.exists()
    assert target.read_bytes() == v2_bytes
    assert indexed_docs.get("replaceable.pdf") == 5


def test_vault_delete_preserves_file_when_index_delete_fails(tmp_path, monkeypatch):
    papers = tmp_path / "papers"
    papers.mkdir(parents=True)
    pdf_file = papers / "safe_paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 content")

    monkeypatch.setattr(api, "PAPERS_DIR", papers)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)

    def failing_delete_doc(filename):
        raise RuntimeError("Index deletion database lock error")

    monkeypatch.setattr(api, "delete_document", failing_delete_doc)

    client = TestClient(api.app)
    response = client.delete("/api/vault/delete/safe_paper.pdf")
    assert response.status_code == 500
    assert "ChromaDB delete failed" in response.json()["detail"]
    # Physical file MUST be preserved!
    assert pdf_file.exists()


def test_vault_delete_nonexistent_document_returns_404(tmp_path, monkeypatch):
    papers = tmp_path / "papers"
    papers.mkdir(parents=True)

    monkeypatch.setattr(api, "PAPERS_DIR", papers)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)
    monkeypatch.setattr(api, "delete_document", lambda fn: 0)

    client = TestClient(api.app)
    response = client.delete("/api/vault/delete/nonexistent.pdf")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_vault_delete_fails_when_file_unlink_fails(tmp_path, monkeypatch):
    papers = tmp_path / "papers"
    papers.mkdir(parents=True)
    pdf_file = papers / "locked_paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 locked")

    monkeypatch.setattr(api, "PAPERS_DIR", papers)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)
    monkeypatch.setattr(api, "delete_document", lambda fn: 2)

    def failing_unlink(missing_ok=True):
        raise OSError("File locked by another process")

    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: failing_unlink() if self.name == "locked_paper.pdf" else None)

    client = TestClient(api.app)
    response = client.delete("/api/vault/delete/locked_paper.pdf")
    assert response.status_code == 500
    assert "Physical file deletion failed" in response.json()["detail"]
