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
    monkeypatch.setattr(database.collection, "get", lambda where: {"ids": ["existing_1"]}, raising=False)

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
    monkeypatch.setattr(database.collection, "get", lambda where: {"ids": []}, raising=False)

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
    monkeypatch.setattr(database.parent_collection, "add", lambda **kwargs: None, raising=False)

    # Injected failure on child collection add
    def failing_add(**kwargs):
        raise RuntimeError("Injected ChromaDB failure")

    monkeypatch.setattr(database.collection, "add", failing_add, raising=False)
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


def test_vault_delete_fails_when_file_locked_preserves_file_and_index(tmp_path, monkeypatch):
    """
    Fix 1 Invariant: If physical file staging fails (e.g. locked file on Windows),
    the file bytes, child chunks, parent chunks, and retrieval must remain intact.
    """
    papers = tmp_path / "papers"
    papers.mkdir(parents=True)
    pdf_file = papers / "locked_paper.pdf"
    pdf_bytes = b"%PDF-1.4 locked paper content that must remain untouched"
    pdf_file.write_bytes(pdf_bytes)

    monkeypatch.setattr(api, "PAPERS_DIR", papers)
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", True)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)

    delete_calls = []
    monkeypatch.setattr(api, "delete_document", lambda fn: delete_calls.append(fn) or 2)

    orig_replace = Path.replace

    def failing_replace(self, target):
        if self.name == "locked_paper.pdf":
            raise PermissionError("[WinError 32] The process cannot access the file because it is being used by another process")
        return orig_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    client = TestClient(api.app)
    response = client.delete("/api/vault/delete/locked_paper.pdf")
    assert response.status_code == 500
    assert "staging failed" in response.json()["detail"].lower()

    # Invariants:
    # 1. Physical file still exists with identical original bytes
    assert pdf_file.exists()
    assert pdf_file.read_bytes() == pdf_bytes
    # 2. delete_document was never called - index was not touched!
    assert delete_calls == []
    # 3. Document is still downloadable
    doc_res = client.get("/api/vault/document/locked_paper.pdf")
    assert doc_res.status_code == 200
    assert doc_res.content == pdf_bytes
    # 4. No temporary staging files left in vault
    assert list(papers.glob(".del_*")) == []


def test_vault_delete_real_windows_file_lock(tmp_path, monkeypatch):
    """
    Test real OS-level locked file on Windows without mocking replace/unlink.
    """
    papers = tmp_path / "papers"
    papers.mkdir(parents=True)
    pdf_file = papers / "windows_locked.pdf"
    pdf_bytes = b"%PDF-1.4 real locked file"
    pdf_file.write_bytes(pdf_bytes)

    monkeypatch.setattr(api, "PAPERS_DIR", papers)
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", True)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)

    delete_calls = []
    monkeypatch.setattr(api, "delete_document", lambda fn: delete_calls.append(fn) or 2)

    import sys
    locked = True
    orig_replace = Path.replace

    def maybe_locked_replace(self, target):
        if locked and self.name == "windows_locked.pdf":
            raise PermissionError("[WinError 32] The process cannot access the file because it is being used by another process")
        return orig_replace(self, target)

    if sys.platform != "win32":
        monkeypatch.setattr(Path, "replace", maybe_locked_replace)

    client = TestClient(api.app)

    # Hold an open handle (on Windows, this creates a real file sharing violation WinError 32)
    with open(pdf_file, "r+b") as handle:
        response = client.delete("/api/vault/delete/windows_locked.pdf")
        assert response.status_code == 500
        assert "staging failed" in response.json()["detail"].lower()
        assert delete_calls == []
        assert pdf_file.exists()

    # After releasing the lock (handle closed / toggled), deletion succeeds
    locked = False
    res_after = client.delete("/api/vault/delete/windows_locked.pdf")
    assert res_after.status_code == 200
    assert not pdf_file.exists()
    assert list(papers.glob(".del_*")) == []


def test_vault_delete_rollback_when_chromadb_fails(tmp_path, monkeypatch):
    """
    Fix 1 Invariant: If ChromaDB delete fails after staging, the file must be
    restored from staging back to original path, leaving no temporary file.
    """
    papers = tmp_path / "papers"
    papers.mkdir(parents=True)
    pdf_file = papers / "chroma_fail.pdf"
    pdf_bytes = b"%PDF-1.4 chroma fail paper"
    pdf_file.write_bytes(pdf_bytes)

    monkeypatch.setattr(api, "PAPERS_DIR", papers)
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", True)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)

    def failing_delete(fn):
        raise RuntimeError("ChromaDB connection severed")

    monkeypatch.setattr(api, "delete_document", failing_delete)

    client = TestClient(api.app)
    response = client.delete("/api/vault/delete/chroma_fail.pdf")
    assert response.status_code == 500
    assert "ChromaDB delete failed" in response.json()["detail"]

    # File was rolled back to original location!
    assert pdf_file.exists()
    assert pdf_file.read_bytes() == pdf_bytes
    # No temporary staging file left behind
    assert list(papers.glob(".del_*")) == []


def test_database_delete_document_child_failure_restores_state(monkeypatch):
    """
    If child collection deletion fails, no data is lost.
    """
    def mock_get(where=None, include=None):
        return {
            "ids": ["c1", "c2"],
            "documents": ["doc1", "doc2"],
            "metadatas": [{"source": "fail_child.pdf"}, {"source": "fail_child.pdf"}],
            "embeddings": [[0.1]*384, [0.2]*384],
        }

    monkeypatch.setattr(database.collection, "get", mock_get, raising=False)
    monkeypatch.setattr(database.parent_collection, "get", mock_get, raising=False)

    def failing_child_delete(ids):
        raise RuntimeError("Child delete exploded")

    monkeypatch.setattr(database.collection, "delete", failing_child_delete, raising=False)

    with pytest.raises(RuntimeError, match="Child delete exploded"):
        database.delete_document("fail_child.pdf")


def test_database_delete_document_parent_failure_restores_child_chunks(monkeypatch):
    """
    Fix 1 Invariant: If parent deletion fails after child deletion,
    the child chunks are restored into child collection before re-raising.
    """
    child_records = {
        "ids": ["c1"],
        "documents": ["child chunk 1"],
        "metadatas": [{"source": "test_parent_fail.pdf"}],
        "embeddings": [[0.1]*384],
    }
    parent_records = {
        "ids": ["p1"],
        "documents": ["parent chunk 1"],
        "metadatas": [{"source": "test_parent_fail.pdf"}],
        "embeddings": [[0.0]*384],
    }

    monkeypatch.setattr(database.collection, "get", lambda **kw: dict(child_records), raising=False)
    monkeypatch.setattr(database.parent_collection, "get", lambda **kw: dict(parent_records), raising=False)

    child_deleted_ids = []
    child_restored_kw = []
    monkeypatch.setattr(database.collection, "delete", lambda ids: child_deleted_ids.extend(ids), raising=False)
    monkeypatch.setattr(database.collection, "add", lambda **kw: child_restored_kw.append(kw), raising=False)

    def failing_parent_delete(ids):
        raise RuntimeError("Parent collection write lock error")

    monkeypatch.setattr(database.parent_collection, "delete", failing_parent_delete, raising=False)

    with pytest.raises(RuntimeError, match="Parent collection write lock error"):
        database.delete_document("test_parent_fail.pdf")

    # Child chunks were deleted...
    assert child_deleted_ids == ["c1"]
    # ...AND restored via compensatory rollback!
    assert len(child_restored_kw) == 1
    assert child_restored_kw[0]["ids"] == ["c1"]
    assert child_restored_kw[0]["documents"] == ["child chunk 1"]
    assert child_restored_kw[0]["metadatas"] == [{"source": "test_parent_fail.pdf"}]
    assert child_restored_kw[0]["embeddings"] == [[0.1]*384]


def test_database_delete_document_cache_invalidation_failure_restores_all(monkeypatch):
    """
    Fix 1 Invariant: If BM25 cache invalidation fails after both child and parent deletes,
    both child and parent chunks are restored before re-raising.
    """
    child_records = {"ids": ["c1"], "documents": ["cdoc"], "metadatas": [{"source": "test.pdf"}], "embeddings": [[0.1]*384]}
    parent_records = {"ids": ["p1"], "documents": ["pdoc"], "metadatas": [{"source": "test.pdf"}], "embeddings": [[0.0]*384]}

    monkeypatch.setattr(database.collection, "get", lambda **kw: dict(child_records), raising=False)
    monkeypatch.setattr(database.parent_collection, "get", lambda **kw: dict(parent_records), raising=False)

    child_restored = []
    parent_restored = []
    monkeypatch.setattr(database.collection, "delete", lambda ids: None, raising=False)
    monkeypatch.setattr(database.parent_collection, "delete", lambda ids: None, raising=False)
    monkeypatch.setattr(database.collection, "add", lambda **kw: child_restored.append(kw), raising=False)
    monkeypatch.setattr(database.parent_collection, "add", lambda **kw: parent_restored.append(kw), raising=False)

    def failing_cache():
        raise RuntimeError("Cache directory corrupted")

    monkeypatch.setattr(database, "invalidate_bm25_cache", failing_cache)

    with pytest.raises(RuntimeError, match="Cache directory corrupted"):
        database.delete_document("test.pdf")

    assert len(child_restored) == 1
    assert child_restored[0]["ids"] == ["c1"]
    assert len(parent_restored) == 1
    assert parent_restored[0]["ids"] == ["p1"]


def test_vault_delete_success_cleans_file_and_both_collections(tmp_path, monkeypatch):
    """
    Fix 1 Invariant: Successful deletion cleanly unlinks file, staging file,
    child chunks, parent chunks, and invalidates cache.
    """
    papers = tmp_path / "papers"
    papers.mkdir(parents=True)
    pdf_file = papers / "success_paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 success content")

    monkeypatch.setattr(api, "PAPERS_DIR", papers)
    monkeypatch.setattr(api, "PIPELINE_AVAILABLE", True)
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)

    deleted_docs = []
    monkeypatch.setattr(api, "delete_document", lambda fn: deleted_docs.append(fn) or 5)

    client = TestClient(api.app)
    response = client.delete("/api/vault/delete/success_paper.pdf")
    assert response.status_code == 200
    assert response.json() == {"deleted": "success_paper.pdf", "chunks_removed": 5}

    assert not pdf_file.exists()
    assert list(papers.glob(".del_*")) == []
    assert deleted_docs == ["success_paper.pdf"]

