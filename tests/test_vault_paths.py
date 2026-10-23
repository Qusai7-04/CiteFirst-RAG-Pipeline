from urllib.parse import quote

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(api, "CRAG_AVAILABLE", False)
    return TestClient(api.app), tmp_path


def test_upload_list_delete_normal_pdf(vault):
    client, root = vault
    uploaded = client.post("/api/vault/upload", files={"file": ("paper.pdf", b"pdf-body", "application/pdf")})
    assert uploaded.status_code == 200
    assert uploaded.json()["filename"] == "paper.pdf"
    assert (root / "papers" / "paper.pdf").read_bytes() == b"pdf-body"
    assert [d["filename"] for d in client.get("/api/vault/list").json()["documents"]] == ["paper.pdf"]
    downloaded = client.get("/api/vault/document/paper.pdf")
    assert downloaded.status_code == 200
    assert downloaded.content == b"pdf-body"

    deleted = client.delete("/api/vault/delete/paper.pdf")
    assert deleted.status_code == 200
    assert not (root / "papers" / "paper.pdf").exists()


@pytest.mark.parametrize("filename", ["../../etc/passwd", r"..\..\outside.pdf", "/tmp/outside.pdf", r"C:\temp\outside.pdf"])
def test_reject_untrusted_names_before_writing_or_deleting(vault, filename):
    client, root = vault
    outside = root / "outside.pdf"
    outside.write_bytes(b"keep")

    uploaded = client.post("/api/vault/upload", files={"file": (filename, b"bad", "application/pdf")})
    assert uploaded.status_code == 400
    assert outside.read_bytes() == b"keep"

    # A slash in a path parameter may be rejected by routing before the handler.
    response = client.delete("/api/vault/delete/" + quote(filename, safe=""))
    expected_statuses = (400, 404) if "/" in filename else (400,)
    assert response.status_code in expected_statuses
    with pytest.raises(HTTPException) as error:
        api.vault_delete(filename)
    assert error.value.status_code == 400
    assert outside.read_bytes() == b"keep"


@pytest.mark.needs_symlinks
def test_symlinks_cannot_escape_paper_storage(vault):
    client, root = vault
    papers = root / "papers"
    papers.mkdir()
    outside = root / "outside.pdf"
    outside.write_bytes(b"keep")
    (papers / "linked.pdf").symlink_to(outside)

    assert client.post("/api/vault/upload", files={"file": ("linked.pdf", b"bad")}).status_code == 400
    assert client.delete("/api/vault/delete/linked.pdf").status_code == 400
    assert client.get("/api/vault/document/linked.pdf").status_code == 400
    assert outside.read_bytes() == b"keep"
    assert client.get("/api/vault/list").json()["documents"] == []

    (papers / "linked.pdf").unlink()
    papers.rmdir()
    papers.symlink_to(root, target_is_directory=True)
    assert client.post("/api/vault/upload", files={"file": ("other.pdf", b"bad")}).status_code == 400
    assert not (root / "other.pdf").exists()


def test_invalid_delete_never_touches_index(vault, monkeypatch):
    client, _ = vault
    monkeypatch.setattr(api, "CRAG_AVAILABLE", True)
    monkeypatch.setattr(api, "delete_document", lambda _: pytest.fail("index deletion must not run"), raising=False)
    assert client.delete(r"/api/vault/delete/..%5C..%5Coutside.pdf").status_code == 400


def test_delete_without_directory_fd_support(vault, monkeypatch):
    client, root = vault
    monkeypatch.setattr(api.os, "supports_dir_fd", set())
    (root / "papers").mkdir()
    paper = root / "papers" / "paper.pdf"
    paper.write_bytes(b"pdf")
    assert client.delete("/api/vault/delete/paper.pdf").status_code == 200
    assert not paper.exists()
