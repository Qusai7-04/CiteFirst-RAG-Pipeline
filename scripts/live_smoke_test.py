"""
Live Real-Dependency Smoke Test for CiteFirst RAG Pipeline.
Exercises:
1. Fresh target vault and papers directory.
2. Synthetic 2-page PDF generation (distinct page text).
3. Empty vault query honesty (no fake success / stubs).
4. PDF upload and ChromaDB parent/child indexing verification.
5. Duplicate upload rejection (HTTP 409 Conflict).
6. Missing-key query behavior (honest error/failure, no fake success).
7. Live query matching (Page 1 text, 1-based page citation, latency measurement).
8. file_filter enforcement across retrieval.
9. Non-matching query abstention.
10. Server-Sent Events (SSE) streaming stages and terminal event.
11. Path traversal / invalid filename rejection (HTTP 400).
12. Settings read / write / hot-reload / restoration.
13. CLI logic_engine execution.
14. Document deletion and post-delete verification.
15. Clean isolation and teardown.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Ensure ROOT is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz
from fastapi.testclient import TestClient


def create_synthetic_pdf(filepath: Path) -> bytes:
    """Create a minimal 2-page PDF with distinct, factual text on each page."""
    doc = fitz.open()
    # Page 0 (Physical page 1)
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Quantum Teleportation Fundamentals\n\n"
        "Quantum teleportation is a technique for transferring quantum information "
        "from a sender to a receiver at a different location. It relies on quantum "
        "entanglement between pairs of particles and classical communication channels. "
        "The protocol was first proposed by Bennett and colleagues in 1993."
    )
    # Page 1 (Physical page 2)
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Hydrodynamics and Turbulent Flow\n\n"
        "Turbulent flow is a fluid motion characterized by chaotic changes in pressure "
        "and flow velocity. It contrasts sharply with laminar flow, in which a fluid flows "
        "in parallel layers with no disruption between them. The Reynolds number is the "
        "primary dimensionless quantity used to predict transitions between laminar and turbulent flow."
    )
    doc.save(str(filepath))
    doc.close()
    return filepath.read_bytes()


def run_smoke_test():
    print("=" * 70)
    print("STARTING CITEFIRST LIVE REAL-DEPENDENCY SMOKE TEST")
    print("=" * 70)

    # 1. Isolated temporary vault & papers folder
    temp_dir = tempfile.mkdtemp(prefix="citefirst_smoke_")
    smoke_vault = Path(temp_dir) / "smoke_vault"
    smoke_papers = Path(temp_dir) / "smoke_papers"
    smoke_vault.mkdir()
    smoke_papers.mkdir()

    os.environ["CITEFIRST_VECTOR_DIR"] = str(smoke_vault.resolve())

    try:
        # Import pipeline modules with isolated environment
        import api
        import database
        import logic_engine

        # Point api PAPERS_DIR to isolated smoke_papers
        api.PAPERS_DIR = smoke_papers

        client = TestClient(api.app)

        print(f"[*] Isolated vault: {smoke_vault}")
        print(f"[*] Isolated papers: {smoke_papers}")

        # 2. Verify empty vault query
        print("\n--- Test 1: Empty Vault Query ---")
        t0 = time.perf_counter()
        res = client.post("/api/query", json={"question": "What is quantum teleportation?"})
        dur = time.perf_counter() - t0
        print(f"Empty vault query returned status={res.status_code} in {dur:.3f}s")
        assert res.status_code == 200
        empty_data = res.json()
        assert empty_data["status"] == "failed"
        assert len(empty_data["facts"]) == 0
        assert "no information detected" in empty_data["answer"].lower()
        print("  -> PASS: Honest empty vault response (no fake success, no stubs)")

        # 3. Create synthetic 2-page PDF
        pdf_path = smoke_papers / "synthetic_research.pdf"
        pdf_bytes = create_synthetic_pdf(pdf_path)
        print(f"\n[*] Created synthetic 2-page PDF: {len(pdf_bytes)} bytes")

        # 4. Upload & index PDF via API
        print("\n--- Test 2: Vault Upload & Real Ingestion ---")
        # Remove from smoke_papers before upload so upload places it
        pdf_path.unlink()
        t0 = time.perf_counter()
        up_res = client.post(
            "/api/vault/upload",
            files={"file": ("synthetic_research.pdf", pdf_bytes, "application/pdf")}
        )
        ingest_dur = time.perf_counter() - t0
        print(f"Upload returned status={up_res.status_code} in {ingest_dur:.3f}s: {up_res.json()}")
        assert up_res.status_code == 200
        assert up_res.json()["indexed"] is True

        # Verify ChromaDB chunk counts and metadata
        child_chunks = database.collection.get(where={"source": "synthetic_research.pdf"})
        parent_chunks = database.parent_collection.get(where={"source": "synthetic_research.pdf"})
        num_children = len(child_chunks.get("ids", []))
        num_parents = len(parent_chunks.get("ids", []))
        print(f"ChromaDB indexed chunks: {num_children} child chunks, {num_parents} parent chunks")
        assert num_children >= 2, f"Expected at least 2 child chunks, got {num_children}"
        assert num_parents >= 2, f"Expected at least 2 parent chunks, got {num_parents}"

        # Verify page_index metadata is 0-based
        pages_found = {m["page_index"] for m in child_chunks["metadatas"]}
        print(f"Child chunk page indices: {pages_found}")
        assert 0 in pages_found and 1 in pages_found, "Must index both page 0 and page 1"
        print("  -> PASS: Accurate parent/child chunks and 0-based page_index metadata")

        # 5. Duplicate Upload Rejection (Step 1 invariant)
        print("\n--- Test 3: Duplicate Upload Rejection (HTTP 409) ---")
        dup_res = client.post(
            "/api/vault/upload",
            files={"file": ("synthetic_research.pdf", b"%PDF-1.4 modified content", "application/pdf")}
        )
        print(f"Duplicate upload returned status={dup_res.status_code}: {dup_res.json()}")
        assert dup_res.status_code == 409
        assert "already exists" in dup_res.json()["detail"].lower()
        # Verify original bytes on disk are preserved
        assert (smoke_papers / "synthetic_research.pdf").read_bytes() == pdf_bytes
        # Verify chunk counts unchanged
        child_recheck = database.collection.get(where={"source": "synthetic_research.pdf"})
        assert len(child_recheck["ids"]) == num_children
        print("  -> PASS: Duplicate upload safely rejected with 409, original bytes & index intact")

        # 6. Missing Key Handling
        print("\n--- Test 4: Missing Key Handling ---")
        saved_key = os.environ.get("OPENROUTER_API_KEY")
        try:
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            no_key_res = client.post("/api/query", json={"question": "What is quantum teleportation?"})
            print(f"Missing key query returned status={no_key_res.status_code}")
            assert no_key_res.status_code == 500
            assert "openrouter_api_key" in no_key_res.json()["detail"].lower()
            print("  -> PASS: Missing key returns honest 500 with clear message (no fake [STUB])")
        finally:
            if saved_key:
                os.environ["OPENROUTER_API_KEY"] = saved_key

        # 7. Live Matching Query & Citations
        print("\n--- Test 5: Live Matching Retrieval & Citations ---")
        t0 = time.perf_counter()
        match_res = client.post("/api/query", json={"question": "What is quantum teleportation?"})
        match_dur = time.perf_counter() - t0
        print(f"Matching query completed in {match_dur:.3f}s (status={match_res.status_code})")
        assert match_res.status_code == 200
        match_data = match_res.json()
        print(f"Query status: {match_data['status']}")
        print(f"Facts count: {len(match_data['facts'])}")
        print(f"Sources: {match_data['sources']}")
        assert "synthetic_research.pdf" in match_data["sources"]
        assert len(match_data["evidence"]) > 0

        # Verify 1-based page citation
        for ev in match_data["evidence"]:
            print(f"Evidence snippet from source={ev.get('source')}, page_index={ev.get('page_index')}, page_number={ev.get('page_number')}")
            assert ev.get("page_number") == ev.get("page_index") + 1
            if ev.get("page_index") == 0:
                assert ev.get("page_number") == 1
        print("  -> PASS: One-based page_number in citations (page_index=0 -> page_number=1)")

        if saved_key and not saved_key.startswith("PASTE_YOUR_"):
            print(f"Generated cited answer: {match_data['answer'][:120]}...")
            assert len(match_data["answer"]) > 20
            print("  -> PASS: Genuinely live OpenRouter generation successful")
        else:
            print("  -> NOTE: OPENROUTER_API_KEY is placeholder/absent; generation unverified (expected without live key)")

        # 8. file_filter Enforcement
        print("\n--- Test 6: file_filter Enforcement ---")
        # Match with correct filter
        ff_res = client.post("/api/query", json={
            "question": "What is quantum teleportation?",
            "file_filter": "synthetic_research.pdf"
        })
        assert ff_res.status_code == 200
        assert "synthetic_research.pdf" in ff_res.json()["sources"]

        # Non-match with mismatched filter
        ff_mismatch = client.post("/api/query", json={
            "question": "What is quantum teleportation?",
            "file_filter": "unrelated_other.pdf"
        })
        assert ff_mismatch.status_code == 200
        assert ff_mismatch.json()["status"] == "failed"
        assert len(ff_mismatch.json()["facts"]) == 0
        print("  -> PASS: file_filter correctly enforces document isolation")

        # 9. Non-matching Query Abstention
        print("\n--- Test 7: Non-matching Query Abstention ---")
        unrelated_res = client.post("/api/query", json={
            "question": "What architectural styles characterized 12th century cathedrals in Normandy?"
        })
        assert unrelated_res.status_code == 200
        unrelated_data = unrelated_res.json()
        print(f"Non-matching status: {unrelated_data['status']}")
        assert unrelated_data["status"] == "failed"
        assert len(unrelated_data["facts"]) == 0
        print("  -> PASS: Pipeline honestly abstains on out-of-vault query")

        # 10. SSE Streaming Events
        print("\n--- Test 8: SSE Streaming Verification ---")
        stream_res = client.post("/api/query/stream", json={"question": "What is turbulent flow?"})
        assert stream_res.status_code == 200
        events = [json.loads(line[6:]) for line in stream_res.text.splitlines() if line.startswith("data: ")]
        event_steps = [e["step"] for e in events]
        print(f"SSE event steps emitted: {event_steps}")
        assert "fetching" in event_steps
        assert "reranking" in event_steps
        assert event_steps[-1] == "done"
        assert "result" in events[-1]
        print("  -> PASS: SSE stream emits real progress stages and terminal done event")

        # 11. Path Traversal Rejection
        print("\n--- Test 9: Path Traversal Rejection ---")
        bad_name_res = client.post(
            "/api/vault/upload",
            files={"file": ("../../bad.pdf", b"bad", "application/pdf")}
        )
        assert bad_name_res.status_code == 400
        bad_del_res = client.delete("/api/vault/delete/..%5C..%5Cbad.pdf")
        assert bad_del_res.status_code == 400
        print("  -> PASS: Path traversal rejected with HTTP 400")

        # 12. Settings Read/Write & Restore
        print("\n--- Test 10: Settings Management ---")
        orig_settings = client.get("/api/settings").json()
        orig_fetch_k = orig_settings["FETCH_K"]
        mod_settings = dict(orig_settings)
        mod_settings["FETCH_K"] = 28
        put_res = client.put("/api/settings", json=mod_settings)
        assert put_res.status_code == 200
        assert client.get("/api/settings").json()["FETCH_K"] == 28
        # Restore
        mod_settings["FETCH_K"] = orig_fetch_k
        client.put("/api/settings", json=mod_settings)
        assert client.get("/api/settings").json()["FETCH_K"] == orig_fetch_k
        print("  -> PASS: Settings updated, validated, and cleanly restored")

        # 13. CLI logic_engine query
        print("\n--- Test 11: CLI Logic Engine Execution ---")
        import subprocess
        cli_proc = subprocess.run(
            [sys.executable, "logic_engine.py", "What is turbulent flow?"],
            cwd=str(ROOT),
            env=os.environ,
            capture_output=True,
            text=True
        )
        print(f"CLI exit code: {cli_proc.returncode}")
        assert cli_proc.returncode == 0
        assert "CITEFIRST PIPELINE" in cli_proc.stdout
        assert "turbulent flow" in cli_proc.stdout.lower()
        print("  -> PASS: Standalone CLI query succeeds with honest output")

        # 14. Document Delete & Consistency
        print("\n--- Test 12: Vault Delete & Post-Delete Verification ---")
        del_res = client.delete("/api/vault/delete/synthetic_research.pdf")
        print(f"Delete returned status={del_res.status_code}: {del_res.json()}")
        assert del_res.status_code == 200
        assert not (smoke_papers / "synthetic_research.pdf").exists()
        # Verify ChromaDB is empty for this source
        post_child = database.collection.get(where={"source": "synthetic_research.pdf"})
        post_parent = database.parent_collection.get(where={"source": "synthetic_research.pdf"})
        assert len(post_child.get("ids", [])) == 0
        assert len(post_parent.get("ids", [])) == 0
        # Post-delete query returns failed
        post_query = client.post("/api/query", json={"question": "What is quantum teleportation?"})
        assert post_query.json()["status"] == "failed"
        # Deleting already-deleted document returns 404
        second_del = client.delete("/api/vault/delete/synthetic_research.pdf")
        assert second_del.status_code == 404
        print("  -> PASS: Physical file and vector chunks cleanly deleted; 404 on duplicate delete")

        print("\n" + "=" * 70)
        print("ALL 12 REAL-DEPENDENCY SMOKE TESTS PASSED CLEANLY")
        print("=" * 70)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"[*] Cleaned up temporary test directory: {temp_dir}")


if __name__ == "__main__":
    run_smoke_test()
