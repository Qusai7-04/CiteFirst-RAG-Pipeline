import importlib.util
import sys
from types import SimpleNamespace


def test_reranking_preserves_page_and_chunk_id(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda: None))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=lambda *args: object()))
    monkeypatch.setitem(sys.modules, "database", SimpleNamespace(
        search_vault=lambda *args, **kwargs: None, DOMAIN_KEYWORDS={},
        check_vault_schema=lambda: "new"))
    spec = importlib.util.spec_from_file_location("logic_engine_evidence_test", "logic_engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    monkeypatch.setattr(engine, "_get_reranker", lambda: SimpleNamespace(predict=lambda pair: 3.0))
    monkeypatch.setattr(engine, "_print_score_histogram", lambda scores: None)
    metas = [{"source": "paper.pdf", "page_index": 1,
              "chunk_id": "doc_p1_child_0", "document_id": "doc", "child_text": "short"}]
    chunks, sources, retained = engine.evaluate_chunks("question", ["parent text"], metas)
    assert chunks == ["parent text"]
    assert sources == ["paper.pdf"]
    assert retained == metas
    assert engine._evidence(chunks, sources, retained) == [{
        "id": "doc_p1_child_0", "document_id": "doc", "source": "paper.pdf",
        "page_index": 1, "page_number": 2, "text": "parent text"}]
    assert engine._evidence(["legacy"], ["old.pdf"], [{}])[0]["page_number"] is None
