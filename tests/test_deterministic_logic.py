import importlib.util
import sys
from types import SimpleNamespace


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chunking_and_rrf_are_deterministic(monkeypatch):
    client = SimpleNamespace(get_or_create_collection=lambda **kwargs: object())
    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **kwargs: client))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *args: object()))
    db = _load("database.py", "database_deterministic_test")
    assert db.split_into_sentence_chunks("Alpha beta. Gamma delta. Epsilon zeta.",
                                         max_words=3, overlap_words=0) == [
        "Alpha beta.", "Gamma delta.", "Epsilon zeta."]
    assert db._reciprocal_rank_fusion([["a", "b"], ["b", "c"]]) == ["b", "a", "c"]


def test_threshold_routing_keeps_only_correct_passages(monkeypatch):
    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda: None))
    monkeypatch.setitem(sys.modules, "database", SimpleNamespace(
        search_vault=lambda *args, **kwargs: None, DOMAIN_KEYWORDS={},
        check_vault_schema=lambda: "new"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=lambda *args: object()))
    engine = _load("logic_engine.py", "logic_engine_threshold_test")
    monkeypatch.setattr(engine, "PHASE_2_ENABLED", False)
    monkeypatch.setattr(engine, "SOFTMAX_MASS_CUTOFF", 1.0)
    monkeypatch.setattr(engine, "_get_reranker", lambda: SimpleNamespace(
        predict=lambda pair: {"good": 2.0, "maybe": -1.0, "bad": -3.0}[pair[1]]))
    metas = [{"source": f"{name}.pdf", "page_index": index, "chunk_id": name}
             for index, name in enumerate(("good", "maybe", "bad"))]
    chunks, sources, retained = engine.evaluate_chunks("query", ["good", "maybe", "bad"], metas)
    assert chunks == ["good"]
    assert sources == ["good.pdf"]
    assert retained == [metas[0]]
