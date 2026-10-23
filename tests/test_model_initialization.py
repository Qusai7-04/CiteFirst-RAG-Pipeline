import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedding_model_is_lazy_and_shared(monkeypatch):
    created = []

    def construct(name):
        time.sleep(0.02)
        created.append(name)
        return object()

    client = SimpleNamespace(get_or_create_collection=lambda **kwargs: object())
    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **kwargs: client))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=construct))
    db = _load("database.py", "database_lazy_test")
    assert created == []
    with ThreadPoolExecutor(max_workers=8) as executor:
        models = list(executor.map(lambda _: db._get_embedding_model(), range(8)))
    assert created == ['BAAI/bge-small-en-v1.5']
    assert all(model is models[0] for model in models)


def test_reranker_is_lazy_retryable_and_shared(monkeypatch):
    created = []

    def construct(name):
        created.append(name)
        if len(created) == 1:
            raise RuntimeError("model unavailable")
        time.sleep(0.02)
        return object()

    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda: None))
    monkeypatch.setitem(sys.modules, "database", SimpleNamespace(
        search_vault=lambda *args, **kwargs: None, DOMAIN_KEYWORDS={},
        check_vault_schema=lambda: "new"))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=construct))
    engine = _load("logic_engine.py", "logic_engine_lazy_test")
    assert created == []
    try:
        engine._get_reranker()
        assert False, "Expected model startup error"
    except RuntimeError as error:
        assert str(error) == "model unavailable"
    with ThreadPoolExecutor(max_workers=8) as executor:
        models = list(executor.map(lambda _: engine._get_reranker(), range(8)))
    assert created == ['BAAI/bge-reranker-base'] * 2
    assert all(model is models[0] for model in models)
