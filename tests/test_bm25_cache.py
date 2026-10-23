import importlib.util
import json
import pickle
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def db(tmp_path, monkeypatch):
    class Collection:
        calls = 0

        def get(self, **kwargs):
            self.calls += 1
            return {"documents": ["Quantum state", "State vector"],
                    "metadatas": [{"source": "one.pdf"}, {"source": "two.pdf"}]}

    collection = Collection()
    client = SimpleNamespace(get_or_create_collection=lambda **kwargs: collection)
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **kwargs: client))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *args: object()))
    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace())

    spec = importlib.util.spec_from_file_location("database_stage2a_test", "database.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BM25_CACHE_PATH = tmp_path / "vector_vault" / "bm25_corpus.json"
    return module, collection


def test_bm25_rebuild_and_cached_load(db):
    module, collection = db
    bm25, docs, metas = module._load_or_build_bm25()
    assert collection.calls == 1
    assert docs == ["Quantum state", "State vector"]
    assert metas[0]["source"] == "one.pdf"
    assert len(bm25.get_scores(["quantum"])) == 2
    cached = json.loads(module.BM25_CACHE_PATH.read_text())
    assert cached["version"] == 1
    assert cached["tokens"] == [["quantum", "state"], ["state", "vector"]]

    cached_bm25, cached_docs, cached_metas = module._load_or_build_bm25()
    assert collection.calls == 1
    assert cached_docs == docs and cached_metas == metas
    assert len(cached_bm25.get_scores(["quantum"])) == 2
    module.invalidate_bm25_cache()
    assert not module.BM25_CACHE_PATH.exists()
    module._load_or_build_bm25()
    assert collection.calls == 2


@pytest.mark.parametrize("content", ['{bad json', '{"version": 999}',
                                       '{"version": 1, "documents": ["x"], "tokens": [[1]], "metadatas": [{}]}'])
def test_bad_cache_rebuilds_from_collection(db, content):
    module, collection = db
    module.BM25_CACHE_PATH.parent.mkdir()
    module.BM25_CACHE_PATH.write_text(content)
    _, docs, _ = module._load_or_build_bm25()
    assert docs == ["Quantum state", "State vector"]
    assert collection.calls == 1
    assert json.loads(module.BM25_CACHE_PATH.read_text())["version"] == 1


def test_legacy_pickle_is_never_deserialized_or_removed(db, tmp_path, monkeypatch):
    module, collection = db
    marker = tmp_path / "executed"

    class Malicious:
        def __reduce__(self):
            return (eval, (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",))

    legacy = module.BM25_CACHE_PATH.with_suffix(".pkl")
    legacy.parent.mkdir()
    legacy.write_bytes(pickle.dumps(Malicious()))
    monkeypatch.setattr(pickle, "load", lambda *args: pytest.fail("unsafe pickle load"))

    _, docs, _ = module._load_or_build_bm25()
    assert docs == ["Quantum state", "State vector"]
    assert collection.calls == 1
    assert module.BM25_CACHE_PATH.exists()
    assert legacy.exists()
    assert not marker.exists()
    module.invalidate_bm25_cache()
    assert legacy.exists()


@pytest.mark.needs_symlinks
def test_cache_symlink_is_replaced_without_reading_target(db, tmp_path):
    module, collection = db
    target = tmp_path / "outside.json"
    target.write_text('{"private": "leave alone"}')
    module.BM25_CACHE_PATH.parent.mkdir()
    module.BM25_CACHE_PATH.symlink_to(target)
    _, docs, _ = module._load_or_build_bm25()
    assert docs == ["Quantum state", "State vector"]
    assert collection.calls == 1
    assert not module.BM25_CACHE_PATH.is_symlink()
    assert target.read_text() == '{"private": "leave alone"}'
