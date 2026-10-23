import importlib.util
import logging
import sys
from types import SimpleNamespace


def test_legacy_schema_logs_warning_without_banner_or_delete_instruction(monkeypatch, caplog, capsys):
    collection = SimpleNamespace(get=lambda **kwargs: {"metadatas": [{"source": "paper.pdf"}]})
    client = SimpleNamespace(get_or_create_collection=lambda **kwargs: collection)
    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **kwargs: client))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *args: object()))
    spec = importlib.util.spec_from_file_location("database_logging_test", "database.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with caplog.at_level(logging.WARNING):
        assert module.check_vault_schema() == "legacy"
    assert "Legacy vault schema detected" in caplog.text
    assert "rm -rf" not in caplog.text
    assert capsys.readouterr().out == ""
