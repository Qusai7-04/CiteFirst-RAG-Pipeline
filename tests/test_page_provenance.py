import importlib.util
import sys
from types import SimpleNamespace


def test_page_chunks_and_ingestion_keep_page_provenance(tmp_path, monkeypatch):
    class Collection:
        def __init__(self):
            self.data = {}

        def get(self, where=None, ids=None, **kwargs):
            if ids:
                return {"ids": ids, "documents": [self.data[ids[0]][0]],
                        "metadatas": [self.data[ids[0]][1]]}
            return {"ids": []}

        def add(self, ids, documents, metadatas, **kwargs):
            self.data.update({key: (document, metadata)
                              for key, document, metadata in zip(ids, documents, metadatas)})

        def query(self, **kwargs):
            return {"documents": [[row[0] for row in self.data.values()]],
                    "metadatas": [[row[1] for row in self.data.values()]],
                    "distances": [[0.1] * len(self.data)]}

    children, parents = Collection(), Collection()
    client = SimpleNamespace(get_or_create_collection=lambda name: children if name == "ieee_papers" else parents)

    class Page:
        def __init__(self, text):
            self.text = text

        def get_text(self, **kwargs):
            return self.text

        def find_tables(self):
            return SimpleNamespace(tables=[])

    class Document:
        def __enter__(self):
            return iter([Page("First page evidence."), Page("Second page evidence.")])

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=lambda path: Document()))
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **kwargs: client))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=lambda *args: SimpleNamespace(
            encode=lambda texts, **kwargs: SimpleNamespace(tolist=lambda: [[0.0]] * len(texts)))))
    spec = importlib.util.spec_from_file_location("database_page_test", "database.py")
    db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(db)
    db.BM25_CACHE_PATH = tmp_path / "cache.json"
    db.process_and_save_pdf(str(tmp_path / "paper.pdf"))

    assert len(children.data) == len(parents.data) == 2
    for child_id, (child_text, child_meta) in children.data.items():
        parent_text, expanded = db._expand_to_parent(child_text, child_meta)
        assert child_meta["page_index"] == expanded["page_index"]
        assert expanded["chunk_id"] == child_id
        assert expanded["document_id"] == child_meta["document_id"]
        assert parent_text == parents.data[child_meta["parent_id"]][0]
    assert {meta["page_index"] for _, meta in children.data.values()} == {0, 1}
    child, parent = db._page_chunks(["First page.", "", "Third page."])
    assert [entry[1] for entry in child] == [0, 2]
    assert [entry[1] for entry in parent] == [0, 2]
    # Identical text on two pages must keep two distinct provenance records.
    children.data = {
        "a": ("Repeated text", {"source": "paper.pdf", "page_index": 0, "chunk_id": "a"}),
        "b": ("Repeated text", {"source": "paper.pdf", "page_index": 1, "chunk_id": "b"}),
    }
    db.BM25_AVAILABLE = False
    result = db.search_vault("repeated", n_results=2)
    assert [meta["page_index"] for meta in result["metadatas"][0]] == [0, 1]
