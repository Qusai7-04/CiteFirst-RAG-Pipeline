"""Shared fixtures and helpers for the CiteFirst test suite."""
import os
import sys
from types import SimpleNamespace
import pytest

# Support offline/model-free test suites where production ML dependencies are not installed
for _mod, _stub in [
    ("dotenv", SimpleNamespace(load_dotenv=lambda *a, **k: None)),
    ("fitz", SimpleNamespace(open=lambda *a, **k: SimpleNamespace())),
    ("chromadb", SimpleNamespace(
        PersistentClient=lambda *a, **k: SimpleNamespace(
            get_or_create_collection=lambda **kw: SimpleNamespace(
                count=lambda: 0,
                query=lambda **kw: {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            )
        )
    )),
    ("sentence_transformers", SimpleNamespace(
        SentenceTransformer=lambda *a, **k: SimpleNamespace(
            encode=lambda texts, *a, **k: __import__("numpy").zeros((len(texts), 384)) if isinstance(texts, list) else __import__("numpy").zeros(384)
        ),
        CrossEncoder=lambda *a, **k: SimpleNamespace(
            predict=lambda pairs, *a, **k: 2.0 if (len(pairs) == 2 and isinstance(pairs[0], str) and isinstance(pairs[1], str)) else __import__("numpy").array([2.0] * len(pairs))
        )
    ))
]:
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = _stub


def _can_symlink(tmp: os.PathLike) -> bool:
    """Return True if the current process can create symlinks in tmp."""
    probe = os.path.join(tmp, ".symlink_probe")
    target = os.path.join(tmp, ".symlink_target")
    try:
        open(target, "w").close()
        os.symlink(target, probe)
        return True
    except OSError:
        return False
    finally:
        for p in (probe, target):
            try:
                os.unlink(p)
            except OSError:
                pass


@pytest.fixture(autouse=True)
def _skip_symlink_tests_on_windows(request, tmp_path):
    """Auto-skip tests that require symlink privileges on unprivileged Windows."""
    marker = request.node.get_closest_marker("needs_symlinks")
    if marker is not None and not _can_symlink(tmp_path):
        pytest.skip("Symlink creation requires elevated privileges on this Windows system")
