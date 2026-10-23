import importlib.util
import sys
from types import SimpleNamespace
import pytest


def test_openrouter_key_is_required_only_at_call_time(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=" answer "))]
            )

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            calls.append({"api_key": api_key, "base_url": base_url})
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda: None))
    fake_openai_mod = SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIError=type("APIError", (Exception,), {}),
        RateLimitError=type("RateLimitError", (Exception,), {}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=lambda *args: SimpleNamespace(encode=lambda *a, **k: object()),
        CrossEncoder=lambda *args: object()
    ))
    monkeypatch.setitem(sys.modules, "database", SimpleNamespace(
        search_vault=lambda *args, **kwargs: None, DOMAIN_KEYWORDS={},
        check_vault_schema=lambda: "new"))

    spec = importlib.util.spec_from_file_location("logic_engine_key_test", "logic_engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)

    assert calls == []

    # 1. Missing key raises at call time
    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
        engine.call_llm("question")

    # 2. Placeholder key raises at call time
    monkeypatch.setenv("OPENROUTER_API_KEY", "PASTE_YOUR_OPENROUTER_API_KEY_HERE")
    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
        engine.call_llm("question")

    # 3. Configured key executes
    monkeypatch.setenv("OPENROUTER_API_KEY", "configured-test-key")
    assert engine.call_llm("question") == "answer"
    assert calls[0]["api_key"] == "configured-test-key"

    # 4. Parameter forwarding to chat
    routed = []
    monkeypatch.setattr(engine.providers, "chat", lambda *args, **kwargs:
                        routed.append((args, kwargs)) or "routed")
    assert engine.call_llm("prompt", system="system", model="custom-model", max_tokens=22, retries=1) == "routed"
    assert routed == [(("prompt", "system"), {"model": "custom-model", "max_tokens": 22, "retries": 1})]
