import sys
import time
from types import SimpleNamespace
import pytest

import providers


def test_provider_selection_and_missing_keys(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert providers.selected_provider() == "openrouter"

    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
        providers.chat("prompt", "system")

    monkeypatch.setenv("OPENROUTER_API_KEY", "PASTE_YOUR_OPENROUTER_API_KEY_HERE")
    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
        providers.chat("prompt", "system")

    monkeypatch.setenv("OPENROUTER_API_KEY", "your_api_key_here")
    with pytest.raises(EnvironmentError, match="OPENROUTER_API_KEY"):
        providers.chat("prompt", "system")


def test_openrouter_chat_success_and_model_forwarding(monkeypatch):
    calls = []
    traces = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                model=kwargs["model"],
                choices=[SimpleNamespace(message=SimpleNamespace(content="  openrouter response text  "), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.0001)
            )

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            calls.append({"api_key": api_key, "base_url": base_url})
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_openai_mod = SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIError=type("APIError", (Exception,), {}),
        RateLimitError=type("RateLimitError", (Exception,), {}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)

    monkeypatch.setenv("OPENROUTER_API_KEY", "valid-openrouter-key")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    # 1. Default model forwarding
    out = providers.chat("test prompt", "test system", on_trace=lambda event, data: traces.append((event, data)))
    assert out == "openrouter response text"
    assert calls[0]["api_key"] == "valid-openrouter-key"
    assert calls[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert calls[1]["model"] == "meta-llama/llama-3.3-70b-instruct"
    assert calls[1]["temperature"] == 0.0
    assert calls[1]["max_tokens"] == 1024
    assert calls[1]["messages"] == [
        {"role": "system", "content": "test system"},
        {"role": "user", "content": "test prompt"}
    ]
    assert len(traces) == 1
    event, data = traces[0]
    assert event == "provider_call"
    assert data["provider"] == "openrouter"
    assert data["model"] == "meta-llama/llama-3.3-70b-instruct"
    assert data["output"] == "openrouter response text"
    assert data["cost"] == 0.0001
    assert data["latency_seconds"] >= 0

    # 2. Custom model forwarding via parameter
    calls.clear()
    traces.clear()
    out = providers.chat("p2", "s2", model="custom/model-id", max_tokens=256)
    assert out == "openrouter response text"
    assert calls[1]["model"] == "custom/model-id"
    assert calls[1]["max_tokens"] == 256


def test_openrouter_custom_base_url_and_env_model(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                model=kwargs["model"],
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")]
            )

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            calls.append({"api_key": api_key, "base_url": base_url})
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_openai_mod = SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIError=type("APIError", (Exception,), {}),
        RateLimitError=type("RateLimitError", (Exception,), {}),
        AuthenticationError=type("AuthenticationError", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://custom.router/v1")
    monkeypatch.setenv("OPENROUTER_MODEL", "custom-env-model")

    providers.chat("prompt", "system")
    assert calls[0]["base_url"] == "https://custom.router/v1"
    assert calls[1]["model"] == "custom-env-model"


def test_openrouter_error_classification_and_retries(monkeypatch):
    state = {"mode": "auth"}
    calls = []
    waits = []

    monkeypatch.setattr(time, "sleep", waits.append)

    class MockAuthError(Exception):
        pass

    class MockRateLimitError(Exception):
        pass

    class MockAPIError(Exception):
        def __init__(self, code, message):
            self.code = code
            self.message = message

    class FailCompletions:
        def create(self, **kwargs):
            calls.append(1)
            mode = state["mode"]
            if mode == "auth":
                raise MockAuthError("Invalid API key provided")
            elif mode == "credits":
                raise MockRateLimitError("User has insufficient credits for request")
            elif mode == "rate_limit_exceeded":
                raise MockRateLimitError("429 Too Many Requests: Rate limit reached")
            elif mode == "forbidden":
                raise MockAPIError(403, "Model access forbidden")
            elif mode == "not_found":
                raise MockAPIError(404, "No such model exists")
            elif mode == "blocked":
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason="content_filter", message=SimpleNamespace(content=""))])
            elif mode == "empty":
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="  "))])
            elif mode == "transient":
                if len(calls) == 1:
                    raise MockRateLimitError("Rate limit temporary")
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="recovered"))])
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="ok"))])

    class MockOpenAI:
        def __init__(self, api_key, base_url):
            self.chat = SimpleNamespace(completions=FailCompletions())

    fake_openai_mod = SimpleNamespace(
        OpenAI=MockOpenAI,
        APIError=MockAPIError,
        RateLimitError=MockRateLimitError,
        AuthenticationError=MockAuthError,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)
    monkeypatch.setenv("OPENROUTER_API_KEY", "valid-key")

    # 1. 401 Authentication
    state["mode"] = "auth"
    calls.clear()
    with pytest.raises(providers.ProviderError, match="authentication failed \\(401\\)"):
        providers.chat("p", "s")

    # 2. 402 Insufficient credits
    state["mode"] = "credits"
    calls.clear()
    with pytest.raises(providers.ProviderError, match="insufficient credits \\(402\\)"):
        providers.chat("p", "s")

    # 3. 403 Forbidden
    state["mode"] = "forbidden"
    calls.clear()
    with pytest.raises(providers.ProviderError, match="forbidden / model access denied \\(403\\)"):
        providers.chat("p", "s")

    # 4. 404 Model Not Found
    state["mode"] = "not_found"
    calls.clear()
    with pytest.raises(providers.ProviderError, match="model not found \\(404\\)"):
        providers.chat("p", "s")

    # 5. Blocked finish reason
    state["mode"] = "blocked"
    calls.clear()
    with pytest.raises(providers.ProviderError, match="blocked with finish reason: content_filter"):
        providers.chat("p", "s")

    # 6. Empty response
    state["mode"] = "empty"
    calls.clear()
    with pytest.raises(providers.ProviderError, match="empty response"):
        providers.chat("p", "s")

    # 7. Transient 429 recovers on retry
    state["mode"] = "transient"
    calls.clear()
    res = providers.chat("p", "s")
    assert res == "recovered"
    assert len(calls) == 2
    assert waits == [2]

    # 8. Unrecoverable rate limit exceeds retries
    state["mode"] = "rate_limit_exceeded"
    calls.clear()
    waits.clear()
    with pytest.raises(providers.ProviderError, match="rate limit / quota exceeded \\(429\\)"):
        providers.chat("p", "s", retries=1)
    assert len(calls) == 2
    assert waits == [2]
