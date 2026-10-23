"""OpenRouter generation provider. Retrieval, prompts, and evaluation remain unchanged."""
from collections.abc import Callable
import logging
import os
import time

log = logging.getLogger(__name__)
OPENROUTER_DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class ProviderError(RuntimeError):
    """A configured generation service could not return a usable answer."""


def selected_provider() -> str:
    """Return the active provider name (OpenRouter)."""
    return "openrouter"


def chat(prompt: str, system: str, model: str | None = None,
         max_tokens: int = 1024, retries: int = 2, on_trace: Callable | None = None) -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key or not key.strip() or key.startswith("PASTE_YOUR_") or key.startswith("your_"):
        raise EnvironmentError("OPENROUTER_API_KEY not found or not configured. Add it to your .env file.")

    from openai import OpenAI, APIError, RateLimitError, AuthenticationError

    base_url = os.getenv("OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL).strip()
    client = OpenAI(api_key=key.strip(), base_url=base_url)

    configured_model = (os.getenv("OPENROUTER_MODEL") or "").strip() or OPENROUTER_DEFAULT_MODEL
    target_model = (model or "").strip() or configured_model

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(1, retries + 2):
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            latency = time.perf_counter() - t0
            if not response.choices:
                raise ProviderError("OpenRouter returned no choices")
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason in {"content_filter", "error"}:
                raise ProviderError(f"OpenRouter generation blocked with finish reason: {finish_reason}")
            content = choice.message.content
            if content is None or not content.strip():
                raise ProviderError("OpenRouter returned an empty response")
            text = content.strip()

            if on_trace:
                usage_dict = {}
                cost = None
                if hasattr(response, "usage") and response.usage:
                    u = response.usage
                    usage_dict = {
                        "prompt_tokens": getattr(u, "prompt_tokens", None),
                        "completion_tokens": getattr(u, "completion_tokens", None),
                        "total_tokens": getattr(u, "total_tokens", None),
                    }
                    cost = getattr(u, "cost", None)
                on_trace("provider_call", {
                    "provider": "openrouter",
                    "model": target_model,
                    "model_returned": getattr(response, "model", target_model),
                    "prompt": prompt,
                    "system": system,
                    "max_tokens": max_tokens,
                    "output": text,
                    "finish_reason": finish_reason,
                    "usage": usage_dict,
                    "cost": cost,
                    "latency_seconds": round(latency, 3),
                })
            return text

        except AuthenticationError as exc:
            if on_trace:
                on_trace("provider_call", {
                    "provider": "openrouter",
                    "model": target_model,
                    "prompt": prompt,
                    "system": system,
                    "max_tokens": max_tokens,
                    "error": "AuthenticationError",
                    "error_message": str(exc),
                    "latency_seconds": round(time.perf_counter() - t0, 3)
                })
            raise ProviderError(f"OpenRouter authentication failed (401): {exc}") from exc
        except RateLimitError as exc:
            err_msg = str(exc)
            if "credit" in err_msg.lower() or "payment" in err_msg.lower():
                if on_trace:
                    on_trace("provider_call", {
                        "provider": "openrouter", "model": target_model, "prompt": prompt,
                        "system": system, "max_tokens": max_tokens, "error": "InsufficientCredits",
                        "error_message": err_msg, "latency_seconds": round(time.perf_counter() - t0, 3)
                    })
                raise ProviderError(f"OpenRouter payment required / insufficient credits (402): {err_msg}") from exc
            if attempt <= retries:
                wait = 2 * attempt
                log.warning("OpenRouter rate limit hit (%s). Retrying in %ds...", err_msg, wait)
                time.sleep(wait)
                continue
            if on_trace:
                on_trace("provider_call", {
                    "provider": "openrouter", "model": target_model, "prompt": prompt,
                    "system": system, "max_tokens": max_tokens, "error": "RateLimitError",
                    "error_message": err_msg, "latency_seconds": round(time.perf_counter() - t0, 3)
                })
            raise ProviderError(f"OpenRouter rate limit / quota exceeded (429): {err_msg}") from exc
        except APIError as exc:
            code = getattr(exc, "code", getattr(exc, "status_code", None))
            err_msg = getattr(exc, "message", str(exc))
            if code == 401:
                if on_trace:
                    on_trace("provider_call", {"provider": "openrouter", "model": target_model, "prompt": prompt, "system": system, "max_tokens": max_tokens, "error": "AuthenticationError", "error_message": err_msg, "latency_seconds": round(time.perf_counter() - t0, 3)})
                raise ProviderError(f"OpenRouter authentication failed (401): {err_msg}") from exc
            elif code == 402 or "credit" in err_msg.lower() or "payment" in err_msg.lower():
                if on_trace:
                    on_trace("provider_call", {"provider": "openrouter", "model": target_model, "prompt": prompt, "system": system, "max_tokens": max_tokens, "error": "InsufficientCredits", "error_message": err_msg, "latency_seconds": round(time.perf_counter() - t0, 3)})
                raise ProviderError(f"OpenRouter payment required / insufficient credits (402): {err_msg}") from exc
            elif code == 403:
                if on_trace:
                    on_trace("provider_call", {"provider": "openrouter", "model": target_model, "prompt": prompt, "system": system, "max_tokens": max_tokens, "error": "Forbidden", "error_message": err_msg, "latency_seconds": round(time.perf_counter() - t0, 3)})
                raise ProviderError(f"OpenRouter forbidden / model access denied (403): {err_msg}") from exc
            elif code == 404:
                if on_trace:
                    on_trace("provider_call", {"provider": "openrouter", "model": target_model, "prompt": prompt, "system": system, "max_tokens": max_tokens, "error": "ModelNotFound", "error_message": err_msg, "latency_seconds": round(time.perf_counter() - t0, 3)})
                raise ProviderError(f"OpenRouter model not found (404): {err_msg}") from exc
            elif code in (429, 502, 503, 504) and attempt <= retries:
                wait = 2 * attempt
                log.warning("OpenRouter transient error (%s). Retrying in %ds...", code, wait)
                time.sleep(wait)
                continue
            if on_trace:
                on_trace("provider_call", {"provider": "openrouter", "model": target_model, "prompt": prompt, "system": system, "max_tokens": max_tokens, "error": f"APIError_{code}", "error_message": err_msg, "latency_seconds": round(time.perf_counter() - t0, 3)})
            raise ProviderError(f"OpenRouter API error ({code}): {err_msg}") from exc
        except ProviderError:
            raise
        except Exception as exc:
            if on_trace:
                on_trace("provider_call", {"provider": "openrouter", "model": target_model, "prompt": prompt, "system": system, "max_tokens": max_tokens, "error": type(exc).__name__, "error_message": str(exc), "latency_seconds": round(time.perf_counter() - t0, 3)})
            raise ProviderError(f"OpenRouter request failed: {type(exc).__name__}: {exc}") from exc
