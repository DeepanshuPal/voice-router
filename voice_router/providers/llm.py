"""LLM leg: thin pass-through to LiteLLM, which already normalizes 100+ LLM
APIs. We add model fallback across the candidates you configure.

Optional: pip install voice-router[llm]
"""

from __future__ import annotations

from .base import ProviderError, ProviderUnavailable


async def chat_completion(messages: list[dict], models: list[str], **kwargs) -> dict:
    try:
        import litellm
    except ImportError as e:
        raise ProviderUnavailable(
            "litellm not installed - run `pip install voice-router[llm]`"
        ) from e

    last: Exception | None = None
    for model in models:
        try:
            resp = await litellm.acompletion(model=model, messages=messages, **kwargs)
            return {
                "model": model,
                "content": resp.choices[0].message.content,
                "usage": getattr(resp, "usage", None) and dict(resp.usage),
            }
        except Exception as e:  # litellm raises provider-specific exceptions
            last = e
    raise ProviderError(f"all LLM candidates failed: {last}")
