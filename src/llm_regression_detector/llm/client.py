"""Unified async LLM client.

Wraps ``litellm.Router`` with a tiered free-tier fallback chain:

1. Hugging Face Inference Providers (default model: ``openai/gpt-oss-20b:cheapest``)
2. Google Gemini free tier (default model: ``gemini-2.0-flash``)
3. Ollama (fully local; default model: ``llama3.2:3b``)

Every model id is configurable via ``Settings`` (env vars ``LRD_HF_MODEL`` etc),
so the project is genuinely model-agnostic — swapping to ``Llama-3.1-70B`` or
``mistralai/Mistral-Nemo-Instruct`` is a one-line change.

When no real credentials are present, ``build_client`` returns a deterministic
mock so the full pipeline still runs end-to-end during development.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from litellm.router import Router

from llm_regression_detector.config import Settings

DEFAULT_MODEL_ALIAS = "default"

_HF_API_BASE = "https://router.huggingface.co/v1"


class LLMError(Exception):
    """Raised when the router cannot produce a completion from any provider."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Provider-agnostic completion result."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async LLM interface used by the runner and the judge."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


class RouterClient:
    """Production client — routes through ``litellm.Router`` with fallbacks."""

    def __init__(self, router: Router) -> None:
        self._router = router

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        try:
            resp: Any = await self._router.acompletion(  # pyright: ignore[reportUnknownMemberType]
                model=DEFAULT_MODEL_ALIAS,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"All providers in fallback chain failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            content=choice.message.content or "",
            model=getattr(resp, "model", DEFAULT_MODEL_ALIAS),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency_ms,
        )


def _build_router(settings: Settings) -> Router:
    """Assemble the fallback chain from whatever credentials are configured.

    All model ids come from ``Settings`` — there are no hardcoded provider
    models in this function body.
    """
    model_list: list[dict[str, Any]] = []

    if settings.hf_token is not None:
        model_list.append(
            {
                "model_name": DEFAULT_MODEL_ALIAS,
                "litellm_params": {
                    "model": settings.hf_model,
                    "api_base": _HF_API_BASE,
                    "api_key": settings.hf_token.get_secret_value(),
                },
            }
        )

    if settings.gemini_api_key is not None:
        model_list.append(
            {
                "model_name": DEFAULT_MODEL_ALIAS,
                "litellm_params": {
                    "model": settings.gemini_model,
                    "api_key": settings.gemini_api_key.get_secret_value(),
                },
            }
        )

    # Ollama is always available as a local fallback (no key required).
    model_list.append(
        {
            "model_name": DEFAULT_MODEL_ALIAS,
            "litellm_params": {
                "model": settings.ollama_model,
                "api_base": settings.ollama_base_url,
            },
        }
    )

    return Router(
        model_list=model_list,
        num_retries=2,
        retry_after=1,
        routing_strategy="simple-shuffle",
    )


def build_client(settings: Settings) -> LLMClient:
    """Construct the appropriate LLM client for the current environment.

    Returns a mock client when no real credentials are configured, so tests and
    local development remain fully offline and free.
    """
    if settings.is_mock_mode:
        from llm_regression_detector.llm.mock import MockLLMClient

        return MockLLMClient()

    return RouterClient(_build_router(settings))
