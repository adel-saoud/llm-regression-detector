"""Unified async LLM client.

Wraps ``litellm.Router`` with a tiered fallback chain.  Priority order:

0. Custom provider: ``LRD_CUSTOM_MODEL`` (any litellm model string, e.g.
   ``anthropic/claude-haiku-4-5``, ``ollama/qwen3.5:0.8b``, ``openai/gpt-4o``).
   When set, this slots in as the highest-priority tier.  Use
   ``LRD_CUSTOM_API_KEY`` / ``LRD_CUSTOM_API_BASE`` as companions.
   ``ollama/`` models auto-detect the API base from ``LRD_OLLAMA_BASE_URL``.
1. Hugging Face Inference Providers (``LRD_HF_MODEL``, requires ``HF_TOKEN``)
2. Google Gemini free tier (``LRD_GEMINI_MODEL``, requires ``GEMINI_API_KEY``)
3. Ollama fully local (``LRD_OLLAMA_MODEL``, no key required)

The judge gets its own single-model client from ``LRD_JUDGE_MODEL``, so the model
under test never grades its own answers. Ollama tiers receive ``LRD_REASONING_EFFORT``
as litellm's ``reasoning_effort``, which litellm sends to Ollama as ``think``.

Every model id comes from ``Settings`` (env vars): nothing is hardcoded.

When no real credentials are present, ``build_client`` returns a deterministic
mock so the full pipeline runs end-to-end offline during development.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from litellm.router import Router
from pydantic import SecretStr

from llm_regression_detector.config import Settings

DEFAULT_MODEL_ALIAS = "default"
"""Router ``model_name`` of the primary tier; every call asks for it and litellm falls back."""

_HF_API_BASE = "https://router.huggingface.co/v1"
"""OpenAI-compatible endpoint of Hugging Face Inference Providers."""


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
    ) -> LLMResponse:
        """Return one chat completion for ``messages`` (OpenAI chat format)."""
        ...


class RouterClient:
    """Production client: routes through ``litellm.Router`` with fallbacks."""

    def __init__(self, router: Router) -> None:
        self._router = router

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Call the primary tier through the router; ``LLMError`` once every tier failed."""
        start = time.perf_counter()
        try:
            response: Any = await self._router.acompletion(  # pyright: ignore[reportUnknownMemberType]
                model=DEFAULT_MODEL_ALIAS,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"All providers in fallback chain failed: {exc}") from exc

        latency_ms = (time.perf_counter() - start) * 1000.0
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=choice.message.content or "",
            model=getattr(response, "model", DEFAULT_MODEL_ALIAS),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency_ms,
        )


def _build_router(settings: Settings) -> Router:
    """Assemble the fallback chain from whatever credentials are configured.

    Builds an ordered list of provider tiers.  The first tier claims
    ``DEFAULT_MODEL_ALIAS`` (the name callers use); subsequent tiers get unique
    group names wired into litellm's ``fallbacks`` cascade.

    All model ids come from ``Settings``: no provider models are hardcoded here.
    Two retries per tier, one second apart, absorb free-tier rate limits before
    the router moves to the next tier.
    """
    # (router model_name for a fallback tier, litellm_params)
    tiers: list[tuple[str, dict[str, Any]]] = []

    # Tier 0: custom model (explicit override, highest priority)
    if settings.custom_model is not None:
        tiers.append(
            (
                "custom",
                _model_params(
                    settings,
                    settings.custom_model,
                    api_key=settings.custom_api_key,
                    api_base=settings.custom_api_base,
                ),
            )
        )

    # Tier 1: Hugging Face Inference Providers
    if settings.hf_token is not None:
        tiers.append(
            (
                "hf",
                {
                    "model": settings.hf_model,
                    "api_base": _HF_API_BASE,
                    "api_key": settings.hf_token.get_secret_value(),
                },
            )
        )

    # Tier 2: Google Gemini
    if settings.gemini_api_key is not None:
        tiers.append(
            (
                "gemini",
                {
                    "model": settings.gemini_model,
                    "api_key": settings.gemini_api_key.get_secret_value(),
                },
            )
        )

    # Tier 3: Ollama (always present unless custom model is already an Ollama model)
    custom_is_ollama = settings.custom_model is not None and _is_ollama(settings.custom_model)
    if not custom_is_ollama:
        tiers.append(("ollama", _model_params(settings, settings.ollama_model)))

    model_list: list[dict[str, Any]] = []
    fallback_names: list[str] = []
    for i, (name, litellm_params) in enumerate(tiers):
        group = DEFAULT_MODEL_ALIAS if i == 0 else name
        model_list.append({"model_name": group, "litellm_params": litellm_params})
        if i > 0:
            fallback_names.append(group)

    fallbacks: list[dict[str, list[str]]] = (
        [{DEFAULT_MODEL_ALIAS: fallback_names}] if fallback_names else []
    )

    return Router(
        model_list=model_list,
        fallbacks=fallbacks,  # pyright: ignore[reportArgumentType]
        num_retries=2,
        retry_after=1,
    )


def _is_ollama(model: str) -> bool:
    return model.startswith(("ollama/", "ollama_chat/"))


def _model_params(
    settings: Settings,
    model: str,
    *,
    api_key: SecretStr | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """litellm params for one model; Ollama models get the local base URL and the
    thinking setting, the only provider where ``reasoning_effort`` is verified."""
    params: dict[str, Any] = {"model": model}
    if api_key is not None:
        params["api_key"] = api_key.get_secret_value()
    if api_base is not None:
        params["api_base"] = api_base
    if _is_ollama(model):
        params.setdefault("api_base", settings.ollama_base_url)
        params["reasoning_effort"] = settings.reasoning_effort.value
    return params


def build_client(settings: Settings) -> LLMClient:
    """Construct the appropriate LLM client for the current environment.

    Returns a mock client when no real credentials are configured, so tests and
    local development remain fully offline and free.
    """
    if settings.is_mock_mode:
        from llm_regression_detector.llm.mock import MockLLMClient

        return MockLLMClient()

    return RouterClient(_build_router(settings))


def build_judge_client(settings: Settings) -> LLMClient:
    """The judge's client: ``LRD_JUDGE_MODEL`` alone, with no fallback to the model under test.

    A fallback could hand grading to the model being graded, which is the bias the
    separate judge exists to avoid. The mock answers both roles offline.
    """
    if settings.is_mock_mode:
        from llm_regression_detector.llm.mock import MockLLMClient

        return MockLLMClient()

    params = _model_params(
        settings,
        settings.judge_model,
        api_key=settings.judge_api_key,
        api_base=settings.judge_api_base,
    )
    return RouterClient(
        Router(
            model_list=[{"model_name": DEFAULT_MODEL_ALIAS, "litellm_params": params}],
            num_retries=2,
            retry_after=1,
        )
    )
