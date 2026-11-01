# pyright: reportPrivateUsage=false
from __future__ import annotations

import pytest

from llm_regression_detector.config import Settings
from llm_regression_detector.llm.client import (  # pyright: ignore[reportPrivateUsage]
    DEFAULT_MODEL_ALIAS,
    LLMResponse,
    RouterClient,
    _build_router,
    build_client,
    build_judge_client,
)
from llm_regression_detector.llm.mock import MockLLMClient


def test_build_client_returns_mock_when_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HF_TOKEN", "GEMINI_API_KEY", "LRD_CUSTOM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    client = build_client(settings)
    assert isinstance(client, MockLLMClient)


def test_build_router_with_hf_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LRD_CUSTOM_MODEL", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    # HF + always-on Ollama fallback
    assert len(router.model_list) == 2  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def test_build_router_with_all_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("GEMINI_API_KEY", "gem_x")
    monkeypatch.delenv("LRD_CUSTOM_MODEL", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    assert len(router.model_list) == 3  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def test_build_router_cascade_groups_with_hf_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF → Ollama cascade: each tier must be in its own model group."""
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LRD_CUSTOM_MODEL", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    names = [m["model_name"] for m in router.model_list]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert DEFAULT_MODEL_ALIAS in names
    assert "ollama" in names
    assert names.count(DEFAULT_MODEL_ALIAS) == 1  # pyright: ignore[reportUnknownMemberType]


def test_build_router_cascade_groups_with_all_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """HF → Gemini → Ollama cascade: three distinct model groups."""
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("GEMINI_API_KEY", "gem_x")
    monkeypatch.delenv("LRD_CUSTOM_MODEL", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    names = [m["model_name"] for m in router.model_list]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert names == [DEFAULT_MODEL_ALIAS, "gemini", "ollama"]


def test_build_router_gemini_primary_when_no_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini becomes the primary (DEFAULT_MODEL_ALIAS) when HF_TOKEN is absent."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gem_x")
    monkeypatch.delenv("LRD_CUSTOM_MODEL", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    names = [m["model_name"] for m in router.model_list]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert names == [DEFAULT_MODEL_ALIAS, "ollama"]


def test_custom_model_is_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    """LRD_CUSTOM_MODEL slots in as the primary (DEFAULT_MODEL_ALIAS) tier."""
    monkeypatch.setenv("LRD_CUSTOM_MODEL", "anthropic/claude-haiku-4-5")
    monkeypatch.setenv("LRD_CUSTOM_API_KEY", "sk-ant-x")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    names = [m["model_name"] for m in router.model_list]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    # custom → ollama (always-on fallback)
    assert names == [DEFAULT_MODEL_ALIAS, "ollama"]
    primary = router.model_list[0]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert primary["litellm_params"]["model"] == "anthropic/claude-haiku-4-5"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def test_custom_model_with_hf_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom model is primary; HF and Ollama follow as fallbacks."""
    monkeypatch.setenv("LRD_CUSTOM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("LRD_CUSTOM_API_KEY", "sk-x")
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    names = [m["model_name"] for m in router.model_list]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert names == [DEFAULT_MODEL_ALIAS, "hf", "ollama"]


def test_custom_ollama_model_no_duplicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """ollama/ custom model: Ollama not added again as a separate fallback tier."""
    monkeypatch.setenv("LRD_CUSTOM_MODEL", "ollama/qwen3.5:0.8b")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    names = [m["model_name"] for m in router.model_list]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert names == [DEFAULT_MODEL_ALIAS]  # sole tier, no duplicate Ollama


def test_custom_ollama_auto_detects_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """ollama/ prefix auto-populates api_base from LRD_OLLAMA_BASE_URL."""
    monkeypatch.setenv("LRD_CUSTOM_MODEL", "ollama/qwen3.5:0.8b")
    monkeypatch.setenv("LRD_OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    primary = router.model_list[0]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    assert primary["litellm_params"]["api_base"] == "http://localhost:11434"  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def test_custom_model_disables_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting LRD_CUSTOM_MODEL means real mode even without HF/Gemini keys."""
    monkeypatch.setenv("LRD_CUSTOM_MODEL", "ollama/qwen3.5:0.8b")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.is_mock_mode is False
    client = build_client(settings)
    assert not isinstance(client, MockLLMClient)


def _litellm_params(router: object) -> list[dict[str, object]]:
    return [m["litellm_params"] for m in router.model_list]  # type: ignore[attr-defined]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportAttributeAccessIssue]


def test_ollama_tier_turns_thinking_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thinking models spend max_tokens on thinking unless litellm sends think: false."""
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LRD_CUSTOM_MODEL", raising=False)
    monkeypatch.delenv("LRD_REASONING_EFFORT", raising=False)
    hf, ollama = _litellm_params(_build_router(Settings(_env_file=None)))  # type: ignore[call-arg]
    assert ollama["reasoning_effort"] == "none"
    assert "reasoning_effort" not in hf  # only verified for Ollama


def test_reasoning_effort_setting_reaches_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LRD_CUSTOM_MODEL", "ollama_chat/qwen3.5:0.8b")
    monkeypatch.setenv("LRD_REASONING_EFFORT", "high")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (custom,) = _litellm_params(_build_router(Settings(_env_file=None)))  # type: ignore[call-arg]
    assert custom["reasoning_effort"] == "high"


def test_judge_client_uses_only_the_judge_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """No fallback: a failing judge must not hand grading to the model under test."""
    monkeypatch.setenv("LRD_CUSTOM_MODEL", "ollama/qwen3.5:0.8b")
    monkeypatch.setenv("LRD_JUDGE_MODEL", "ollama/qwen3.5:4b")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = build_judge_client(Settings(_env_file=None))  # type: ignore[call-arg]
    assert isinstance(client, RouterClient)
    (judge,) = _litellm_params(client._router)
    assert judge["model"] == "ollama/qwen3.5:4b"
    assert judge["api_base"] == "http://localhost:11434"
    assert judge["reasoning_effort"] == "none"


def test_judge_client_is_mock_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HF_TOKEN", "GEMINI_API_KEY", "LRD_CUSTOM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    assert isinstance(build_judge_client(Settings(_env_file=None)), MockLLMClient)  # type: ignore[call-arg]


async def test_mock_client_returns_classifier_json() -> None:
    client = MockLLMClient()
    resp = await client.complete([{"role": "user", "content": "I want a refund please"}])
    assert isinstance(resp, LLMResponse)
    assert "billing" in resp.content
    assert resp.model == "mock/deterministic"


async def test_mock_client_judge_response_when_prompt_mentions_judge() -> None:
    client = MockLLMClient()
    resp = await client.complete(
        [{"role": "user", "content": "Score and judge this output"}],
    )
    assert "summary_score" in resp.content


async def test_mock_client_is_deterministic() -> None:
    client = MockLLMClient()
    msg = [{"role": "user", "content": "Same input every time"}]
    a = await client.complete(msg)
    b = await client.complete(msg)
    assert a.content == b.content
    assert a.latency_ms == b.latency_ms


async def test_router_client_propagates_litellm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RouterClient wraps litellm exceptions in LLMError."""
    from llm_regression_detector.llm.client import LLMError

    class _BoomRouter:
        async def acompletion(self, **_: object) -> object:
            raise RuntimeError("provider down")

    client = RouterClient(_BoomRouter())  # type: ignore[arg-type]
    with pytest.raises(LLMError):
        await client.complete([{"role": "user", "content": "hi"}])
