# pyright: reportPrivateUsage=false
from __future__ import annotations

import pytest

from llm_regression_detector.config import Settings
from llm_regression_detector.llm.client import (  # pyright: ignore[reportPrivateUsage]
    LLMResponse,
    RouterClient,
    _build_router,
    build_client,
)
from llm_regression_detector.llm.mock import MockLLMClient


def test_build_client_returns_mock_when_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("HF_TOKEN", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    client = build_client(settings)
    assert isinstance(client, MockLLMClient)


def test_build_router_with_hf_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    # HF + always-on Ollama fallback
    assert len(router.model_list) == 2  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


def test_build_router_with_all_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("GEMINI_API_KEY", "gem_x")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    router = _build_router(settings)
    assert len(router.model_list) == 3  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


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
