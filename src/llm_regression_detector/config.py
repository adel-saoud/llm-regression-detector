"""Runtime configuration: env-driven, validated by pydantic-settings.

Resolution order: explicit kwargs → environment variables → ``.env`` file → defaults.
"""

from __future__ import annotations

import difflib
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    """Levels accepted by ``LRD_LOG_LEVEL``; names match the stdlib ``logging`` constants."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class WebhookPlatform(StrEnum):
    """Payload format sent to ``LRD_WEBHOOK_URL``; ``generic`` posts the raw alert JSON."""

    SLACK = "slack"
    GOOGLE_CHAT = "google_chat"
    DISCORD = "discord"
    GENERIC = "generic"


class ReasoningEffort(StrEnum):
    """Values of litellm's ``reasoning_effort``, which it forwards to Ollama as ``think``.

    ``none`` sends ``think: false``. Thinking models such as Qwen3.5 think by default and
    can spend the whole ``max_tokens`` budget on it, returning an empty answer.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


_PLATFORM_HOSTS: dict[WebhookPlatform, tuple[str, ...]] = {
    WebhookPlatform.SLACK: ("slack.com",),
    WebhookPlatform.GOOGLE_CHAT: ("googleapis.com",),
    WebhookPlatform.DISCORD: ("discord.com", "discordapp.com"),
}
"""Registrable domains of each platform's incoming-webhook URLs."""


def detect_platform(url: str) -> WebhookPlatform:
    """Infer the webhook platform from the URL host.

    The host must equal a known domain or end with ``.<domain>``, so a look-alike
    such as ``slack.com.example.net`` falls through to ``generic``.
    """
    host = (urlparse(url).hostname or "").lower()
    for platform, domains in _PLATFORM_HOSTS.items():
        if any(host == domain or host.endswith(f".{domain}") for domain in domains):
            return platform
    return WebhookPlatform.GENERIC


ENV_PREFIX = "LRD_"
"""Prefix of every setting read from the environment, except the vendor credentials."""


class UnknownSettingError(ValueError):
    """An ``LRD_*`` environment variable matches no setting, usually a typo or an old name."""


class Settings(BaseSettings):
    """Top-level settings; instantiate once at process start.

    Every field is read from ``LRD_<FIELD NAME>``. The two provider credentials keep
    the names their vendors' own tools read (``HF_TOKEN``, ``GEMINI_API_KEY``) so one
    secret serves both. ``extra="forbid"`` rejects unknown keys in ``.env``; unknown
    ``LRD_*`` names in the process environment are rejected by
    ``_reject_unknown_env_vars``, because pydantic-settings ignores those even with
    ``extra="forbid"``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix=ENV_PREFIX,
        env_ignore_empty=True,
        extra="forbid",
        case_sensitive=False,
        validate_default=True,
    )

    # ── Custom provider (highest-priority tier when set) ────────────────────
    custom_model: str | None = Field(
        default=None,
        description="Any litellm model string; slots in as highest-priority provider",
    )
    custom_api_key: SecretStr | None = Field(
        default=None, description="API key for the custom model provider"
    )
    custom_api_base: str | None = Field(
        default=None,
        description="Base URL for the custom provider (auto-set for ollama/ models)",
    )

    # ── Provider credentials (any one is enough; mock mode if none) ────────
    hf_token: SecretStr | None = Field(
        default=None, description="Hugging Face token", validation_alias="HF_TOKEN"
    )
    gemini_api_key: SecretStr | None = Field(
        default=None, description="Google AI Studio key", validation_alias="GEMINI_API_KEY"
    )
    ollama_base_url: str = Field(default="http://localhost:11434")

    # ── Model selection (truly model-agnostic: override per provider) ─────
    hf_model: str = Field(
        default="openai/openai/gpt-oss-20b:cheapest",
        description="LiteLLM model id for HF Inference Providers (with provider policy)",
    )
    gemini_model: str = Field(default="gemini/gemini-2.0-flash")
    ollama_model: str = Field(
        default="ollama/qwen3.5:0.8b",
        description="Model under test on the Ollama tier",
    )
    reasoning_effort: ReasoningEffort = Field(
        default=ReasoningEffort.NONE,
        description="Thinking budget sent to Ollama models (system under test and judge)",
    )

    # ── Judge (a separate model, so no model grades its own answers) ───────
    judge_model: str = Field(
        default="ollama/qwen3.5:4b",
        description="litellm model string of the judge; must differ from the model under test",
    )
    judge_api_key: SecretStr | None = Field(default=None, description="API key for the judge")
    judge_api_base: str | None = Field(
        default=None,
        description="Base URL for the judge provider (auto-set for ollama/ models)",
    )

    # ── Judge consensus (multi-call majority vote dampens judge variance) ──
    judge_consensus_n: int = Field(
        default=1,
        ge=1,
        le=7,
        description="Number of judge calls per case; majority vote when > 1",
    )

    # ── Cost estimation ($ per million tokens; defaults are free tier) ─────
    cost_input_usd: float = Field(default=0.0, ge=0.0)
    cost_output_usd: float = Field(default=0.0, ge=0.0)

    # ── Webhook ────────────────────────────────────────────────────────────
    webhook_url: str | None = Field(default=None)
    webhook_platform: WebhookPlatform | None = Field(
        default=None,
        description="Override auto-detected platform",
    )

    # ── Runtime ────────────────────────────────────────────────────────────
    log_level: LogLevel = Field(default=LogLevel.INFO)
    concurrency: int = Field(default=10, ge=1, le=100)
    db_path: Path = Field(
        default=Path("evals/runs.db"),
        description="Run history the dashboard reads; `lrd dashboard --db` sets it",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_env_vars(cls, data: Any) -> Any:
        """Fail on an ``LRD_*`` variable that maps to no field, naming the closest valid one."""
        known = {
            (
                f.validation_alias if isinstance(f.validation_alias, str) else ENV_PREFIX + name
            ).upper()
            for name, f in cls.model_fields.items()
        }
        unknown = sorted(
            key
            for key in os.environ
            if key.upper().startswith(ENV_PREFIX) and key.upper() not in known
        )
        if unknown:
            hints: list[str] = []
            for key in unknown:
                close = difflib.get_close_matches(key.upper(), known, n=1)
                hints.append(f"{key} (did you mean {close[0]}?)" if close else key)
            msg = f"Unknown environment variable(s): {', '.join(hints)}."
            raise UnknownSettingError(msg)
        return data

    @property
    def is_mock_mode(self) -> bool:
        """True when no real provider credentials are configured."""
        if self.custom_model is not None:
            return False
        return self.hf_token is None and self.gemini_api_key is None

    @property
    def primary_model(self) -> str:
        """The first model of the fallback chain under test: the one that answers when it is up."""
        if self.custom_model is not None:
            return self.custom_model
        if self.hf_token is not None:
            return self.hf_model
        if self.gemini_api_key is not None:
            return self.gemini_model
        return self.ollama_model

    @model_validator(mode="after")
    def _reject_self_judging(self) -> Self:
        """A model grading its own answers rates them higher (self-preference bias).

        Panickssery et al., 2024, "LLM Evaluators Recognize and Favor Their Own
        Generations", https://arxiv.org/abs/2404.13076. The mock answers both roles,
        so the check only applies to real providers.
        """
        if not self.is_mock_mode and _same_model(self.judge_model, self.primary_model):
            msg = (
                f"LRD_JUDGE_MODEL ({self.judge_model}) is the model under test: "
                "set LRD_JUDGE_MODEL to a different model."
            )
            raise ValueError(msg)
        return self

    @property
    def resolved_webhook_platform(self) -> WebhookPlatform | None:
        """The explicit ``webhook_platform`` override, or one auto-detected from the URL."""
        if self.webhook_url is None:
            return None
        return self.webhook_platform or detect_platform(self.webhook_url)

    @model_validator(mode="after")
    def _cap_mock_mode_concurrency(self) -> Self:
        """The mock answers in-process, so extra concurrency only adds event-loop overhead."""
        if self.is_mock_mode and self.concurrency > 50:
            self.concurrency = 50
        return self


def _same_model(a: str, b: str) -> bool:
    """True when two litellm strings name one model; ``ollama/`` and ``ollama_chat/`` are one."""

    def canonical(model: str) -> str:
        return model.removeprefix("ollama_chat/").removeprefix("ollama/").lower()

    return canonical(a) == canonical(b)
