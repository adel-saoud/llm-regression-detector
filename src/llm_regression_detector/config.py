"""Runtime configuration — env-driven, validated by pydantic-settings.

Resolution order: explicit kwargs → environment variables → ``.env`` file → defaults.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self
from urllib.parse import urlparse

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class WebhookPlatform(StrEnum):
    SLACK = "slack"
    GOOGLE_CHAT = "google_chat"
    DISCORD = "discord"
    GENERIC = "generic"


_SLACK_HOSTS = ("slack.com",)
_GOOGLE_CHAT_HOSTS = ("googleapis.com",)
_DISCORD_HOSTS = ("discord.com", "discordapp.com")


def detect_platform(url: str) -> WebhookPlatform:
    """Infer webhook platform from a URL host (suffix-match, not substring)."""
    host = (urlparse(url).hostname or "").lower()
    if any(host == h or host.endswith(f".{h}") for h in _SLACK_HOSTS):
        return WebhookPlatform.SLACK
    if any(host == h or host.endswith(f".{h}") for h in _GOOGLE_CHAT_HOSTS):
        return WebhookPlatform.GOOGLE_CHAT
    if any(host == h or host.endswith(f".{h}") for h in _DISCORD_HOSTS):
        return WebhookPlatform.DISCORD
    return WebhookPlatform.GENERIC


class Settings(BaseSettings):
    """Top-level settings; instantiate once at process start."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    # ── Custom provider override (takes priority over all built-in tiers) ──
    # Accepts any litellm model string, e.g.:
    #   "ollama/llama3.2:3b"              → fully local, no key needed
    #   "anthropic/claude-haiku-4-5"      → Anthropic API
    #   "openai/gpt-4o"                   → OpenAI API
    #   "vertex_ai/gemini-2.0-flash"      → Google Cloud Vertex AI
    #   "github/gpt-4o"                   → GitHub Copilot API
    # When set, this model becomes the primary in the fallback chain.
    # LRD_CUSTOM_API_KEY and LRD_CUSTOM_API_BASE are optional companions.
    custom_model: str | None = Field(
        default=None,
        description="Any litellm model string; slots in as highest-priority provider",
        validation_alias=AliasChoices("LRD_CUSTOM_MODEL", "custom_model"),
    )
    custom_api_key: SecretStr | None = Field(
        default=None,
        description="API key for the custom model provider",
        validation_alias=AliasChoices("LRD_CUSTOM_API_KEY", "custom_api_key"),
    )
    custom_api_base: str | None = Field(
        default=None,
        description="Base URL for the custom provider (auto-set for ollama/ models)",
        validation_alias=AliasChoices("LRD_CUSTOM_API_BASE", "custom_api_base"),
    )

    # ── Provider credentials (any one is enough; mock mode if none) ────────
    hf_token: SecretStr | None = Field(default=None, description="Hugging Face token")
    gemini_api_key: SecretStr | None = Field(default=None, description="Google AI Studio key")
    ollama_base_url: str = Field(default="http://localhost:11434")

    # ── Model selection (truly model-agnostic — override per provider) ─────
    hf_model: str = Field(
        default="openai/openai/gpt-oss-20b:cheapest",
        description="LiteLLM model id for HF Inference Providers (with provider policy)",
        validation_alias=AliasChoices("LRD_HF_MODEL", "hf_model"),
    )
    gemini_model: str = Field(
        default="gemini/gemini-2.0-flash",
        validation_alias=AliasChoices("LRD_GEMINI_MODEL", "gemini_model"),
    )
    ollama_model: str = Field(
        default="ollama/llama3.2:3b",
        validation_alias=AliasChoices("LRD_OLLAMA_MODEL", "ollama_model"),
    )

    # ── Judge consensus (multi-call majority vote dampens judge variance) ──
    judge_consensus_n: int = Field(
        default=1,
        ge=1,
        le=7,
        description="Number of judge calls per case; majority vote when > 1",
        validation_alias=AliasChoices("LRD_JUDGE_CONSENSUS_N", "judge_consensus_n"),
    )

    # ── Cost estimation ($/M tokens; defaults are ~free-tier) ──────────────
    cost_per_million_input_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("LRD_COST_INPUT_USD", "cost_per_million_input_usd"),
    )
    cost_per_million_output_usd: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("LRD_COST_OUTPUT_USD", "cost_per_million_output_usd"),
    )

    # ── Webhook ────────────────────────────────────────────────────────────
    webhook_url: str | None = Field(default=None)
    webhook_platform: WebhookPlatform | None = Field(
        default=None,
        description="Override auto-detected platform",
    )

    # ── Runtime ────────────────────────────────────────────────────────────
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        validation_alias=AliasChoices("LRD_LOG_LEVEL", "log_level"),
    )
    concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        validation_alias=AliasChoices("LRD_CONCURRENCY", "concurrency"),
    )

    @property
    def is_mock_mode(self) -> bool:
        """True when no real provider credentials are configured."""
        if self.custom_model is not None:
            return False
        return self.hf_token is None and self.gemini_api_key is None

    @property
    def resolved_webhook_platform(self) -> WebhookPlatform | None:
        """The explicit ``webhook_platform`` override, or one auto-detected from the URL."""
        if self.webhook_url is None:
            return None
        return self.webhook_platform or detect_platform(self.webhook_url)

    @model_validator(mode="after")
    def _cap_mock_mode_concurrency(self) -> Self:
        # mock is in-process; cap concurrency to avoid event-loop saturation.
        if self.is_mock_mode and self.concurrency > 50:
            self.concurrency = 50
        return self
