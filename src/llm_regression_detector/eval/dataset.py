"""Pydantic models for the eval pipeline + dataset/prompt loaders."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(StrEnum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class GoldenCase(BaseModel):
    """One labelled test case from the golden dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    input_email: str
    expected_category: Category
    expected_summary_keywords: list[str] = Field(default_factory=list[str])
    difficulty: Difficulty
    notes: str | None = None


class FewShot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    input: str
    output: str


class PromptSpec(BaseModel):
    """Versioned prompt definition loaded from YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    description: str | None = None
    system: str
    user_template: str = Field(description="Format string with `{email}` placeholder")
    few_shots: list[FewShot] = Field(default_factory=list[FewShot])

    def render_messages(self, email: str) -> list[dict[str, str]]:
        """Build the chat-completion messages array for one input email."""
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system}]
        for shot in self.few_shots:
            messages.append({"role": "user", "content": shot.input})
            messages.append({"role": "assistant", "content": shot.output})
        messages.append({"role": "user", "content": self.user_template.format(email=email)})
        return messages


class JudgeScore(BaseModel):
    """LLM-as-Judge output for a single case."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    category_match: bool
    summary_score: int = Field(ge=1, le=5)
    rationale: str = ""


class CaseResult(BaseModel):
    """Outcome of running one golden case through the prompt + judge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    predicted_category: str
    predicted_summary: str
    judge: JudgeScore
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    raw_output: str
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.judge.category_match and self.judge.summary_score >= 3


class CategoryAccuracy(BaseModel):
    """Per-category breakdown — exposes regressions hidden in the aggregate."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    category: str
    accuracy: float = Field(ge=0.0, le=1.0)
    cases_total: int = Field(ge=0)
    cases_passed: int = Field(ge=0)


class RunSummary(BaseModel):
    """Aggregate metrics for an eval run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accuracy: float = Field(ge=0.0, le=1.0)
    accuracy_ci_low: float = Field(ge=0.0, le=1.0, description="Wilson 95% CI lower bound")
    accuracy_ci_high: float = Field(ge=0.0, le=1.0, description="Wilson 95% CI upper bound")
    avg_summary_score: float = Field(ge=1.0, le=5.0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    latency_p99_ms: float = Field(ge=0.0)
    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    cases_total: int = Field(ge=0)
    cases_passed: int = Field(ge=0)
    per_category: list[CategoryAccuracy] = Field(default_factory=list[CategoryAccuracy])


class EvalRun(BaseModel):
    """A complete eval run — what we persist and diff against."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    prompt_name: str
    prompt_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    results: list[CaseResult]
    summary: RunSummary

    @field_validator("results")
    @classmethod
    def _non_empty(cls, v: list[CaseResult]) -> list[CaseResult]:
        if not v:
            raise ValueError("EvalRun must contain at least one CaseResult")
        return v


# ── Loaders ────────────────────────────────────────────────────────────────


def load_dataset(path: Path | str) -> list[GoldenCase]:
    """Load a JSON golden-dataset file into a list of ``GoldenCase``."""
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Dataset at {path} must be a JSON array")
    return [GoldenCase.model_validate(item) for item in cast(list[Any], raw)]


def load_prompt(path: Path | str) -> PromptSpec:
    """Load a YAML prompt file into a ``PromptSpec``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PromptSpec.model_validate(raw)


__all__ = [
    "CaseResult",
    "Category",
    "CategoryAccuracy",
    "Difficulty",
    "EvalRun",
    "FewShot",
    "GoldenCase",
    "JudgeScore",
    "PromptSpec",
    "RunSummary",
    "load_dataset",
    "load_prompt",
]
