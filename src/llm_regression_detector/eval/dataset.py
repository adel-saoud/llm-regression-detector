"""Pydantic models for the eval pipeline + dataset/prompt loaders."""

from __future__ import annotations

import json
import warnings
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

DATASET_SCHEMA_VERSION = 2
"""Version written in the ``schema_version`` field of a v2 dataset file.

A bare JSON array is read as schema v1, so files written before v2 keep loading.
"""

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
"""Dialect pydantic generates; stated in each published schema so validators pick it."""

_SCHEMA_BASE_URL = (
    "https://raw.githubusercontent.com/adel-saoud/llm-regression-detector/main/schemas"
)

DATASET_SCHEMA_ID = f"{_SCHEMA_BASE_URL}/golden-dataset.v{DATASET_SCHEMA_VERSION}.schema.json"
"""``$id`` of the dataset contract. The schema version is in the name, so a breaking
change publishes a new file instead of rewriting this one; consumers pin a release tag."""

PREDICTION_SCHEMA_ID = f"{_SCHEMA_BASE_URL}/predictions.v1.schema.json"
"""``$id`` of one predictions JSONL line, the input of ``lrd score``."""


NonEmptyStr = Annotated[str, Field(min_length=1)]
"""A string that must not be empty, so a blank rubric criterion fails validation."""


class Difficulty(StrEnum):
    """Optional case tag for slicing results; it does not change how a case is scored."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class GoldenCase(BaseModel):
    """One labelled test case from the golden dataset.

    The fields that are set decide how the case is scored, and the checks add up:
    ``expected_category`` is compared to the predicted label by exact string match,
    ``reference_answer`` and ``expected_summary_keywords`` guide the judge's 1 to 5
    quality score, and every ``rubric`` criterion gets its own pass or fail verdict.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        # Mirrors ``_has_something_to_judge`` so non-Python producers get the same rule.
        json_schema_extra={
            "anyOf": [
                {"required": ["expected_category"]},
                {"required": ["reference_answer"]},
                {"required": ["rubric"]},
            ]
        },
    )

    id: str
    topic: str | None = None
    input_text: str
    expected_category: str | None = Field(
        default=None,
        description="Gold label, compared to the predicted category by exact string match.",
    )
    expected_summary_keywords: list[str] = Field(default_factory=list[str])
    reference_answer: str | None = Field(
        default=None,
        description="A good answer the judge compares the predicted answer against.",
    )
    rubric: list[NonEmptyStr] | None = Field(
        default=None,
        min_length=1,
        description="Criteria the answer must meet; the judge returns pass or fail for each.",
    )
    difficulty: Difficulty | None = None
    notes: str | None = None
    source: dict[str, str] | None = Field(
        default=None,
        description="Provenance of the case, e.g. {'origin': 'trace-to-golden', 'trace': 'id'}.",
    )

    @model_validator(mode="after")
    def _has_something_to_judge(self) -> Self:
        """A case with no category, reference answer or rubric would pass on any output."""
        if self.expected_category is None and self.reference_answer is None and self.rubric is None:
            raise ValueError(
                f"case {self.id!r} has nothing to judge: "
                "set expected_category, reference_answer or rubric"
            )
        return self


class GoldenDataset(BaseModel):
    """A schema v2 dataset file: ``{"schema_version": 2, "cases": [...]}``.

    The wrapper makes the version explicit so loaders never guess it from fields.
    Its JSON Schema, ``schemas/golden-dataset.v2.schema.json``, is the contract that
    producers in other repos (trace-to-golden) pin; ``$id`` carries the version.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        title="Golden dataset v2",
        json_schema_extra={"$schema": JSON_SCHEMA_DIALECT, "$id": DATASET_SCHEMA_ID},
    )

    schema_version: Literal[2]
    cases: list[GoldenCase] = Field(min_length=1)


class PredictionRecord(BaseModel):
    """One line of a predictions JSONL file: an answer the system under test already produced.

    Used by offline scoring (``lrd score``) when the system is an agent or service
    that generates its own answers, so there is no prompt to re-run. Keys mirror the
    JSON output keys of a prompt (``category``, ``summary``, ``answer``).
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        title="Prediction v1",
        json_schema_extra={"$schema": JSON_SCHEMA_DIALECT, "$id": PREDICTION_SCHEMA_ID},
    )

    id: str = Field(description="Id of the golden case this prediction answers.")
    category: str | None = None
    summary: str | None = None
    answer: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    error: str | None = Field(
        default=None, description="Set when the system failed on this case; the case fails."
    )


class FewShot(BaseModel):
    """One example exchange, sent as a user and assistant turn before the real input."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    input: str
    output: str


class OutputFormat(StrEnum):
    """How the model output of a prompt is read.

    ``json``: a JSON object with optional ``category``, ``summary`` and ``answer`` keys.
    ``text``: the whole output is the answer.
    """

    JSON = "json"
    TEXT = "text"


class PromptSpec(BaseModel):
    """Versioned prompt definition loaded from YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    description: str | None = None
    system: str
    user_template: str = Field(description="Format string with an `{input_text}` placeholder")
    few_shots: list[FewShot] = Field(default_factory=list[FewShot])
    output_format: OutputFormat = OutputFormat.JSON
    dataset: str | None = Field(
        default=None,
        description="Golden dataset this prompt is evaluated on, relative to the prompt file.",
    )

    @model_validator(mode="after")
    def _warn_on_legacy_placeholder(self) -> Self:
        """``{email}`` dates from the email classifier; it still renders until removed."""
        if "{email}" in self.user_template:
            warnings.warn(
                f"Prompt {self.name!r}: the {{email}} placeholder is deprecated, "
                "use {input_text}. {email} will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self

    def render_messages(self, input_text: str) -> list[dict[str, str]]:
        """Build the chat-completion messages array for one case input."""
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system}]
        for shot in self.few_shots:
            messages.append({"role": "user", "content": shot.input})
            messages.append({"role": "assistant", "content": shot.output})
        user_content = self.user_template.format(input_text=input_text, email=input_text)
        messages.append({"role": "user", "content": user_content})
        return messages


class CriterionVerdict(BaseModel):
    """The judge's pass or fail verdict on one rubric criterion."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    criterion: str
    passed: bool
    rationale: str = ""


class JudgeScore(BaseModel):
    """Scores for a single case.

    ``category_match`` is ``None`` when the case has no ``expected_category``.
    ``summary_score`` keeps its v1 name so stored runs still load; it rates the
    summary or the answer, whichever the prompt produced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    category_match: bool | None = None
    summary_score: int = Field(ge=1, le=5)
    rationale: str = ""
    criteria: list[CriterionVerdict] = Field(default_factory=list[CriterionVerdict])

    @property
    def rubric_pass_rate(self) -> float | None:
        """Share of rubric criteria passed, or ``None`` when the case has no rubric."""
        if not self.criteria:
            return None
        return sum(c.passed for c in self.criteria) / len(self.criteria)


class CaseResult(BaseModel):
    """Outcome of running one golden case through the prompt + judge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    topic: str | None = None
    predicted_category: str = Field(description="Empty when the prompt produced no category")
    predicted_summary: str
    predicted_answer: str | None = None
    judge: JudgeScore
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    raw_output: str
    error: str | None = None

    @property
    def passed(self) -> bool:
        """Every check the case defines must pass: category, quality score and each criterion.

        The quality bar is 3 of 5, the judge prompt's "partially correct" grade.
        """
        return (
            self.judge.category_match is not False
            and self.judge.summary_score >= 3
            and all(c.passed for c in self.judge.criteria)
        )


class CategoryAccuracy(BaseModel):
    """Per-category breakdown: exposes regressions hidden in the aggregate."""

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
    """A complete eval run: what we persist and diff against."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    prompt_name: str
    prompt_version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    results: list[CaseResult] = Field(min_length=1)
    summary: RunSummary


# ── Loaders ────────────────────────────────────────────────────────────────


def load_dataset(path: Path | str) -> list[GoldenCase]:
    """Load a golden-dataset file into a list of ``GoldenCase``.

    Accepts a v2 object (``{"schema_version": 2, "cases": [...]}``) or a v1 bare array.
    """
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        cases = [GoldenCase.model_validate(item) for item in cast(list[Any], raw)]
    elif isinstance(raw, dict):
        version = cast(dict[str, Any], raw).get("schema_version")
        if version != DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Dataset at {path} has schema_version {version}; "
                f"supported: {DATASET_SCHEMA_VERSION} (object) or a bare JSON array (v1)"
            )
        cases = list(GoldenDataset.model_validate(raw).cases)
    else:
        raise ValueError(f"Dataset at {path} must be a JSON array (v1) or object (v2)")
    duplicates = sorted(i for i, n in Counter(c.id for c in cases).items() if n > 1)
    if duplicates:
        raise ValueError(f"Dataset at {path} has duplicate case ids: {', '.join(duplicates)}")
    return cases


def load_predictions(path: Path | str) -> list[PredictionRecord]:
    """Load a predictions JSONL file: one ``PredictionRecord`` per non-blank line.

    Errors name the 1-based line number so a producer can find the bad record.
    """
    records: list[PredictionRecord] = []
    seen: set[str] = set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = PredictionRecord.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"Predictions at {path}, line {line_number}: {exc}") from exc
        if record.id in seen:
            raise ValueError(
                f"Predictions at {path}, line {line_number}: duplicate id {record.id!r}"
            )
        seen.add(record.id)
        records.append(record)
    return records


def load_prompt(path: Path | str) -> PromptSpec:
    """Load a YAML prompt file into a ``PromptSpec``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PromptSpec.model_validate(raw)


def prompt_dataset_path(prompt_path: Path, prompt: PromptSpec) -> Path | None:
    """Where the prompt's ``dataset`` key points, resolved against the prompt's folder.

    Relative to the prompt file, not the working directory, so the pairing holds
    whatever folder ``lrd`` or CI runs from.
    """
    if prompt.dataset is None:
        return None
    return prompt_path.parent / prompt.dataset


__all__ = [
    "DATASET_SCHEMA_ID",
    "DATASET_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_ID",
    "CaseResult",
    "CategoryAccuracy",
    "CriterionVerdict",
    "Difficulty",
    "EvalRun",
    "FewShot",
    "GoldenCase",
    "GoldenDataset",
    "JudgeScore",
    "OutputFormat",
    "PredictionRecord",
    "PromptSpec",
    "RunSummary",
    "load_dataset",
    "load_predictions",
    "load_prompt",
    "prompt_dataset_path",
]
