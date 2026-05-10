"""Slow-drift detector — flags accuracy declining gradually over many runs.

Single PRs catch sudden regressions. Slow drift (a slow tightening of the
prompt over weeks, or a model provider quietly retraining) is what eval
pipelines miss. This module exposes a thin wrapper around the statistical
helper so the CLI / dashboard can report drift in addition to PR-level diffs.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from llm_regression_detector.eval.dataset import EvalRun
from llm_regression_detector.eval.stats import detect_slow_drift


class DriftReport(BaseModel):
    """Output of running the slow-drift check over a window of stored runs."""

    model_config = ConfigDict(extra="forbid")

    has_drift: bool
    moving_average: float = Field(ge=0.0, le=1.0)
    std_dev: float = Field(ge=0.0)
    threshold: float = Field(description="Acceptable lower bound (MA - k·σ)")
    latest_accuracy: float = Field(ge=0.0, le=1.0)
    window_size: int = Field(ge=0)


def analyze_drift(
    historical: Sequence[EvalRun],
    latest: EvalRun,
    *,
    k_sigma: float = 2.0,
) -> DriftReport:
    """Run the drift check over a window of past ``EvalRun``s.

    ``historical`` should be the runs **before** ``latest`` (newest first or
    oldest first — order doesn't affect the moving-average computation).
    """
    accuracies = [r.summary.accuracy for r in historical]
    signal = detect_slow_drift(accuracies, latest.summary.accuracy, k_sigma=k_sigma)
    return DriftReport(
        has_drift=signal.has_drift,
        moving_average=signal.moving_average,
        std_dev=signal.std_dev,
        threshold=signal.threshold,
        latest_accuracy=signal.latest,
        window_size=len(accuracies),
    )
