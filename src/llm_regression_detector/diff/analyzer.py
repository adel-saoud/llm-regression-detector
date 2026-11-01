"""Baseline-vs-candidate comparison.

Detects regressions (cases that flipped pass→fail) and improvements (fail→pass),
computes a signed accuracy delta, and maps the result to an ``AlertSeverity`` via
configurable thresholds.

Severity is CI-aware: when the 95% Wilson confidence intervals of baseline and
candidate overlap, a drop that would be CRITICAL is downgraded to WARNING,
because it may be judge or sampling noise rather than a real regression. This
is what keeps a small dataset from failing CI on noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from llm_regression_detector.eval.dataset import CaseResult, EvalRun
from llm_regression_detector.eval.stats import WilsonInterval, intervals_overlap
from llm_regression_detector.notify.base import AlertPayload, AlertSeverity


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Accuracy changes at which severity escalates, as fractions (``0.03`` is 3 points).

    Severity scores the drop only, so a gain never escalates by itself:

    - a drop of at least ``critical_pp`` is CRITICAL when the intervals separate,
      WARNING when they overlap;
    - a drop of at least ``warning_pp`` is WARNING;
    - any case that flips from pass to fail is at least WARNING, even when the net
      delta is a gain, because a net gain can hide individual regressions;
    - a gain of at least ``critical_pp`` stays INFO but sets ``DiffReport.large_gain``,
      since a jump that size more often means a judge or dataset change than a
      better prompt.
    """

    warning_pp: float = 0.03
    critical_pp: float = 0.08


class CaseDelta(BaseModel):
    """How one case changed between baseline and candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    case_id: str
    baseline_passed: bool
    candidate_passed: bool
    baseline_summary_score: int
    candidate_summary_score: int

    @property
    def is_regression(self) -> bool:
        """Passed in the baseline, fails in the candidate."""
        return self.baseline_passed and not self.candidate_passed

    @property
    def is_improvement(self) -> bool:
        """Failed in the baseline, passes in the candidate."""
        return not self.baseline_passed and self.candidate_passed


class DiffReport(BaseModel):
    """Full structured comparison between two ``EvalRun`` instances."""

    model_config = ConfigDict(extra="forbid")

    baseline_run_id: str
    candidate_run_id: str
    accuracy_delta: float = Field(description="Signed (candidate - baseline)")
    is_significant: bool = Field(
        description="True iff baseline & candidate Wilson 95% CIs do not overlap"
    )
    severity: AlertSeverity
    regressions: list[CaseDelta] = Field(default_factory=list[CaseDelta])
    improvements: list[CaseDelta] = Field(default_factory=list[CaseDelta])
    unchanged: list[CaseDelta] = Field(default_factory=list[CaseDelta])

    large_gain: bool = Field(
        default=False,
        description="Accuracy rose by at least critical_pp: check the judge or the dataset",
    )


class Analyzer:
    """Pure functions over two ``EvalRun``s: no I/O, easy to test."""

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self._thresholds = thresholds or Thresholds()

    def diff(self, baseline: EvalRun, candidate: EvalRun) -> DiffReport:
        """Compare case by case on the ids both runs share, then grade the accuracy delta."""
        baseline_index = {r.case_id: r for r in baseline.results}
        candidate_index = {r.case_id: r for r in candidate.results}
        shared_ids = sorted(baseline_index.keys() & candidate_index.keys())

        regressions: list[CaseDelta] = []
        improvements: list[CaseDelta] = []
        unchanged: list[CaseDelta] = []

        for case_id in shared_ids:
            delta = _build_case_delta(baseline_index[case_id], candidate_index[case_id])
            if delta.is_regression:
                regressions.append(delta)
            elif delta.is_improvement:
                improvements.append(delta)
            else:
                unchanged.append(delta)

        accuracy_delta = candidate.summary.accuracy - baseline.summary.accuracy
        baseline_ci = WilsonInterval(
            point=baseline.summary.accuracy,
            low=baseline.summary.accuracy_ci_low,
            high=baseline.summary.accuracy_ci_high,
        )
        candidate_ci = WilsonInterval(
            point=candidate.summary.accuracy,
            low=candidate.summary.accuracy_ci_low,
            high=candidate.summary.accuracy_ci_high,
        )
        is_significant = not intervals_overlap(baseline_ci, candidate_ci)
        severity = self._severity(
            accuracy_delta,
            has_regressions=bool(regressions),
            is_significant=is_significant,
        )

        return DiffReport(
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            accuracy_delta=accuracy_delta,
            is_significant=is_significant,
            severity=severity,
            large_gain=accuracy_delta >= self._thresholds.critical_pp,
            regressions=regressions,
            improvements=improvements,
            unchanged=unchanged,
        )

    def _severity(
        self,
        accuracy_delta: float,
        *,
        has_regressions: bool,
        is_significant: bool,
    ) -> AlertSeverity:
        """Map the drop to a severity; see ``Thresholds`` for the rules."""
        drop = max(0.0, -accuracy_delta)
        if drop >= self._thresholds.critical_pp:
            return AlertSeverity.CRITICAL if is_significant else AlertSeverity.WARNING
        if drop >= self._thresholds.warning_pp or has_regressions:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO


def _build_case_delta(baseline: CaseResult, candidate: CaseResult) -> CaseDelta:
    """Pass state and quality score of one case in both runs."""
    return CaseDelta(
        case_id=baseline.case_id,
        baseline_passed=baseline.passed,
        candidate_passed=candidate.passed,
        baseline_summary_score=baseline.judge.summary_score,
        candidate_summary_score=candidate.judge.summary_score,
    )


def to_alert_payload(report: DiffReport, *, report_url: str | None = None) -> AlertPayload:
    """Convert a structured diff into a platform-neutral webhook alert."""
    delta_pp = report.accuracy_delta * 100
    significance_marker = "" if report.is_significant else " (within noise)"
    if report.severity == AlertSeverity.CRITICAL:
        title = f"🚨 Critical regression: accuracy {delta_pp:+.2f} pp{significance_marker}"
    elif report.severity == AlertSeverity.WARNING:
        title = f"⚠️  Regression detected: accuracy {delta_pp:+.2f} pp{significance_marker}"
    elif report.large_gain:
        title = f"🔎 Large gain: accuracy {delta_pp:+.2f} pp, check the judge or dataset"
    else:
        title = f"✅ No regression: accuracy {delta_pp:+.2f} pp"

    summary = (
        f"{len(report.regressions)} regression(s), {len(report.improvements)} improvement(s) "
        f"vs baseline `{report.baseline_run_id}`. "
        f"Wilson CI{'s do not overlap' if report.is_significant else 's overlap (noise possible)'}."
    )

    return AlertPayload(
        severity=report.severity,
        title=title,
        summary=summary,
        accuracy_delta=report.accuracy_delta,
        regressions=[r.case_id for r in report.regressions],
        improvements=[i.case_id for i in report.improvements],
        report_url=report_url,
        run_id=report.candidate_run_id,
    )
