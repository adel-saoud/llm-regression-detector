"""GitHub-flavoured markdown formatter for the PR comment."""

from __future__ import annotations

from llm_regression_detector.diff.analyzer import DiffReport
from llm_regression_detector.diff.drift import DriftReport
from llm_regression_detector.eval.dataset import EvalRun
from llm_regression_detector.eval.stats import MIN_DRIFT_HISTORY
from llm_regression_detector.notify.base import AlertSeverity

_MAX_LISTED_REGRESSIONS = 15
"""Regressed case ids listed before "and N more", so a broken prompt keeps the comment short."""

_MAX_LISTED_IMPROVEMENTS = 10
"""Improved case ids listed; improvements matter less than regressions in review."""

_LARGE_GAIN_NOTE = (
    "🔎 **Large gain: check the judge or dataset.** A jump this size more often comes "
    "from a changed judge model or dataset than from a better prompt."
)
"""Shown when accuracy rises by at least ``Thresholds.critical_pp``; severity stays INFO."""

_SEVERITY_HEADER: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: "✅ **No regression detected**",
    AlertSeverity.WARNING: "⚠️ **Regression detected**",
    AlertSeverity.CRITICAL: "🚨 **Critical regression: merge blocked**",
}


def render_pr_comment(
    candidate: EvalRun,
    diff: DiffReport | None,
    *,
    drift: DriftReport | None = None,
    report_url: str | None = None,
) -> str:
    """Build the body of a GitHub PR comment summarising the eval run.

    Without a baseline the comment only says the run becomes the baseline; the
    drift section appears once enough history exists for the check to run.
    """
    lines: list[str] = ["## LLM Regression Report", ""]

    if diff is None:
        lines.append("ℹ️ No baseline found for this prompt: recording as the new baseline.")
    else:
        lines.append(_SEVERITY_HEADER[diff.severity])
        lines.append("")
        if diff.large_gain:
            lines.append(_LARGE_GAIN_NOTE)
            lines.append("")
        delta_pp = diff.accuracy_delta * 100
        significance_note = (
            "✓ statistically significant" if diff.is_significant else "⚠ within noise"
        )
        lines.append("| Metric | Baseline | Candidate | Δ |")
        lines.append("| --- | --- | --- | --- |")
        baseline_accuracy = candidate.summary.accuracy - diff.accuracy_delta
        lines.append(
            f"| Accuracy | {baseline_accuracy * 100:.1f}% | "
            f"**{candidate.summary.accuracy * 100:.1f}%** | "
            f"`{delta_pp:+.2f} pp` ({significance_note}) |"
        )
        lines.append(
            f"| 95% CI (candidate) | n/a | "
            f"{candidate.summary.accuracy_ci_low * 100:.1f}–"
            f"{candidate.summary.accuracy_ci_high * 100:.1f}% | n/a |"
        )
        lines.append(
            f"| Cases passed | n/a | {candidate.summary.cases_passed}/"
            f"{candidate.summary.cases_total} | n/a |"
        )
        lines.append(
            f"| Avg summary score | n/a | {candidate.summary.avg_summary_score:.2f} / 5 | n/a |"
        )
        lines.append(
            f"| Latency p50 / p95 / p99 | n/a | "
            f"{candidate.summary.latency_p50_ms:.0f} / "
            f"{candidate.summary.latency_p95_ms:.0f} / "
            f"{candidate.summary.latency_p99_ms:.0f} ms | n/a |"
        )
        if candidate.summary.estimated_cost_usd > 0:
            lines.append(
                f"| Estimated cost | n/a | ${candidate.summary.estimated_cost_usd:.4f} | n/a |"
            )
        lines.append("")

        if candidate.summary.per_category:
            lines.append("### Per-category accuracy")
            lines.append("")
            lines.append("| Category | Accuracy | Passed |")
            lines.append("| --- | --- | --- |")
            for category in candidate.summary.per_category:
                lines.append(
                    f"| `{category.category}` | {category.accuracy * 100:.1f}% | "
                    f"{category.cases_passed}/{category.cases_total} |"
                )
            lines.append("")

        if diff.regressions:
            lines.append(f"### Regressions ({len(diff.regressions)})")
            lines.append("")
            for regression in diff.regressions[:_MAX_LISTED_REGRESSIONS]:
                lines.append(f"- `{regression.case_id}`: was PASS, now FAIL")
            if len(diff.regressions) > _MAX_LISTED_REGRESSIONS:
                hidden_count = len(diff.regressions) - _MAX_LISTED_REGRESSIONS
                lines.append(f"- _…and {hidden_count} more_")
            lines.append("")

        if diff.improvements:
            lines.append(f"### Improvements ({len(diff.improvements)})")
            lines.append("")
            for improvement in diff.improvements[:_MAX_LISTED_IMPROVEMENTS]:
                lines.append(f"- `{improvement.case_id}`: was FAIL, now PASS")
            lines.append("")

    if drift is not None and drift.window_size >= MIN_DRIFT_HISTORY:
        lines.append("### Slow-drift check")
        lines.append("")
        if drift.has_drift:
            lines.append(
                f"⚠ Latest accuracy ({drift.latest_accuracy * 100:.1f}%) is below the "
                f"`MA − 2σ` band ({drift.threshold * 100:.1f}%) over "
                f"{drift.window_size} prior runs."
            )
        else:
            lines.append(
                f"✓ No drift over {drift.window_size} prior runs · "
                f"MA {drift.moving_average * 100:.1f}% ± {drift.std_dev * 100:.2f}%."
            )
        lines.append("")

    if report_url:
        lines.append(f"[📊 View full report]({report_url})")
        lines.append("")

    lines.append(
        f"<sub>Run `{candidate.run_id}` · prompt "
        f"`{candidate.prompt_name}@{candidate.prompt_version}` · "
        f"{candidate.timestamp.strftime('%Y-%m-%d %H:%M UTC')}</sub>"
    )
    return "\n".join(lines)
