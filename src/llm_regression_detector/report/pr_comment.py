"""GitHub-flavoured markdown formatter for the PR comment."""

from __future__ import annotations

from llm_regression_detector.diff.analyzer import DiffReport
from llm_regression_detector.diff.drift import DriftReport
from llm_regression_detector.eval.dataset import EvalRun
from llm_regression_detector.notify.base import AlertSeverity

_SEVERITY_HEADER: dict[AlertSeverity, str] = {
    AlertSeverity.INFO: "✅ **No regression detected**",
    AlertSeverity.WARNING: "⚠️ **Regression detected**",
    AlertSeverity.CRITICAL: "🚨 **Critical regression — merge blocked**",
}


def render_pr_comment(
    candidate: EvalRun,
    diff: DiffReport | None,
    *,
    drift: DriftReport | None = None,
    report_url: str | None = None,
) -> str:
    """Build the body of a GitHub PR comment summarising the eval run."""
    lines: list[str] = ["## LLM Regression Report", ""]

    if diff is None:
        lines.append("ℹ️ No baseline found for this prompt — recording as the new baseline.")
    else:
        lines.append(_SEVERITY_HEADER[diff.severity])
        lines.append("")
        delta_pp = diff.accuracy_delta * 100
        sig_note = "✓ statistically significant" if diff.is_significant else "⚠ within noise"
        lines.append("| Metric | Baseline | Candidate | Δ |")
        lines.append("| --- | --- | --- | --- |")
        baseline_acc = candidate.summary.accuracy - diff.accuracy_delta
        lines.append(
            f"| Accuracy | {baseline_acc * 100:.1f}% | "
            f"**{candidate.summary.accuracy * 100:.1f}%** | "
            f"`{delta_pp:+.2f} pp` ({sig_note}) |"
        )
        lines.append(
            f"| 95% CI (candidate) | — | "
            f"{candidate.summary.accuracy_ci_low * 100:.1f}–"
            f"{candidate.summary.accuracy_ci_high * 100:.1f}% | — |"
        )
        lines.append(
            f"| Cases passed | — | {candidate.summary.cases_passed}/"
            f"{candidate.summary.cases_total} | — |"
        )
        lines.append(
            f"| Avg summary score | — | {candidate.summary.avg_summary_score:.2f} / 5 | — |"
        )
        lines.append(
            f"| Latency p50 / p95 / p99 | — | "
            f"{candidate.summary.latency_p50_ms:.0f} / "
            f"{candidate.summary.latency_p95_ms:.0f} / "
            f"{candidate.summary.latency_p99_ms:.0f} ms | — |"
        )
        if candidate.summary.estimated_cost_usd > 0:
            lines.append(
                f"| Estimated cost | — | ${candidate.summary.estimated_cost_usd:.4f} | — |"
            )
        lines.append("")

        if candidate.summary.per_category:
            lines.append("### Per-category accuracy")
            lines.append("")
            lines.append("| Category | Accuracy | Passed |")
            lines.append("| --- | --- | --- |")
            for cat in candidate.summary.per_category:
                lines.append(
                    f"| `{cat.category}` | {cat.accuracy * 100:.1f}% | "
                    f"{cat.cases_passed}/{cat.cases_total} |"
                )
            lines.append("")

        if diff.regressions:
            lines.append(f"### Regressions ({len(diff.regressions)})")
            lines.append("")
            for r in diff.regressions[:15]:
                lines.append(f"- `{r.case_id}` — was PASS, now FAIL")
            if len(diff.regressions) > 15:
                lines.append(f"- _…and {len(diff.regressions) - 15} more_")
            lines.append("")

        if diff.improvements:
            lines.append(f"### Improvements ({len(diff.improvements)})")
            lines.append("")
            for i in diff.improvements[:10]:
                lines.append(f"- `{i.case_id}` — was FAIL, now PASS")
            lines.append("")

    if drift is not None and drift.window_size >= 5:
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
