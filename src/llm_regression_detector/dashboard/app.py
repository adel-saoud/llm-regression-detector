# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Streamlit dashboard — accuracy timeline, CI band, drift, side-by-side diff.

Reads the SQLite store written by the runner. Path is configurable via the
``LRD_DB_PATH`` env var (set automatically when launched via ``lrd dashboard``).

Type-checker note: pandas/plotly don't ship type stubs, and Streamlit's signature
typing leaks ``Unknown`` through DataFrame chains. Strict pyright is silenced
file-wide here — every other module remains fully strict.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from llm_regression_detector.diff.analyzer import Analyzer
from llm_regression_detector.diff.drift import analyse_drift
from llm_regression_detector.eval.dataset import EvalRun
from llm_regression_detector.storage.sqlite import SQLiteStorage

st.set_page_config(
    page_title="LLM Regression Detector",
    page_icon="📊",
    layout="wide",
)

DB_PATH = Path(os.environ.get("LRD_DB_PATH", "evals/runs.db"))


@st.cache_data(ttl=10)
def _load_recent(prompt_name: str | None, limit: int = 50) -> list[EvalRun]:
    async def _go() -> list[EvalRun]:
        storage = SQLiteStorage(DB_PATH)
        await storage.initialize()
        if prompt_name is None:
            return []
        return await storage.recent(prompt_name, limit=limit)

    return asyncio.run(_go())


@st.cache_data(ttl=10)
def _list_prompts() -> list[str]:
    import sqlite3

    if not DB_PATH.exists():
        return []
    with sqlite3.connect(DB_PATH) as db:
        rows = db.execute("SELECT DISTINCT prompt_name FROM eval_runs ORDER BY 1").fetchall()
    return [r[0] for r in rows]


def _runs_to_df(runs: list[EvalRun]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": r.run_id,
                "version": r.prompt_version,
                "timestamp": r.timestamp,
                "accuracy": r.summary.accuracy,
                "ci_low": r.summary.accuracy_ci_low,
                "ci_high": r.summary.accuracy_ci_high,
                "summary_score": r.summary.avg_summary_score,
                "p50_ms": r.summary.latency_p50_ms,
                "p95_ms": r.summary.latency_p95_ms,
                "p99_ms": r.summary.latency_p99_ms,
                "cost_usd": r.summary.estimated_cost_usd,
                "passed": r.summary.cases_passed,
                "total": r.summary.cases_total,
            }
            for r in runs
        ]
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("📊 LLM Regression Detector")
st.sidebar.caption(f"DB: `{DB_PATH}`")

prompts = _list_prompts()
if not prompts:
    st.title("No eval runs yet")
    st.markdown(
        "Run `uv run lrd run --prompt prompts/classifier_v1.yaml` to generate the first baseline."
    )
    st.stop()

prompt_name = st.sidebar.selectbox("Prompt", prompts)
runs = _load_recent(prompt_name)
df = _runs_to_df(runs)

# ── Header / KPIs ───────────────────────────────────────────────────────────

st.title(f"📊 {prompt_name}")
if df.empty:
    st.info("No runs for this prompt yet.")
    st.stop()

latest = df.iloc[0]
prev = df.iloc[1] if len(df) > 1 else None

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Accuracy",
    f"{latest['accuracy'] * 100:.1f}%",
    delta=(
        f"{(latest['accuracy'] - prev['accuracy']) * 100:+.2f} pp" if prev is not None else None
    ),
    help=f"95% Wilson CI {latest['ci_low'] * 100:.1f}–{latest['ci_high'] * 100:.1f}%",
)
col2.metric("Summary score", f"{latest['summary_score']:.2f} / 5")
col3.metric("Latency p50/p95", f"{latest['p50_ms']:.0f} / {latest['p95_ms']:.0f} ms")
col4.metric("Cases", f"{int(latest['passed'])}/{int(latest['total'])}")
col5.metric(
    "Est. cost",
    f"${latest['cost_usd']:.4f}" if latest["cost_usd"] > 0 else "—",
)

# ── Drift summary ───────────────────────────────────────────────────────────

if len(runs) >= 6:
    drift = analyse_drift(runs[1:], runs[0])
    if drift.has_drift:
        st.error(
            f"⚠ **Slow drift detected** — latest {drift.latest_accuracy * 100:.1f}% < "
            f"threshold {drift.threshold * 100:.1f}% "
            f"(MA {drift.moving_average * 100:.1f}%, σ {drift.std_dev * 100:.2f}%)"
        )
    else:
        st.caption(
            f"No slow drift over {drift.window_size} prior runs · "
            f"MA {drift.moving_average * 100:.1f}% ± {drift.std_dev * 100:.2f}%"
        )

# ── Accuracy timeline with CI band ──────────────────────────────────────────

st.subheader("Accuracy over time (95% CI band)")
df_sorted = df.sort_values("timestamp")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df_sorted["timestamp"],
        y=df_sorted["ci_high"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=df_sorted["timestamp"],
        y=df_sorted["ci_low"],
        mode="lines",
        line={"width": 0},
        fill="tonexty",
        fillcolor="rgba(88, 166, 255, 0.15)",
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=df_sorted["timestamp"],
        y=df_sorted["accuracy"],
        mode="lines+markers",
        name="accuracy",
        line={"color": "#58a6ff", "width": 2},
        customdata=df_sorted[["run_id", "version", "passed", "total"]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "version: %{customdata[1]}<br>"
            "accuracy: %{y:.1%}<br>"
            "passed: %{customdata[2]}/%{customdata[3]}<extra></extra>"
        ),
    )
)
fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
fig.update_layout(height=340, margin={"l": 0, "r": 0, "t": 20, "b": 0}, hovermode="x unified")
st.plotly_chart(fig, width="stretch")

# ── Latency percentile timeline ─────────────────────────────────────────────

with st.expander("Latency percentiles over time", expanded=False):
    fig_lat = px.line(
        df_sorted,
        x="timestamp",
        y=["p50_ms", "p95_ms", "p99_ms"],
        markers=True,
    )
    fig_lat.update_layout(height=280, margin={"l": 0, "r": 0, "t": 20, "b": 0})
    fig_lat.update_yaxes(title="ms")
    st.plotly_chart(fig_lat, width="stretch")

# ── Side-by-side diff ───────────────────────────────────────────────────────

st.subheader("Compare runs")
if len(runs) < 2:
    st.info("At least two runs are needed for a diff.")
else:
    col_a, col_b = st.columns(2)
    baseline_id = col_a.selectbox("Baseline", df["run_id"].iloc[1:].tolist())
    candidate_id = col_b.selectbox("Candidate", df["run_id"].tolist())

    runs_by_id = {r.run_id: r for r in runs}
    baseline = runs_by_id.get(baseline_id) if baseline_id else None
    candidate = runs_by_id.get(candidate_id) if candidate_id else None

    if baseline and candidate and baseline.run_id != candidate.run_id:
        diff = Analyzer().diff(baseline, candidate)
        delta_pp = diff.accuracy_delta * 100
        severity_colour = {"info": "🟢", "warning": "🟡", "critical": "🔴"}
        sig_marker = "" if diff.is_significant else " (within noise)"
        st.markdown(
            f"### {severity_colour[diff.severity.value]} {diff.severity.value.upper()} "
            f"— accuracy `{delta_pp:+.2f} pp`{sig_marker}"
        )

        # Per-category comparison
        if baseline.summary.per_category and candidate.summary.per_category:
            base_cats = {c.category: c.accuracy for c in baseline.summary.per_category}
            cand_cats = {c.category: c.accuracy for c in candidate.summary.per_category}
            cats_df = pd.DataFrame(
                [
                    {
                        "category": cat,
                        "baseline": base_cats.get(cat, 0.0),
                        "candidate": cand_cats.get(cat, 0.0),
                        "delta_pp": (cand_cats.get(cat, 0.0) - base_cats.get(cat, 0.0)) * 100,
                    }
                    for cat in sorted(base_cats.keys() | cand_cats.keys())
                ]
            )
            with st.expander("Per-category accuracy", expanded=True):
                st.dataframe(
                    cats_df.style.format(
                        {"baseline": "{:.1%}", "candidate": "{:.1%}", "delta_pp": "{:+.2f}"}
                    ),
                    width="stretch",
                )

        tab_reg, tab_imp, tab_all = st.tabs(
            [
                f"Regressions ({len(diff.regressions)})",
                f"Improvements ({len(diff.improvements)})",
                "All cases",
            ]
        )
        with tab_reg:
            if diff.regressions:
                st.dataframe(
                    pd.DataFrame([d.model_dump() for d in diff.regressions]),
                    width="stretch",
                )
            else:
                st.success("No regressions.")
        with tab_imp:
            if diff.improvements:
                st.dataframe(
                    pd.DataFrame([d.model_dump() for d in diff.improvements]),
                    width="stretch",
                )
            else:
                st.info("No improvements.")
        with tab_all:
            cand_df = pd.DataFrame(
                [
                    {
                        "case_id": r.case_id,
                        "passed": r.passed,
                        "predicted_category": r.predicted_category,
                        "summary_score": r.judge.summary_score,
                        "latency_ms": round(r.latency_ms, 1),
                    }
                    for r in candidate.results
                ]
            )
            st.dataframe(cand_df, width="stretch")
