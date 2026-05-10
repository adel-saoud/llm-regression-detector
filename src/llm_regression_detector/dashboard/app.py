"""Streamlit dashboard — accuracy timeline, CI band, drift, side-by-side diff.

Reads the SQLite store written by the runner. Path is configurable via the
``LRD_DB_PATH`` env var (set automatically when launched via ``lrd dashboard``).

This file is excluded from pyright strict mode (see pyproject.toml [tool.pyright])
because pandas, plotly, and streamlit don't ship type stubs.
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
from llm_regression_detector.diff.drift import analyze_drift
from llm_regression_detector.eval.dataset import EvalRun
from llm_regression_detector.storage.sqlite import SQLiteStorage

st.set_page_config(
    page_title="LLM Regression Detector",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Remove Streamlit chrome (footer, top deploy bar) */
    footer { visibility: hidden; }
    [data-testid="stHeader"] { display: none; }

    /* Main content area — breathing room */
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* KPI cards — subtle blue-tinted card */
    [data-testid="stMetric"] {
        background: rgba(88, 166, 255, 0.05);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }

    /* KPI label — uppercase small-caps style */
    [data-testid="stMetricLabel"] p {
        font-size: 0.68rem !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #8b95a5 !important;
    }

    /* KPI value — tighter tracking, bolder */
    [data-testid="stMetricValue"] p {
        font-size: 1.9rem !important;
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }

    /* Section subheaders — clean weight hierarchy, no decoration */
    h3 {
        font-size: 0.95rem !important;
        font-weight: 700;
        letter-spacing: 0.01em;
        text-transform: uppercase;
        opacity: 0.55;
        margin-top: 0.25rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Caption text — better line-height */
    [data-testid="stCaptionContainer"] p { line-height: 1.6; }

    /* Sidebar — subtle tint that works in both light and dark mode */
    [data-testid="stSidebar"] { background-color: rgba(0, 0, 0, 0.04); }
    [data-testid="stSidebar"] > div:first-child { background-color: transparent; }
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.03); }
    }

    /* Sidebar nav links — clean list */
    [data-testid="stSidebarUserContent"] a {
        text-decoration: none;
        color: #4b91e2;
        font-size: 0.85rem;
    }
    [data-testid="stSidebarUserContent"] a:hover { text-decoration: underline; }
    </style>
    """,
    unsafe_allow_html=True,
)

DB_PATH = Path(os.environ.get("LRD_DB_PATH", "evals/runs.db"))


_CASE_CATEGORY = {"b": "Billing", "a": "Account", "t": "Technical", "g": "General"}


def _humanize(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def _expand_case_id(case_id: str) -> str:
    """b001 → 'Billing #1', a012 → 'Account #12', unknown prefix → unchanged."""
    prefix = case_id[:1].lower()
    num = case_id[1:].lstrip("0") or "0"
    label = _CASE_CATEGORY.get(prefix, "")
    return f"{label} #{num}" if label else case_id


def _short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[:n].rstrip() + "…"


def _compact_tokens(n: int) -> str:
    """Format a token count to fit comfortably inside a KPI card."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M tok"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K tok"
    return f"{n} tok"


def _severity_card(severity: str, delta_pp: float, sig_marker: str) -> None:
    color = {"info": "#2da44e", "warning": "#9a6700", "critical": "#cf222e"}[severity]
    icon = {"info": "🟢", "warning": "🟡", "critical": "🔴"}[severity]
    direction = "drop" if delta_pp < 0 else "gain"
    st.markdown(
        f'<div style="background:{color}18;border:1px solid {color}44;border-radius:10px;'
        f'padding:12px 16px;margin:10px 0;">'
        f'<span style="font-size:1.15rem;font-weight:700;color:{color};">'
        f"{icon} {severity.upper()}</span>"
        f'<span style="font-size:0.95rem;opacity:0.85;"> — '
        f"{abs(delta_pp):.1f} % pts {direction}{sig_marker}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


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
                "prompt_tokens": r.summary.total_prompt_tokens,
                "completion_tokens": r.summary.total_completion_tokens,
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

prompt_name = st.sidebar.selectbox("Select a prompt", prompts, format_func=_humanize)
runs = _load_recent(prompt_name)
df = _runs_to_df(runs)

# Sidebar status badge — show latest diff result at a glance
if len(runs) >= 2 and not df.empty:
    _status_diff = Analyzer().diff(runs[1], runs[0])
    _delta = _status_diff.accuracy_delta * 100
    _sev = _status_diff.severity.value
    _icon = {"info": "🟢", "warning": "🟡", "critical": "🔴"}[_sev]
    _color = {"info": "#2da44e", "warning": "#9a6700", "critical": "#cf222e"}[_sev]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Latest change**")
    st.sidebar.markdown(
        f'<span style="color:{_color};font-weight:700;">{_icon} {_sev.upper()}</span> '
        f"· `{_delta:+.1f} % pts`  \n"
        f'<span style="font-size:0.8rem;color:#6b7280;">'
        f"{runs[0].prompt_version} vs {runs[1].prompt_version}</span>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Navigate**  \n"
    "[📊 Overview](#overview)  \n"
    "[📈 Accuracy history](#accuracy-history)  \n"
    "[🔍 Version comparison](#version-comparison)"
)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown('<a id="overview"></a>', unsafe_allow_html=True)
st.title(f"📊 {_humanize(prompt_name)}")

if df.empty:
    st.info("No runs for this prompt yet.")
    st.stop()

latest = df.iloc[0]
prev = df.iloc[1] if len(df) > 1 else None

st.caption(
    f"Prompt ID: `{prompt_name}` · "
    f"{int(latest['total'])} test cases · "
    f"{len(runs)} run{'s' if len(runs) != 1 else ''} recorded"
)

# ── KPIs ────────────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Accuracy",
    f"{latest['accuracy'] * 100:.1f}%",
    delta=(
        f"{(latest['accuracy'] - prev['accuracy']) * 100:+.1f} % pts" if prev is not None else None
    ),
    help=(
        f"Share of test cases the model answered correctly · "
        f"95% confidence range: {latest['ci_low'] * 100:.1f}–{latest['ci_high'] * 100:.1f}% · "
        "delta vs previous run"
    ),
)
col2.metric(
    "Quality score",
    f"{latest['summary_score']:.2f} / 5",
    help=(
        "An AI judge grades each answer from 1 (wrong) to 5 (perfect). "
        "This is the average across all test cases."
    ),
)
total_tokens = int(latest["prompt_tokens"]) + int(latest["completion_tokens"])
col3.metric(
    "Median latency",
    f"{latest['p50_ms']:.0f} ms",
    help=(
        f"Median (p50) response time per LLM call · "
        f"p95: {latest['p95_ms']:.0f} ms · "
        f"p99: {latest['p99_ms']:.0f} ms"
    ),
)
col4.metric(
    "Tests passed",
    f"{int(latest['passed'])} / {int(latest['total'])}",
    help="Number of test cases where the AI's prediction matched the expected answer",
)
col5.metric(
    "Run cost",
    f"${latest['cost_usd']:.4f}" if latest["cost_usd"] > 0 else _compact_tokens(total_tokens),
    delta="$0 · free tier" if latest["cost_usd"] == 0 else None,
    delta_color="off",
    help=(
        "Estimated API cost based on token counts × $/M rates. "
        f"This run used {int(latest['prompt_tokens']):,} input + "
        f"{int(latest['completion_tokens']):,} output tokens. "
        "Set LRD_COST_INPUT_USD / LRD_COST_OUTPUT_USD to see real spend."
    ),
)

# ── Drift summary ───────────────────────────────────────────────────────────

if len(runs) >= 6:
    drift = analyze_drift(runs[1:], runs[0])
    if drift.has_drift:
        st.error(
            f"⚠ **Gradual quality drop detected** — "
            f"current accuracy {drift.latest_accuracy * 100:.1f}% "
            f"has fallen below the expected floor of {drift.threshold * 100:.1f}% "
            f"(average over the last {drift.window_size} runs: "
            f"{drift.moving_average * 100:.1f}%)"
        )
    else:
        st.caption(
            f"✓ Quality stable — no gradual decline detected across the last "
            f"{drift.window_size} runs "
            f"(avg {drift.moving_average * 100:.1f}%, "
            f"±{drift.std_dev * 100:.1f}% variation)"
        )

# ── Two-column body: Accuracy chart | Version comparison ────────────────────

st.markdown(
    "<hr style='margin:1.5rem 0 1.25rem;border:none;border-top:1px solid rgba(128,128,128,0.25);'>",
    unsafe_allow_html=True,
)
col_left, col_right = st.columns([3, 2], gap="large")

# ── Left: Accuracy history chart ────────────────────────────────────────────

with col_left:
    st.subheader("Accuracy history")
    st.caption(
        "Each point is one eval run. The shaded band is the 95% confidence range — "
        "if two bands overlap, the difference is within statistical noise."
    )
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    x_labels = df_sorted["version"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=df_sorted["ci_high"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
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
            x=x_labels,
            y=df_sorted["accuracy"],
            mode="lines+markers",
            name="accuracy",
            line={"color": "#58a6ff", "width": 2},
            marker={"size": 8},
            customdata=df_sorted[["run_id", "version", "passed", "total", "timestamp"]],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "accuracy: %{y:.1%}<br>"
                "passed: %{customdata[2]}/%{customdata[3]}<br>"
                "run: %{customdata[0]}<br>"
                "time: %{customdata[4]|%Y-%m-%d %H:%M}<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    fig.update_xaxes(type="category")
    fig.update_layout(height=320, margin={"l": 0, "r": 0, "t": 20, "b": 0}, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    with st.expander("Response time history", expanded=False):
        lat_df = pd.DataFrame(
            {
                "version": df_sorted["version"],
                "p50 (median)": df_sorted["p50_ms"],
                "p95": df_sorted["p95_ms"],
                "p99": df_sorted["p99_ms"],
            }
        )
        fig_lat = px.line(
            lat_df,
            x="version",
            y=["p50 (median)", "p95", "p99"],
            markers=True,
            labels={
                "version": "Version",
                "value": "Response time (ms)",
                "variable": "Percentile",
            },
        )
        fig_lat.update_xaxes(type="category")
        fig_lat.update_layout(height=260, margin={"l": 0, "r": 0, "t": 20, "b": 0})
        fig_lat.update_yaxes(title="ms")
        st.plotly_chart(fig_lat, width="stretch")

# ── Right: Version comparison panel ─────────────────────────────────────────

with col_right:
    st.subheader("Version comparison")

    if len(runs) < 2:
        st.info("At least two runs are needed for a comparison.")
    else:
        version_label = {
            r.run_id: f"{r.prompt_version}  ·  {r.timestamp.strftime('%b %d, %H:%M')}" for r in runs
        }

        baseline_id = st.selectbox(
            "Before (baseline)",
            df["run_id"].iloc[1:].tolist(),
            format_func=lambda rid: version_label.get(str(rid)) or str(rid),
        )
        candidate_id = st.selectbox(
            "After (candidate)",
            df["run_id"].tolist(),
            format_func=lambda rid: version_label.get(str(rid)) or str(rid),
        )

        runs_by_id = {r.run_id: r for r in runs}
        baseline = runs_by_id.get(baseline_id) if baseline_id else None
        candidate = runs_by_id.get(candidate_id) if candidate_id else None

        if baseline and candidate and baseline.run_id != candidate.run_id:
            diff = Analyzer().diff(baseline, candidate)
            delta_pp = diff.accuracy_delta * 100
            sig_marker = "" if diff.is_significant else " (within noise)"
            _severity_card(diff.severity.value, delta_pp, sig_marker)
            _cand_summary = {r.case_id: r.predicted_summary for r in candidate.results}

            # Per-category breakdown
            if baseline.summary.per_category and candidate.summary.per_category:
                base_cats = {c.category: c.accuracy for c in baseline.summary.per_category}
                cand_cats = {c.category: c.accuracy for c in candidate.summary.per_category}
                cats_df = pd.DataFrame(
                    [
                        {
                            "Category": cat,
                            "Before": base_cats.get(cat, 0.0),
                            "After": cand_cats.get(cat, 0.0),
                            "Change (% pts)": (cand_cats.get(cat, 0.0) - base_cats.get(cat, 0.0))
                            * 100,
                        }
                        for cat in sorted(base_cats.keys() | cand_cats.keys())
                    ]
                )
                with st.expander("Accuracy by category", expanded=True):
                    st.dataframe(
                        cats_df.style.format(
                            {
                                "Before": "{:.1%}",
                                "After": "{:.1%}",
                                "Change (% pts)": "{:+.1f}",
                            }
                        ),
                        width="stretch",
                    )

            tab_reg, tab_imp, tab_all = st.tabs(
                [
                    f"⬇ Got worse ({len(diff.regressions)})",
                    f"⬆ Got better ({len(diff.improvements)})",
                    "All test cases",
                ]
            )
            with tab_reg:
                if diff.regressions:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Case": _expand_case_id(d.case_id),
                                    "Topic": _short(_cand_summary.get(d.case_id, "")),
                                    "Score before": d.baseline_summary_score,
                                    "Score after": d.candidate_summary_score,
                                    "Change": (
                                        d.candidate_summary_score - d.baseline_summary_score
                                    ),
                                }
                                for d in diff.regressions
                            ]
                        ),
                        width="stretch",
                    )
                else:
                    st.success("No regressions — quality held across all test cases.")
            with tab_imp:
                if diff.improvements:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Case": _expand_case_id(d.case_id),
                                    "Topic": _short(_cand_summary.get(d.case_id, "")),
                                    "Score before": d.baseline_summary_score,
                                    "Score after": d.candidate_summary_score,
                                    "Change": (
                                        d.candidate_summary_score - d.baseline_summary_score
                                    ),
                                }
                                for d in diff.improvements
                            ]
                        ),
                        width="stretch",
                    )
                else:
                    st.info("No improvements detected.")
            with tab_all:
                if candidate:
                    cand_df = pd.DataFrame(
                        [
                            {
                                "Case": _expand_case_id(r.case_id),
                                "Topic": _short(r.predicted_summary),
                                "Result": "✓ Pass" if r.passed else "✗ Fail",
                                "Expected": _CASE_CATEGORY.get(r.case_id[:1].lower(), "?"),
                                "AI prediction": r.predicted_category.title(),
                                "Score": r.judge.summary_score,
                                "Latency (ms)": round(r.latency_ms, 1),
                            }
                            for r in candidate.results
                        ]
                    )
                    st.dataframe(cand_df, width="stretch")
