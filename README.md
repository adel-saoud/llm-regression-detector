# LLM Regression Detector

> **Automated quality gate for LLM prompts — catches regressions in CI before they reach users, with statistical rigour that won't fire false alarms.**

[![CI](https://github.com/adelsaoud/llm-regression-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/adelsaoud/llm-regression-detector/actions/workflows/ci.yml)
[![Eval](https://github.com/adelsaoud/llm-regression-detector/actions/workflows/eval.yml/badge.svg)](https://github.com/adelsaoud/llm-regression-detector/actions/workflows/eval.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/types-pyright%20strict%200%20errors-1f5fff)](https://github.com/microsoft/pyright)
[![Coverage](https://img.shields.io/badge/coverage-88%25-brightgreen)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![Demo: baseline 88.7% → degraded 60.4% → CRITICAL regression detected](demo.gif)

## What this is

> Only **52% of enterprises** run any form of evaluation on their LLM systems.
> *(LangChain, State of AI Agents 2026)*

When you ship an LLM-powered feature, you're constantly tweaking the prompt — adding examples, rephrasing instructions, adjusting tone. Every change *could* silently break quality. Most teams only find out when users complain.

**This project is a CI gate that catches those drops automatically.** On every pull request that touches a prompt, it:

1. Runs the new prompt against a hand-labelled golden dataset (53 real examples)
2. Scores each output with an LLM-as-Judge
3. Compares results against the stored baseline using **Wilson 95% confidence intervals**
4. Posts a regression report as a PR comment and Slack/Discord alert
5. **Exits non-zero** (blocking the merge) if the drop is statistically significant

Think of it as a test suite for your prompts — the same discipline as unit tests, applied to LLM quality.

> Inspired by the eval pipeline I built for [DaiLY at Decathlon France](#) (25K users,
> 98% accuracy in production). This is the same pattern, open-sourced as a standalone tool.

---

## Why it's non-trivial to build well

Most teams try this with a simple accuracy check and run into the same problems:

| Problem | How this project handles it |
|---|---|
| **Small datasets make raw % comparisons unreliable** | Wilson 95% CIs — if the intervals overlap, it's noise, not a regression. No false alarms. |
| **Aggregate accuracy hides category-level collapses** | Per-category breakdown surfaced in every report. |
| **Gradual drift isn't caught by PR-level diffs** | Slow-drift detector: moving-average band (`MA − k·σ`) over the last N runs flags slow decay. |
| **LLM judge calls are noisy** | Optional N-call majority vote (`LRD_JUDGE_CONSENSUS_N=3`) dampens judge variance at 3× cost. |
| **Webhook delivery fails silently on transient errors** | `tenacity` retry with exponential backoff + jitter on every platform. |
| **Adding a provider locks you in** | `litellm` Router — swap any model with one env var, zero code changes. |

---

## See it in action

The repo ships two prompts on purpose:

- `classifier_v1.yaml` — with few-shot examples → **88.7% accuracy**
- `classifier_v2_degraded.yaml` — examples stripped → **60.4% accuracy**

Run the self-eval demo (no API keys needed, ~3 seconds):

```bash
git clone https://github.com/adelsaoud/llm-regression-detector
cd llm-regression-detector
uv sync --all-extras
rm -f evals/runs.db

# baseline
uv run lrd run -p prompts/classifier_v1.yaml --no-diff --no-notify

# degraded candidate — fires CRITICAL
uv run lrd run -p prompts/classifier_v2_degraded.yaml --no-notify
```

Expected result:

```
▶ baseline (v1)   88.7%  (95% CI 77.4–94.7%)
▶ candidate (v2)  60.4%  (95% CI 46.9–72.4%)

CRITICAL · accuracy -28.30 pp significant · regressions=16 · improvements=1
CRITICAL regression — exiting non-zero.
```

The CIs don't overlap (94.7% vs 72.4%) so severity is `CRITICAL · significant` — not a warning, not noise.
The per-category breakdown shows *where* it broke: `billing` −42 pp, `account` −41 pp, `technical` −35 pp,
while `general` (+7 pp) would mask all three in the aggregate alone.

> These numbers come from the deterministic mock provider (no API key required). Real models
> produce similar shapes; exact values vary — which is why the system reports statistical
> significance rather than raw deltas.

---

## Run it for free

You need a Hugging Face account (no credit card). That's it.

```bash
# 1. clone & install
git clone https://github.com/adelsaoud/llm-regression-detector
cd llm-regression-detector
uv sync --all-extras

# 2. add your free HF token (https://huggingface.co/settings/tokens)
cp .env.example .env
# edit .env and set HF_TOKEN=hf_...

# 3. run a baseline
uv run lrd run --prompt prompts/classifier_v1.yaml --report evals/report.html

# 4. open the dashboard
uv run lrd dashboard
```

**No API key at all?** The project ships a deterministic mock provider — the full pipeline
runs offline with zero network calls. Every test in the suite uses it.

**Prefer local models?** Install [Ollama](https://ollama.com), pull `llama3.2:3b`, skip step 2.

---

## What's inside

```
src/llm_regression_detector/
  config.py          env-driven settings (pydantic-settings)
  llm/               provider-agnostic client — litellm Router + deterministic mock
  eval/              async runner · LLM-as-Judge · Wilson CI · percentiles · slow-drift
  diff/              CI-aware regression detector + severity logic
  notify/            Slack · Google Chat · Discord · generic JSON — shared retry policy
  storage/           SQLite run history with schema versioning + forward migration
  report/            HTML report (Jinja2) + GitHub PR comment (markdown)
  dashboard/         Streamlit dashboard — timeline, CI bands, drift chart
  cli.py             `lrd run` · `lrd diff` · `lrd report` · `lrd dashboard`

prompts/             versioned prompt YAMLs — the "code" under test
golden_dataset/      53 hand-labelled examples across 4 categories
tests/               79 tests · 88% coverage · fully hermetic (no network, no keys)
.github/workflows/   ci.yml (lint + type + tests) · eval.yml (runs on prompt changes)
```

Full architecture diagram and module map → [`docs/architecture.md`](docs/architecture.md)

---

## Tech stack

| | |
|---|---|
| **Python 3.11+**, `uv`, `hatchling` | Modern packaging — installs in seconds |
| **`litellm` Router** | One API for 100+ models. HF → Gemini → Ollama → mock fallback chain |
| **`pydantic` v2** | Runtime-validated models everywhere; `frozen=True, extra="forbid"` |
| **`typer` + `rich`** | Ergonomic CLI with pretty tables |
| **`aiosqlite`** | Async SQLite for run history; schema-versioned with forward migration |
| **`tenacity`** | Exponential backoff + jitter on webhook delivery |
| **`structlog`** | Structured JSON logs; run-scoped context via `bind()` |
| **`streamlit` + `plotly`** | Dashboard with accuracy timeline and CI band charts |
| **`ruff`** | Lint + format in one tool |
| **`pyright` strict** | 0 errors, 0 warnings — full type coverage including tests |

---

## Development

```bash
uv sync --all-extras
uv run pre-commit install

uv run ruff check --fix .   # lint + autofix
uv run ruff format .        # format
uv run pyright              # type-check (strict, 0 errors)
uv run pytest               # 79 tests, 88% coverage, gate at 85%
```

---

## Honest limitations

- **Judge variance is dampened, not eliminated.** Majority vote at `consensus_n=3` helps; pairwise judging ("is A better than B?") would be the next tier — not implemented.
- **Binary CI only.** Wilson interval covers pass/fail. Paired-bootstrap on the summary score (1–5) would give a tighter signal — not implemented.
- **53 cases catches large regressions; subtle ones (≤5 pp) need 200+.** The CIs need room to separate cleanly. Documented; not pretending otherwise.
- **No adversarial robustness.** This evaluates classifier quality, not resistance to prompt injection.
- **Free-tier rate limits.** Sustained bursts may need a paid tier; the Router retries with backoff but can't conjure quota.

---

## License

[MIT](LICENSE) — use it, fork it, ship it.
