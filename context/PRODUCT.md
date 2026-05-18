# Product Context

**Objective**: A CI/CD pipeline that catches LLM quality regressions before they reach production.

## Objective & Vision

- **Elevator pitch**: On every prompt change, run the new prompt against a golden dataset, score outputs with LLM-as-Judge, diff against a baseline using Wilson 95% CI, and block the merge if quality drops.
- **Long-term vision**: A self-contained, zero-cost eval framework that any ML team can drop into their CI pipeline.
- **Value proposition**: Statistically rigorous regression detection with a full audit trail — not just accuracy deltas.

## Target Audience

- **Usage context**: CLI-first (CI runners, local terminal); optional Streamlit dashboard for interactive exploration.
- **Primary users**: ML engineers iterating on prompts who need a CI gate against quality regressions.
- **Secondary users**: Contributors extending the golden dataset, adding new notifiers, or integrating new LLM providers.

## Features & Scope

- **Key features**: Eval runner, LLM-as-Judge scoring (majority vote), Wilson CI diff, severity classification, HTML report, PR comment, Slack/Discord/Google Chat alerts, SQLite history, Streamlit dashboard, slow-drift detection, `lrd init` scaffolder.
- **Out of scope**: Multi-tenant / SaaS, real-time monitoring, fine-tuning pipelines, Python < 3.11, paid API defaults.
- **Future roadmap**: Pairwise judging (head-to-head baseline vs candidate), bootstrap CI on summary scores, dataset snapshot versioning for reproducible runs.

## Business Logic & Rules

- **Core domain rules**: A run is CRITICAL only when Wilson 95% CIs do not overlap — not just on percentage-point delta alone.
- **Edge cases**: CRITICAL fires only when Wilson 95% CIs do not overlap — in practice ~30+ cases. Smaller runs always produce WARNING regardless of delta size.
- **Compliance / Privacy**: No user data in golden dataset. API keys via env vars only, never hardcoded.

## Success Criteria & Constraints

- **Usage limits**: $0 — default config runs entirely on free tiers (HF Inference, Gemini, Ollama) or the deterministic mock.
- **Success metrics**: `uv run pytest` passes in under 30s; eval on 50 cases completes in under 60s on mock.
- **Absolute constraints**: No paid model defaults. No `utils.py`. No bare `except Exception`.
