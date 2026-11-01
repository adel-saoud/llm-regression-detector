# Project structure

```
src/llm_regression_detector/
├── config.py          Settings, all read from environment variables
├── llm/               litellm Router client and the deterministic mock
├── eval/              Dataset models and loaders, runner, judge, Wilson CI, percentiles
├── diff/              CI-aware regression analyzer and slow-drift detector
├── notify/            Slack, Google Chat, Discord and generic webhooks, one retry policy
├── storage/           SQLite run history, schema-versioned with forward migration
├── report/            HTML report (Jinja2 template inside the package) and PR comment
├── dashboard/         Streamlit dashboard, shipped in the wheel, launched by `lrd dashboard`
├── ci.py              Which prompts a change set affects (used by eval.yml)
└── cli.py             lrd run, score, diff, report, pr-comment, init, affected-prompts, schema, dashboard

prompts/               Versioned prompt YAMLs, the "code" under test. Each names its dataset.
prompts/demo/          The deliberately degraded prompt for the demo, never picked by CI
golden_dataset/        53 hand-labelled incident cases (schema v1), 8 free-text support cases (v2)
schemas/               JSON Schemas of a v2 dataset file and of a predictions line (`lrd schema`)
scripts/               ci_eval_affected.sh (eval.yml) and self_eval_demo.sh
tests/                 Unit and integration tests, no network and no keys
docs/                  Architecture, configuration, dataset guide, ADRs (adr/), demo tape
.github/workflows/     ci.yml (gate), eval.yml (live eval on PRs)
```

`prompts/`, `golden_dataset/` and `schemas/` stay at the repository root on purpose: the
CLI takes them as paths, so they are examples and fixtures, not package data. Only files
the installed package loads at run time (the report template and the dashboard) live
under `src/`.

## Stack

| Area | Library | Role |
|---|---|---|
| Core | `litellm` | One API for 100+ model providers, with fallbacks |
| | `pydantic` v2, `pydantic-settings` | Validated models and environment configuration |
| | `typer`, `rich` | CLI and terminal tables |
| | `aiosqlite` | Async SQLite run history |
| | `tenacity`, `httpx` | Webhook delivery with exponential backoff and jitter |
| | `structlog` | Structured logs |
| | `jinja2` | HTML report |
| Dashboard (extra) | `streamlit`, `plotly`, `pandas` | Accuracy timeline, CI band, version comparison |
| Tooling | `uv`, `uv_build` | Environment, lock file, build backend |
| | `ruff`, `pyright` (strict) | Lint, format, types |
| | `pytest`, `pytest-cov` | Tests, 90% coverage floor |
| | `pre-commit`, `gitleaks`, `commitizen` | Local gate, secret scan, Conventional Commits and releases |
