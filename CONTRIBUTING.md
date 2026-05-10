# Contributing

## Setup
```bash
uv sync --all-extras
uv run pre-commit install
```

## Run quality gates locally
```bash
uv run ruff check --fix .
uv run ruff format .
uv run pyright
uv run pytest
```

CI runs the same gates. Coverage threshold is **85%**.

## Commits
Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`).
`commitizen` is wired into `pre-commit` to enforce this.

## Adding an LLM provider
Providers route through `litellm` — most are already supported. To add one to
the fallback chain, append a new branch in `_build_router` in
`src/llm_regression_detector/llm/client.py` and surface its credentials + model
id as fields on `Settings`. Keep it $0 — no paid-tier defaults.

## Adding a webhook platform
1. Implement the `Notifier` Protocol from
   `src/llm_regression_detector/notify/base.py`.
2. Use `post_with_retry` from
   `src/llm_regression_detector/notify/transport.py` for the actual delivery —
   that's where the shared retry/backoff/jitter policy lives.
3. Register the new platform in `WebhookPlatform` (config) and the platform map
   in `notify/__init__.py`.

## Evolving stored data
`EvalRun` rows are persisted as JSON tagged with a schema version. When you add
a non-additive change to the schema, bump `CURRENT_SCHEMA_VERSION` in
`src/llm_regression_detector/storage/sqlite.py` and add a branch to
`_migrate_payload` that fills the new field with a sensible default. Add a
matching migration test in `tests/unit/test_storage_migration.py`.
