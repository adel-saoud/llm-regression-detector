# Agent Guidelines

## Principles

1. **Simplicity First**: Keep dependencies minimal. Favor readable solutions over clever ones.
2. **Proactive Partnership**: Anticipate issues and flag them before they ship. Challenge decisions that contradict these guidelines.
3. **Statistical Rigour**: All regression claims require Wilson 95% CI. Never produce CRITICAL severity on small-N runs.
4. **Type Safety**: `pyright` strict mode, 0 errors, 0 warnings. Every function signature is annotated.
5. **Commit Hygiene**: Never commit, push, or amend without explicit instruction. User commits only.

## Context Navigation

Read the following context files as needed to understand the project's core rules and guidelines.
Suggest to the user to update the context files as needed to reflect the project's current state.

- **[context/PRODUCT.md](context/PRODUCT.md)** (What): Core objective, feature scope, and constraints.
- **[context/PLATFORM.md](context/PLATFORM.md)** (Where): Technology stack, infrastructure, and architectural principles.
- **[context/PROCESS.md](context/PROCESS.md)** (How): Quality gates, development workflow, and code standards.

## Critical Gotchas

- **Adding a field to `RunSummary`/`EvalRun`**: 4-step change — model → `conftest.py::make_run` → bump `CURRENT_SCHEMA_VERSION` + add `_migrate_payload` branch → add fixture row in `test_storage_migration.py`.
- **Severity is CI-aware**: CRITICAL fires only when Wilson 95% CIs do not overlap. In practice this needs ~30+ cases — small-N tests produce WARNING regardless of delta size.
- **`dashboard/app.py`**: Excluded from pyright — no type stubs for streamlit/plotly/pandas.
- **`git diff --name-only`**: Lists deleted files — use `--diff-filter=AM` when detecting changed prompts in CI.
