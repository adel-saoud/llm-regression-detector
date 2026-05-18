# Process Context

**Workflow**: Feature branches → PR → squash-merge to `main`. All 4 quality gates must pass before a task is done.

## Quality Assurance

- **Testing approach**: Unit tests for all domain logic; integration tests for the full pipeline. No real API calls in tests — use `MockLLMClient` or `_ScriptedClient`.
- **Coverage goals**: ≥ 85% on non-dashboard code (enforced by `pytest --cov`). Dashboard excluded via `tool.coverage.run.omit`.
- **Linting strategy**: `ruff check --fix` then `ruff format` — both must exit 0. `pre-commit` enforces this on every commit (`uv run pre-commit install` on first setup).
- **Type safety**: `pyright` strict, 0 errors, 0 warnings. `dashboard/app.py` is the only exclusion.

## Development Workflow

- **Commit rules**: Conventional Commits — `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`. User commits only; agent never commits or pushes.
- **Issue tracking**: GitHub Issues / PR comments.
- **Version control**: Feature branches `feat/<slug>`, fix branches `fix/<slug>`. Squash-merge preferred.
- **Code review**: PR description follows `.github/PULL_REQUEST_TEMPLATE.md`. Eval CI runs automatically on `prompts/**` or `golden_dataset/**` changes.

## Code Standards & Conventions

- **Code formatting**: `ruff format` (enforced in CI via `--check`).
- **Directory structure**: `src/llm_regression_detector/` for all importable code. No `utils.py` — helpers live in their domain module.
- **Documentation**: No comments that restate what well-named code already says. Inline comments for non-obvious statistical or protocol choices only.
- **Naming conventions**: Snake case throughout. `structlog.bind()` for run-scoped log context — never repeat `run_id=...` per call.
