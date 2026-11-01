# Platform Context

**Stack**: Python 3.12+ · `uv` · `uv_build` · `src/` layout. Check existing libraries before adding new dependencies.

## Core Technology Stack

- **Languages**: Python 3.12+ (uses `Self`, `StrEnum`, `datetime.UTC`, `|` union types)
- **Tooling**: `uv` (env + run), `uv_build` (build), `ruff` (lint + format), `pyright` strict (type-check), `pre-commit` with `gitleaks` and `commitizen`
- **Backend**: `litellm` (LLM routing), `pydantic` v2 (validation), `typer` + `rich` (CLI), `aiosqlite` (storage), `jinja2` (report templating)
- **Frontend**: `streamlit` + `plotly` + `pandas` (dashboard, optional dep, excluded from pyright)
- **HTTP & Retry**: `httpx` + `tenacity` (webhook delivery with jitter backoff)
- **Logging**: `structlog` with `bind()` for run-scoped context
- **Testing**: `pytest` + `pytest-asyncio` + `pytest-cov`

## Architecture & Patterns

- **Data flow**: Async batch: `asyncio.gather` + semaphore for concurrent LLM calls
- **Design patterns**: Protocols (PEP 544) over ABCs; `Notifier` and `LLMClient` are structural interfaces
- **State management**: Immutable Pydantic models (`frozen=True, extra="forbid"`) across all module boundaries
- **System architecture**: Single-repo CLI tool; SQLite local file for run history
- **Statistical logic**: Pure, dependency-free functions isolated in `eval/stats.py`

## Infrastructure & Deployment

- **Containerization**: optional `Dockerfile` (Python 3.14 slim, non-root, entry point `lrd`); CI and local runs use `uv` directly
- **Environment**: Runs locally or on any GitHub Actions runner (`ubuntu-latest`)
- **Secret management**: Env vars only (`HF_TOKEN`, `GEMINI_API_KEY`, `LRD_CUSTOM_API_KEY`, `LRD_WEBHOOK_URL`), loaded via `pydantic-settings` from the shell or `.env`
- **CI/CD pipelines**: pre-commit (ruff, pyright, gitleaks) then pytest on Python 3.12, 3.13 and 3.14 (90% gate), then a wheel smoke test; eval workflow on `prompts/**` or `golden_dataset/**` changes
- **Observability**: `structlog` structured logs; Datadog/Sentry out of scope

## Third-Party Integrations

- **LLM Providers**: litellm router: `LRD_CUSTOM_MODEL` → HF Inference → Gemini (free) → Ollama → Mock
- **Notifications**: Slack (incoming webhook), Google Chat, Discord, generic JSON, all via `Notifier` Protocol
- **Storage**: `aiosqlite` with explicit schema versioning and `_migrate_payload` for forward compatibility
- **Report**: Jinja2 HTML report + GitHub-flavoured Markdown PR comment
