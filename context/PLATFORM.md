# Platform Context

**Stack**: Python 3.11+ · `uv` · `hatchling` · `src/` layout. Check existing libraries before adding new dependencies.

## Core Technology Stack

- **Languages**: Python 3.11+ (uses `Self`, `StrEnum`, `tomllib`, `match`, `|` union types)
- **Tooling**: `uv` (env + run), `hatchling` (build), `ruff` (lint + format), `pyright` strict (type-check), `pre-commit`
- **Backend**: `litellm` (LLM routing), `pydantic` v2 (validation), `typer` + `rich` (CLI), `aiosqlite` (storage), `jinja2` (report templating)
- **Frontend**: `streamlit` + `plotly` + `pandas` (dashboard, optional dep — excluded from pyright)
- **HTTP & Retry**: `httpx` + `tenacity` (webhook delivery with jitter backoff)
- **Logging**: `structlog` with `bind()` for run-scoped context
- **Testing**: `pytest` + `pytest-asyncio` + `pytest-cov`

## Architecture & Patterns

- **Data flow**: Async batch — `asyncio.gather` + semaphore for concurrent LLM calls
- **Design patterns**: Protocols (PEP 544) over ABCs; `Notifier` and `LLMClient` are structural interfaces
- **State management**: Immutable Pydantic models (`frozen=True, extra="forbid"`) across all module boundaries
- **System architecture**: Single-repo CLI tool; SQLite local file for run history
- **Statistical logic**: Pure, dependency-free functions isolated in `eval/stats.py`

## Infrastructure & Deployment

- **Containerization**: None (pure Python, `uv` manages the env)
- **Environment**: Runs locally or on any GitHub Actions runner (`ubuntu-latest`)
- **Secret management**: Env vars only — `LRD_*` prefix, loaded via `pydantic-settings`
- **CI/CD pipelines**: `ruff check → ruff format → pyright → pytest` (85% gate); eval workflow on `prompts/**` or `golden_dataset/**` changes
- **Observability**: `structlog` structured logs; Datadog/Sentry out of scope

## Third-Party Integrations

- **LLM Providers**: litellm router — `LRD_CUSTOM_MODEL` → HF Inference → Gemini (free) → Ollama → Mock
- **Notifications**: Slack (incoming webhook), Google Chat, Discord, generic JSON — all via `Notifier` Protocol
- **Storage**: `aiosqlite` with explicit schema versioning and `_migrate_payload` for forward compatibility
- **Report**: Jinja2 HTML report + GitHub-flavoured Markdown PR comment
