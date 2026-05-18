Run all quality gates in order and report results.

```bash
uv run ruff check --fix . && \
uv run ruff format . && \
uv run pyright && \
uv run pytest -q
```

Report each gate as ✅ passed or ❌ failed with the relevant error output.
If any gate fails, stop and explain what needs fixing before proceeding.
