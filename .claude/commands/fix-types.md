Fix all pyright strict-mode type errors in the codebase.

```bash
uv run pyright 2>&1
```

For each error:
1. Read the file and understand the context — don't guess fixes blindly.
2. Apply the minimal change that satisfies strict mode (prefer type annotations and narrowing over `# type: ignore`).
3. Never use `Any` as a band-aid unless the module is already excluded from pyright (e.g. `dashboard/app.py`).
4. After all fixes, re-run `uv run pyright` to confirm 0 errors, 0 warnings.
5. Then run `uv run pytest -q` to confirm no tests broke.

Banned workarounds: `cast(Any, ...)`, `# type: ignore` without a specific error code, loosening `frozen=True` or `extra="forbid"`.
