## Summary
<!-- One or two sentences. What changes and why. -->

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Prompt change (will trigger eval pipeline)
- [ ] Docs / chore

## Eval impact
<!-- If this changes a prompt, the eval pipeline will run automatically.
     Paste the resulting accuracy delta or a link to the eval report. -->

## Checklist
- [ ] `uv run ruff check .` passes
- [ ] `uv run pyright` passes
- [ ] `uv run pytest` passes (coverage ≥ 85%)
- [ ] No paid-tier defaults introduced
- [ ] No hardcoded model IDs or webhook platforms
