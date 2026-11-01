---
status: accepted
date: 2026-10-31
---

# A versioned dataset contract: wrapper object plus a published JSON Schema

## Context and Problem Statement

Dataset v1 was a bare JSON array of classification cases. Free-text cases needed new
fields (`reference_answer`, `rubric`, `source`), and other repos now write datasets
for this one: trace-to-golden turns reviewed production traces into cases. A format
read only by this repo's pydantic model gives those producers nothing to validate
against and no rule for what may change.

## Considered Options

- Keep the bare array and add optional fields
- A wrapper object `{"schema_version": 2, "cases": [...]}`, with the pydantic model as the only definition
- The wrapper object plus a generated JSON Schema with a versioned `$id`, published as the contract

## Decision Outcome

Chosen option: the wrapper object plus a published, versioned JSON Schema, because
the version field lets the loader reject a format it does not know, and a
language-neutral schema lets producers validate without importing this package.

- File: `schemas/golden-dataset.v2.schema.json`, JSON Schema 2020-12.
- `$id`: `https://raw.githubusercontent.com/adel-saoud/llm-regression-detector/main/schemas/golden-dataset.v2.schema.json`
  (`DATASET_SCHEMA_ID`, `eval/dataset.py`).
- Generated from `GoldenDataset` by `uv run lrd schema`; a test fails if the file drifts.
- Consumers pin the same path at a release tag, the first planned being `v0.2.0`.
- An additive change keeps the file and `$id` and ships in a minor release. A breaking
  change publishes `golden-dataset.v3.schema.json` with `schema_version` 3.

The full policy is in `docs/golden-dataset-guide.md`, "The schema contract".

### Consequences

- Good: v1 bare arrays still load (`load_dataset`, `eval/dataset.py`), and unknown
  `schema_version` values are rejected, so a v3 file never loads silently as v2.
- Good: trace-to-golden validates against the schema without importing this package.
- Bad: two artefacts to keep in step, the model and the schema file; the drift test
  enforces it.
- Bad: the `$id` on `main` moves with the branch; only a tagged URL is stable, and no
  tag exists yet.
