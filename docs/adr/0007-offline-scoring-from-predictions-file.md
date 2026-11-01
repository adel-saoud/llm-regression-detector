---
status: accepted
date: 2026-10-31
---

# Score agents offline from a predictions JSONL file

## Context and Problem Statement

`lrd run` re-runs a prompt. Agents and services generate their own answers through
tools, retrieval and several model calls, so there is no single prompt to re-run.
They still need the same judge, diff and exit codes as prompts. agent-mesh, for
example, exports its answers per suite.

## Considered Options

- Embed each system's answers in the dataset file
- Let `lrd` call the agent over HTTP
- A separate predictions JSONL file, one `{"id", "category"?, "answer"?}` line per case, graded by `lrd score`

## Decision Outcome

Chosen option: a separate predictions JSONL file graded by `lrd score`, because it
keeps the dataset (the labels) apart from any one system's answers, lets any
language produce the file, and needs no network contract with the agent.
`PredictionRecord` (`eval/dataset.py`) is published as
`schemas/predictions.v1.schema.json`. A case id with no prediction fails that case
with `MISSING_PREDICTION_ERROR` (`eval/runner.py`) instead of being skipped, so a
system that silently drops cases cannot raise its score.

### Consequences

- Good: one dataset grades any number of systems; the diff runs on stored runs keyed
  by `--name`.
- Good: dropped cases count as failures, so coverage gaps show up in accuracy.
- Bad: the predictions file has no `schema_version` field; its schema `$id` carries v1,
  and the first breaking change has to add one.
- Bad: the file is produced outside `lrd`, so latency and token counts of the agent are
  not measured.
