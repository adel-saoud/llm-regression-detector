---
status: accepted
date: 2026-10-31
---

# Each prompt names its dataset, and CI re-runs every affected prompt

## Context and Problem Statement

`eval.yml` ran only the first changed prompt, always against the incident dataset
(`CHANGELOG.md`, Unreleased, Changed). A second prompt on another dataset was never
evaluated, and a dataset-only change re-ran nothing relevant.

## Considered Options

- Pair prompts and datasets by file name
- A central config file mapping each prompt to its dataset
- A `dataset` key in each prompt YAML, relative to the prompt file

## Decision Outcome

Chosen option: a `dataset` key in each prompt YAML, relative to the prompt file,
because the pairing then lives next to the prompt it describes, survives renames of
either file in one edit, and does not depend on the working directory
(`prompt_dataset_path`, `eval/dataset.py`). `lrd affected-prompts` (`ci.py`) selects
every prompt whose YAML changed or whose dataset changed, and `eval.yml` evaluates
each on its own dataset with one PR comment section per prompt.

### Consequences

- Good: a dataset-only pull request re-evaluates exactly the prompts that use it.
- Bad: a dataset shared by many prompts makes a dataset change cost one full run per
  prompt, each with a judge call per case.
- Neutral: only the top level of `prompts/` is scanned, so the deliberately degraded
  demo prompt lives in `prompts/demo/` and never fails a dataset pull request.
- `lrd run --dataset` no longer defaults to the incident dataset; a prompt with no
  `dataset` key and no flag stops with an error.
