---
status: accepted
date: 2026-10-31
---

# No pairwise judge yet

## Context and Problem Statement

The judge grades each answer on its own (pointwise). A pairwise judge would see the
baseline and candidate answers to the same case and say which is better, which is
often more sensitive to small quality differences. It was considered and not built.

## Considered Options

- Pointwise grading only (current)
- Pairwise grading of baseline versus candidate on every case
- Pairwise grading only on cases whose pointwise verdict changed

## Decision Outcome

Chosen option: pointwise grading only, for now, because the gate decision rests on
pass rates with Wilson intervals (ADR 0001), which need an absolute verdict per case,
and pointwise scores can be stored once and diffed against any later run. Pairwise
grading needs both answers at grading time and one more judge call per compared case.

### Consequences

- Good: each run is graded once and compared with any stored baseline without
  re-judging.
- Bad: small quality differences that do not flip a pass or fail verdict are missed.
- Bad: pairwise judges have their own position bias, so building it later means
  judging both orders and doubling those calls.
- Open: when a pairwise judge becomes worth its cost; the inputs exist, since stored
  runs keep each predicted summary and answer (`CaseResult`, `eval/dataset.py`).
