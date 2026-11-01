---
status: accepted
date: 2026-10-31
---

# The category check is done in code, the judge only grades text

## Context and Problem Statement

Classification cases carry an `expected_category`. The judge model used to decide
`category_match`, which added judge noise to the one check that has an exact answer
(`CHANGELOG.md`, Unreleased, Changed).

## Considered Options

- Let the judge decide whether the category matches
- Exact string match in code, with the judge grading only summaries and free-text answers

## Decision Outcome

Chosen option: exact string match in code, because a label comparison has a right
answer and needs no model, and a deterministic check keeps classification results
stable across judge models. `category_match` (`eval/scorer.py`) compares the
prediction with `expected_category`; a judge that still returns `category_match` is
ignored (`test_category_match_is_deterministic_not_judged`,
`tests/unit/test_judge_consensus.py`).

### Consequences

- Good: the classification part of accuracy does not move when the judge model changes.
- Bad: no partial credit and no synonyms. `P1` and `p1` differ; the prompt must ask
  for the exact label.
- Neutral: the judge is still part of the gate. A case passes only with a matching
  category, a judge score of at least 3 of 5 and every rubric criterion passed
  (`CaseResult.passed`, `eval/dataset.py`), so a judge failure that falls back to
  score 1 fails the case.
