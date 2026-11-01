---
status: accepted
date: 2026-10-31
---

# A rubric is a list of pass or fail criteria, and every criterion must pass

## Context and Problem Statement

Free-text answers have no exact label to compare. Dataset v2 added a `rubric` to
cases, and the format had to say what a criterion looks like and when a case passes.

## Considered Options

- One free-text rubric graded as a whole
- A 1 to 5 scale per criterion
- A list of pass or fail criteria, with a pass-rate threshold
- A list of pass or fail criteria, all of which must pass

## Decision Outcome

Chosen option: a list of pass or fail criteria, all of which must pass, because a
binary verdict per criterion is easier for a judge model to give consistently than a
scale, each failed criterion names what is missing, and "all must pass" matches how
reviewers read a checklist. The judge returns one verdict per criterion, aligned by
position; criteria it leaves out count as failed (`_parse_judge_response`,
`eval/scorer.py`). With `LRD_JUDGE_CONSENSUS_N` above 1 each criterion is decided by
majority vote.

### Consequences

- Good: the HTML report and dashboard show which criterion failed, not only a score.
- Bad: one strict criterion fails the whole case; there is no partial credit beyond
  `JudgeScore.rubric_pass_rate`, which is reported but does not gate.
- Bad: criteria are plain strings matched by position, so reordering a rubric changes
  which verdict belongs to which criterion in stored runs.
- Open: optional ids or weights per criterion, and how they would stay compatible with
  plain strings. A `rubric-pass-threshold` can be added later without breaking
  existing datasets.
