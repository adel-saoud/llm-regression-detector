---
status: accepted
date: 2026-10-31
---

# Direction-aware severity: only drops and flipped cases raise it

## Context and Problem Statement

Severity used the absolute accuracy delta, so a large gain with a few regressed
cases could be scored CRITICAL and fail a build that improved the prompt
(`CHANGELOG.md`, Unreleased, Fixed). A gain can also hide cases that flipped from
pass to fail, and a sudden large gain more often points to a changed judge or
dataset than to a better prompt.

## Considered Options

- Score the absolute delta (the previous behaviour)
- Score the drop only, ignore gains entirely
- Score the drop only, keep WARNING when any case flips to fail, and flag large gains

## Decision Outcome

Chosen option: score the drop only, keep WARNING for flipped cases, and flag large
gains, because a gate should never fail an improvement, yet a net gain must not
hide individual regressions, and an implausible jump deserves a human look rather
than silence. A gain of at least `critical_pp` stays INFO and sets
`DiffReport.large_gain` (`diff/analyzer.py`); the CLI, HTML report, PR comment,
alert title and dashboard print "Large gain: check the judge or dataset".

### Consequences

- Good: an improving prompt never fails CI. Covered by the gain, drop and mixed tests
  in `tests/unit/test_diff.py`.
- Good: a mixed change (net gain, some cases regressed) is WARNING, so the regressed
  cases are listed in the PR comment.
- Neutral: thresholds stay `warning_pp = 0.03` and `critical_pp = 0.08`
  (`Thresholds`, `diff/analyzer.py`); `critical_pp` now also sets the large-gain flag.
- Bad: `large_gain` is a hint, not a gate. A broken judge that inflates scores still
  passes the build if no one reads the flag.
