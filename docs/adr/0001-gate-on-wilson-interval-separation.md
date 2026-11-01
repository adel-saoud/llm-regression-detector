---
status: accepted
date: 2026-10-31
---

# Gate on Wilson interval separation, not on a raw accuracy delta

## Context and Problem Statement

A golden set has tens of cases, not thousands. On 53 cases one flipped case moves
accuracy by 1.9 points, so a fixed delta threshold fires on noise. The gate has to
tell a real quality drop from sampling noise before it fails a build.

## Considered Options

- A fixed accuracy-delta threshold
- A two-proportion test with a p-value
- Wilson 95% intervals of baseline and candidate, CRITICAL only when they do not overlap

## Decision Outcome

Chosen option: Wilson 95% intervals, CRITICAL only when they do not overlap, because
the Wilson interval stays valid at small n and near 0% or 100% where the normal
approximation breaks, it needs no dependency beyond stdlib `math`
(`eval/stats.py`), and "the intervals do not overlap" can be read off the report
without statistics training. A large drop whose intervals overlap is downgraded to
WARNING (`Analyzer._severity`, `diff/analyzer.py`).

### Consequences

- Good: small datasets do not raise false CRITICAL alerts. On local Ollama the degraded
  prompt costs 16.98 points and stays WARNING in 3 of 3 runs, because the intervals
  overlap (`results/manifest.json`, `runs.ollama.summary.diff.severity`).
- Bad: small datasets catch only large drops. At 50 cases the intervals separate at
  a 20 to 26 point drop, and a 5-point drop needs about 430 cases at a 95% baseline
  and 870 at 85% (`results/manifest.json`, `sensitivity`).
- Non-overlap is stricter than a two-proportion test at the same level, so some real
  drops are reported as WARNING. Accepted: a WARNING still reaches the PR comment.
- The 1 to 5 judge score has no interval yet; a bootstrap CI would cover it and is not built.
