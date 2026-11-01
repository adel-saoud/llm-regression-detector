---
status: accepted
date: 2026-10-31
---

# A separate judge model that can never be the model under test

## Context and Problem Statement

One client served both the answers and the judge, so the model under test graded its
own output. LLM judges rate their own generations higher (Panickssery et al., 2024,
"LLM Evaluators Recognize and Favor Their Own Generations",
https://arxiv.org/abs/2404.13076). The fallback chain made it worse: when the
primary judge tier failed, grading could move to whatever model answered.

## Considered Options

- Keep one shared client (the previous behaviour)
- A separate judge setting, with a warning when it equals the model under test
- A separate judge client with no fallback, and startup refuses a judge equal to the model under test

## Decision Outcome

Chosen option: a separate judge client with no fallback, refused at startup when it
equals the model under test, because a warning is easy to miss in CI logs and any
fallback could hand grading back to the graded model. `LRD_JUDGE_MODEL` (default
`ollama/qwen3.5:4b`) builds its own single-model client (`build_judge_client`,
`llm/client.py`); `Settings._reject_self_judging` compares it with
`Settings.primary_model`, treating `ollama/` and `ollama_chat/` as one provider
(`config.py`). The mock answers both roles, so the check skips mock mode.

### Consequences

- Good: no model grades its own answers. Covered by
  `test_judge_equal_to_model_under_test_is_rejected` and
  `test_judge_client_uses_only_the_judge_model`.
- Bad: with HF or Gemini under test the default judge is still local Ollama, so those
  users must set `LRD_JUDGE_MODEL`; `eval.yml` sets it to `gemini/gemini-2.0-flash`.
- Bad: if the judge provider is down, the run fails instead of degrading.
- Measured: on local Ollama the judge score rises from 3.94 to 4.89 while accuracy
  falls from 37.7% to 20.8% (`results/manifest.json`, `runs.ollama.summary`). A
  separate judge removes self-preference, not leniency; the category check (ADR 0004)
  still carries the gate.
