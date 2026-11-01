# Golden Dataset Guide

A golden dataset is a fixed, hand-labelled test set that your LLM is evaluated against on every run. It is the ground truth that makes regression detection possible: the detector compares how well the new prompt scores against these labels relative to the stored baseline, and uses Wilson 95% confidence intervals to decide whether any drop is real or just noise.

---

## File format

Two file shapes load. Both give the same `GoldenCase` model.

| Shape | Version | When to use |
|:--|:--|:--|
| `{"schema_version": 2, "cases": [...]}` | v2 | New datasets. Required for free-text answers. |
| `[...]` (bare JSON array) | v1 | Existing classification datasets. Still supported. |

Any other `schema_version` is rejected with an error, so a future v3 file never loads silently as v2. Case `id` values must be unique within a file.

### The schema contract

[`schemas/golden-dataset.v2.schema.json`](../schemas/golden-dataset.v2.schema.json) is the canonical, machine-readable contract for dataset files. Other repos that write datasets, such as [trace-to-golden](https://github.com/adel-saoud/trace-to-golden), validate against it.

| Property | Value |
|---|---|
| File | `schemas/golden-dataset.v2.schema.json` |
| Dialect | JSON Schema 2020-12 (`$schema`) |
| `$id` | `https://raw.githubusercontent.com/adel-saoud/llm-regression-detector/main/schemas/golden-dataset.v2.schema.json` |
| Contract version | 2, the `schema_version` every file must carry |
| Source | Generated from `GoldenDataset` with `uv run lrd schema --output schemas/golden-dataset.v2.schema.json`; a test fails if the file drifts from the model |

How consumers pin it: keep a copy, and compare it with the file at a release tag of this repo, for example `https://raw.githubusercontent.com/adel-saoud/llm-regression-detector/v0.2.0/schemas/golden-dataset.v2.schema.json`.

How it changes:

- An additive change (a new optional field) keeps the file name and `$id`, ships in a minor release, and is listed in the CHANGELOG.
- A breaking change (a removed or renamed field, a stricter rule) publishes `golden-dataset.v3.schema.json` with a new `$id` and `schema_version` 3, in a release that bumps the minor version before 1.0 and the major version after it. The loader rejects any version it does not know, so a v3 file never loads silently as v2.

## Case fields

| Field | Type | Required | Description |
|:--|:--|:--|:--|
| `id` | string | yes | Stable unique identifier (e.g. `b001`). Never reuse or reorder. |
| `input_text` | string | yes | The input passed to your prompt as `{input_text}`. |
| `topic` | string | no | Human-readable label shown in reports and the dashboard. |
| `expected_category` | string | see below | Gold label. Compared to the predicted `category` by exact string match. |
| `reference_answer` | string | see below | A good answer. The judge rates the predicted answer against it. |
| `rubric` | list of strings | see below | Criteria the answer must meet. The judge returns pass or fail for each one. |
| `expected_summary_keywords` | list of strings | no | Keywords the summary or answer should cover. Feeds the 1 to 5 score. |
| `difficulty` | `easy` \| `medium` \| `hard` \| `adversarial` | no | For your own slicing. |
| `notes` | string | no | Free text for dataset maintainers. Not sent to any model. |
| `source` | object of strings | no | Provenance, e.g. `{"origin": "trace-to-golden", "trace": "abc123"}`. Not sent to any model. |

Each case needs at least one of `expected_category`, `reference_answer` or `rubric`. Unknown fields are rejected, so a typo such as `rubrik` fails loudly instead of being ignored.

## How a case is scored

The fields a case sets decide which checks run. The checks add up, and a case passes only when every check it defines passes.

| Check | Runs when | Pass rule | Who decides |
|:--|:--|:--|:--|
| Category | `expected_category` is set | predicted `category` equals it exactly | code (no model call) |
| Quality score | always | `summary_score` of 3 or more out of 5 | judge model |
| Rubric | `rubric` is set | every criterion passes | judge model, one verdict per criterion |

With `LRD_JUDGE_CONSENSUS_N` above 1, the judge runs N times; the quality score is the median and each criterion is decided by majority vote.

A case can combine checks. An assistant that must pick an outcome and also answer well can set both:

```json
{
  "schema_version": 2,
  "cases": [
    {
      "id": "ret001",
      "topic": "Return window for an unopened kettle",
      "input_text": "I bought an electric kettle 10 days ago and never opened the box. Can I still return it?",
      "expected_category": "ANSWER",
      "reference_answer": "Yes. Unopened items can be returned within 30 days of delivery.",
      "rubric": [
        "States that the return is allowed",
        "Mentions the 30-day return window"
      ],
      "source": {"origin": "hand-written"}
    },
    {
      "id": "prd001",
      "input_text": "How many cups does the 1.8 litre drip coffee maker make?",
      "reference_answer": "The 1.8 litre drip coffee maker brews up to 12 cups."
    }
  ]
}
```

A full example lives in [`golden_dataset/store_support_answers.json`](../golden_dataset/store_support_answers.json), paired with [`prompts/store_support_v1.yaml`](../prompts/store_support_v1.yaml).

### v1 example (classification only)

```json
[
  {
    "id": "b001",
    "topic": "Subscription renewal charge dispute",
    "input_text": "I was charged twice for my subscription this month. Please refund the duplicate payment immediately.",
    "expected_category": "billing"
  },
  {
    "id": "t001",
    "topic": "App crashes on login",
    "input_text": "Every time I try to log in on iOS 17 the app crashes after the loading screen. I have tried reinstalling.",
    "expected_category": "technical"
  }
]
```

## Prompt output formats

`output_format` in the prompt YAML tells the runner how to read the model output.

| `output_format` | Model output | What is scored |
|:--|:--|:--|
| `json` (default) | JSON object with optional `category`, `summary` and `answer` keys | `category` for the category check; `answer`, or `summary` when there is no `answer`, for the judge |
| `text` | plain text | the whole output is the answer |

A case with `expected_category` fails with an error when the output has no string `category`.

The user template placeholder is `{input_text}`. The old `{email}` placeholder still renders but raises a `DeprecationWarning` and will be removed in a future release.

## Pairing a prompt with its dataset

Each prompt YAML names the dataset it is evaluated on with a `dataset` key, relative to the prompt file:

```yaml
name: store-support
version: v1
dataset: ../golden_dataset/store_support_answers.json
```

- `lrd run --prompt X` uses that dataset. `--dataset` overrides it. With neither, the run stops with an error instead of guessing.
- In CI, `lrd affected-prompts <changed files...>` lists the prompts whose YAML changed or whose dataset changed. A pull request that only edits a dataset re-evaluates every prompt that uses it.
- `lrd affected-prompts` scans the top level of `prompts/` only. Prompts in subfolders are never picked by CI; `prompts/demo/` holds the prompt built to regress for the self-eval demo, so a dataset change never runs it.
- `lrd init` writes the key for you.

## Writing rubric criteria

- **One observable fact per criterion.** "Mentions the 30-day return window" can be checked; "Gives a good answer" cannot.
- **Write negative criteria too.** "Does not repeat the card number" catches leaks a reference answer never would.
- **Keep it short.** Two to five criteria per case. Every criterion must pass, so a long list makes cases brittle.
- **Order is stable.** Verdicts are matched to criteria by position; reordering criteria between runs makes per-criterion history meaningless.

---

## Scoring answers that already exist

When the system under test is an agent or service that produces its own answers, there is no prompt to re-run. Export its answers to a predictions file and score them with `lrd score`:

```bash
uv run lrd score --dataset coordinator.dataset.json --predictions coordinator.predictions.jsonl --version "$GITHUB_SHA"
```

The predictions file is JSON Lines: one object per line, matched to a case by `id`. Contract: [`schemas/predictions.v1.schema.json`](../schemas/predictions.v1.schema.json).

| Field | Type | Required | Description |
|:--|:--|:--|:--|
| `id` | string | yes | Id of the golden case this answers. |
| `category` | string | no | Predicted label, checked against `expected_category`. |
| `answer` | string | no | The reply the user saw. Rated by the judge. |
| `summary` | string | no | Rated by the judge when there is no `answer`. |
| `latency_ms` | number | no | Recorded latency, for the latency percentiles. |
| `prompt_tokens`, `completion_tokens` | integer | no | Recorded token use, for the cost estimate. |
| `error` | string | no | Set when the system failed on this case. The case fails. |

```jsonl
{"id": "ret001", "category": "ANSWER", "answer": "Yes, unopened items can be returned within 30 days.", "latency_ms": 840}
{"id": "ord001", "category": "CLARIFY", "answer": "Which order do you mean? Please share the order number."}
```

Rules:

- A case with no prediction fails as an error, so a system that crashes on some cases cannot pass by answering fewer of them.
- A prediction whose `id` is not in the dataset stops the run with an error.
- Blank lines are skipped. Duplicate ids, invalid JSON and unknown keys stop the run with the line number.
- `--name` keys the stored baseline that later runs diff against. It defaults to the dataset file name up to its first dot (`coordinator` above). `--version` is a free label, for example a commit SHA.
- Judge, diff, report, alerts and exit codes are the same as `lrd run`.

---

## How many cases do you need?

The detector uses the Wilson 95% confidence interval. When the CIs of two runs overlap, the delta is noise: severity is downgraded automatically. This means dataset size directly controls your sensitivity floor.

Smallest accuracy drop that reaches CRITICAL (the two intervals stop overlapping) for a baseline at 85% and at 95%. `uv run python scripts/reproduce.py` computes it with `wilson_interval` and records it in the `sensitivity` section of [`results/manifest.json`](../results/manifest.json):

| Cases | Baseline 85% | Baseline 95% |
|:--|:--|:--|
| 25 | 40 pp | 32 pp |
| 50 | 26 pp | 20 pp |
| 100 | 17 pp | 13 pp |
| 200 | 11.5 pp | 8 pp |
| 500 | 6.8 pp | 4.6 pp |

Smaller drops still show up as WARNING, with the regressed cases listed. Start with 30 to 50 cases to catch broken prompts, and grow the set toward a few hundred as you find failure modes in production.

---

## Picking good test cases

- **Cover every category proportionally.** If 30% of real traffic is `billing`, aim for 30% billing cases.
- **Include edge cases.** Ambiguous inputs that sit between two categories are where prompts regress most.
- **Use realistic inputs.** Copy-paste from real logs (anonymized) rather than writing synthetic examples: synthetic cases are too clean.
- **Avoid duplicates.** Near-identical inputs inflate your apparent accuracy without adding signal.
- **Fix labels, not inputs.** If a case is mislabelled, correct the label. Do not delete the case or reshuffle IDs: that breaks run-to-run comparability.

---

## Topic labels and the dashboard

The `topic` field is purely descriptive: it does not affect scoring. The dashboard displays it in the per-case breakdown table so you can quickly see which specific inputs regressed between runs. A good topic label is one sentence that would make sense out of context: `"Double charge dispute"` is better than `"billing test 3"`.

---

## Quick start with `lrd init`

```bash
uv run lrd init
```

The command prompts for a task name and category list, then writes a prompt YAML and a starter v1 dataset with one placeholder case for each of the first five categories. Replace the placeholder `input_text` strings with real examples before running your first evaluation.

---

## `expected_category` must match exactly

The detector compares `expected_category` to the `category` field in your prompt's JSON output using string equality. If your prompt outputs `"Billing"` but the dataset says `"billing"`, every billing case will count as a miss. Always verify that the values match: case, spelling, and whitespace, before establishing a baseline.
