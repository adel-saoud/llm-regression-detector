# Golden Dataset Guide

A golden dataset is a fixed, hand-labelled test set that your LLM is evaluated against on every run. It is the ground truth that makes regression detection possible: the detector compares how well the new prompt scores against these labels relative to the stored baseline, and uses Wilson 95% confidence intervals to decide whether any drop is real or just noise.

---

## JSON schema

Each file is a JSON array of case objects.

| Field | Type | Description |
|:--|:--|:--|
| `id` | string | Stable unique identifier for the case (e.g. `b001`). Never reuse or reorder. |
| `topic` | string | Human-readable label shown in the dashboard's per-case breakdown. |
| `input_email` | string | The raw input passed to your LLM (rename mentally to match your task). |
| `expected_category` | string | The ground-truth label. Must exactly match what your prompt outputs. |

### Full example

```json
[
  {
    "id": "b001",
    "topic": "Subscription renewal charge dispute",
    "input_email": "I was charged twice for my subscription this month. Please refund the duplicate payment immediately.",
    "expected_category": "billing"
  },
  {
    "id": "t001",
    "topic": "App crashes on login",
    "input_email": "Every time I try to log in on iOS 17 the app crashes after the loading screen. I have tried reinstalling.",
    "expected_category": "technical"
  },
  {
    "id": "a001",
    "topic": "Cannot reset password",
    "input_email": "I forgot my password and the reset email never arrives, even after checking spam.",
    "expected_category": "account"
  }
]
```

---

## How many cases do you need?

The detector uses the Wilson 95% confidence interval. When the CIs of two runs overlap, the delta is noise — severity is downgraded automatically. This means dataset size directly controls your sensitivity floor:

| Cases | What you can reliably catch |
|:--|:--|
| 20–30 | Large regressions (15+ percentage-point drops). Good enough to start. |
| 30–50 | Solid coverage. Catches 10 pp drops comfortably. Recommended minimum. |
| 50–100 | Comfortable. Catches 7–8 pp drops. |
| 200+ | Required for subtle regressions (≤5 pp). Needed for production-grade gates. |

Start with 30 cases. Add more as you identify failure modes in production.

---

## Picking good test cases

- **Cover every category proportionally.** If 30% of real traffic is `billing`, aim for 30% billing cases.
- **Include edge cases.** Ambiguous inputs that sit between two categories are where prompts regress most.
- **Use realistic inputs.** Copy-paste from real logs (anonymized) rather than writing synthetic examples — synthetic cases are too clean.
- **Avoid duplicates.** Near-identical inputs inflate your apparent accuracy without adding signal.
- **Fix labels, not inputs.** If a case is mislabelled, correct the label. Do not delete the case or reshuffle IDs — that breaks run-to-run comparability.

---

## Topic labels and the dashboard

The `topic` field is purely descriptive — it does not affect scoring. The dashboard displays it in the per-case breakdown table so you can quickly see which specific inputs regressed between runs. A good topic label is one sentence that would make sense out of context: `"Double charge dispute"` is better than `"billing test 3"`.

---

## Quick start with `lrd init`

```bash
uv run lrd init
```

The command prompts for a task name and category list, then writes a prompt YAML and a starter dataset with one placeholder case per category. Replace the placeholder `input_email` strings with real examples before running your first evaluation.

---

## `expected_category` must match exactly

The detector compares `expected_category` to the `category` field in your prompt's JSON output using string equality. If your prompt outputs `"Billing"` but the dataset says `"billing"`, every billing case will count as a miss. Always verify that the values match — case, spelling, and whitespace — before establishing a baseline.
