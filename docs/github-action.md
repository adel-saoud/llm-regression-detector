# GitHub Action (planned)

Status: **planned, not published.** There is no `action.yml` yet. This page fixes the input contract early, because once the Action is on the Marketplace every input name becomes a public API that workflows pin.

Today the same pipeline runs from [`.github/workflows/eval.yml`](../.github/workflows/eval.yml) with `uv run lrd run`.

## Planned inputs

| Input | Required | Default | Maps to | Notes |
|:--|:--|:--|:--|:--|
| `prompt` | one of `prompt` or `predictions` | | `lrd run --prompt` | Path to the prompt YAML under test. |
| `predictions` | one of `prompt` or `predictions` | | `lrd score --predictions` | Predictions JSONL of answers the system already produced. Selects offline scoring. |
| `name` | no | dataset file name up to its first dot | `lrd score --name` | Offline scoring only: keys the stored baseline. |
| `version` | no | `${{ github.sha }}` | `lrd score --version` | Offline scoring only: label of the run. |
| `dataset` | with `predictions`, or when the prompt has no `dataset` key | the prompt's `dataset` key | `--dataset` | v1 bare array or v2 `{"schema_version": 2, ...}`. Detected from the file, so there is no schema-version input. |
| `judge-consensus-n` | no | `1` | `--consensus-n` / `LRD_JUDGE_CONSENSUS_N` | Judge calls per case. Median quality score, majority vote per rubric criterion. |
| `fail-on` | no | `critical` | `--fail-on-critical` | `critical`, `warning` or `none`. Today the CLI only supports `critical`; `warning` needs a CLI flag first. |
| `report-path` | no | `evals/report.html` | `--report` | HTML report, uploaded as a workflow artifact. |
| `comment-on-pr` | no | `true` | `lrd pr-comment` | Sticky PR comment with the diff summary. |
| `baseline-cache-key` | no | `eval-baseline-${{ github.base_ref }}` | `actions/cache` key | Where the baseline `runs.db` is restored from. |

## Not inputs, on purpose

- **Model choice and API keys** come from environment variables (`LRD_CUSTOM_MODEL`, `LRD_CUSTOM_API_KEY`, `HF_TOKEN`, `GEMINI_API_KEY`) set from repository secrets. Keeping secrets out of `with:` avoids them showing up in workflow files and logs.
- **Rubric pass threshold.** Every criterion must pass. A `rubric-pass-threshold` input can be added later without breaking existing workflows; removing one cannot.
- **Dataset schema version.** The loader reads it from the file.
- **A mode input.** Setting `predictions` selects offline scoring; setting both `prompt` and `predictions` is an error.

## Planned outputs

| Output | Meaning |
|:--|:--|
| `severity` | `info`, `warning` or `critical` for the diff against the baseline, the values of `AlertSeverity`. |
| `large-gain` | `true` when accuracy rose by at least the critical threshold: check the judge or the dataset before trusting it. |
| `accuracy` | Pass rate of the candidate run, 0 to 1. |
| `run-id` | Id of the stored run, for follow-up `lrd report` calls. |
