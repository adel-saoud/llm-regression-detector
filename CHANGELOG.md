# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-01

Below 1.0 a breaking change bumps the minor version (`major_version_zero` in the commitizen config).

### Added

- `DiffReport.large_gain`, set when accuracy rises by at least `critical_pp`. The CLI, HTML report, PR comment, alert title and dashboard print "Large gain: check the judge or dataset".
- Startup check on settings: an unknown `LRD_*` environment variable stops every command with exit code 1 and names the closest valid setting.
- `LRD_DB_PATH` setting, read by the dashboard.
- Versioned schema contract: `schemas/golden-dataset.v2.schema.json` and `schemas/predictions.v1.schema.json` carry a JSON Schema 2020-12 `$schema` and a versioned `$id`. Pin them at a release tag; see [docs/golden-dataset-guide.md](docs/golden-dataset-guide.md#the-schema-contract).
- `scripts/reproduce.py` regenerates every README number into `results/manifest.json`, with environment, Ollama version and model digests, dataset and prompt hashes, and the sample size of each metric. It runs local Ollama with no key (model under test `qwen3.5:0.8b`, judge `qwen3.5:4b`); `--mock` runs the offline mock and `--db PATH` keeps the runs for dashboard screenshots. The `Try it` commands, the demo GIF and CI stay on the mock.
- A test that fails when a README number differs from the manifest.
- `LRD_JUDGE_MODEL`, `LRD_JUDGE_API_KEY` and `LRD_JUDGE_API_BASE`: the judge runs on its own single-model client with no fallback, so the model under test never grades its own answers. Default `ollama/qwen3.5:4b`.
- `LRD_REASONING_EFFORT` (`none`, `low`, `medium`, `high`; default `none`), sent to Ollama models through litellm's `reasoning_effort`, which litellm forwards as Ollama's `think`. Qwen3.5 thinks by default and could spend the whole token budget before answering.

- Dataset schema v2: a `{"schema_version": 2, "cases": [...]}` file format for free-text answers. Cases gain `reference_answer`, `rubric` (pass or fail criteria) and `source` (provenance). See [docs/golden-dataset-guide.md](docs/golden-dataset-guide.md).
- Per-criterion rubric verdicts in results (`JudgeScore.criteria`, `JudgeScore.rubric_pass_rate`); with `LRD_JUDGE_CONSENSUS_N` above 1 each criterion is decided by majority vote.
- `output_format` in prompt YAML: `json` (default) or `text`. JSON output may carry an `answer` key next to `category` and `summary`.
- `lrd score` grades answers a system already produced, read from a predictions JSONL file (`{"id", "category"?, "answer"?}` per line), with the same judge, diff and exit codes as `lrd run`. For agents and services that generate their own answers.
- `lrd schema` prints the JSON Schema of a v2 dataset file, or of one predictions line with `--kind predictions`; the generated copies live in `schemas/`.
- Example free-text dataset `golden_dataset/store_support_answers.json` and prompt `prompts/store_support_v1.yaml`.
- Rubric pass count in the HTML report and dashboard case tables.
- Planned GitHub Action inputs in [docs/github-action.md](docs/github-action.md).
- `dataset` key in prompt YAML, relative to the prompt file, so each prompt names the dataset it is evaluated on. `lrd init` writes it.
- `lrd affected-prompts <changed files...>` lists the prompts a change set needs to re-evaluate: changed prompt files and prompts whose dataset changed.

### Changed

- `expected_category` is optional. Each case needs at least one of `expected_category`, `reference_answer` or `rubric`.
- The category check is an exact string match done in code. The judge no longer decides `category_match`, which removes judge noise from classification results.
- Cases without a category are grouped under `uncategorized` in the per-category breakdown.
- Dataset loading rejects duplicate case ids and unknown `schema_version` values.
- `lrd run --dataset` no longer defaults to `golden_dataset/incidents.json`. It falls back to the prompt's `dataset` key, and stops with an error when neither is set.
- `eval.yml` evaluates every affected prompt, each on its own dataset, and posts one PR comment with a section per prompt. Before, it ran only the first changed prompt, always against the incident dataset.
- The deliberately degraded demo prompt moved to `prompts/demo/incident_triage_v2_degraded.yaml`, out of the folder CI scans, so dataset pull requests do not fail on it.
- **BREAKING:** Python 3.12 is now the minimum (`requires-python = ">=3.12"`). Python 3.11 is no longer supported. Development uses 3.14 and CI tests 3.12, 3.13 and 3.14.
- Build backend switched from hatchling to `uv_build`. The wheel content is unchanged: the package, its report template and the dashboard.
- Development dependencies moved from the `dev` extra to the `dev` dependency group, so `pip install llm-regression-detector[dev]` no longer exists. `uv sync` installs them.
- Tooling aligned with the shared repo standard: ruff 0.17, pyright 1.1.414, pytest 9, pytest-cov 7, coverage floor raised from 85% to 90%, pre-commit hooks pinned by commit SHA, gitleaks secret scan added, pyright run from the project venv.
- `ci.yml` runs pre-commit, tests on a Python 3.12 to 3.14 matrix and a wheel smoke test; every Action is pinned by commit SHA and the workflow token is read-only.
- `eval.yml` no longer runs on pull requests from forks (it reads provider secrets) and can be started by hand with a base branch.
- CLI, report and PR comment text use colons instead of em-dashes; empty PR comment cells read `n/a`.
- README restructured. Configuration moved to `docs/configuration.md`, project structure and stack to `docs/project-structure.md`, the demo tape to `docs/demo.tape`.
- Docker image on Python 3.14 and uv 0.13.5.
- **BREAKING:** settings read from the environment use the `LRD_` prefix everywhere: `OLLAMA_BASE_URL`, `WEBHOOK_URL` and `WEBHOOK_PLATFORM` are now `LRD_OLLAMA_BASE_URL`, `LRD_WEBHOOK_URL` and `LRD_WEBHOOK_PLATFORM`. The old names are no longer read. `HF_TOKEN` and `GEMINI_API_KEY` keep their vendor names.
- **BREAKING:** `Settings.cost_per_million_input_usd` and `cost_per_million_output_usd` are renamed `cost_input_usd` and `cost_output_usd`, matching `LRD_COST_INPUT_USD` and `LRD_COST_OUTPUT_USD`.
- **BREAKING:** an unknown key in `.env` is an error instead of being ignored.
- **BREAKING:** the judge no longer shares the model under test. Startup fails when `LRD_JUDGE_MODEL` equals the first model of the chain under test; with HF or Gemini under test, set `LRD_JUDGE_MODEL` (the default judge is local Ollama). `eval.yml` judges with `gemini/gemini-2.0-flash`, or the `LRD_JUDGE_MODEL` repository variable.
- **BREAKING:** `LRD_OLLAMA_MODEL` defaults to `ollama/qwen3.5:0.8b` instead of `ollama/llama3.2:3b`.
- Severity scores the drop only. A gain never raises it; cases that flipped from pass to fail still give WARNING even when net accuracy rose.
- README results, the sensitivity table and the dataset-size guidance are regenerated from `results/manifest.json`. The demo GIF and dashboard screenshot are re-recorded from the same run.

### Deprecated

- The `{email}` placeholder in `user_template`. Use `{input_text}`. `{email}` still renders and emits a `DeprecationWarning`.

### Removed

- **BREAKING:** `DiffReport.has_regression`, unused. Read `severity` or `regressions` instead.
- The dashboard no longer turns a case id such as `b001` into `Billing #1` when the case has no `topic`; it shows the id. The mapping only fit the retired email classifier dataset.
- The unused `Category` enum (`p0` to `p3`) from `llm_regression_detector.eval.dataset`. Categories have been free strings since dataset schema v2.

### Fixed

- A large accuracy gain with a few regressed cases could be scored CRITICAL, because severity used the absolute delta.
- The HTML report showed the red CRITICAL banner for INFO diffs: the template compared against `ok`, which is not an `AlertSeverity` value.
- The dashboard printed two constant docstrings on the page (Streamlit magic).

### Docs

- `docs/DECISIONS.md` is replaced by Architecture Decision Records in `docs/adr/` (MADR 4 format, indexed in [docs/adr/README.md](docs/adr/README.md)). All ten are accepted.

### Compatibility

- v1 datasets (bare JSON arrays) load unchanged.
- Stored runs load unchanged: the new result fields are optional, so the run history schema version stays at 1.
