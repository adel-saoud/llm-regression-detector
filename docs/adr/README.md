# Architecture Decision Records

One file per significant decision, in [MADR 4](https://adr.github.io/madr/) format.
`proposed` means the reasoning awaits the maintainer's confirmation; `accepted` means confirmed.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-gate-on-wilson-interval-separation.md) | Gate on Wilson interval separation, not on a raw accuracy delta | accepted |
| [0002](0002-direction-aware-severity.md) | Direction-aware severity: only drops and flipped cases raise it | accepted |
| [0003](0003-separate-judge-model.md) | A separate judge model that can never be the model under test | accepted |
| [0004](0004-category-check-in-code.md) | The category check is done in code, the judge only grades text | accepted |
| [0005](0005-versioned-dataset-schema-contract.md) | A versioned dataset contract: wrapper object plus a published JSON Schema | accepted |
| [0006](0006-mock-for-ci-ollama-for-results.md) | The mock for CI and "Try it", local Ollama with repeats for published results | accepted |
| [0007](0007-offline-scoring-from-predictions-file.md) | Score agents offline from a predictions JSONL file | accepted |
| [0008](0008-no-pairwise-judge-yet.md) | No pairwise judge yet | accepted |
| [0009](0009-each-prompt-names-its-dataset.md) | Each prompt names its dataset, and CI re-runs every affected prompt | accepted |
| [0010](0010-rubric-pass-fail-all-must-pass.md) | A rubric is a list of pass or fail criteria, and every criterion must pass | accepted |
