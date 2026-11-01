# Extending llm-regression-detector

Setup, the quality gate and commit rules are in the shared
[contributing guide](https://github.com/adel-saoud/.github/blob/main/CONTRIBUTING.md).
This page covers the extension points specific to this project.

## Adding an LLM provider

Providers route through `litellm`, which already supports most of them. To add one to
the fallback chain, append a new branch in `_build_router` in
`src/llm_regression_detector/llm/client.py` and expose its credentials and model id as
fields on `Settings`. Keep it free to run: no paid-tier defaults.

## Adding a webhook platform

1. Implement the `Notifier` Protocol from
   `src/llm_regression_detector/notify/base.py`.
2. Use `post_with_retry` from `src/llm_regression_detector/notify/transport.py` for the
   delivery. The shared retry, backoff and jitter policy lives there.
3. Register the new platform in `WebhookPlatform` and its webhook domains in
   `_PLATFORM_HOSTS` (both in `config.py`), then map it to the notifier in
   `_NOTIFIER_BY_PLATFORM` in `notify/__init__.py`.

## Evolving stored data

`EvalRun` rows are stored as JSON tagged with a schema version. For a non-additive
change to the schema, bump `CURRENT_SCHEMA_VERSION` in
`src/llm_regression_detector/storage/sqlite.py` and add a branch to `_migrate_payload`
that fills the new field with a sensible default. Add a matching migration test in
`tests/unit/test_storage_migration.py`.
