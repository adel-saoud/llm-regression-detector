#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Self-eval demo
#
# Proves the regression detector catches a real quality drop:
#   1. Run the baseline prompt (v1, with few-shot examples)
#   2. Run the deliberately degraded prompt (v2, no few-shots, weaker spec)
#   3. Diff them — expect a drop and an alert-worthy severity
#
# Works with $0 setup:
#   - if HF_TOKEN / GEMINI_API_KEY is set, uses real free-tier models
#   - otherwise falls back to the deterministic mock client
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")/.."

DB="evals/runs.db"
DATASET="golden_dataset/incidents.json"

echo "▶ Step 1: baseline (v1, few-shots intact)"
uv run lrd run \
    --prompt prompts/incident_triage_v1.yaml \
    --dataset "$DATASET" \
    --db "$DB" \
    --report evals/report_v1.html \
    --no-diff \
    --no-notify

echo
echo "▶ Step 2: candidate (v2, deliberately degraded)"
# Capture the exit code so the wrap-up prints, but propagate a non-zero status
# at the end. CI relies on that signal to gate merges.
set +e
uv run lrd run \
    --prompt prompts/incident_triage_v2_degraded.yaml \
    --dataset "$DATASET" \
    --db "$DB" \
    --report evals/report_v2.html \
    --notify
demo_exit=$?
set -e

echo
echo "✅ Demo complete. Reports written to:"
echo "    evals/report_v1.html"
echo "    evals/report_v2.html"

exit "$demo_exit"
