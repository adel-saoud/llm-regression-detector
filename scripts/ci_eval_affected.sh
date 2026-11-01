#!/usr/bin/env bash
# Evaluate every prompt affected by the changes since a base ref, each on the
# dataset its `dataset` key names. Used by .github/workflows/eval.yml.
#
# Usage: scripts/ci_eval_affected.sh <base-ref>
# Writes evals/report-<prompt file stem>.html per prompt, and pr_comment.md with
# one section per prompt when at least one prompt ran. Runs every affected prompt,
# then exits non-zero if any run failed (exit 2 = CRITICAL regression).
# Portable to macOS bash 3.2 so it can be dry-run locally.
set -euo pipefail

base_ref="${1:?usage: scripts/ci_eval_affected.sh <base-ref>}"
mkdir -p evals

changed_files="evals/changed_files.txt"
affected="evals/affected_prompts.txt"
comment_parts="evals/pr_comment.parts.md"

git diff -z --name-only --diff-filter=AM "${base_ref}...HEAD" > "$changed_files"
if [ ! -s "$changed_files" ]; then
  echo "No added or modified files since ${base_ref}."
  exit 0
fi
xargs -0 uv run lrd affected-prompts < "$changed_files" > "$affected"

: > "$comment_parts"
status=0
# fd 3 keeps `uv run` from reading the prompt list on stdin.
while IFS= read -r prompt <&3; do
  stem="$(basename "${prompt%.*}")"
  name="$(uv run python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["name"])' "$prompt")"

  echo "::group::lrd run ${prompt}"
  uv run lrd run \
    --prompt "$prompt" \
    --report "evals/report-${stem}.html" \
    --diff \
    --save \
    --notify \
    --fail-on-critical || status=$?
  echo "::endgroup::"

  printf '### `%s`\n\n' "$prompt" >> "$comment_parts"
  uv run lrd pr-comment --prompt-name "$name" >> "$comment_parts" \
    || echo "_No stored run for ${name}._" >> "$comment_parts"
  printf '\n' >> "$comment_parts"
done 3< "$affected"

if [ -s "$comment_parts" ]; then
  mv "$comment_parts" pr_comment.md
else
  echo "No prompt is affected by these changes."
fi
exit "$status"
