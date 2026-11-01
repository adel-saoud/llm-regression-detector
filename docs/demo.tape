# Records docs/demo.gif for the README, on the mock model: no API key, no cost.
#
# Run from the repository root:  vhs docs/demo.tape
# Requires:                      brew install vhs (pulls ttyd and ffmpeg)
#
# The hidden setup moves .env and evals/runs.db aside so the run is always the
# mock and always starts without a baseline; the hidden cleanup puts them back.
# If a recording is interrupted, restore them from .env.demo.bak and
# evals/runs.db.demo.bak.

Output docs/demo.gif

Require uv

Set Shell "zsh"
Set FontSize 13
Set Width 1100
Set Height 620
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set Framerate 30

Hide
Type "setopt interactive_comments; [ -f .env ] && mv .env .env.demo.bak; [ -f evals/runs.db ] && mv evals/runs.db evals/runs.db.demo.bak; uv sync -q; clear"
Enter
Sleep 3s
Show

Type "# llm-regression-detector: fail the build only when an LLM quality drop is real"
Sleep 1.5s
Enter
Enter

Type "# 1. Baseline: incident triage prompt v1"
Sleep 800ms
Enter
Type@40ms "uv run lrd run -p prompts/incident_triage_v1.yaml --no-diff --no-notify"
Enter
Sleep 7s

Enter
Type "# 2. Candidate: vague severity definitions, no few-shot examples"
Sleep 1.5s
Enter
Type@40ms "uv run lrd run -p prompts/demo/incident_triage_v2_degraded.yaml --no-notify"
Enter
Sleep 8s

# Hold on the CRITICAL verdict so viewers can read it.
Sleep 4s

Hide
Type "rm -f evals/runs.db; [ -f evals/runs.db.demo.bak ] && mv evals/runs.db.demo.bak evals/runs.db; [ -f .env.demo.bak ] && mv .env.demo.bak .env; clear"
Enter
Sleep 500ms
