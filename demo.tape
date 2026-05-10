# demo.tape — generates demo.gif for the README hero section.
#
# Regenerate with:  vhs demo.tape
# Requires:         brew install vhs ttyd ffmpeg
#
# Records a full self-eval demo in mock mode — no API keys, no cost,
# reproducible on any contributor's machine.

Output demo.gif

Require uv

Set Shell "zsh"
Set FontSize 13
Set Width 1100
Set Height 620
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set Framerate 30

# ── Setup (hidden): force mock mode + clean state ─────────────────────────
Hide
Type "setopt interactive_comments; rm -f evals/runs.db; mv .env .env.demo.bak 2>/dev/null || true; clear"
Enter
Sleep 500ms
Show

# ── Intro ─────────────────────────────────────────────────────────────────
Type "# llm-regression-detector — catches LLM quality drops in CI ($0 to run)"
Sleep 1.5s
Enter
Enter

# ── Step 1: baseline ──────────────────────────────────────────────────────
Type "# 1. Baseline: prompt v1 with few-shot examples"
Sleep 800ms
Enter
Type@40ms "uv run lrd run -p prompts/classifier_v1.yaml --no-diff --no-notify"
Enter
Sleep 4500ms

# ── Step 2: degraded candidate ────────────────────────────────────────────
Enter
Type "# 2. Candidate: v2 — few-shots removed (deliberately degraded)"
Sleep 1.5s
Enter
Type@40ms "uv run lrd run -p prompts/classifier_v2_degraded.yaml --no-notify"
Enter
Sleep 5500ms

# Hold on the CRITICAL output so viewers can read it
Sleep 3s

# ── Cleanup (hidden) ──────────────────────────────────────────────────────
Hide
Type "mv .env.demo.bak .env 2>/dev/null || true"
Enter
