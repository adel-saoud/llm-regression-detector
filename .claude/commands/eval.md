Run a full eval against the golden dataset using the mock LLM (no API keys needed).

Steps:
1. Run the baseline prompt and save the result to the local DB.
2. Run the degraded prompt to simulate a regression.
3. Diff the two runs and show the severity verdict.

```bash
uv run lrd run \
  --prompt prompts/incident_triage_v1.yaml \
  --dataset golden_dataset/incidents.json \
  --report evals/report_v1.html \
  --save --no-diff --no-notify

uv run lrd run \
  --prompt prompts/incident_triage_v2_degraded.yaml \
  --dataset golden_dataset/incidents.json \
  --report evals/report_v2.html \
  --diff --save --no-notify
```

After both runs complete, summarise:
- Accuracy delta (baseline → candidate)
- Severity verdict (OK / WARNING / CRITICAL)
- Number of regressions and improvements
- Location of generated HTML reports
