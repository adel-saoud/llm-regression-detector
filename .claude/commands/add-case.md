Add one or more labelled cases to the golden dataset.

Ask the user for:
- The incident description (input text)
- The expected category: p0 / p1 / p2 / p3
- Difficulty: easy / medium / hard / adversarial
- Optional: expected summary keywords, notes

Then:
1. Open `golden_dataset/incidents.json`
2. Generate a new unique ID following the existing pattern (e.g. `p0042`)
3. Append the new case(s) in valid JSON format
4. Verify the dataset still loads: `uv run python -c "from llm_regression_detector.eval.dataset import load_dataset; load_dataset('golden_dataset/incidents.json'); print('OK')"`
5. Run the dataset tests: `uv run pytest tests/unit/test_dataset.py -v`

Remind the user: adding hard/adversarial cases is the best way to surface prompt weaknesses before they reach production.
