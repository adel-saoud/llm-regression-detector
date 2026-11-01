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

For a free-text dataset (schema v2, such as `golden_dataset/store_support_answers.json`), ask for a `reference_answer` and/or `rubric` criteria instead of a category, append the case inside the `cases` array, and run `uv run pytest tests/unit/test_dataset_v2.py -v`. Each case needs at least one of `expected_category`, `reference_answer` or `rubric`; see `docs/golden-dataset-guide.md`.

Remind the user: adding hard/adversarial cases is the best way to surface prompt weaknesses before they reach production.
