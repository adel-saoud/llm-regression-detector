Scaffold a new versioned prompt file based on an existing one.

Ask the user:
1. Which existing prompt to use as the base (default: `prompts/incident_triage_v1.yaml`)
2. What version number / suffix the new file should have
3. What change they want to make (stronger system message, different few-shots, etc.)

Then:
- Copy the base prompt YAML to the new path
- Apply the requested change
- Print a `lrd run` command the user can run to immediately eval the new prompt against the golden dataset
- Remind them: a PR that modifies `prompts/` will trigger the CI eval workflow automatically
