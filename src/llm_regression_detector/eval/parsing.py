"""Tolerant parsing of LLM outputs.

LLMs occasionally wrap their JSON in code fences, prepend reasoning, or trail
prose after the closing brace. ``extract_json_object`` recovers the first valid
JSON object regardless, and ``parse_prediction`` turns the output of the prompt
under test into a ``Prediction`` according to the prompt's ``output_format``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from llm_regression_detector.eval.dataset import OutputFormat


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Return the first JSON object found in ``raw``, or ``None`` if none parses.

    Handles three common shapes:
      1. Pure JSON: ``{"key": "value"}``
      2. Code-fenced: ```` ```json\\n{"key": "value"}\\n``` ````
      3. JSON embedded in prose: locates ``{`` ... ``}`` and tries to parse.
    """
    text = raw.strip()
    if not text:
        return None

    text = _strip_code_fence(text)

    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = _try_brace_substring(text)

    if not isinstance(parsed, dict):
        return None
    return cast(dict[str, Any], parsed)


def _strip_code_fence(text: str) -> str:
    """Drop a leading ```` ```lang ```` line and a trailing fence, if present."""
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline != -1:
        text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _try_brace_substring(text: str) -> Any | None:
    """Parse from the first ``{`` to the last ``}``, which skips prose on both sides."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class Prediction:
    """What the prompt under test produced for one case.

    ``category`` is ``None`` when the output has no string ``category`` key.
    ``answer`` is the free-text reply: the ``answer`` key in JSON mode, or the whole
    output in text mode.
    """

    category: str | None = None
    summary: str = ""
    answer: str | None = None

    @property
    def judged_text(self) -> str:
        """The text the judge rates: the answer when there is one, else the summary."""
        return self.answer if self.answer is not None else self.summary


def parse_prediction(raw: str, output_format: OutputFormat) -> Prediction | None:
    """Read a ``Prediction`` from raw model output, or ``None`` when JSON was expected
    but no JSON object could be recovered."""
    if output_format is OutputFormat.TEXT:
        return Prediction(answer=raw.strip())
    parsed = extract_json_object(raw)
    if parsed is None:
        return None
    category = parsed.get("category")
    summary = parsed.get("summary")
    answer = parsed.get("answer")
    return Prediction(
        category=category if isinstance(category, str) else None,
        summary=summary if isinstance(summary, str) else "",
        answer=answer if isinstance(answer, str) else None,
    )
