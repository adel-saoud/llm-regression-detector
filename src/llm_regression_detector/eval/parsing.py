"""Tolerant JSON extraction from LLM outputs.

LLMs occasionally wrap their JSON in code fences, prepend reasoning, or trail
prose after the closing brace. This module provides a single helper that
recovers the first valid JSON object regardless.
"""

from __future__ import annotations

import json
from typing import Any, cast


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Return the first JSON object found in ``raw``, or ``None`` if none parses.

    Handles three common shapes:
      1. Pure JSON: ``{"key": "value"}``
      2. Code-fenced: ```` ```json\\n{"key": "value"}\\n``` ````
      3. JSON embedded in prose — locates ``{`` ... ``}`` and tries to parse.
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
    if not text.startswith("```"):
        return text
    first_newline = text.find("\n")
    if first_newline != -1:
        text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _try_brace_substring(text: str) -> Any | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
