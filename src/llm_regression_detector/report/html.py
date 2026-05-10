"""Standalone HTML report renderer.

Loads the Jinja2 template from ``templates/report.html.j2`` (separation of
concerns: presentation lives in templates, rendering logic stays here).
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from llm_regression_detector.diff.analyzer import DiffReport
from llm_regression_detector.eval.dataset import EvalRun

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.html.j2"

_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_html(
    candidate: EvalRun,
    diff: DiffReport | None = None,
    baseline: EvalRun | None = None,
) -> str:
    """Render an HTML report. ``diff`` and ``baseline`` are optional (first-run case)."""
    template = _env.get_template(_TEMPLATE_NAME)
    return template.render(candidate=candidate, diff=diff, baseline=baseline)


def write_html(
    path: Path | str,
    candidate: EvalRun,
    diff: DiffReport | None = None,
    baseline: EvalRun | None = None,
) -> Path:
    """Render and persist the HTML report. Creates parent dirs as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(candidate, diff, baseline), encoding="utf-8")
    return out
