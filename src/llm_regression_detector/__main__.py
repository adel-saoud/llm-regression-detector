"""Allow ``python -m llm_regression_detector`` as an alias for the ``lrd`` CLI."""

from __future__ import annotations

from llm_regression_detector.cli import app

if __name__ == "__main__":
    app()
