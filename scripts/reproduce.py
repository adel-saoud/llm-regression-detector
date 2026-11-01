"""Regenerate every number the README shows, and record how it was produced.

Usage:
    uv run python scripts/reproduce.py          # local Ollama, 3 repeats, no API key
    uv run python scripts/reproduce.py --mock   # deterministic mock, offline, about 10 s
    uv run python scripts/reproduce.py --db runs.db   # also keep the runs for the dashboard

Both modes run the real ``lrd run`` command on the baseline and the degraded demo
prompt, read the two stored runs back, diff them with the same ``Analyzer`` CI uses,
and write one section of ``results/manifest.json``. The other section is kept, so a
mock run never erases the Ollama numbers and the reverse. The README quotes the
``ollama`` section.

Ollama answers can differ between runs even at temperature 0, because requests
batched together on one server change the arithmetic. A single run would hide that,
so the Ollama mode repeats the pair (``--repeats``, default 3) and records every
repeat plus the median and range of each number.

The default run needs Ollama at ``LRD_OLLAMA_BASE_URL`` with ``LRD_OLLAMA_MODEL`` (the
model under test) and ``LRD_JUDGE_MODEL`` pulled; both names come from ``Settings``, so
switching models is a configuration change. ``lrd`` starts in an empty temporary
directory with provider keys and other ``LRD_*`` settings removed, so a local ``.env``
cannot change which models answer.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from llm_regression_detector.config import ENV_PREFIX, Settings
from llm_regression_detector.diff.analyzer import Analyzer, DiffReport
from llm_regression_detector.eval.dataset import EvalRun, load_prompt
from llm_regression_detector.eval.stats import intervals_overlap, wilson_interval
from llm_regression_detector.llm import mock
from llm_regression_detector.storage.sqlite import SQLiteStorage

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "manifest.json"
BASELINE_PROMPT = ROOT / "prompts" / "incident_triage_v1.yaml"
CANDIDATE_PROMPT = ROOT / "prompts" / "demo" / "incident_triage_v2_degraded.yaml"
GIT = shutil.which("git") or "git"
PACKAGES = ("llm-regression-detector", "litellm", "pydantic", "pydantic-settings")

SENSITIVITY_BASELINES = (0.85, 0.95)
SENSITIVITY_SIZES = (25, 50, 100, 200, 500)
TARGET_DROP = 0.05
"""Inputs of the sensitivity table in docs/golden-dataset-guide.md and the README limits:
the smallest drop each dataset size can call CRITICAL, and the size a 5-point drop needs."""


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run on the deterministic mock instead of local Ollama (offline smoke check).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Keep every run in this new SQLite file, for `lrd dashboard --db` screenshots.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="Baseline and candidate pairs to run (default 3 on Ollama, 1 on the mock).",
    )
    args = parser.parse_args()
    repeats: int = args.repeats or (1 if args.mock else 3)
    if args.db is not None and args.db.exists():
        sys.exit(f"{args.db} exists: pass a new path so each diff compares only this script's runs")

    settings = Settings()
    section = "mock" if args.mock else "ollama"
    env = _clean_env()
    if not args.mock:
        env |= {
            f"{ENV_PREFIX}CUSTOM_MODEL": settings.ollama_model,
            f"{ENV_PREFIX}JUDGE_MODEL": settings.judge_model,
            f"{ENV_PREFIX}OLLAMA_BASE_URL": settings.ollama_base_url,
            f"{ENV_PREFIX}REASONING_EFFORT": settings.reasoning_effort.value,
        }
    runs: list[list[dict[str, Any]]] = []
    severities: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        db = args.db.resolve() if args.db is not None else Path(tmp) / "runs.db"
        for i in range(1, repeats + 1):
            print(f"repeat {i}/{repeats}", flush=True)
            _lrd_run(BASELINE_PROMPT, db, Path(tmp), env, diff=False)
            _lrd_run(CANDIDATE_PROMPT, db, Path(tmp), env, diff=True)
            candidate, baseline = asyncio.run(_two_latest(db, load_prompt(BASELINE_PROMPT).name))
            diff = Analyzer().diff(baseline, candidate)
            runs.append(_metrics(baseline, candidate, diff))
            severities.append(diff.severity.value)

    manifest: dict[str, Any] = (
        json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    )
    manifest["command"] = "uv run python scripts/reproduce.py [--mock]"
    manifest.setdefault("runs", {})[section] = {
        "environment": _environment(),
        "model": _mock_model() if args.mock else _ollama_models(settings),
        "seed": "none: the mock derives each answer from the SHA-256 of the prompt messages"
        if args.mock
        else "none: temperature 0 (greedy decoding); outputs can still shift with the "
        "Ollama version, the model digest or the hardware",
        "dataset": _file_entry(ROOT / _dataset_path(BASELINE_PROMPT)),
        "prompts": {
            "baseline": _file_entry(BASELINE_PROMPT),
            "candidate": _file_entry(CANDIDATE_PROMPT),
        },
        "repeats": runs,
        "summary": _summary(runs),
    }
    manifest["sensitivity"] = _sensitivity()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)} [{section}]: severity per repeat {severities}")


def _clean_env() -> dict[str, str]:
    """The current environment without provider credentials or ``LRD_*`` settings."""
    drop = {"HF_TOKEN", "GEMINI_API_KEY"}
    return {k: v for k, v in os.environ.items() if k not in drop and not k.startswith(ENV_PREFIX)}


def _lrd_run(prompt: Path, db: Path, cwd: Path, env: dict[str, str], *, diff: bool) -> None:
    """``lrd run`` as a user types it; exit code 2 (CRITICAL) is an expected result."""
    cmd = [
        sys.executable,
        "-m",
        "llm_regression_detector",
        "run",
        "--prompt",
        str(prompt),
        "--db",
        str(db),
        "--no-notify",
        "--diff" if diff else "--no-diff",
    ]
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False)  # noqa: S603
    if result.returncode not in (0, 2):
        sys.exit(f"lrd run failed with exit code {result.returncode}: {prompt}")


async def _two_latest(db: Path, prompt_name: str) -> tuple[EvalRun, EvalRun]:
    """Candidate and baseline, newest first, as ``lrd run`` stored them."""
    runs = await SQLiteStorage(db).recent(prompt_name, limit=2)
    if len(runs) != 2:
        sys.exit(f"expected 2 stored runs for {prompt_name}, found {len(runs)}")
    return runs[0], runs[1]


def _environment() -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in PACKAGES},
        "hardware": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }


def _git_commit() -> str:
    """HEAD, or ``uncommitted`` when the working tree differs from it."""

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603
            [GIT, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return "uncommitted" if git("status", "--porcelain") else git("rev-parse", "HEAD")


def _mock_model() -> dict[str, Any]:
    digest = hashlib.sha256(Path(mock.__file__).read_bytes()).hexdigest()
    return {
        "name": "mock",
        "version": f"llm_regression_detector.llm.mock sha256:{digest}",
        "role": "answers and judge",
    }


def _ollama_models(settings: Settings) -> dict[str, Any]:
    """Server version and the digest of each model, as the Ollama API reports them."""

    def get(path: str) -> Any:
        url = settings.ollama_base_url.rstrip("/") + path
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310  # local URL from settings
            return json.load(response)

    digests = {m["name"]: m["digest"] for m in get("/api/tags")["models"]}

    def entry(model: str, role: str) -> dict[str, str]:
        name = model.split("/", 1)[1]
        if name not in digests:
            sys.exit(f"{name} is not pulled: run `ollama pull {name}`")
        return {"name": model, "digest": f"sha256:{digests[name]}", "role": role}

    return {
        "server": f"ollama {get('/api/version')['version']}",
        "under_test": entry(settings.ollama_model, "answers"),
        "judge": entry(settings.judge_model, "judge"),
        "reasoning_effort": settings.reasoning_effort.value,
        "concurrency": settings.concurrency,
        "latency_note": (
            "Latency is not reported: a local Ollama server is often shared with other "
            "clients, so timings can include queueing. Accuracy and judge scores do not "
            "depend on timing."
        ),
    }


def _dataset_path(prompt: Path) -> Path:
    spec = load_prompt(prompt)
    assert spec.dataset is not None, f"{prompt} names no dataset"
    return (prompt.parent / spec.dataset).resolve().relative_to(ROOT)


def _file_entry(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _metrics(baseline: EvalRun, candidate: EvalRun, diff: DiffReport) -> list[dict[str, Any]]:
    """Each number with the sample it was measured on."""
    out: list[dict[str, Any]] = []
    for label, run in (("baseline", baseline), ("candidate", candidate)):
        s = run.summary
        out += [
            {
                "name": f"{label}.accuracy",
                "value": round(s.accuracy, 4),
                "n": s.cases_total,
                "passed": s.cases_passed,
            },
            {
                "name": f"{label}.accuracy_ci95",
                "value": [round(s.accuracy_ci_low, 4), round(s.accuracy_ci_high, 4)],
                "n": s.cases_total,
                "method": "Wilson",
            },
            {
                "name": f"{label}.judge_score_mean",
                "value": round(s.avg_summary_score, 2),
                "n": s.cases_total,
                "scale": "1 to 5",
            },
        ]
        out += [
            {
                "name": f"{label}.accuracy.{c.category}",
                "value": round(c.accuracy, 4),
                "n": c.cases_total,
                "passed": c.cases_passed,
            }
            for c in s.per_category
        ]
    n = len(diff.regressions) + len(diff.improvements) + len(diff.unchanged)
    out += [
        {"name": "diff.accuracy_delta_pp", "value": round(diff.accuracy_delta * 100, 2), "n": n},
        {"name": "diff.severity", "value": diff.severity.value, "n": n},
        {"name": "diff.intervals_overlap", "value": not diff.is_significant, "n": n},
        {"name": "diff.regressed_cases", "value": len(diff.regressions), "n": n},
        {"name": "diff.improved_cases", "value": len(diff.improvements), "n": n},
    ]
    return out


def _summary(runs: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Median, min and max of every scalar number across repeats; counts for labels."""
    out: dict[str, Any] = {"repeats": len(runs)}
    for metric in runs[0]:
        name = metric["name"]
        values = [next(m["value"] for m in run if m["name"] == name) for run in runs]
        if isinstance(values[0], bool | str):
            out[name] = {str(v).lower(): values.count(v) for v in dict.fromkeys(values)}
        elif isinstance(values[0], int | float):
            out[name] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "n": metric["n"],
            }
    return out


def _separates(baseline: float, n: int, lost: int) -> bool:
    """True when losing ``lost`` of ``n`` cases from ``baseline`` accuracy is CRITICAL-sized."""
    passed = round(baseline * n)
    return not intervals_overlap(wilson_interval(passed, n), wilson_interval(passed - lost, n))


def _sensitivity() -> dict[str, Any]:
    """What a dataset of each size can detect, by the same non-overlap rule as the analyzer."""
    min_drop: dict[str, dict[str, float]] = {}
    min_cases: dict[str, int] = {}
    for baseline in SENSITIVITY_BASELINES:
        key = f"{baseline:.2f}"
        min_drop[key] = {}
        for n in SENSITIVITY_SIZES:
            lost = next(k for k in range(1, n + 1) if _separates(baseline, n, k))
            min_drop[key][str(n)] = round(lost / n * 100, 1)
        n = 10
        while not _separates(baseline, n, round(TARGET_DROP * n)):
            n += 1
        min_cases[key] = n
    return {
        "method": (
            "Wilson 95% intervals of the baseline and the candidate do not overlap, "
            "the condition the analyzer needs for CRITICAL; baseline pass count rounded"
        ),
        "min_drop_pp_by_cases": min_drop,
        "min_cases_for_5pp_drop": min_cases,
    }


if __name__ == "__main__":
    main()
