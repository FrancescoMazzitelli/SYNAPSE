#!/usr/bin/env python3
"""Collect wall-clock times of completed SYNAPSE survey runs.

Two sources are supported:

* ``results/<model>/<dataset>/log.txt`` -- the original 200-agent runs.
* ``ablation/results/**/manifest.json`` -- the ablation runs (any agent count,
  memory condition and temperature), when they exist.

The output is a tidy CSV (``costs/data/runtimes.csv``) used by
``costs/estimate_costs.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

COSTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = COSTS_DIR.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_ABLATION_DIR = REPO_ROOT / "ablation" / "results"
DEFAULT_OUTPUT = COSTS_DIR / "data" / "runtimes.csv"

# Columns added by the survey pipeline that are not questionnaire answers.
METADATA_COLUMNS = {"agent_id", "serial_number", "agent_bio"}


def _number(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _count_questions(results_csv: Path) -> int | None:
    """Count answer columns in a result CSV (all columns except metadata)."""
    try:
        columns = pd.read_csv(results_csv, nrows=0).columns
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] cannot read header of {results_csv}: {exc}", file=sys.stderr)
        return None
    return int(sum(column not in METADATA_COLUMNS for column in columns))


def _count_rows(csv_path: Path) -> int | None:
    try:
        return int(pd.read_csv(csv_path, usecols=[0]).shape[0])
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] cannot count rows of {csv_path}: {exc}", file=sys.stderr)
        return None


def _first_result_csv(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("*_results.csv"))
    return candidates[0] if candidates else None


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def collect_legacy(results_dir: Path) -> list[dict[str, Any]]:
    """Parse the original ``results/<model>/<dataset>/log.txt`` runs."""
    rows: list[dict[str, Any]] = []
    if not results_dir.exists():
        return rows

    for log_path in sorted(results_dir.rglob("log.txt")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        duration_hours = _number(r"^Duration \(hours\):\s*([\d.]+)", text)
        if duration_hours is None:
            print(f"[warn] no duration in {log_path}", file=sys.stderr)
            continue

        rel = log_path.relative_to(results_dir)
        if len(rel.parts) < 3:
            print(f"[warn] unexpected log layout: {log_path}", file=sys.stderr)
            continue
        model, dataset = rel.parts[0], rel.parts[1]
        run_dir = log_path.parent

        result_csv = _first_result_csv(run_dir)
        n_questions = _count_questions(result_csv) if result_csv else None

        oracle_csv = run_dir / "oracle" / "sampled_oracle.csv"
        n_agents = _count_rows(oracle_csv) if oracle_csv.exists() else None
        if n_agents is None and result_csv is not None:
            n_agents = _count_rows(result_csv)

        if n_agents is None or n_questions is None:
            print(
                f"[warn] missing agent/question count for {model}/{dataset}; "
                "row will be skipped",
                file=sys.stderr,
            )
            continue

        start = re.search(r"^Start Time:\s*(.+)$", text, flags=re.MULTILINE)
        end = re.search(r"^End Time:\s*(.+)$", text, flags=re.MULTILINE)
        rows.append(
            {
                "run_id": f"{model}|{dataset}|legacy",
                "source": "legacy",
                "model": model,
                "dataset": dataset,
                "n_agents": int(n_agents),
                "n_questions": int(n_questions),
                "duration_hours": float(duration_hours),
                "started_at": start.group(1).strip() if start else "",
                "finished_at": end.group(1).strip() if end else "",
                "temperature": None,
                "memory": None,
                "results_csv": _display_path(result_csv) if result_csv else "",
                "log_path": _display_path(log_path),
                "notes": "",
            }
        )
    return rows


def collect_ablation(ablation_dir: Path) -> list[dict[str, Any]]:
    """Parse ablation manifests, which carry their own run metadata."""
    rows: list[dict[str, Any]] = []
    if not ablation_dir.exists():
        return rows

    for manifest_path in sorted(ablation_dir.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] cannot read {manifest_path}: {exc}", file=sys.stderr)
            continue

        if manifest.get("status") != "ok":
            continue
        duration_s = manifest.get("duration_s")
        n_agents = manifest.get("n_agents")
        n_questions = manifest.get("n_questions")
        if duration_s is None or n_agents is None or n_questions is None:
            print(
                f"[warn] incomplete manifest {manifest_path}; row will be skipped",
                file=sys.stderr,
            )
            continue

        run_id = manifest.get("run_id")
        if not run_id:
            run_id = str(manifest_path.parent.relative_to(ablation_dir))
        outputs = manifest.get("outputs") or {}
        results_csv = outputs.get("results_csv") or outputs.get("results") or ""
        log_path = outputs.get("log") or ""
        memory = manifest.get("memory")
        if isinstance(memory, dict):
            memory = memory.get("name") or memory.get("mode")
        rows.append(
            {
                "run_id": str(run_id),
                "source": "ablation",
                "model": str(manifest.get("model", "")),
                "dataset": str(manifest.get("dataset", "")),
                "n_agents": int(n_agents),
                "n_questions": int(n_questions),
                "duration_hours": float(duration_s) / 3600.0,
                "started_at": str(manifest.get("started_at") or ""),
                "finished_at": str(manifest.get("finished_at") or ""),
                "temperature": manifest.get("temperature"),
                "memory": memory,
                "results_csv": str(results_csv),
                "log_path": str(log_path),
                "notes": str(manifest.get("notes") or ""),
            }
        )
    return rows


def collect_runtimes(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    ablation_dir: Path = DEFAULT_ABLATION_DIR,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    rows = collect_legacy(results_dir) + collect_ablation(ablation_dir)
    if not rows:
        raise RuntimeError(
            "No completed runs found. Expected results/*/*/log.txt or "
            "ablation/results/**/manifest.json."
        )

    frame = pd.DataFrame(rows)
    # Keep the first occurrence of duplicate run ids, but tell the user.
    duplicated = frame["run_id"].duplicated(keep="first")
    if duplicated.any():
        print(
            f"[warn] dropping {int(duplicated.sum())} duplicate run_id rows",
            file=sys.stderr,
        )
        frame = frame.loc[~duplicated].copy()

    frame = frame.sort_values(
        ["dataset", "model", "n_agents", "source", "run_id"], kind="stable"
    ).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory with the original <model>/<dataset>/log.txt runs.",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=DEFAULT_ABLATION_DIR,
        help="Directory with ablation manifests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = collect_runtimes(args.results_dir, args.ablation_dir, args.output)
    print(f"Wrote {len(frame)} runs to {args.output}")
    if not frame.empty:
        print(
            "Sources: "
            + ", ".join(
                f"{name}={count}"
                for name, count in frame["source"].value_counts().sort_index().items()
            )
        )
        print(
            "Agent counts: "
            + ", ".join(
                f"{int(value)}={count}"
                for value, count in frame["n_agents"].value_counts().sort_index().items()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
