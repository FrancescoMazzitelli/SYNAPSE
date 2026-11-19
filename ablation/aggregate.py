#!/usr/bin/env python3
"""Collect ablation runs and (optionally) evaluate them.

The driver writes one ``manifest.json`` per cell. This script:

1. flattens every manifest into ``<out-dir>/runs.csv`` (one row per run);
2. with ``--evaluate``, calls the repository's ``evaluate.run_single_evaluation``
   on each completed run, writing ``<out-dir>/<model>/<dataset>/.../evaluation_report.txt``
   and ``<out-dir>/metrics.csv``;
3. aggregates metrics per ``(dataset, model, temperature, memory)`` across seeds,
   with mean/std, into ``<out-dir>/summary_by_condition.csv``.

Examples
--------
    python ablation/aggregate.py
    python ablation/aggregate.py --evaluate
    python ablation/aggregate.py --evaluate --dataset data_lyon_EDGT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

RUN_COLUMNS = [
    "run_id", "status", "model", "chat_model", "dataset", "seed", "temperature",
    "memory", "n_agents", "n_questions", "n_responses", "n_expected",
    "duration_s", "started_at", "finished_at", "run_dir",
    "results_csv", "results_json", "oracle", "log", "config", "error",
]

METRIC_COLUMNS = [
    "n_agents", "n_matched_cols", "n_categorical", "n_numeric", "n_free_text",
    "entropy_oracle_median", "entropy_agent_median", "kl_median",
    "bleu_global", "rouge1", "rouge2", "rougeL", "meteor", "bertscore_f1",
    "success_rate", "avg_field_coverage", "tgc", "median_tgc",
    "factual_accuracy", "median_factual_accuracy",
    "relevance_mean", "relevance_median",
]


def flatten_manifest(manifest: dict) -> Dict:
    outputs = manifest.get("outputs") or {}
    memory = manifest.get("memory") or {}
    return {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "model": manifest.get("model"),
        "chat_model": manifest.get("chat_model"),
        "dataset": manifest.get("dataset"),
        "seed": manifest.get("seed"),
        "temperature": manifest.get("temperature"),
        "memory": memory.get("name", memory.get("mode")),
        "n_agents": manifest.get("n_agents"),
        "n_questions": manifest.get("n_questions"),
        "n_responses": manifest.get("n_responses"),
        "n_expected": manifest.get("n_expected"),
        "duration_s": manifest.get("duration_s"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "run_dir": manifest.get("run_dir"),
        "results_csv": outputs.get("results_csv"),
        "results_json": outputs.get("results_json"),
        "oracle": outputs.get("oracle"),
        "log": outputs.get("log"),
        "config": outputs.get("config"),
        "error": manifest.get("error"),
    }


def collect(results_dir: Path, dataset: str = None, model: str = None) -> pd.DataFrame:
    rows: List[Dict] = []
    for manifest_path in sorted(results_dir.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] unreadable manifest {manifest_path}: {exc}")
            continue
        if dataset and manifest.get("dataset") != dataset:
            continue
        model_name = manifest.get("model")
        if model and model_name != model and f"ollama/{model_name}" != model:
            continue
        rows.append(flatten_manifest(manifest))
    return pd.DataFrame(rows, columns=RUN_COLUMNS)


def evaluate_runs(
    runs: pd.DataFrame,
    results_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    try:
        from evaluate import run_single_evaluation
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Cannot import evaluate.py (check its dependencies: rouge-score, "
            "tabulate, scipy, bert-score)."
        ) from exc

    metrics = []
    completed = runs[(runs["status"] == "ok") & runs["results_csv"].notna()]
    for _, row in completed.iterrows():
        run_dir = Path(row["run_dir"]).resolve() if row["run_dir"] else results_dir
        # Mirror the run layout under out_dir: <model>/<dataset>/<seed>/<cell>/
        if run_dir.is_relative_to(results_dir):
            rel = run_dir.relative_to(results_dir)
        else:
            rel = Path(str(row["run_id"]).replace("|", "/"))
        report_path = out_dir / rel / "evaluation_report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        run_info = {
            "model": row["model"],
            "dataset": row["dataset"],
            "results": row["results_csv"],
            "oracle": row["oracle"],
            "log": row["log"] if isinstance(row["log"], str) else "",
        }
        print(f"\n[eval] {row['run_id']}")
        try:
            result = run_single_evaluation(run_info, out_dir, report_path_override=str(report_path))
        except Exception as exc:  # noqa: BLE001 - one failed run must not stop the batch
            print(f"[error] evaluation failed for {row['run_id']}: {exc}")
            result = {"error": str(exc)}
        result.update({
            "run_id": row["run_id"],
            "model": row["model"],
            "dataset": row["dataset"],
            "seed": row["seed"],
            "temperature": row["temperature"],
            "memory": row["memory"],
        })
        metrics.append(result)

    return pd.DataFrame(metrics)


def summarise(metrics_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    if df.empty:
        return df
    keys = ["dataset", "model", "temperature", "memory"]
    available = [c for c in METRIC_COLUMNS if c in df.columns]
    if not available:
        return pd.DataFrame()
    grouped = df.groupby(keys, dropna=False)[available]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std().add_suffix("_std")
    n = grouped.count().iloc[:, :1].rename(columns={available[0]: "n_seeds"})
    summary = pd.concat([mean, std, n], axis=1).reset_index()
    summary = summary.sort_values(keys).reset_index(drop=True)
    summary.to_csv(out_path, index=False)
    return summary


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Collect and evaluate ablation runs.")
    ap.add_argument("--results-dir", default="ablation/results",
                    help="Root written by run_ablation.py")
    ap.add_argument("--out-dir", default="ablation/evaluations",
                    help="Where runs.csv / metrics.csv / reports go")
    ap.add_argument("--dataset", default=None, help="Only this dataset")
    ap.add_argument("--model", default=None, help="Only this model")
    ap.add_argument("--evaluate", action="store_true",
                    help="Run evaluate.py on every completed run (slow)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir)
    if not results_dir.is_dir():
        print(f"[error] results directory not found: {results_dir}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect(results_dir, dataset=args.dataset, model=args.model)
    if runs.empty:
        print(f"[error] no manifest.json found under {results_dir}")
        return 1

    runs_path = out_dir / "runs.csv"
    runs.to_csv(runs_path, index=False)
    print(f"[ok] {len(runs)} run(s) -> {runs_path}")
    print(runs.groupby(["model", "dataset", "status"]).size().to_string())

    if not args.evaluate:
        return 0

    metrics = evaluate_runs(runs, results_dir, out_dir)
    metrics_path = out_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\n[ok] metrics -> {metrics_path}")

    summary = summarise(metrics_path, out_dir / "summary_by_condition.csv")
    if not summary.empty:
        print(f"[ok] summary -> {out_dir / 'summary_by_condition.csv'}")
        cols = [c for c in [
            "dataset", "model", "temperature", "memory", "n_seeds",
            "success_rate_mean", "success_rate_std",
            "tgc_mean", "tgc_std",
            "kl_median_mean", "kl_median_std",
            "factual_accuracy_mean", "factual_accuracy_std",
        ] if c in summary.columns]
        print(summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
