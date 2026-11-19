#!/usr/bin/env python3
"""Evaluate the ablation runs against their oracle.

This module reuses the *same* normalisation, column filtering and metric code as
``evaluate.py`` (the full-study evaluation suite), but computes the
entropy / KL-divergence part plus a per-agent answer-overlap metric, and
aggregates them over the ablation conditions. ``evaluate.py`` itself is never
modified.

Outputs (under ``ablation/evaluations/`` by default):

* ``na_columns_common.csv``        -- columns that are N/A in every oracle of a dataset;
* ``metrics_entropy_kl.csv``       -- one row per completed run;
* ``per_variable_entropy_kl.csv``  -- one row per run x evaluated variable;
* ``answer_overlap_by_agent.csv``  -- per-agent overlap with the oracle;
* ``summary_by_condition.csv``     -- mean/std over seeds, per
  ``dataset x model x temperature x memory``;
* ``summary_by_memory_temperature.csv`` -- the same means collapsed over
  datasets and models;
* ``summary_entropy_kl.md``        -- English Markdown summary tables.

Example
-------
    python ablation/evaluate_ablation.py
    python ablation/evaluate_ablation.py --limit 1 --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation import aggregate  # noqa: E402

NA_SENTINELS = frozenset({
    "", "N/A", "NA", "NAN", "NONE", "NAT", "NULL", "<NR>",
    "-1", "-7", "-8", "-9",
})

METRIC_COLUMNS = [
    "n_agents", "n_matched_cols", "n_entropy_cols", "n_dropped_common_na",
    "entropy_oracle_median", "entropy_agent_median", "delta_entropy_median",
    "kl_median", "kl_mean", "kl_max",
    "answer_overlap", "answer_overlap_median", "n_overlap_fields",
    "n_binarized", "n_dropped_nr", "n_dropped_excluded",
    "n_dropped_constant", "n_dropped_zero_overlap",
]

SUMMARY_COLUMNS = [
    "entropy_oracle_median", "entropy_agent_median",
    "delta_entropy_median", "kl_median", "answer_overlap",
]


# ---------------------------------------------------------------------------
# evaluate.py import shim
# ---------------------------------------------------------------------------

def _install_lightweight_dependencies() -> None:
    """Expose tiny stand-ins for optional NLP imports used by evaluate.py.

    The entropy/KL path only needs ``scipy.stats.entropy``. The other two
    modules are imported at module load time but are not used here.
    """
    if "scipy" not in sys.modules:
        scipy = types.ModuleType("scipy")
        stats = types.ModuleType("scipy.stats")

        def scipy_entropy(pk, qk=None, base=None):
            pk_arr = np.asarray(pk, dtype=float)
            if qk is None:
                mask = pk_arr > 0
                value = -np.sum(pk_arr[mask] * np.log(pk_arr[mask]))
            else:
                qk_arr = np.asarray(qk, dtype=float)
                mask = pk_arr > 0
                value = float(np.sum(pk_arr[mask] * np.log(pk_arr[mask] / qk_arr[mask])))
            return value / math.log(base) if base else value

        stats.entropy = scipy_entropy  # type: ignore[attr-defined]
        scipy.stats = stats  # type: ignore[attr-defined]
        sys.modules["scipy"] = scipy
        sys.modules["scipy.stats"] = stats

    if "rouge_score" not in sys.modules:
        rouge_score = types.ModuleType("rouge_score")
        rouge_scorer = types.ModuleType("rouge_score.rouge_scorer")

        class _RougeScorer:  # pragma: no cover - never instantiated here
            def __init__(self, *args, **kwargs):
                pass

        rouge_scorer.RougeScorer = _RougeScorer  # type: ignore[attr-defined]
        rouge_score.rouge_scorer = rouge_scorer  # type: ignore[attr-defined]
        sys.modules["rouge_score"] = rouge_score
        sys.modules["rouge_score.rouge_scorer"] = rouge_scorer

    if "tabulate" not in sys.modules:
        tabulate = types.ModuleType("tabulate")

        def _tabulate(rows, **kwargs):  # pragma: no cover - not used here
            return str(rows)

        tabulate.tabulate = _tabulate  # type: ignore[attr-defined]
        sys.modules["tabulate"] = tabulate


def load_evaluate_module():
    """Import evaluate.py with full-study defaults and light dependencies."""
    _install_lightweight_dependencies()
    import evaluate  # noqa: PLC0415

    evaluate.CATEGORICAL_MAX_UNIQUE = 20
    evaluate.FREE_TEXT_MIN_UNIQUE = 5
    evaluate.NR_THRESHOLD = 0.9
    evaluate.AGENT_VAR_MIN_UNIQUE = 2
    evaluate.EXCLUDED_COLS = set()
    return evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def local_output(manifest_path: Path, recorded: Optional[str], results_dir: Path) -> Optional[Path]:
    """Resolve an absolute /root/SYNAPSE output path to the local copy."""
    if not recorded:
        return None
    name = Path(recorded).name
    candidate = manifest_path.parent / name
    if candidate.exists():
        return candidate

    recorded_path = Path(recorded)
    if recorded_path.exists():
        return recorded_path

    parts = recorded_path.parts
    if "results" in parts:
        idx = len(parts) - 1 - parts[::-1].index("results")
        fallback = results_dir.parent.joinpath(*parts[idx:])
        if fallback.exists():
            return fallback
    return candidate


def _median(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return round(float(series.median()), 4)


def _round(value) -> float:
    return round(float(value), 4) if value is not None and not pd.isna(value) else float("nan")


def _is_na(value) -> bool:
    """True for blank / N/A / non-response values."""
    if pd.isna(value):
        return True
    return str(value).strip().upper() in NA_SENTINELS


def _normalised_name(name) -> str:
    """Same key as evaluate.load_and_map: lowercase + spaces to underscores."""
    return str(name).lower().replace(" ", "_")


def find_common_na_columns(results_dir: Path) -> (Dict[str, set], List[Dict]):
    """Find columns that are N/A in *every* run of a dataset.

    Detection is done on the raw per-run oracle and agent CSVs. The returned
    dictionary is keyed by dataset and uses normalised column names, so it can
    be applied directly to the aligned columns built by ``load_and_map``.
    """
    by_dataset = defaultdict(lambda: {"results": [], "oracle": []})

    for manifest_path in sorted(results_dir.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if manifest.get("status") != "ok":
            continue
        dataset = manifest.get("dataset")
        outputs = manifest.get("outputs") or {}
        results_csv = local_output(manifest_path, outputs.get("results_csv"), results_dir)
        oracle_csv = local_output(manifest_path, outputs.get("oracle"), results_dir)
        if not results_csv or not oracle_csv or not results_csv.exists() or not oracle_csv.exists():
            continue

        res_raw = pd.read_csv(results_csv, dtype=str)
        ora_raw = pd.read_csv(oracle_csv, dtype=str)
        by_dataset[dataset]["results"].append(
            {c for c in res_raw.columns if res_raw[c].map(_is_na).all()}
        )
        by_dataset[dataset]["oracle"].append(
            {c for c in ora_raw.columns if ora_raw[c].map(_is_na).all()}
        )

    common: Dict[str, set] = {}
    details: List[Dict] = []
    for dataset, sets in sorted(by_dataset.items()):
        for source in ("results", "oracle"):
            per_run = sets[source]
            intersection = set.intersection(*per_run) if per_run else set()
            if source == "oracle":
                common[dataset] = {_normalised_name(c) for c in intersection}
            for col in sorted(intersection):
                details.append({
                    "dataset": dataset,
                    "source": source,
                    "column": col,
                    "n_runs_checked": len(per_run),
                })
    return common, details


def compute_answer_overlap(
    res: pd.DataFrame,
    ora: pd.DataFrame,
    evaluate,
    agent_ids: Optional[List[str]] = None,
) -> (List[float], List[Dict]):
    """Token overlap between each agent answer and its oracle counterpart.

    For every agent, the score is the fraction of retained survey fields whose
    normalised answer shares at least one token with the oracle answer. Fields
    with an empty/N-A oracle value are skipped. This reuses ``norm_str`` and
    ``tokens_overlap`` from ``evaluate.py``, i.e. the same matching rule used by
    TGC and factual accuracy in the full study.
    """
    per_agent: List[float] = []
    details: List[Dict] = []
    columns = [c for c in res.columns if c in ora.columns]

    for i, (_, row) in enumerate(res.iterrows()):
        ora_row = ora.iloc[i]
        hits = 0
        total = 0
        for col in columns:
            ov = ora_row[col]
            if not evaluate.norm_str(ov):
                continue
            total += 1
            hit = bool(evaluate.tokens_overlap(row[col], ov))
            hits += int(hit)
            details.append({
                "agent_index": i,
                "agent_id": agent_ids[i] if agent_ids and i < len(agent_ids) else f"Agent_{i}",
                "variable": col,
                "agent_answer": str(row[col]),
                "oracle_answer": str(ov),
                "overlap": int(hit),
            })
        per_agent.append(hits / total if total else float("nan"))
    return per_agent, details


# ---------------------------------------------------------------------------
# Core evaluation of one run
# ---------------------------------------------------------------------------

def evaluate_run(
    manifest_path: Path,
    results_dir: Path,
    evaluate,
    with_per_variable: bool = True,
    global_na_columns: Optional[Dict[str, set]] = None,
) -> (Dict, List[Dict], List[Dict]):
    """Compute entropy, KL and per-agent answer overlap for one run."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    results_csv = local_output(manifest_path, outputs.get("results_csv"), results_dir)
    oracle_csv = local_output(manifest_path, outputs.get("oracle"), results_dir)

    if results_csv is None or oracle_csv is None:
        raise FileNotFoundError(f"missing results/oracle for {manifest_path}")
    if not results_csv.exists() or not oracle_csv.exists():
        raise FileNotFoundError(f"missing local file: {results_csv or oracle_csv}")

    dataset = manifest.get("dataset")
    evaluate._load_dataset_dictionary(dataset)

    res_ids_raw = pd.read_csv(results_csv, dtype=str)
    agent_ids = (
        res_ids_raw["agent_id"].astype(str).tolist()
        if "agent_id" in res_ids_raw.columns else None
    )

    res, ora, col_map, _meta, n = evaluate.load_and_map(str(results_csv), str(oracle_csv))

    # Remove columns that are N/A in every oracle of this dataset. They are
    # dropped from both sides so they cannot affect entropy or KL.
    n_dropped_common_na = 0
    if global_na_columns:
        na_keys = global_na_columns.get(dataset, set())
        drop_na = [c for c in res.columns if _normalised_name(c) in na_keys]
        if drop_na:
            res.drop(columns=drop_na, inplace=True)
            ora.drop(columns=drop_na, inplace=True)
            n_dropped_common_na = len(drop_na)

    res, ora = evaluate.resolve_code_label_format(res, ora)

    n_binarized = len(evaluate.binarize_flag_cols(res, ora))

    nr_cols = evaluate.identify_nr_columns(ora, evaluate.NR_THRESHOLD)
    excluded = [c for c in evaluate.EXCLUDED_COLS if c in ora.columns]
    for cols in (nr_cols, excluded):
        drop = [c for c in cols if c in res.columns]
        res.drop(columns=drop, inplace=True)
        ora.drop(columns=drop, inplace=True)

    # Oracle entropy is computed on the model-invariant column set, exactly as
    # in evaluate.run_single_evaluation().
    ora_base = ora.copy()
    canonical_cols = evaluate.select_entropy_kl_columns(ora_base, ora_base)

    constant_cols = evaluate.identify_agent_constant_columns(res, ora)
    zero_overlap_cols = evaluate.identify_zero_overlap_columns(res, ora)
    for cols in (constant_cols, zero_overlap_cols):
        drop = [c for c in cols if c in res.columns]
        res.drop(columns=drop, inplace=True)
        ora.drop(columns=drop, inplace=True)

    entropy_kl_cols = [c for c in canonical_cols if c in res.columns and c in ora.columns]

    entropy_df = evaluate.compute_entropy_table(
        res, ora, entropy_kl_cols, max_cats=evaluate.CATEGORICAL_MAX_UNIQUE
    )
    kl_df = evaluate.compute_kl_table(
        res, ora, entropy_kl_cols, max_cats=evaluate.CATEGORICAL_MAX_UNIQUE
    )
    canonical_entropy_df = evaluate.compute_entropy_table(
        ora_base, ora_base, canonical_cols, max_cats=evaluate.CATEGORICAL_MAX_UNIQUE
    )

    overlap_scores, overlap_details = compute_answer_overlap(
        res, ora, evaluate, agent_ids[:n] if agent_ids else None
    )

    memory = manifest.get("memory") or {}
    metrics = {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "model": manifest.get("model"),
        "chat_model": manifest.get("chat_model"),
        "dataset": dataset,
        "seed": manifest.get("seed"),
        "temperature": manifest.get("temperature"),
        "memory": memory.get("name", memory.get("mode")) if isinstance(memory, dict) else memory,
        "n_agents": n,
        "n_matched_cols": len(col_map),
        "n_entropy_cols": len(entropy_kl_cols),
        "n_dropped_common_na": n_dropped_common_na,
        "entropy_oracle_median": _median(canonical_entropy_df["Entropy (Oracle)"]),
        "entropy_agent_median": _median(entropy_df["Entropy (Agent)"]),
        "delta_entropy_median": _median(entropy_df["Delta Entropy"]),
        "kl_median": _median(kl_df["KL Divergence (Oracle || Agent)"]),
        "kl_mean": _round(kl_df["KL Divergence (Oracle || Agent)"].mean())
        if not kl_df.empty else float("nan"),
        "kl_max": _round(kl_df["KL Divergence (Oracle || Agent)"].max())
        if not kl_df.empty else float("nan"),
        "answer_overlap": _round(pd.Series(overlap_scores).mean()),
        "answer_overlap_median": _round(pd.Series(overlap_scores).median()),
        "n_overlap_fields": len(overlap_details) // n if n else 0,
        "n_binarized": n_binarized,
        "n_dropped_nr": len(nr_cols),
        "n_dropped_excluded": len(excluded),
        "n_dropped_constant": len(constant_cols),
        "n_dropped_zero_overlap": len(zero_overlap_cols),
        "error": "",
    }

    per_variable: List[Dict] = []
    if with_per_variable and not entropy_df.empty:
        kl_by_col = dict(zip(
            kl_df["Variable"],
            kl_df["KL Divergence (Oracle || Agent)"],
        ))
        ratings = dict(zip(kl_df["Variable"], kl_df["Rating"]))
        for _, row in entropy_df.iterrows():
            col = row["Variable"]
            per_variable.append({
                "run_id": metrics["run_id"],
                "model": metrics["model"],
                "dataset": dataset,
                "seed": metrics["seed"],
                "temperature": metrics["temperature"],
                "memory": metrics["memory"],
                "variable": col,
                "categories": int(row["# Categories"]),
                "entropy_oracle": row["Entropy (Oracle)"],
                "entropy_agent": row["Entropy (Agent)"],
                "delta_entropy": row["Delta Entropy"],
                "kl_oracle_agent": _round(kl_by_col.get(col, float("nan"))),
                "rating": ratings.get(col, ""),
            })

    for detail in overlap_details:
        detail.update({
            "run_id": metrics["run_id"],
            "model": metrics["model"],
            "dataset": dataset,
            "seed": metrics["seed"],
            "temperature": metrics["temperature"],
            "memory": metrics["memory"],
        })
    return metrics, per_variable, overlap_details


def evaluate_all(
    runs: pd.DataFrame,
    results_dir: Path,
    out_dir: Path,
    evaluate,
    verbose: bool = False,
    with_per_variable: bool = True,
    global_na_columns: Optional[Dict[str, set]] = None,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """Evaluate every successful run listed in ``runs``."""
    metrics_rows: List[Dict] = []
    variable_rows: List[Dict] = []
    overlap_rows: List[Dict] = []
    manifest_by_id = {row["run_id"]: None for _, row in runs.iterrows()}

    for manifest_path in sorted(results_dir.rglob("manifest.json")):
        try:
            run_id = json.loads(manifest_path.read_text(encoding="utf-8")).get("run_id")
        except Exception:  # noqa: BLE001
            continue
        manifest_by_id[run_id] = manifest_path

    completed = runs[(runs["status"] == "ok") & runs["run_id"].notna()]
    total = len(completed)
    for i, (_, row) in enumerate(completed.iterrows(), 1):
        run_id = row["run_id"]
        manifest_path = manifest_by_id.get(run_id)
        if manifest_path is None:
            print(f"[warn] no manifest found for {run_id}")
            continue
        if verbose:
            print(f"[eval {i}/{total}] {run_id}")
        try:
            metrics, per_variable, per_agent_overlap = evaluate_run(
                manifest_path, results_dir, evaluate, with_per_variable,
                global_na_columns,
            )
        except Exception as exc:  # noqa: BLE001 - one bad run must not stop the batch
            print(f"[error] {run_id}: {exc}")
            metrics = {
                "run_id": run_id,
                "status": "failed",
                "model": row["model"],
                "dataset": row["dataset"],
                "seed": row["seed"],
                "temperature": row["temperature"],
                "memory": row["memory"],
                "error": str(exc),
            }
            per_variable = []
            per_agent_overlap = []
        metrics_rows.append(metrics)
        variable_rows.extend(per_variable)
        overlap_rows.extend(per_agent_overlap)

    metrics_df = pd.DataFrame(metrics_rows)
    variables_df = pd.DataFrame(variable_rows)
    overlap_df = pd.DataFrame(overlap_rows)
    return metrics_df, variables_df, overlap_df


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of the per-run medians, grouped by ablation condition."""
    if metrics.empty:
        return pd.DataFrame()

    keys = ["dataset", "model", "temperature", "memory"]
    agg: Dict[str, List[str]] = {col: ["mean", "std"] for col in SUMMARY_COLUMNS}
    agg["n_agents"] = ["first"]
    grouped = metrics.groupby(keys, dropna=False).agg(agg)
    grouped.columns = [f"{col}_{stat}" if stat != "first" else "n_agents" for col, stat in grouped.columns]
    grouped["n_runs"] = metrics.groupby(keys, dropna=False).size()
    summary = grouped.reset_index()
    return summary.sort_values(keys).reset_index(drop=True)


def summarise_compact(metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse datasets/models to show the memory x temperature effect."""
    if metrics.empty:
        return pd.DataFrame()
    keys = ["memory", "temperature"]
    agg = {col: "mean" for col in SUMMARY_COLUMNS}
    grouped = metrics.groupby(keys, dropna=False).agg(agg)
    grouped["n_runs"] = metrics.groupby(keys, dropna=False).size()
    summary = grouped.reset_index().sort_values(keys)
    return summary.reset_index(drop=True)


def markdown_table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    if df.empty:
        return "(no data)"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: float_fmt.format(v) if pd.notna(v) else "NA")
    headers = list(out.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in headers) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, compact: pd.DataFrame, summary: pd.DataFrame) -> Path:
    report = [
        "# Ablation evaluation: entropy and KL divergence",
        "",
        "All metrics are computed with the same normalisation, column filtering",
        "and formulas as `evaluate.py` in the full study. Columns that are N/A",
        "in every oracle of a dataset are removed globally from both results and",
        "oracle before evaluation (see `na_columns_common.csv`). Entropy is",
        "normalised Shannon entropy; KL is KL(oracle || agent), so lower is better.",
        "`answer_overlap` is the per-agent fraction of retained fields whose",
        "answer shares a token with the oracle, averaged over agents (higher is",
        "better); per-agent values are in `answer_overlap_by_agent.csv`.",
        "`delta_entropy` is agent entropy minus oracle entropy: negative means",
        "the agents are less diverse than the oracle, positive means more diverse.",
        "",
        "## By memory condition and temperature (averaged over datasets and models)",
        "",
        markdown_table(compact),
        "",
        "## By dataset, model, memory and temperature (mean over seeds)",
        "",
        markdown_table(summary),
        "",
    ]
    path = out_dir / "summary_entropy_kl.md"
    path.write_text("\n".join(report), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="ablation/results",
                    help="Root directory containing the ablation manifests")
    ap.add_argument("--out-dir", default="ablation/evaluations",
                    help="Where CSV and Markdown summaries are written")
    ap.add_argument("--dataset", default=None, help="Only evaluate this dataset")
    ap.add_argument("--model", default=None, help="Only evaluate this model")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N runs (smoke test)")
    ap.add_argument("--no-per-variable", action="store_true",
                    help="Do not write the per-variable detail table")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print one line per evaluated run")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir)
    if not results_dir.is_dir():
        print(f"[error] results directory not found: {results_dir}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = aggregate.collect(results_dir, dataset=args.dataset, model=args.model)
    if runs.empty:
        print(f"[error] no manifest.json found under {results_dir}")
        return 1
    if args.limit:
        runs = runs.head(args.limit)

    runs_path = out_dir / "runs.csv"
    runs.to_csv(runs_path, index=False)
    print(f"[ok] {len(runs)} run(s) -> {runs_path}")

    global_na_columns, na_details = find_common_na_columns(results_dir)
    na_path = out_dir / "na_columns_common.csv"
    pd.DataFrame(na_details, columns=["dataset", "source", "column", "n_runs_checked"]).to_csv(
        na_path, index=False
    )
    if na_details:
        counts = pd.DataFrame(na_details).groupby(["dataset", "source"]).size()
        print("[ok] globally N/A columns -> " + str(na_path))
        print(counts.to_string())
    else:
        print(f"[ok] no globally N/A columns -> {na_path}")

    evaluate = load_evaluate_module()
    metrics, variables, overlap = evaluate_all(
        runs, results_dir, out_dir, evaluate,
        verbose=args.verbose,
        with_per_variable=not args.no_per_variable,
        global_na_columns=global_na_columns,
    )
    if metrics.empty:
        print("[error] no run could be evaluated")
        return 1

    metrics_path = out_dir / "metrics_entropy_kl.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"[ok] per-run metrics -> {metrics_path}")

    if not variables.empty:
        variables_path = out_dir / "per_variable_entropy_kl.csv"
        variables.to_csv(variables_path, index=False)
        print(f"[ok] per-variable metrics -> {variables_path}")

    if not overlap.empty:
        overlap_path = out_dir / "answer_overlap_by_agent.csv"
        overlap.to_csv(overlap_path, index=False)
        print(f"[ok] per-agent overlap -> {overlap_path}")

    summary = summarise(metrics)
    summary_path = out_dir / "summary_by_condition.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[ok] condition summary -> {summary_path}")

    compact = summarise_compact(metrics)
    compact_path = out_dir / "summary_by_memory_temperature.csv"
    compact.to_csv(compact_path, index=False)

    report_path = write_report(out_dir, compact, summary)
    print(f"[ok] markdown summary -> {report_path}")
    print()
    print(markdown_table(compact, float_fmt="{:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
