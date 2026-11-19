#!/usr/bin/env python3
"""Estimate SYNAPSE runtime from the measured 200-agent runs.

No regression is used. For every dataset we take the mean duration of the
available 200-agent runs, divide by 200 to get the average time per simulated
person, then multiply by the dataset's ``n_persons``. The result is expressed
in raw hours and in 8-hour working days, and compared with the fieldwork duration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COSTS_DIR = Path(__file__).resolve().parent
DEFAULT_RUNTIMES = COSTS_DIR / "data" / "runtimes.csv"
DEFAULT_METADATA = COSTS_DIR / "data" / "dataset_metadata.csv"
DEFAULT_OUTPUT_DIR = COSTS_DIR / "outputs"
AGENTS_PER_RUN = 200

REQUIRED_RUN_COLUMNS = {"dataset", "model", "n_agents", "duration_hours"}
REQUIRED_METADATA_COLUMNS = {"dataset", "label", "n_persons", "fieldwork_days"}


def load_tables(runtimes_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(runtimes_path)
    metadata = pd.read_csv(metadata_path)

    missing_runs = REQUIRED_RUN_COLUMNS - set(runs.columns)
    if missing_runs:
        raise ValueError(f"{runtimes_path} is missing columns: {sorted(missing_runs)}")
    missing_metadata = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing_metadata:
        raise ValueError(f"{metadata_path} is missing columns: {sorted(missing_metadata)}")

    runs["n_agents"] = pd.to_numeric(runs["n_agents"], errors="coerce")
    runs["duration_hours"] = pd.to_numeric(runs["duration_hours"], errors="coerce")
    runs = runs.dropna(subset=["n_agents", "duration_hours"])
    runs = runs[runs["n_agents"] == AGENTS_PER_RUN]
    runs = runs[runs["duration_hours"] > 0]

    metadata["n_persons"] = pd.to_numeric(metadata["n_persons"], errors="coerce")
    metadata["fieldwork_days"] = pd.to_numeric(metadata["fieldwork_days"], errors="coerce")
    metadata = metadata.dropna(subset=["n_persons"])

    if runs.empty:
        raise RuntimeError(f"No {AGENTS_PER_RUN}-agent runs with valid duration in {runtimes_path}")

    runs = runs[runs["dataset"].isin(metadata["dataset"])].copy()
    if runs.empty:
        raise RuntimeError("No measured datasets found in the metadata table")
    return runs, metadata


def estimate(runs: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    mean_per_model = (
        runs.groupby(["dataset", "model"])
        .agg(n_runs=("duration_hours", "size"), mean_hours_200=("duration_hours", "mean"))
        .reset_index()
    )
    mean_per_model["hours_per_person"] = mean_per_model["mean_hours_200"] / AGENTS_PER_RUN

    meta_columns = metadata[["dataset", "label", "n_persons", "fieldwork_days"]]
    table = meta_columns.merge(mean_per_model, on="dataset", how="inner")
    table["estimated_hours"] = table["hours_per_person"] * table["n_persons"]
    table["estimated_workdays_8h"] = table["estimated_hours"] / 8.0
    table["workdays_vs_fieldwork"] = table["estimated_workdays_8h"] / table["fieldwork_days"]
    return table.sort_values(["dataset", "model"]).reset_index(drop=True)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if pd.api.types.is_number(value) and not isinstance(value, bool):
                if pd.isna(value):
                    values.append("n/a")
                else:
                    number = float(value)
                    values.append(f"{int(number):,}" if number.is_integer() else f"{number:,.2f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(table: pd.DataFrame, output_dir: Path) -> None:
    view = table[
        [
            "label",
            "model",
            "n_runs",
            "n_persons",
            "hours_per_person",
            "estimated_hours",
            "estimated_workdays_8h",
            "fieldwork_days",
            "workdays_vs_fieldwork",
        ]
    ].rename(
        columns={
            "label": "dataset",
            "n_runs": "runs",
            "n_persons": "n_persons",
            "hours_per_person": "h/person",
            "estimated_hours": "estimated h (raw)",
            "estimated_workdays_8h": "workdays (8h)",
            "fieldwork_days": "fieldwork days",
            "workdays_vs_fieldwork": "x fieldwork",
        }
    )

    lines = []
    for dataset, group in table.groupby("dataset", sort=True):
        label = group["label"].iloc[0]
        low = group["estimated_workdays_8h"].min()
        high = group["estimated_workdays_8h"].max()
        fieldwork = group["fieldwork_days"].iloc[0]
        fieldwork_text = "n/a" if pd.isna(fieldwork) else f"{float(fieldwork):,.0f}"
        lines.append(
            f"- **{label}** (`{dataset}`): {low:,.1f}–{high:,.1f} 8-hour workdays across models "
            f"vs {fieldwork_text} fieldwork days."
        )
    comparisons = "\n".join(lines)

    report = f"""# Runtime estimate from the 200-agent runs

## Method

No regression model is used. For each dataset **and model** we take the measured
{AGENTS_PER_RUN}-agent run and compute the time per simulated person:

```
hours_per_person   = mean(duration_hours) / {AGENTS_PER_RUN}
estimated_hours    = hours_per_person * n_persons        # raw value
estimated_workdays = estimated_hours / 8                 # 8-hour working day
```

`n_persons` comes from `dataset_metadata.csv`. Every row below is one dataset
and one model, so model-specific runtimes are kept separate. Fieldwork days are
used only as context for the comparison and never to compute the estimate.

## Results

{markdown_table(view)}

### Comparison in 8-hour workdays

{comparisons}

## Notes

- No model factors, questionnaire-length effects or regression coefficients are
  estimated; each value is a direct linear rescale of that model's 200-agent
  run.
- Absolute values depend on the machine and on the model used for the run.
- `estimated h (raw)` is unnormalised; `workdays (8h)` divides those hours by 8.
  Fieldwork days remain calendar days, so the ratio compares 8-hour workdays
  with calendar fieldwork days.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtimes", type=Path, default=DEFAULT_RUNTIMES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-scan results/ and ablation/results/ before estimating.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.refresh:
        from collect_runtimes import collect_runtimes

        print("Collecting run times from results/ ...")
        collect_runtimes(output_path=args.runtimes)

    runs, metadata = load_tables(args.runtimes, args.metadata)
    result = estimate(runs, metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "estimate.csv", index=False)
    write_report(result, args.output_dir)

    print(
        result[
            [
                "dataset",
                "model",
                "hours_per_person",
                "n_persons",
                "estimated_workdays_8h",
                "fieldwork_days",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote {args.output_dir / 'estimate.csv'} and {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
