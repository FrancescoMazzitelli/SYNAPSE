# costs — runtime estimate from the 200-agent runs

This folder estimates SYNAPSE runtime with a simple linear rescale of the
measured 200-agent runs. There is no regression, no model/dataset effects and
no questionnaire-length model.

For each dataset **and model**:

```
hours_per_person   = duration_hours of the 200-agent run / 200
estimated_hours    = hours_per_person * n_persons        # raw
estimated_workdays = estimated_hours / 8                 # 8-hour working day
```

`n_persons` comes from `data/dataset_metadata.csv`. Each dataset-model pair has
its own row. Both the raw hours and the 8-hour working days are reported, and
compared with the dataset's `fieldwork_days`.

## Run

```bash
python costs/estimate_costs.py
# or re-collect the timings first:
python costs/estimate_costs.py --refresh
# or:
bash costs/run_costs.sh
```

## Files

| File | Contents |
|---|---|
| `data/runtimes.csv` | Measured runs used as input |
| `data/dataset_metadata.csv` | Dataset `n_persons` and `fieldwork_days` |
| `collect_runtimes.py` | Optional collector for `runtimes.csv` |
| `estimate_costs.py` | The whole calculation and report |
| `outputs/estimate.csv` | One row per dataset-model pair with hours and days |
| `outputs/report.md` | Short report with method and comparison |

## Output columns

- `model`, `dataset`: the pair the estimate refers to.
- `n_runs`: number of 200-agent runs averaged (normally 1).
- `hours_per_person`: that model's measured hours divided by 200.
- `n_persons`: target sample size from the metadata.
- `estimated_hours`: raw linear extrapolation.
- `estimated_workdays_8h`: the same value divided by 8 (8-hour working day).
- `fieldwork_days` / `workdays_vs_fieldwork`: context comparison.

## Assumption

Time per simulated person is constant when scaling from 200 agents to
`n_persons`. No model factors or questionnaire effects are estimated: each
model uses only its own measured 200-agent run. Fieldwork days remain calendar
days, so the ratio compares 8-hour workdays with calendar fieldwork days.
