# Runtime estimate from the 200-agent runs

## Method

For each dataset **and model** we take the measured
200-agent run and compute the time per simulated person:

```
hours_per_person   = mean(duration_hours) / 200
estimated_hours    = hours_per_person * n_persons        # raw value
estimated_workdays = estimated_hours / 8                 # 8-hour working day
```

`n_persons` comes from `dataset_metadata.csv`. Every row below is one dataset
and one model, so model-specific runtimes are kept separate. Fieldwork days are
used only as context for the comparison and never to compute the estimate.

## Results

| dataset | model | runs | n_persons | h/person | estimated h (raw) | workdays (8h) | fieldwork days | x fieldwork |
|---|---|---|---|---|---|---|---|---|
| NYC Citywide Mobility Survey (CMS) | gemma2:27b | 1 | 2,966 | 0.05 | 145.63 | 18.20 | 51 | 0.36 |
| NYC Citywide Mobility Survey (CMS) | gemma2:9b | 1 | 2,966 | 0.04 | 112.41 | 14.05 | 51 | 0.28 |
| NYC Citywide Mobility Survey (CMS) | gemma3:27b | 1 | 2,966 | 0.11 | 334.12 | 41.76 | 51 | 0.82 |
| NYC Citywide Mobility Survey (CMS) | gpt-oss:20b | 1 | 2,966 | 0.29 | 858.95 | 107.37 | 51 | 2.11 |
| NYC Citywide Mobility Survey (CMS) | llama3.1:8b | 1 | 2,966 | 0.02 | 71.04 | 8.88 | 51 | 0.17 |
| NYC Citywide Mobility Survey (CMS) | llama3.3:70b | 1 | 2,966 | 0.31 | 905.82 | 113.23 | 51 | 2.22 |
| NYC Citywide Mobility Survey (CMS) | mistral-nemo:12b | 1 | 2,966 | 0.03 | 90.31 | 11.29 | 51 | 0.22 |
| NYC Citywide Mobility Survey (CMS) | mistral:7b | 1 | 2,966 | 0.02 | 59.32 | 7.42 | 51 | 0.15 |
| 6t VTC Mobility Survey | gemma2:27b | 1 | 1,501 | 0.05 | 70.10 | 8.76 | 20 | 0.44 |
| 6t VTC Mobility Survey | gemma2:9b | 1 | 1,501 | 0.03 | 44.20 | 5.53 | 20 | 0.28 |
| 6t VTC Mobility Survey | gemma3:27b | 1 | 1,501 | 0.10 | 151.83 | 18.98 | 20 | 0.95 |
| 6t VTC Mobility Survey | gpt-oss:20b | 1 | 1,501 | 0.18 | 270.86 | 33.86 | 20 | 1.69 |
| 6t VTC Mobility Survey | llama3.1:8b | 1 | 1,501 | 0.02 | 34.30 | 4.29 | 20 | 0.21 |
| 6t VTC Mobility Survey | llama3.3:70b | 1 | 1,501 | 0.32 | 479.57 | 59.95 | 20 | 3.00 |
| 6t VTC Mobility Survey | mistral-nemo:12b | 1 | 1,501 | 0.03 | 38.73 | 4.84 | 20 | 0.24 |
| 6t VTC Mobility Survey | mistral:7b | 1 | 1,501 | 0.05 | 68.82 | 8.60 | 20 | 0.43 |
| EDGT Lyon | gemma2:27b | 1 | 28,230 | 0.07 | 2,050.91 | 256.36 | 173 | 1.48 |
| EDGT Lyon | gemma2:9b | 1 | 28,230 | 0.05 | 1,490.54 | 186.32 | 173 | 1.08 |
| EDGT Lyon | gemma3:27b | 1 | 28,230 | 0.16 | 4,384.12 | 548.01 | 173 | 3.17 |
| EDGT Lyon | gpt-oss:20b | 1 | 28,230 | 0.46 | 12,903.93 | 1,612.99 | 173 | 9.32 |
| EDGT Lyon | llama3.1:8b | 1 | 28,230 | 0.14 | 3,819.52 | 477.44 | 173 | 2.76 |
| EDGT Lyon | llama3.3:70b | 1 | 28,230 | 0.54 | 15,163.74 | 1,895.47 | 173 | 10.96 |
| EDGT Lyon | mistral-nemo:12b | 1 | 28,230 | 0.04 | 1,246.35 | 155.79 | 173 | 0.90 |
| EDGT Lyon | mistral:7b | 1 | 28,230 | 0.04 | 1,043.10 | 130.39 | 173 | 0.75 |

### Comparison in 8-hour workdays

- **NYC Citywide Mobility Survey (CMS)** (`data_NYC_mobility`): 7.4–113.2 8-hour workdays across models vs 51 fieldwork days.
- **6t VTC Mobility Survey** (`data_VTC_survey`): 4.3–59.9 8-hour workdays across models vs 20 fieldwork days.
- **EDGT Lyon** (`data_lyon_EDGT`): 130.4–1,895.5 8-hour workdays across models vs 173 fieldwork days.

## Notes

- No model factors, questionnaire-length effects or regression coefficients are
  estimated; each value is a direct linear rescale of that model's 200-agent
  run.
- Absolute values depend on the machine and on the model used for the run.
- `estimated h (raw)` is unnormalised; `workdays (8h)` divides those hours by 8.
  Fieldwork days remain calendar days, so the ratio compares 8-hour workdays
  with calendar fieldwork days.
