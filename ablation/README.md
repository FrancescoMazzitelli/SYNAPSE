# Ablation study (`ablation/`)

Pilot ablation for the SYNAPSE survey engine. The study is driven by a single
external JSON file ([`spec.json`](spec.json)); the code here does **not**
re-implement the pipeline, it composes the existing modules:

| Stage | Reused code / new wrapper |
|---|---|
| Header/question mapping | `preprocess.map_generic_data`, `preprocess.get_questions_generic` |
| dtype inference (cached) | `synthesize.infer_dtypes_with_llm` |
| Persona sampling (paper procedure) | `ablation.agents.sample_representative_population_seeded` reuses `synthesize._compute_row_weights`, `create_system_prompt`, `SurveyAgent` |
| Survey execution (batches, timeouts, retries) | `gp_survey_engine.AgenticSurveyPipeline` + new `ablation.pipeline.run_gp_survey` |
| Memory variants | `ablation.memory.MemoryAwareSurveyPipeline` (subclass, only memory is overridden) |
| Output serialisation | `postprocess.ProcessSurveyResponse`, new `ablation.pipeline.postprocess_response` |
| Metrics | `evaluate.run_single_evaluation` |

No existing method at the repository root is modified. The only root change is
a one-line import fix in `gp_survey_engine.py` (`Optional` was used but never
imported); everything else new lives in `ablation/`.

## What the study varies

* **Memory condition**: `none` / sliding `window` (5 = paper baseline) / `full`.
* **Temperature**: `0.0` vs. a positive value (e.g. `0.7`).
* **Model**: the cheap models listed in the spec.
* **Dataset**: the subset listed in the spec.
* **Seed**: the persona-sampling seed (and the prompt's option sampling); the
  number of personas is fixed by `agents`.

Everything else is held fixed (the supplied bios, prompts, batching, validation,
retry policy, question set). The number of calls is real: every executed cell
sends one LLM request per agent per question to `llm_base_url`. The number of
runs is

```
models x datasets x seeds x temperatures x memory_conditions
```

With the example spec that is `3 x 3 x 3 x 2 x 3 = 162` runs of
50 agents × all questions per dataset. Use the CLI filters for a smaller pilot.

## Quick start

```bash
# 1. Inspect the plan and the LLM-call estimate (no LLM is called for agents)
python ablation/run_ablation.py --dry-run

# 2. Small pilot: one model, two datasets, one seed
python ablation/run_ablation.py \
    --models llama3.1:8b \
    --datasets data_lyon_EDGT,data_MyDailyTravelData \
    --seeds 0 \
    --max-runs 6

# 3. Full plan (resumable: completed cells are skipped)
nohup python ablation/run_ablation.py > ablation/run.log 2>&1 &

# 4. Collect manifests (and optionally evaluate every run)
python ablation/aggregate.py
python ablation/aggregate.py --evaluate
```

`run_ablation.py` extracts the `configs/Generic/data_*.zip` archives on demand,
so no separate extraction step is needed.

## Spec reference (`spec.json`)

| Key | Meaning |
|---|---|
| `config_folder` | Config directory (default `configs/Generic`) |
| `output_dir` / `cache_dir` | Run output and mapping-cache directories |
| `llm_base_url` | Ollama-compatible API host, e.g. `http://192.168.250.40:15888` |
| `llm_api_base` | OpenAI-compatible URL passed to Langroid (default `<llm_base_url>/v1`) |
| `llm_api_key` | API key for that endpoint (`ollama` by default) |
| `datasets` | Dataset folder names, e.g. `data_lyon_EDGT` |
| `models` | Ollama models, with or without the `ollama/` prefix |
| `agents` | Number of sampled personas (50 in the pilot) |
| `seeds` | Persona-sampling seeds (the paper pipeline used `42` when no seed was given; the driver defaults to `[0]`) |
| `temperatures` | e.g. `[0.0, 0.7]` |
| `memory_conditions` | Strings (`"none"`, `"full"`, `"window5"`) or dicts `{name, mode, window}` |
| `questions` | `"all"`, a list of headers, or `{dataset: [...]}` |
| `batch_size`, `timeout_per_batch` | Passed to `ablation.pipeline.run_gp_survey` |
| `bio_model` | `null` (default in the spec) = reuse the supplied bios, never regenerate them |
| `dtype_model` | Model used once per dataset to infer answer dtypes (cached; `null` falls back to `config.json`'s chat model) |
| `model_overrides` | Extra keys merged into `config.json["model"]` (e.g. `{"seed": 7}` if your backend supports it) |
| `resume` | Skip cells whose `manifest.json` says `ok` |

Notes:

* Personas are sampled with the paper's inverse-frequency weights via the new
  `ablation.agents.sample_representative_population_seeded`; `seeds` really
  changes the sampling. Sampling one seed twice yields the same 50 respondents.
* That weight is the product of the inverse marginal frequencies of **all**
  sociodemographic labels, so it is very concentrated and the sample is
  effectively deterministic for some datasets. Measured for 50 agents
  (`eff_n = 1 / sum(w^2)`, `n*max_w` must be <= 1 for pandas>=3):

  | dataset | labels | eff_n | n*max_w |
  |---|---|---|---|
  | data_pmus_yaounde | 14 | 12.6 | 8.6 |
  | data_hanoi | 17 | 6.0 | 16.4 |
  | data_lyon_EDGT | 49 | 4.5 | 11.8 |
  | data_VTC_survey | 16 | 2.3 | 31.2 |
  | data_NYC_mobility | 175 | 1.0 | 50.0 |

  Consequences: (a) the `seeds` replicate mainly the *49 non-dominant*
  respondents, not the whole sample, so seed-level variance is not a measure of
  sampling uncertainty of the population; (b) on `data_NYC_mobility` the
  dominant respondent is always included, so it is a poor choice for the seed
  dimension (it is kept out of the example spec for this reason). This is
  inherited from the paper pipeline and is not changed here.
* `sampled_oracle.csv` in `configs/Generic/` is a scratch file of the existing
  pipeline; each run copies it into its own directory. Do not run two driver
  processes in parallel on the same `config_folder`.
* The 200 bios in `configs/Generic/bio/bio_<dataset>.json` are supplied as input
  and only read. Each of the 50 sampled rows is mapped back to the original
  200-row paper sample (seed 42), so it keeps its own `GEN_<i>` bio. They are
  never regenerated or overwritten; `bio_model: null` enforces this.
* Every executed run calls the LLM at `llm_base_url`: once per agent per
  question (plus one retry at most per question). `api_base` and `api_key`
  from the spec are injected into each run's `config.json`, so Langroid talks
  to the remote endpoint instead of `localhost:11434`. Dtype inference uses
  the same host via `ablation.dtypes.infer_dtypes_with_llm_at`.
* The pipeline does not set the backend's sampling `seed`; use
  `model_overrides` if your Langroid/Ollama version supports it. Temperature is
  the primary generation knob of the study.

## Output layout

```
ablation/results/<model>/<dataset>/seed-<s>/mem-<name>_temp-<t>/
    <timestamp>_results.csv      # reused postprocessor output
    <timestamp>_results.json
    batch_*_results.csv/json
    sampled_oracle.csv           # the 50 personas of this seed
    config.json                  # effective config for this run
    log.txt                      # timing, counts, condition
    manifest.json                # machine-readable record (used for resume/aggregate)

ablation/results/plan.json       # planned cells of the last invocation
ablation/results/spec.json       # spec snapshot of the last invocation
```

### Evaluation of the completed runs

`ablation/aggregate.py` flattens the manifests into
`ablation/evaluations/runs.csv`. The dedicated
`ablation/evaluate_ablation.py` computes the entropy / KL-divergence part of the
full study against each run's oracle, reusing the functions in `evaluate.py`
without modifying that file:

```bash
python ablation/evaluate_ablation.py
python ablation/evaluate_ablation.py --dataset data_VTC_survey --model gemma2:9b
```

It writes `na_columns_common.csv` (columns that are N/A in every oracle of a
dataset; removed globally from results and oracle), `metrics_entropy_kl.csv`
(one row per run), `per_variable_entropy_kl.csv` (one row per run x variable),
`answer_overlap_by_agent.csv` (overlap with the oracle for each agent and field),
`summary_by_condition.csv` (mean/std over seeds per
`dataset × model × temperature × memory`),
`summary_by_memory_temperature.csv` and `summary_entropy_kl.md` under
`ablation/evaluations/`.

Normalised Shannon entropy and KL(oracle || agent) use exactly the same
normalisation, column filtering and formulas as `evaluate.py`; a check on a
full-study run reproduces its published summary row. `delta_entropy` is
agent entropy minus oracle entropy (negative = agents less diverse than the
oracle).

`answer_overlap` is an additional per-agent metric: for each agent it is the
fraction of retained survey fields whose answer shares at least one normalised
token with the oracle answer (same `tokens_overlap` rule as TGC/factual in the
full study). Fields with an empty/N-A oracle value are skipped. The run-level
value is the mean over agents (the median is also reported), and per-agent
values are written to `answer_overlap_by_agent.csv`.

`aggregate.py --evaluate` still requires the full NLP dependencies
(`scipy`, `rouge-score`, `tabulate`, `bert-score`); the new script only needs
the already-installed `numpy` and `pandas`.

## Root files: what changed

The obsolete OD-retrieval feature was removed everywhere: its package
was deleted, its config block and dependencies were dropped, and the related
argument, prompt context and coordinate/route helpers were removed from
`main.py` and `gp_survey_engine.py`. No survey-generation behaviour remains
changed by that feature.

The seeded sampler and the ablation-specific builders remain new methods in
`ablation/agents.py`; `ablation.pipeline` holds the small new execution loop
that accepts `pipeline_factory`. Stable respondent ids for the shared bio cache
are computed inside `ablation.agents._stable_serial_numbers`.
