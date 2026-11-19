#!/usr/bin/env python3
"""Ablation study driver for the SYNAPSE survey pipeline.

The study is defined entirely by an external spec file (JSON). Each run is the
existing pipeline with three knobs moved:

* ``model`` + ``temperature``  -> ``config.json["model"]`` overrides
* ``memory``                  -> :class:`ablation.memory.MemoryAwareSurveyPipeline`
* agent sampling ``seed``      -> ``random_state`` of the paper sampling procedure

Everything else (question mapping, batched survey execution, post-processing)
is the production code. Every executed cell really calls the LLM at
``llm_base_url`` from the spec; the 200 pre-generated bios in
``configs/Generic/bio/`` are reused and never regenerated.

No file at the repository root is modified by a run: overrides are passed in
memory, and the per-run ``sampled_oracle.csv`` is copied next to the results.

Examples
--------
    # show the plan and the estimated cost, without calling any LLM
    python ablation/run_ablation.py --spec ablation/spec.json --dry-run

    # execute the real study: one LLM call per agent x question, per cell
    python ablation/run_ablation.py --spec ablation/spec.json

    # run only one model / two datasets on one seed
    python ablation/run_ablation.py --spec ablation/spec.json \
        --models llama3.1:8b --datasets data_lyon_EDGT,data_MyDailyTravelData \
        --seeds 0

    # resume an interrupted pilot (completed runs are skipped)
    python ablation/run_ablation.py --spec ablation/spec.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess import (  # noqa: E402
    extract_personal_agent_info,
    get_questions_generic,
    map_generic_data,
    safe_csv_read,
)
from synthesize import (  # noqa: E402
    detect_socio_demographics,
    load_config,
)
from postprocess import ProcessSurveyResponse  # noqa: E402
from ablation.agents import build_agents  # noqa: E402
from ablation.dtypes import infer_dtypes_with_llm_at  # noqa: E402
from ablation.memory import condition_tag, parse_memory_condition, pipeline_factory  # noqa: E402
from ablation.pipeline import postprocess_response, run_gp_survey  # noqa: E402


DEFAULT_SPEC = Path(__file__).resolve().parent / "spec.json"

SPEC_DEFAULTS = {
    "config_folder": "configs/Generic",
    "output_dir": "ablation/results",
    "cache_dir": "ablation/cache",
    "llm_base_url": "http://localhost:11434",
    "llm_api_base": None,
    "llm_api_key": "ollama",
    "datasets": [],
    "models": [],
    "agents": 50,
    "seeds": [0],
    "temperatures": [0.0, 0.7],
    "memory_conditions": [
        {"name": "none", "mode": "none"},
        {"name": "window5", "mode": "window", "window": 5},
        {"name": "full", "mode": "full"},
    ],
    "questions": "all",
    "batch_size": 10,
    "timeout_per_batch": 600000,
    "bio_model": "__config__",
    "dtype_model": None,
    "model_overrides": {},
    "resume": True,
}


# ----------------------------------------------------------------------
# Spec handling
# ----------------------------------------------------------------------
def load_spec(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        user_spec = json.load(f)
    if not isinstance(user_spec, dict):
        raise ValueError(f"Spec must be a JSON object: {path}")

    spec = {**SPEC_DEFAULTS, **user_spec}
    spec["config_folder"] = _resolve(spec["config_folder"])
    spec["output_dir"] = _resolve(spec["output_dir"])
    spec["cache_dir"] = _resolve(spec["cache_dir"])
    spec["llm_base_url"] = str(spec["llm_base_url"]).rstrip("/")
    spec["llm_api_base"] = str(
        spec.get("llm_api_base") or f"{spec['llm_base_url']}/v1"
    ).rstrip("/")

    if not spec["datasets"]:
        raise ValueError("Spec must list at least one entry in 'datasets'")
    if not spec["models"]:
        raise ValueError("Spec must list at least one entry in 'models'")
    if int(spec["agents"]) <= 0:
        raise ValueError("'agents' must be a positive integer")

    spec["seeds"] = [int(s) for s in _as_list(spec["seeds"])]
    spec["temperatures"] = [float(t) for t in _as_list(spec["temperatures"])]
    spec["memory_conditions"] = [parse_memory_condition(c) for c in _as_list(spec["memory_conditions"])]
    if not spec["seeds"]:
        raise ValueError("Spec must list at least one entry in 'seeds'")
    if not spec["temperatures"]:
        raise ValueError("Spec must list at least one entry in 'temperatures'")
    if not spec["memory_conditions"]:
        raise ValueError("Spec must list at least one entry in 'memory_conditions'")
    names = [c["name"] for c in spec["memory_conditions"]]
    if len(set(names)) != len(names):
        raise ValueError(f"'memory_conditions' names must be unique, got {names}")

    spec["batch_size"] = int(spec["batch_size"])
    spec["timeout_per_batch"] = float(spec["timeout_per_batch"])
    spec["agents"] = int(spec["agents"])
    spec["resume"] = bool(spec["resume"])
    return spec


def _resolve(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else (REPO_ROOT / p))


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _cli_filter(values: Optional[List[str]], allowed: List[str], label: str) -> List[str]:
    if not values:
        return allowed
    selected = []
    for chunk in values:
        selected.extend(x.strip() for x in chunk.split(",") if x.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        print(f"[warn] ignoring unknown {label}: {unknown}")
    return [x for x in allowed if x in selected]


# ----------------------------------------------------------------------
# Datasets and question mapping (cached)
# ----------------------------------------------------------------------
def ensure_dataset(config_folder: Path, dataset: str) -> Tuple[Path, Path]:
    """Extract ``data_<x>.zip`` if needed and return (survey, dictionary)."""
    data_dir = config_folder / dataset
    survey = data_dir / "survey.csv"
    dictionary = data_dir / "dictionary.csv"
    if survey.exists() and dictionary.exists():
        return survey, dictionary

    zip_path = config_folder / f"{dataset}.zip"
    if zip_path.exists():
        print(f"[data] extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(config_folder)

    if not (survey.exists() and dictionary.exists()):
        raise FileNotFoundError(
            f"Dataset {dataset!r} not found: expected {survey} and {dictionary} "
            f"(or {zip_path})."
        )
    return survey, dictionary


def _file_signature(path: Path) -> Dict[str, int]:
    st = path.stat()
    return {"mtime": int(st.st_mtime), "size": st.st_size}


def _ollama_model_ready(
    model: str, base_url: str = "http://localhost:11434", timeout: float = 3.0
) -> bool:
    """Check that the configured Ollama server is up and the model is pulled."""
    model_name = model.split("/", 1)[-1]
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
    except Exception:  # noqa: BLE001
        return False
    return model_name in names or f"{model_name}:latest" in names


def _serialize_mapping(mapping: dict) -> dict:
    out = {}
    for key, value in mapping.items():
        desc, question, answers = value[:3]
        dtype = value[3] if len(value) >= 4 else None
        out[key] = {
            "description": desc,
            "question": question,
            "answers": sorted(list(answers), key=str),
            "dtype": dtype,
        }
    return out


def _deserialize_mapping(raw: dict) -> dict:
    mapping = {}
    for key, item in raw.items():
        answers = set(item.get("answers") or [])
        dtype = item.get("dtype")
        if dtype:
            mapping[key] = (item["description"], item["question"], answers, dtype)
        else:
            mapping[key] = (item["description"], item["question"], answers)
    return mapping


def prepare_dataset(
    config_folder: Path,
    cache_dir: Path,
    dataset: str,
    dtype_model: Optional[str],
    model_config: dict,
    llm_base_url: str,
) -> Tuple[dict, dict, dict]:
    """Return (mapping, questions, detected_labels) for a dataset, cached on disk.

    The cache stores the dtype classification (an LLM call per dataset) and the
    sociodemographic labels, so the pilot does not pay that cost on every run.
    """
    survey, dictionary = ensure_dataset(config_folder, dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset}.mapping.json"
    dtype_model = dtype_model or model_config.get("chat_model", "gpt-oss:20b")
    signature = {
        "survey": _file_signature(survey),
        "dictionary": _file_signature(dictionary),
        "dtype_model": dtype_model,
    }

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if (cached.get("signature") == signature
                    and cached.get("labels")
                    and cached.get("dtype_source", "llm") == "llm"):
                mapping = _deserialize_mapping(cached["mapping"])
                questions = get_questions_generic(mapping)
                print(f"[cache] {dataset}: {len(questions)} questions, "
                      f"{len(cached['labels'].get('labels', []))} demographic labels")
                return mapping, questions, cached["labels"]
        except Exception as exc:  # noqa: BLE001 - a stale cache must not stop the run
            print(f"[cache] ignoring unreadable cache for {dataset}: {exc}")

    print(f"[map] {dataset}: mapping headers and inferring dtypes ...")
    mapping = map_generic_data(str(survey), str(dictionary))
    dtype_source = "llm"
    if _ollama_model_ready(dtype_model, llm_base_url):
        mapping = infer_dtypes_with_llm_at(mapping, dtype_model, llm_base_url)
    else:
        from synthesize import _fallback_dtype

        print(f"[map] {dataset}: '{dtype_model}' not available at {llm_base_url}; using the "
              f"static dtype fallback (result NOT cached)")
        mapping = _fallback_dtype(mapping)
        dtype_source = "fallback"

    desc_df = safe_csv_read(str(dictionary))
    labels = extract_personal_agent_info(desc_df)
    if not labels:
        labels = detect_socio_demographics(mapping_dict=mapping, model_config=model_config)

    if dtype_source == "llm":
        payload = {
            "signature": signature,
            "dtype_source": dtype_source,
            "labels": labels,
            "mapping": _serialize_mapping(mapping),
        }
        cache_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    questions = get_questions_generic(mapping)
    print(f"[map] {dataset}: {len(questions)} questions")
    return mapping, questions, labels


def select_questions(questions: dict, selector, dataset: str) -> dict:
    """Apply the spec's ``questions`` selector (``"all"`` or a list of headers)."""
    if selector is None or selector == "all":
        return questions
    if isinstance(selector, dict):
        selector = selector.get(dataset, "all")
        if selector == "all":
            return questions
    names = _as_list(selector)
    wanted = {str(n) for n in names}
    unknown = sorted(wanted - set(questions))
    if unknown:
        print(f"[warn] {dataset}: {len(unknown)} question(s) not in the dataset: {unknown[:5]}")
    selected = {k: v for k, v in questions.items() if k in wanted}
    if not selected:
        raise ValueError(f"{dataset}: no valid question selected (wanted {sorted(wanted)[:5]}...)")
    return selected


# ----------------------------------------------------------------------
# Run identity / layout
# ----------------------------------------------------------------------
def model_names(model: str) -> Tuple[str, str]:
    """Return (chat_model, directory_name) for a spec model entry."""
    dir_name = model.split("/", 1)[1] if model.startswith("ollama/") else model
    return f"ollama/{dir_name}", dir_name


def run_dir_for(output_dir, model_dir, dataset, seed, condition, temperature) -> Path:
    return (
        Path(output_dir)
        / model_dir
        / dataset
        / f"seed-{seed}"
        / condition_tag(condition, temperature)
    )


def run_id_for(model_dir, dataset, seed, condition, temperature) -> str:
    return f"{model_dir}|{dataset}|seed-{seed}|{condition_tag(condition, temperature)}"


def is_complete(run_dir: Path) -> bool:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if manifest.get("status") != "ok":
        return False
    outputs = manifest.get("outputs") or {}
    return all(outputs.get(key) and Path(outputs[key]).exists()
               for key in ("results_csv", "results_json"))


def effective_config(config_folder: Path, chat_model: str, temperature: float,
                     n_agents: int, model_overrides: dict, spec: dict) -> dict:
    cfg = json.loads((config_folder / "config.json").read_text(encoding="utf-8"))
    cfg.setdefault("model", {}).update(model_overrides or {})
    cfg["model"]["chat_model"] = chat_model
    cfg["model"]["temperature"] = temperature
    cfg["model"]["api_base"] = spec["llm_api_base"]
    cfg["model"]["api_key"] = spec.get("llm_api_key", "ollama")
    cfg.setdefault("synthesis", {})["subsample"] = n_agents
    return cfg


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------
def execute_run(
    *,
    run_dir: Path,
    run_id: str,
    spec: dict,
    dataset: str,
    model_dir: str,
    chat_model: str,
    temperature: float,
    seed: int,
    condition: dict,
    mapping: dict,
    questions: dict,
    agents: list,
    labels: dict,
    oracle_path: Path,
) -> dict:
    started_at = time.time()
    date_str = time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    _clean_run_dir(run_dir)

    config = effective_config(
        Path(spec["config_folder"]), chat_model, temperature, spec["agents"],
        spec["model_overrides"], spec,
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    oracle_dst = run_dir / "sampled_oracle.csv"
    if oracle_path.exists():
        shutil.copyfile(oracle_path, oracle_dst)

    # Makes the ">60 possible answers" subsampling in the prompt reproducible
    # for a given seed, and lets a rerun of the same cell match it.
    random.seed(seed)

    batch_size = spec["batch_size"]
    postprocessor = ProcessSurveyResponse(
        str(spec["config_folder"]),
        batch_size=batch_size,
        RUN_FOLDER=str(run_dir),
        source="ABLATION",
        date_str=date_str,
    )

    result_queue: Queue = Queue()
    stop_event = threading.Event()
    survey_thread = threading.Thread(
        target=run_gp_survey,
        args=(
            result_queue,
            stop_event,
            mapping,
            questions,
            agents,
            batch_size,
            spec["timeout_per_batch"],
        ),
        kwargs={
            "pipeline_factory": pipeline_factory(condition),
        },
    )
    post_thread = threading.Thread(
        target=postprocess_response,
        args=(result_queue, stop_event, postprocessor, date_str),
    )

    survey_thread.start()
    post_thread.start()
    survey_thread.join()
    post_thread.join()

    write_success = postprocessor.write_results(str(run_dir), date_str)
    finished_at = time.time()
    # One serialized row per agent (columns are the questions)
    n_responses = len(postprocessor.synthetic_asdict)
    expected = len(agents)

    if n_responses == 0:
        status = "error"
    elif n_responses < expected:
        status = "partial"
    else:
        status = "ok"

    results_csv = run_dir / f"{date_str}_results.csv"
    results_json = run_dir / f"{date_str}_results.json"

    log_path = run_dir / "log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Ablation run: {run_id}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {chat_model}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Memory condition: {condition['name']} "
                f"(mode={condition['mode']}, window={condition.get('window')})\n")
        f.write(f"Sampling seed: {seed}\n")
        f.write(f"Agents: {len(agents)} (sampled with the paper procedure)\n")
        f.write(f"Questions: {len(questions)}\n")
        f.write(f"Demographic labels: {labels.get('labels', [])}\n")
        f.write(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started_at))}\n")
        f.write(f"End: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(finished_at))}\n")
        f.write(f"Duration (hours): {(finished_at - started_at) / 3600:.3f}\n")
        f.write(f"Responses: {n_responses}/{expected}\n")
        f.write(f"Successful write to disk: {write_success}\n")

    manifest = {
        "run_id": run_id,
        "status": status,
        "model": model_dir,
        "chat_model": chat_model,
        "dataset": dataset,
        "seed": seed,
        "temperature": temperature,
        "memory": condition,
        "n_agents": len(agents),
        "n_questions": len(questions),
        "n_responses": n_responses,
        "n_expected": expected,
        "run_dir": str(run_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(finished_at)),
        "duration_s": round(finished_at - started_at, 1),
        "outputs": {
            "results_csv": str(results_csv) if results_csv.exists() else None,
            "results_json": str(results_json) if results_json.exists() else None,
            "oracle": str(oracle_dst) if oracle_dst.exists() else None,
            "log": str(log_path),
            "config": str(run_dir / "config.json"),
        },
    }
    _write_manifest(run_dir, manifest)
    print(f"[run] {run_id}: {status} ({n_responses}/{expected} responses, "
          f"{(finished_at - started_at) / 60:.1f} min)")
    return manifest


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _clean_run_dir(run_dir: Path) -> None:
    """Remove artifacts of a previous attempt in the same cell (used with --force)."""
    for pattern in ("*_results.csv", "*_results.json", "batch_*_results.csv",
                    "batch_*_results.json", "manifest.json", "log.txt",
                    "config.json", "sampled_oracle.csv"):
        for path in run_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------
def enumerate_cells(spec: dict, models: List[str], datasets: List[str],
                    seeds: List[int]) -> List[dict]:
    cells = []
    for model in models:
        chat_model, model_dir = model_names(model)
        for dataset in datasets:
            for seed in seeds:
                for temperature in spec["temperatures"]:
                    for condition in spec["memory_conditions"]:
                        cells.append({
                            "model": model,
                            "chat_model": chat_model,
                            "model_dir": model_dir,
                            "dataset": dataset,
                            "seed": seed,
                            "temperature": temperature,
                            "condition": condition,
                            "run_dir": run_dir_for(
                                spec["output_dir"], model_dir, dataset, seed, condition, temperature
                            ),
                            "run_id": run_id_for(model_dir, dataset, seed, condition, temperature),
                        })
    return cells


def print_plan(cells: List[dict], n_questions: Dict[str, int], n_agents: int) -> None:
    n = len(cells)
    calls = sum(n_agents * n_questions.get(c["dataset"], 0) for c in cells)
    print("\n" + "=" * 68)
    print("ABLATION PLAN")
    print("=" * 68)
    print(f"  runs                : {n}")
    print(f"  agent runs          : {n * n_agents}")
    print(f"  LLM calls (min)     : {calls}  (each agent answers all questions once)")
    print(f"  LLM calls (max)     : {2 * calls}  (worst case: one retry per question)")
    print("  breakdown:")
    by_dataset: Dict[str, int] = {}
    for cell in cells:
        by_dataset[cell["dataset"]] = by_dataset.get(cell["dataset"], 0) + 1
    for dataset, count in by_dataset.items():
        print(f"    {dataset:<32} {count:>4} runs  x {n_questions.get(dataset, '?')} questions")
    print("=" * 68 + "\n")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run (or plan) the SYNAPSE ablation study from a JSON spec."
    )
    ap.add_argument("--spec", default=str(DEFAULT_SPEC),
                    help="Path to the JSON spec (default: ablation/spec.json)")
    ap.add_argument("--models", action="append", default=None,
                    help="Comma-separated subset of spec models")
    ap.add_argument("--datasets", action="append", default=None,
                    help="Comma-separated subset of spec datasets")
    ap.add_argument("--seeds", action="append", default=None,
                    help="Comma-separated subset of spec seeds")
    ap.add_argument("--max-runs", type=int, default=None,
                    help="Only execute the first N planned runs (pilot)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and cost estimate, then exit")
    ap.add_argument("--force", action="store_true",
                    help="Re-run cells even if manifest.json says they completed")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    spec_path = Path(args.spec)
    spec = load_spec(spec_path)

    models = _cli_filter(args.models, list(spec["models"]), "models")
    datasets = _cli_filter(args.datasets, list(spec["datasets"]), "datasets")
    seeds = _cli_filter(args.seeds, [str(s) for s in spec["seeds"]], "seeds")
    seeds = [int(s) for s in seeds]
    if not (models and datasets and seeds):
        print("[error] filters selected no models/datasets/seeds; nothing to do.")
        return 2

    config_folder = Path(spec["config_folder"])
    categories: Dict[str, tuple] = {}
    questions_by_dataset: Dict[str, int] = {}
    model_config, _, _, _ = load_config(spec["config_folder"])
    dtype_model = spec["dtype_model"]

    def prepare(dataset: str):
        if dataset not in categories:
            mapping, questions, labels = prepare_dataset(
                config_folder,
                Path(spec["cache_dir"]),
                dataset,
                dtype_model,
                model_config,
                spec["llm_base_url"],
            )
            questions = select_questions(questions, spec["questions"], dataset)
            categories[dataset] = (mapping, questions, labels)
            questions_by_dataset[dataset] = len(questions)
        return categories[dataset]

    # Prepare every selected dataset once (also needed for the cost estimate).
    for dataset in datasets:
        prepare(dataset)

    cells = enumerate_cells(spec, models, datasets, seeds)
    if args.max_runs is not None:
        cells = cells[: max(0, args.max_runs)]

    print_plan(cells, questions_by_dataset, spec["agents"])

    out_root = Path(spec["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "plan.json").write_text(
        json.dumps(
            {
                "spec_file": str(spec_path.resolve()),
                "spec": {k: v for k, v in spec.items()},
                "runs": [
                    {
                        "run_id": cell["run_id"],
                        "model": cell["model_dir"],
                        "dataset": cell["dataset"],
                        "seed": cell["seed"],
                        "temperature": cell["temperature"],
                        "memory": cell["condition"],
                        "run_dir": str(cell["run_dir"]),
                    }
                    for cell in cells
                ],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    spec_snapshot = out_root / "spec.json"
    if spec_path.resolve() != spec_snapshot.resolve():
        shutil.copyfile(spec_path, spec_snapshot)

    if args.dry_run:
        print(f"Dry run: plan written to {out_root / 'plan.json'}")
        return 0

    resume = spec["resume"] and not args.force
    n_done = n_skipped = n_failed = 0

    # Groups share agents: same model, dataset, seed and temperature.
    grouped = itertools.groupby(
        cells,
        key=lambda c: (c["model"], c["chat_model"], c["model_dir"], c["dataset"],
                       c["seed"], c["temperature"]),
    )
    for (_model, chat_model, model_dir, dataset, seed, temperature), group in grouped:
        group_cells = list(group)
        pending = [c for c in group_cells if not (resume and is_complete(c["run_dir"]))]
        n_skipped += len(group_cells) - len(pending)
        if not pending:
            print(f"[skip] {model_dir}/{dataset}/seed-{seed}/temp-{temperature:g}: all cells done")
            continue

        mapping, questions, labels = categories[dataset]
        oracle_path = config_folder / "sampled_oracle.csv"
        print(f"\n[group] {model_dir} / {dataset} / seed={seed} / temp={temperature:g} "
              f"({len(pending)} cell(s))")

        try:
            survey_path, dict_path = ensure_dataset(config_folder, dataset)
            model_overrides = {
                **spec["model_overrides"],
                "chat_model": chat_model,
                "temperature": temperature,
                "api_base": spec["llm_api_base"],
                "api_key": spec.get("llm_api_key", "ollama"),
            }
            agent_kwargs = {
                "config_folder": spec["config_folder"],
                "map": mapping,
                "data_path": str(survey_path),
                "desc_path": str(dict_path),
                "model_config_override": model_overrides,
                "subsample_override": spec["agents"],
                "random_state": seed,
                "detected_labels": labels,
                "oracle_path": str(oracle_path),
            }
            # "__config__" means "keep config.json['synthesis']['bio_model']".
            if spec.get("bio_model", "__config__") != "__config__":
                agent_kwargs["bio_model"] = spec.get("bio_model")
            agents = build_agents(**agent_kwargs)
        except Exception:
            print(f"[error] could not build agents for {model_dir}/{dataset}/seed-{seed}:")
            traceback.print_exc()
            n_failed += len(pending)
            continue

        for cell in pending:
            try:
                manifest = execute_run(
                    run_dir=cell["run_dir"],
                    run_id=cell["run_id"],
                    spec=spec,
                    dataset=dataset,
                    model_dir=model_dir,
                    chat_model=chat_model,
                    temperature=temperature,
                    seed=seed,
                    condition=cell["condition"],
                    mapping=mapping,
                    questions=questions,
                    agents=agents,
                    labels=labels,
                    oracle_path=oracle_path,
                )
                if manifest["status"] == "ok":
                    n_done += 1
                else:
                    n_failed += 1
            except Exception:
                print(f"[error] run failed: {cell['run_id']}")
                traceback.print_exc()
                n_run_dir = Path(cell["run_dir"])
                n_run_dir.mkdir(parents=True, exist_ok=True)
                _write_manifest(n_run_dir, {
                    "run_id": cell["run_id"],
                    "status": "error",
                    "model": model_dir,
                    "dataset": dataset,
                    "seed": seed,
                    "temperature": temperature,
                    "memory": cell["condition"],
                    "error": traceback.format_exc(),
                })
                n_failed += 1

        del agents

    print("\n" + "=" * 68)
    print(f"Done. completed={n_done} skipped={n_skipped} failed={n_failed}")
    print(f"Results: {out_root}")
    print("=" * 68)
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
