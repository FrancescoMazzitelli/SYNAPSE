"""
LLM-as-a-Judge with Likert Scale evaluation.

Evaluates survey responses across all models/datasets in results/.
Each agent's complete survey response is evaluated as a whole by a judge LLM.

Usage:
  python llm_judge.py
  python llm_judge.py --model llama3.3:70b --dataset data_lyon_EDGT
  python llm_judge.py --judge-model qwen3.5:9b --host http://192.168.1.42:11434
  python llm_judge.py --per-cell --max-agents 5 --max-questions 5  # cell-level eval
  python llm_judge.py --mode file --input responses.csv
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Dimensions ──────────────────────────────────────────────────────────────
DEFAULT_DIMENSIONS = {
    "accuracy": "The response is factually correct and free of errors.",
    "relevance": "The response directly addresses the question asked.",
    "coherence": "The response is well-structured, logical, and easy to follow.",
    "completeness": "The response covers all necessary aspects.",
    "fluency": "The response uses natural, grammatically correct language.",
}

LIKERT_LABELS = {
    1: "Very Poor",
    2: "Poor",
    3: "Fair",
    4: "Good",
    5: "Excellent",
}


def build_levels_rubric(likert_max: int) -> str:
    return "\n".join(
        f"  {i} = {LIKERT_LABELS.get(i, f'Level {i}')}"
        for i in range(1, likert_max + 1)
    )


# ── Prompts ─────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are an expert evaluator of AI-generated survey responses.

An AI agent with a given persona answered a set of survey questions.
Rate the agent's responses on a Likert scale from 1 to {likert_max} for each dimension:

{levels_rubric}

Return ONLY valid JSON — no preamble, no markdown fences."""

AGENT_USER_PROMPT = """Agent persona:
{persona}

Survey responses:
{qa_pairs}

Dimensions (score each 1-{likert_max}):
{rubrics}

Evaluate how well the agent answered the survey overall.
Return ONLY valid JSON."""

CELL_SYSTEM_PROMPT = """You are an expert evaluator of survey responses.
Rate the answer on a Likert scale from 1 to {likert_max} for each dimension:

{levels_rubric}

Return ONLY valid JSON — no preamble, no markdown fences."""

CELL_USER_PROMPT = """Question:
{question}

Answer:
{response}

Dimensions (score each 1-{likert_max}):
{rubrics}

Return ONLY valid JSON."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_input(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    elif p.suffix == ".json":
        with open(p) as f:
            return pd.DataFrame(json.load(f))
    elif p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    sys.exit(f"Unsupported format: {p.suffix}")


def resolve_columns(df: pd.DataFrame, instruction_col: str, response_col: str):
    cols = df.columns.tolist()
    if instruction_col is None:
        candidates = [c for c in cols if c.lower() in ("prompt", "instruction", "question", "query", "input")]
        if not candidates:
            sys.exit("Could not detect instruction column. Use --instruction-col.")
        instruction_col = candidates[0]
    if response_col is None:
        candidates = [c for c in cols if c.lower() in ("response", "answer", "output", "completion", "generated")]
        if not candidates:
            sys.exit("Could not detect response column. Use --response-col.")
        response_col = candidates[0]
    return instruction_col, response_col


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def parse_dimensions(dim_str: Optional[str]) -> dict:
    if dim_str:
        d = {}
        for name in [x.strip() for x in dim_str.split(",")]:
            d[name] = DEFAULT_DIMENSIONS.get(name, f"Quality in terms of '{name}'.")
        return d
    return dict(DEFAULT_DIMENSIONS)


def format_qa_pairs(row: pd.Series, q_cols: list) -> str:
    lines = []
    for c in q_cols:
        v = row[c]
        if pd.isna(v):
            v = ""
        lines.append(f"  {c}: {str(v).strip()}")
    return "\n".join(lines)


def describe_row(row: pd.Series, q_cols: list) -> str:
    parts = []
    for c in q_cols:
        v = row[c]
        if pd.isna(v):
            v = ""
        v = str(v).strip()
        if len(v) > 120:
            v = v[:117] + "..."
        parts.append(f"{c}={v}")
    return " | ".join(parts)


# ── Discovery ────────────────────────────────────────────────────────────────

METADATA_COLS = frozenset({"agent_id", "serial_number", "agent_bio"})


def discover_runs(results_dir: str, model_filter: str = None, dataset_filter: str = None) -> list[dict]:
    base = Path(results_dir)
    if not base.is_dir():
        sys.exit(f"Directory not found: {results_dir}")

    runs = []
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        model_name = model_dir.name
        if model_filter and model_name != model_filter:
            continue
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            dataset_name = dataset_dir.name
            if dataset_filter and dataset_name != dataset_filter:
                continue
            if not (dataset_dir / "oracle").is_dir():
                continue
            candidates = sorted(dataset_dir.glob("*_results.csv"))
            if not candidates:
                continue
            runs.append({
                "model": model_name,
                "dataset": dataset_name,
                "results": str(candidates[-1]),
            })
    return runs


def load_run_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df


def get_question_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in METADATA_COLS]


# ── Judge clients ────────────────────────────────────────────────────────────

class OllamaJudge:
    def __init__(self, model: str, host: str, temperature: float, max_retries: int, timeout: int):
        import subprocess
        self._subprocess = subprocess
        self._json = __import__("json")
        self.api_url = host.rstrip("/") + "/api/chat"
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def _call(self, system: str, user: str, dimensions: dict) -> dict:
        payload = self._json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": self.temperature},
        })
        for attempt in range(self.max_retries):
            try:
                result = self._subprocess.run(
                    ["curl", "-s", "--max-time", str(self.timeout),
                     "-X", "POST", self.api_url,
                     "-d", payload],
                    capture_output=True, text=True, timeout=self.timeout + 10,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"curl exit code {result.returncode}: {result.stderr.strip()}")
                data = self._json.loads(result.stdout)
                raw = data.get("message", {}).get("content", "{}")
                parsed = extract_json(raw)
                for dim in dimensions:
                    if dim not in parsed:
                        parsed[dim] = None
                    else:
                        try:
                            parsed[dim] = int(parsed[dim])
                        except (ValueError, TypeError):
                            parsed[dim] = None
                return parsed
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"    [Failed] {e}")
                    return {dim: None for dim in dimensions}
        return {dim: None for dim in dimensions}

    def evaluate_agent(self, persona: str, qa_pairs: str, dimensions: dict, likert_max: int) -> dict:
        system = AGENT_SYSTEM_PROMPT.format(likert_max=likert_max, levels_rubric=build_levels_rubric(likert_max))
        user = AGENT_USER_PROMPT.format(persona=persona, qa_pairs=qa_pairs, likert_max=likert_max, rubrics="\n".join(f"  {k}: {v}" for k, v in dimensions.items()))
        return self._call(system, user, dimensions)

    def evaluate_cell(self, question: str, response: str, dimensions: dict, likert_max: int) -> dict:
        system = CELL_SYSTEM_PROMPT.format(likert_max=likert_max, levels_rubric=build_levels_rubric(likert_max))
        user = CELL_USER_PROMPT.format(question=question, response=response, likert_max=likert_max, rubrics="\n".join(f"  {k}: {v}" for k, v in dimensions.items()))
        return self._call(system, user, dimensions)


class OpenAILikeJudge:
    def __init__(self, model: str, base_url: Optional[str], api_key: Optional[str], temperature: float, max_retries: int, timeout: int):
        from openai import OpenAI
        kwargs = {"api_key": api_key or "sk-placeholder", "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

    def _call(self, system: str, user: str, dimensions: dict) -> dict:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                result = extract_json(resp.choices[0].message.content)
                for dim in dimensions:
                    if dim not in result:
                        result[dim] = None
                    else:
                        try:
                            result[dim] = int(result[dim])
                        except (ValueError, TypeError):
                            result[dim] = None
                return result
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"    [Failed] {e}")
                    return {dim: None for dim in dimensions}
        return {dim: None for dim in dimensions}

    def evaluate_agent(self, persona: str, qa_pairs: str, dimensions: dict, likert_max: int) -> dict:
        system = AGENT_SYSTEM_PROMPT.format(likert_max=likert_max, levels_rubric=build_levels_rubric(likert_max))
        user = AGENT_USER_PROMPT.format(persona=persona, qa_pairs=qa_pairs, likert_max=likert_max, rubrics="\n".join(f"  {k}: {v}" for k, v in dimensions.items()))
        return self._call(system, user, dimensions)

    def evaluate_cell(self, question: str, response: str, dimensions: dict, likert_max: int) -> dict:
        system = CELL_SYSTEM_PROMPT.format(likert_max=likert_max, levels_rubric=build_levels_rubric(likert_max))
        user = CELL_USER_PROMPT.format(question=question, response=response, likert_max=likert_max, rubrics="\n".join(f"  {k}: {v}" for k, v in dimensions.items()))
        return self._call(system, user, dimensions)


# ── Stats ────────────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame, dims: list) -> pd.DataFrame:
    rows = []
    for dim in dims:
        scores = df[dim].dropna()
        if scores.empty:
            continue
        dist = scores.value_counts().sort_index()
        rows.append({
            "Dimension": dim,
            "Mean": round(scores.mean(), 2),
            "Std": round(scores.std(), 2),
            "Median": round(scores.median(), 2),
            "Min": int(scores.min()),
            "Max": int(scores.max()),
            "N": len(scores),
            "Distribution": ", ".join(f"{k}: {v}" for k, v in dist.items()),
        })
    return pd.DataFrame(rows)


# ── Auto evaluation ──────────────────────────────────────────────────────────

def run_auto(args):
    print(f"Scanning {args.results_dir} ...")
    runs = discover_runs(args.results_dir, args.model, args.dataset)
    if not runs:
        print("  No runs found.")
        return
    print(f"  Found {len(runs)} runs")

    dimensions = parse_dimensions(args.dimensions)
    dims_list = list(dimensions.keys())

    print(f"Judge: {args.judge_model} @ {args.host}")
    print(f"Strategy: {'per-cell' if args.per_cell else 'per-agent'}")
    print(f"Dimensions: {', '.join(dims_list)} | Likert: 1-{args.likert_max}")

    if args.provider == "ollama":
        judge = OllamaJudge(args.judge_model, args.host, args.temperature, args.max_retries, args.timeout)
    else:
        judge = OpenAILikeJudge(args.judge_model, args.base_url, args.api_key, args.temperature, args.max_retries, args.timeout)

    run_stats_list = []
    Path(args.output).mkdir(parents=True, exist_ok=True)
    all_scores_path = Path(args.output) / "all_judge_scores.csv"

    for ri, run in enumerate(runs):
        print(f"\n{'='*60}")
        print(f"  [{ri+1}/{len(runs)}] {run['model']} / {run['dataset']}")
        print(f"{'='*60}")

        df = load_run_results(run["results"])
        q_cols = get_question_columns(df)
        n_agents = len(df)
        n_questions = len(q_cols)

        out_dir = Path(args.output) / run["model"] / run["dataset"]
        out_dir.mkdir(parents=True, exist_ok=True)
        scores_path = out_dir / "llm_judge_scores.csv"
        stats_path = out_dir / "llm_judge_stats.csv"

        if args.per_cell:
            ma = n_agents if args.max_agents == 0 else min(args.max_agents, n_agents)
            mq = n_questions if args.max_questions == 0 else min(args.max_questions, n_questions)
            total = ma * mq
            print(f"  Agents: {ma} | Questions: {mq} | Evaluations: {total}")
            if total > 500:
                print(f"  Estimated: ~{math.ceil(total * 3 / 60)}min at 3s/call")

            agent_indices = list(range(ma))
            q_indices = list(range(mq))

            header = ["model", "dataset", "question", "agent_id", "response"] + dims_list
            f_out = open(scores_path, "w", buffering=1)
            f_out.write(",".join(header) + "\n")

            errors = 0
            done = 0
            for qi in q_indices:
                q = q_cols[qi]
                for ai in agent_indices:
                    row = df.iloc[ai]
                    response = str(row[q]) if not pd.isna(row[q]) else ""
                    agent_id = row.get("agent_id", ai)

                    scores = judge.evaluate_cell(q, response, dimensions, args.likert_max)
                    if any(v is None for v in scores.values()):
                        errors += 1

                    csv_row = [run["model"], run["dataset"], q, str(agent_id), response[:200]]
                    csv_row += [str(scores.get(d, "")) for d in dims_list]
                    f_out.write(",".join(f'"{x}"' for x in csv_row) + "\n")
                    done += 1
                    if done % 10 == 0 or done == total:
                        print(f"    [{done}/{total}] err={errors}", end="\r", flush=True)
                    if args.sleep > 0:
                        time.sleep(args.sleep)
            f_out.close()
            print()

        else:
            ma = n_agents if args.max_agents == 0 else min(args.max_agents, n_agents)
            total = ma
            print(f"  Agents: {ma}/{n_agents} | Questions per agent: {n_questions} | Evaluations: {total}")

            header = ["model", "dataset", "agent_id", "persona"] + dims_list
            f_out = open(scores_path, "w", buffering=1)
            f_out.write(",".join(header) + "\n")

            errors = 0
            for ai in range(ma):
                row = df.iloc[ai]
                persona = str(row.get("agent_bio", "")) if not pd.isna(row.get("agent_bio")) else ""
                qa = format_qa_pairs(row, q_cols)
                agent_id = row.get("agent_id", ai)

                scores = judge.evaluate_agent(persona, qa, dimensions, args.likert_max)
                if any(v is None for v in scores.values()):
                    errors += 1

                csv_row = [run["model"], run["dataset"], str(agent_id), persona[:200]]
                csv_row += [str(scores.get(d, "")) for d in dims_list]
                f_out.write(",".join(f'"{x}"' for x in csv_row) + "\n")
                print(f"    [{ai+1}/{total}] {scores}")
                if args.sleep > 0:
                    time.sleep(args.sleep)
            f_out.close()

        # Stats from saved file
        run_df = pd.read_csv(scores_path)
        run_stats = compute_stats(run_df, dims_list)
        run_stats.insert(0, "Dataset", run["dataset"])
        run_stats.insert(0, "Model", run["model"])
        run_stats.to_csv(stats_path, index=False)
        run_stats_list.append(run_stats)
        print(f"  Errors: {errors}/{total} | Saved: {scores_path}")

        # Append to global scores
        if ri == 0:
            run_df.to_csv(all_scores_path, index=False)
        else:
            run_df.to_csv(all_scores_path, mode="a", index=False, header=False)

        del df, run_df, f_out

    # Global summary from saved files
    if run_stats_list:
        summary = pd.concat(run_stats_list, ignore_index=True)
        summary.to_csv(Path(args.output) / "judge_summary.csv", index=False)
        if "Mean" in summary.columns:
            print(f"\nPer-model mean scores:")
            pivot = summary.pivot_table(index="Model", columns="Dimension", values="Mean", aggfunc="mean").round(2)
            print(pivot.to_string())

    # Global stats from all_judge_scores.csv (streamed, already on disk)
    if all_scores_path.exists():
        global_df = pd.read_csv(all_scores_path)
        global_stats = compute_stats(global_df, dims_list)
        global_stats.to_csv(Path(args.output) / "all_judge_stats.csv", index=False)
        del global_df

    meta = {
        "results_dir": args.results_dir, "judge_model": args.judge_model,
        "host": args.host, "provider": args.provider,
        "strategy": "per-cell" if args.per_cell else "per-agent",
        "dimensions": dims_list, "likert_max": args.likert_max,
        "n_runs": len(runs), "timeout": args.timeout,
    }
    with open(Path(args.output) / "judge_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll output in {args.output}/")


# ── File mode ────────────────────────────────────────────────────────────────

def run_file(args):
    df = load_input(args.input)
    print(f"Loaded: {args.input} ({len(df)} rows)")

    inst_col, resp_col = resolve_columns(df, args.instruction_col, args.response_col)
    dimensions = parse_dimensions(args.dimensions)
    dims_list = list(dimensions.keys())

    print(f"Judge: {args.judge_model} | Dimensions: {', '.join(dims_list)} | Likert: 1-{args.likert_max}")

    if args.provider == "ollama":
        judge = OllamaJudge(args.judge_model, args.host, args.temperature, args.max_retries, args.timeout)
    else:
        judge = OpenAILikeJudge(args.judge_model, args.base_url, args.api_key, args.temperature, args.max_retries, args.timeout)

    results = []
    errors = 0
    for idx, row in df.iterrows():
        instruction = str(row[inst_col])
        response = str(row[resp_col])
        scores = judge.evaluate_cell(instruction, response, dimensions, args.likert_max)
        if any(v is None for v in scores.values()):
            errors += 1
        r = {"id": idx, "instruction": instruction, "response": response}
        r.update(scores)
        results.append(r)
        print(f"  [{idx+1}/{len(df)}] {scores}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "judge_scores.csv", index=False)
    stats_df = compute_stats(results_df, dims_list)
    stats_df.to_csv(out_dir / "judge_stats.csv", index=False)

    meta = {"input": args.input, "judge_model": args.judge_model,
            "dimensions": dims_list, "n_responses": len(results_df), "n_errors": errors}
    with open(out_dir / "judge_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nEvaluated: {len(results_df)} | Errors: {errors}")
    print(stats_df.to_string(index=False))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="LLM-as-a-Judge with Likert scale")
    p.add_argument("--mode", default="auto", choices=["auto", "file"])

    p.add_argument("--results-dir", default="results")
    p.add_argument("--model", help="Filter by model")
    p.add_argument("--dataset", help="Filter by dataset")
    p.add_argument("--max-agents", type=int, default=0, help="Max agents (0=all)")
    p.add_argument("--max-questions", type=int, default=0, help="Max questions (0=all, per-cell only)")
    p.add_argument("--per-cell", action="store_true", help="Evaluate each cell instead of whole agent")

    p.add_argument("--input", help="Input file for file mode")
    p.add_argument("--instruction-col", help="Question column name")
    p.add_argument("--response-col", help="Answer column name")

    p.add_argument("--output", default="evaluations/llm_judge")
    p.add_argument("--provider", default="ollama", choices=["ollama", "openai"])
    p.add_argument("--judge-model", default="qwen3.5:9b", help="Judge model (default: qwen3.5:9b)")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--base-url", help="OpenAI-compatible base URL")
    p.add_argument("--api-key", help="API key")
    p.add_argument("--dimensions", help=f"Comma-separated. Default: {', '.join(DEFAULT_DIMENSIONS.keys())}")
    p.add_argument("--likert-max", type=int, default=5, choices=[3, 4, 5, 6, 7, 10])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds (default: 300)")
    p.add_argument("--sleep", type=float, default=0.0)

    args = p.parse_args()

    if args.mode == "auto":
        run_auto(args)
    else:
        if not args.input:
            sys.exit("--input is required in file mode.")
        run_file(args)


if __name__ == "__main__":
    main()