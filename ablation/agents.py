"""Agent construction for the ablation study.

This is a *new* function, not a change to ``synthesize.build_generic_agents``.
It reuses the production helpers (config loading, socio-demographic detection,
system-prompt tools and ``SurveyAgent``) while adding only what the study needs:

* model / temperature override;
* subsample size;
* a real sampling seed;
* **reuse of the 200 bios already provided in** ``configs/Generic/bio/``.

Those bios are never regenerated. The 50 sampled rows are mapped back to the
original 200-row paper sample (seed 42) so each persona keeps the bio generated
for its own row.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import langroid as lr
import langroid.language_models as lm
import numpy as np
import pandas as pd

from preprocess import extract_personal_agent_info, safe_csv_read
from synthesize import (
    SurveyAgent,
    _compute_row_weights,
    _discreteNumericTool,
    _multipleAnswerTool,
    _singleAnswerTool,
    create_system_prompt,
    detect_socio_demographics,
    load_config,
)

_UNSET = object()


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def sample_representative_population_seeded(
    data_path: str,
    n_sample: int,
    demographic_labels: List[str],
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Paper sampling procedure with a working ``seed`` argument.

    ``synthesize.sample_representative_population`` accepts ``random_state`` but
    ignores it (hard-coded to 42). Rather than edit that method, this new helper
    reuses its private weight computation and honours the seed. The numpy
    fallback reproduces the pre-pandas-3 behaviour the paper runs used.
    """
    df = safe_csv_read(data_path) if data_path.lower().endswith(".csv") else pd.read_excel(data_path)
    actual_seed = 42 if seed is None else seed
    if n_sample >= len(df):
        return df.sample(frac=1, random_state=actual_seed).reset_index(drop=True)

    weights = _compute_row_weights(df, demographic_labels)
    try:
        sampled = df.sample(n=n_sample, weights=weights, replace=False, random_state=actual_seed)
    except ValueError as exc:
        if "replace=False" not in str(exc):
            raise
        rng = np.random.RandomState(actual_seed)
        try:
            idx = rng.choice(len(df), size=n_sample, replace=False, p=weights.to_numpy())
        except ValueError as nested:
            raise ValueError(
                f"Weighted sampling failed for {data_path} (n_sample={n_sample}): "
                "the demographic weights are too concentrated. Reduce 'agents' or "
                "the set of sociodemographic labels."
            ) from nested
        sampled = df.iloc[idx]
    return sampled.reset_index(drop=True)


def _row_identity(row: pd.Series) -> str:
    """Stable id for a dataset row, independent of its order in a sample."""
    raw = row.get("SERIALNO")
    if raw is not None and not (isinstance(raw, float) and pd.isna(raw)) and str(raw).strip():
        return "SERIAL::" + str(raw).strip()
    payload = "|".join(f"{col}={row[col]}" for col in row.index)
    return "GEN::" + hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------
# Pre-generated bios
# ----------------------------------------------------------------------
def load_cached_bios(config_folder: str, dataset: str) -> Dict[str, str]:
    """Read the bios supplied by the user (never written back by the ablation)."""
    dataset_name = os.path.basename(os.path.dirname(dataset.rstrip("/\\"))) or dataset
    path = Path(config_folder) / "bio" / f"bio_{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing pre-generated bios: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        bios = json.load(handle)
    if not isinstance(bios, dict) or not bios:
        raise ValueError(f"No bios found in {path}")
    return bios


def cached_bio_keys_for_sample(
    sampled_df: pd.DataFrame,
    data_path: str,
    demographic_labels: List[str],
    cached_bios: Dict[str, str],
    seed: int,
) -> List[str]:
    """Map each sampled row to one of the provided ``GEN_*`` bios.

    The provided 200 bios were produced from the paper's 200-agent sample
    (seed 42). We rebuild that reference sample, record which ``GEN_<i>`` key
    belongs to which row, and reuse the matching bio for our 50 agents.
    """
    reference = sample_representative_population_seeded(
        data_path=data_path,
        n_sample=min(200, len(sampled_df) + 200),
        demographic_labels=demographic_labels,
        seed=42,
    )
    identity_to_key = {
        _row_identity(row): f"GEN_{idx}" for idx, (_, row) in enumerate(reference.iterrows())
    }

    available = set(cached_bios)
    chosen: List[str] = []
    rng = np.random.RandomState(seed)
    for _, row in sampled_df.iterrows():
        key = identity_to_key.get(_row_identity(row))
        if key not in available:
            # Deterministic fallback; still only uses bios the user supplied.
            remaining = sorted(available - set(chosen))
            if not remaining:
                remaining = sorted(available)
            key = remaining[rng.randint(len(remaining))]
        chosen.append(key)
    return chosen


# ----------------------------------------------------------------------
# Agent builder
# ----------------------------------------------------------------------
def build_agents(
    config_folder: str,
    map: dict,
    data_path: str,
    desc_path: str,
    *,
    model_config_override: Optional[dict] = None,
    subsample_override: Optional[int] = None,
    random_state: Optional[int] = None,
    bio_model=_UNSET,
    detected_labels: Optional[dict] = None,
    oracle_path: Optional[str] = None,
):
    """Build the 50 personas for one ablation cell, reusing the provided bios."""
    model_config, synth_conf, _, _ = load_config(config_folder)
    if model_config_override:
        model_config = {**model_config, **model_config_override}
    header = synth_conf.get("system_message_header", "")
    footer = synth_conf.get("system_message_footer", "")
    subsample = (
        subsample_override if subsample_override is not None else synth_conf.get("subsample", 10)
    )

    desc_df = safe_csv_read(desc_path) if desc_path.endswith(".csv") else pd.read_excel(desc_path)
    labels = detected_labels
    if labels is None:
        labels = extract_personal_agent_info(desc_df)
        if not labels:
            labels = detect_socio_demographics(mapping_dict=map, model_config=model_config)
    print(labels)

    sampled_df = sample_representative_population_seeded(
        data_path=data_path,
        n_sample=subsample,
        demographic_labels=labels["labels"],
        seed=random_state,
    )
    oracle = Path(oracle_path) if oracle_path else Path(config_folder) / "sampled_oracle.csv"
    oracle.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(oracle, index=False)

    cached_bios = load_cached_bios(config_folder, data_path)
    actual_seed = 42 if random_state is None else random_state
    bio_keys = cached_bio_keys_for_sample(
        sampled_df=sampled_df,
        data_path=data_path,
        demographic_labels=labels["labels"],
        cached_bios=cached_bios,
        seed=actual_seed,
    )

    system_prompts: List[Tuple[str, str]] = []
    for idx, (_, row) in enumerate(sampled_df.iterrows()):
        key = bio_keys[idx]
        if key in cached_bios:
            system_prompts.append((cached_bios[key], key))
            continue
        # Only reached if the user explicitly allows regeneration in the spec.
        if bio_model is _UNSET or bio_model is None:
            raise ValueError(
                f"Bio {key} is missing and bio_model is disabled. "
                "Provide the bio file or set bio_model in the spec."
            )
        prompt = create_system_prompt(
            row=row,
            mapping_dict=map,
            demo_labels=labels["labels"],
            config_folder=config_folder,
            bio_model=bio_model,
        )
        system_prompts.append((prompt, key))

    llm_config = lm.OpenAIGPTConfig(**model_config)
    agents = []
    for i, (system_message, serial) in enumerate(system_prompts):
        agent_config = lr.ChatAgentConfig(
            name=f"Agent_{i}",
            llm=llm_config,
            system_message=header + system_message + footer,
            use_tools=True,
            use_functions_api=True,
        )
        agent = SurveyAgent(
            config=agent_config, agent_id=i, bio=system_message, serial_number=serial
        )
        agent.enable_message(_singleAnswerTool)
        agent.enable_message(_multipleAnswerTool)
        agent.enable_message(_discreteNumericTool)
        agents.append(agent)
    return agents
