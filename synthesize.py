import pandas as pd
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import random
import langroid as lr
import langroid.language_models as lm
from langroid.agent.chat_agent import ChatDocument
from preprocess import *
import numpy as np
from ctransformers import AutoModelForCausalLM
import re
from pypdf import PdfReader


def _resolve_answer_label(raw_value: Any, answer_set: set) -> Optional[str]:
    """Resolve a raw data value to the full answer string using the answer set.

    Answer set entries follow the pattern 'key: label' (e.g., '7: 65 years old or older').
    Returns the full entry (e.g., '7: 65 years old or older'), or None if no match.
    """
    if not answer_set:
        return None

    raw_str = str(raw_value).strip()

    for answer in answer_set:
        ans_str = str(answer).strip()
        m = re.match(r'^(.+?)\s*:\s*(.*)$', ans_str)
        if m:
            key_part = m.group(1).strip()
            if raw_str == key_part:
                return ans_str
            try:
                if float(raw_str) == float(key_part):
                    return ans_str
            except ValueError:
                pass

    # Fallback: string contains
    for answer in answer_set:
        if raw_str in str(answer):
            return str(answer).strip()

    return None


def load_config(config_folder:str) -> tuple:
    config_path = Path(config_folder)
    with open(config_path / "config.json", "r") as file:
        config: dict = json.load(file)
    return (config.get("model", {}), config.get("synthesis", {}),
            config.get("survey", {}), config.get("analysis", {}))


class _singleAnswerTool(lr.agent.ToolMessage):
    request: str = "singleAnswerResponse"
    purpose: str = "To respond with the <TEXT> of the answer that you specify."
    TEXT: int

    @classmethod
    def example(cls):
        return [
            cls(TEXT=45),
            ("To respond to the survey question with only one answer", cls(TEXT=5))
        ]


class _multipleAnswerTool(lr.agent.ToolMessage):
    request: str = "multipleAnswerResponse"
    purpose: str = "To respond with a list of <TEXT> of the answers that apply to your response."
    TEXT: Tuple[int]

    @classmethod
    def example(cls):
        return [
            cls(TEXT=(4, 7, 12)),
            ("I want to response with the keys of the 4 answers that apply to me", cls(TEXT=(4, 8, 23, 35))),
            ("Only one answer applys to me.", cls(TEXT=(5,)))
        ]


class _discreteNumericTool(lr.agent.ToolMessage):
    request: str = "discreteNumericResponse"
    purpose: str = "To respond with an appropriate numeric <NUMERIC> value when none of the possible responses make sense."
    NUMERIC: int

    @classmethod
    def example(cls):
        return [
            cls(NUMERIC=43),
            ("I want to respond with my age of 28", cls(NUMERIC=28)),
            ("I want to respond with my yearly salary", cls(NUMERIC=43_000))
        ]


class SurveyAgent(lr.ChatAgent):
    def __init__(self, config: lr.ChatAgentConfig, agent_id: str, bio:str, serial_number: str):
        super().__init__(config)
        self.agent_id = agent_id
        self.bio = bio
        self.serial_number = serial_number
        self.responses = []
        self.question_variables = []
        self.question_dtypes = []
        self.dtype_matches = []
        self.queued_question: str
        self.possible_responses: Dict[int, str]
        self.queued_keys: List[int]
        self.answer_response = None
        self.answer_types = []
        self.survey_complete: bool
        self.survey_failed: bool

    def queue_question(self, variable: str, question_package: Dict[str, str | Dict[int, str]], shuffle_response: bool = False):
        self.question_beginning = question_package["question"]
        self.possible_responses = question_package["response"]
        if shuffle_response:
            items = list(self.possible_responses.items())
            random.shuffle(items)
            self.possible_responses = dict(items)
        self.queued_keys = list(self.possible_responses.keys())
        self.question_variables.append(variable)
        self.question_dtypes.append(question_package["dtype"])
        if self.question_dtypes[-1] == "TEXT":
            self.queued_question = f"{self.question_beginning} Available options: " + "; ".join(
                f"{key}: {value}" for key, value in self.possible_responses.items())
        elif self.question_dtypes[-1] == "NUMERIC":
            self.queued_question = f"{self.question_beginning} Please provide a numeric response or select an alternative: " + "; ".join(
                f"{key}: {value}" for key, value in self.possible_responses.items())

    def llm_response(self, message: Optional[str | ChatDocument] = None) -> Optional[ChatDocument]:
        return super().llm_response(message)

    def ask_question(self):
        self.llm_response(self.queued_question)

    def singleAnswerResponse(self, msg: _singleAnswerTool) -> str:
        self.dtype_matches.append("TEXT" == self.question_dtypes[-1])
        return str(msg.TEXT if msg.TEXT in self.queued_keys else None)

    def multipleAnswerResponse(self, msg: _multipleAnswerTool):
        self.dtype_matches.append("TEXT" == self.question_dtypes[-1])
        return str(_multipleAnswerTool.TEXT if all(key in self.queued_keys for key in _multipleAnswerTool.TEXT) else None)

    def discreteNumericResponse(self, msg: _discreteNumericTool):
        self.dtype_matches.append("NUMERIC" == self.question_dtypes[-1])
        return str(msg.NUMERIC if msg.NUMERIC not in self.queued_keys else None)


def _compute_row_weights(df: pd.DataFrame, demographic_labels: list[str]) -> pd.Series:
    weights = pd.Series(1.0, index=df.index)
    used_cols = []
    for col in demographic_labels:
        if col not in df.columns:
            continue
        non_null_ratio = df[col].notna().mean()
        if non_null_ratio < 0.2:
            continue
        freqs = df[col].value_counts(normalize=True)
        if freqs.empty:
            continue
        inv = df[col].map(freqs).rdiv(1.0)
        inv = inv.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        weights *= inv
        used_cols.append(col)
    if not used_cols or weights.sum() == 0 or np.isnan(weights.sum()):
        return pd.Series(1.0 / len(df), index=df.index)
    return weights / weights.sum()


def sample_representative_population(
        data_path: str,
        n_sample: int,
        demographic_labels: List[str],
        *,
        random_state: int | None = None,
    ) -> pd.DataFrame:
    df = (
        safe_csv_read(data_path)
        if data_path.lower().endswith(".csv")
        else pd.read_excel(data_path)
    )
    if n_sample >= len(df):
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    weights = _compute_row_weights(df, demographic_labels)
    sampled_df = df.sample(n=n_sample, weights=weights, replace=False, random_state=42)
    return sampled_df.reset_index(drop=True)


def detect_socio_demographics(mapping_dict: dict, model_config: dict) -> dict:
    key_desc = {k: v[1] if isinstance(v, tuple) else str(v) for k, v in mapping_dict.items()}
    prompt = f"""
<|system|>
You are a model that returns only JSON.
You MUST respond with a single JSON object with key "labels" containing ALL headers
that represent socio-demographic variables (age, gender, income, education, household composition, employment, etc.).
Do NOT include any explanations, extra text, or examples.
Ensure it is parseable by json.loads() in Python.
<|end|>

<|user|>
Dictionary:
{json.dumps(key_desc, indent=2)}
<|end|>
<|assistant|>
    """
    model = AutoModelForCausalLM.from_pretrained(
        "agents-test-assessment/synth-survey-gen/gpt-oss-20b-Q4_K_M.gguf",
        model_file="gpt-oss-20b-Q4_K_M.gguf",
        model_type='llama',
        context_length=48000,
        gpu_layers=24,
        local_files_only=True
    )
    content = model(prompt)
    print(content)
    try:
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            json_str = match.group(1)
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)
            return json.loads(json_str)
        else:
            print("No JSON found in LLM response.")
            return {"labels": []}
    except json.JSONDecodeError as e:
        print(f"JSON decode failed: {e}. Response was:\n{content}")
        return {"labels": []}


def infer_dtypes_with_llm(
    mapping_dict: dict,
    ollama_model: str = "gpt-oss:20b",
) -> dict:
    """Use an LLM to infer the expected response dtype for each survey variable.

    The mapping_dict has 3-element tuples (desc, question, answer_set).
    This function adds a 4th element (dtype) using LLM classification.
    Falls back to static inference if the LLM call fails.
    """
    import requests

    model_name = ollama_model.replace("ollama/", "", 1)

    # Check if dtype already set (some entries may have 4 elements)
    already_set = all(isinstance(v, tuple) and len(v) >= 4 for v in mapping_dict.values())
    if already_set:
        print("Dtype already set in mapping_dict, skipping LLM inference.")
        return mapping_dict

    lines = []
    for header, (desc, question, answer_set) in mapping_dict.items():
        answers = list(answer_set) if answer_set else []
        answers_str = "; ".join(str(a) for a in answers) if answers else "None (free response)"
        lines.append(
            f"Variable: {header}\n"
            f"  Question: {question}\n"
            f"  Description: {desc}\n"
            f"  Options: {answers_str}"
        )

    prompt = (
        "You are a classifier that determines the expected response type for survey variables.\n"
        "For each variable, choose exactly ONE of these types:\n"
        "- SINGLE: predefined answer options, respondent selects exactly one\n"
        "- MULTIPLE: predefined answer options, respondent can select more than one (e.g. 'select all that apply')\n"
        "- NUMERIC: no predefined options, expects a number (age, income, count, distance, etc.)\n"
        "- TEXT: no predefined options, expects free text (name, description, etc.)\n\n"
        "Respond with ONLY a valid JSON object where keys are variable names and values are the type.\n"
        "Example: {\"AGE\": \"NUMERIC\", \"GENDER\": \"SINGLE\", \"HOBBIES\": \"MULTIPLE\"}\n\n"
        "Variables:\n"
        + "\n\n".join(lines)
        + "\n\nJSON:"
    )

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=180,
        )
        resp.raise_for_status()
        content = resp.json().get("response", "").strip()
        print(f"[LLM dtype] Response snippet: {content[:200]}...")

        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if not match:
            print("No JSON found in LLM dtype response. Using static fallback.")
            return _fallback_dtype(mapping_dict)

        json_str = match.group(1)
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        dtype_map = json.loads(json_str)
        print(f"[LLM dtype] Classified {len(dtype_map)} variables.")
    except Exception as e:
        print(f"LLM dtype inference failed: {e}. Using static fallback.")
        return _fallback_dtype(mapping_dict)

    for header in mapping_dict:
        desc, question, answer_set = mapping_dict[header][:3]
        dtype = dtype_map.get(header, "TEXT")
        mapping_dict[header] = (desc, question, answer_set, dtype)

    return mapping_dict


def _fallback_dtype(mapping_dict: dict) -> dict:
    """Static fallback for dtype inference when LLM is unavailable."""
    from preprocess import infer_dtype

    for header in mapping_dict:
        desc, question, answer_set = mapping_dict[header][:3]
        dtype = infer_dtype(header, answer_set, str(question), str(desc))
        mapping_dict[header] = (desc, question, answer_set, dtype)

    print("Applied static dtype fallback.")
    return mapping_dict


def build_generic_agents(config_folder: str, map: dict, data_path: str, desc_path:str, **kwargs):
    model_config, synth_conf, _, _ = load_config(config_folder)
    header = synth_conf.get("system_message_header", "")
    footer = synth_conf.get("system_message_footer", "")
    subsample = synth_conf.get("subsample", 10)

    desc_df = (
        safe_csv_read(desc_path)
        if desc_path.endswith(".csv")
        else pd.read_excel(desc_path)
    )
    detected_labels = extract_personal_agent_info(desc_df)
    if not detected_labels:
        detected_labels = detect_socio_demographics(mapping_dict=map, model_config=model_config)
    print(detected_labels)
    
    sampled_df = sample_representative_population(
        data_path=data_path, 
        n_sample=subsample, 
        demographic_labels=detected_labels["labels"]
    )
    sampled_df.to_csv(Path(config_folder) / "sampled_oracle.csv", index=False)

    bio_model = synth_conf.get("bio_model", None)

    # Compute serial numbers for the sampled rows
    serial_numbers = [
        row.get("SERIALNO", f"GEN_{idx}")
        for idx, row in sampled_df.iterrows()
    ]

    # Try to load cached bios
    dataset_name = os.path.basename(os.path.dirname(data_path.rstrip("/\\")))
    bio_dir = Path(config_folder) / "bio"
    bio_dir.mkdir(parents=True, exist_ok=True)
    bio_file = bio_dir / f"bio_{dataset_name}.json"

    cached_bios = {}
    if bio_file.exists():
        try:
            with open(bio_file, "r") as f:
                cached_bios = json.load(f)
            print(f"Loaded {len(cached_bios)} cached bios from {bio_file}")
        except Exception as e:
            print(f"Failed to load cached bios: {e}. Regenerating.")
            cached_bios = {}

    # Generate only missing bios
    system_prompts = []
    for idx, row in sampled_df.iterrows():
        serial_number = serial_numbers[idx]
        if serial_number in cached_bios:
            system_prompts.append((cached_bios[serial_number], serial_number))
        else:
            system_message = create_system_prompt(
                row=row,
                mapping_dict=map,
                demo_labels=detected_labels["labels"],
                config_folder=config_folder,
                bio_model=bio_model,
            )
            system_prompts.append((system_message, serial_number))
            cached_bios[serial_number] = system_message

    # Save all bios to cache
    try:
        with open(bio_file, "w") as f:
            json.dump(cached_bios, f, indent=2)
        print(f"Saved {len(cached_bios)} bios to {bio_file}")
    except Exception as e:
        print(f"Failed to save bios to cache: {e}")

    llm_config = lm.OpenAIGPTConfig(**model_config)
    agents = []
    for i, (system_message, serial_number) in enumerate(system_prompts):
        agent_config = lr.ChatAgentConfig(
            name=f"Agent_{i}",
            llm=llm_config,
            system_message=header + system_message + footer,
            use_tools=True,
            use_functions_api=True
        )
        agent = SurveyAgent(config=agent_config, agent_id=i, bio=system_message, serial_number=serial_number)
        agent.enable_message(_singleAnswerTool)
        agent.enable_message(_multipleAnswerTool)
        agent.enable_message(_discreteNumericTool)
        agents.append(agent)
    return agents


def generate_discoursive_bio(profile_data: dict, mapping_data: dict, bio_model: str) -> str:
    """
    Generate a natural, first-person discoursive biography from
    sociodemographic key-value pairs using Ollama.

    Falls back to a descriptive sentence if Ollama is unavailable.
    """
    desc_lines = []
    for key, value in profile_data.items():
        label = mapping_data.get(key, key)
        desc_lines.append(f"{label}: {value}")

    prompt = (
        "Generate a natural, first-person biography, trying to be as specific as possible, "
        "for a survey respondent based on their sociodemographic profile below. "
        "Write ONLY the biography, no extra text.\n\n"
        "Profile:\n" + "\n".join(f"- {l}" for l in desc_lines) + "\n\n"
        "Biography:"
    )

    import requests
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": bio_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7},
            },
            timeout=120,
        )
        resp.raise_for_status()
        bio = resp.json().get("response", "").strip()
        if bio:
            return bio
    except Exception as e:
        print(f"Ollama bio generation failed: {e}. Using fallback.")

    parts = [f"{mapping_data.get(k, k)} is {v}" for k, v in profile_data.items()]
    return "You are a person with the following characteristics: " + "; ".join(parts)


def create_system_prompt(
    row: pd.Series,
    mapping_dict: Dict,
    demo_labels: List[str],
    config_folder: str,
    bio_model: str | None = None,
) -> str:
    profile_data = {}
    for label in demo_labels:
        if label not in row.index:
            continue
        if label not in mapping_dict:
            continue
        val1 = row[label]
        if pd.isna(val1) or val1 is None:
            continue
        # Resolve raw code to full answer string from answer set
        desc_tuple = mapping_dict[label]
        if isinstance(desc_tuple, tuple) and len(desc_tuple) >= 3:
            answer_set = desc_tuple[2]
            resolved = _resolve_answer_label(val1, answer_set)
            if resolved:
                profile_data[label] = resolved
                continue
        profile_data[label] = val1

    mapping_context = {}
    for label in demo_labels:
        if label not in row.index:
            continue
        val2 = row[label]
        if pd.isna(val2) or val2 is None:
            continue
        if label in mapping_dict:
            desc_tuple = mapping_dict[label]
            if isinstance(desc_tuple, tuple):
                mapping_context[label] = desc_tuple[0]
            else:
                mapping_context[label] = str(desc_tuple)

    discoursive_bio = None
    if bio_model:
        discoursive_bio = generate_discoursive_bio(profile_data, mapping_context, bio_model)

    try:
        msg_gen = SystemMessageGenerator(config_folder, "SystemMessage.j2")
        system_message = msg_gen.write_system_message(
            profile=profile_data,
            mapping=mapping_context,
            discoursive_bio=discoursive_bio,
            survey_intro="General Purpose Survey",
        )
    except Exception as e:
        print(f"Template generation failed: {e}. Using fallback prompt.")
        if discoursive_bio:
            system_message = discoursive_bio
        else:
            profile_str = ", ".join([f"{k}: {v}" for k, v in profile_data.items()])
            system_message = f"""You are a survey respondent with the following characteristics:
        {profile_str}

        When answering questions, respond authentically based on your profile.
        Use the provided tools to submit your answers.
        """
    return system_message
