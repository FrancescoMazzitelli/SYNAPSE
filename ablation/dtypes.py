"""Remote dtype inference for the ablation study (new method, no core edits).

``synthesize.infer_dtypes_with_llm`` talks to ``http://localhost:11434``. The
ablation endpoint is configured in ``ablation/spec.json``, so this new function
keeps the same prompt/parsing logic but posts to the configured Ollama host.
"""

from __future__ import annotations

import json
import re
from typing import Dict

import requests

from synthesize import _fallback_dtype


def infer_dtypes_with_llm_at(
    mapping_dict: Dict,
    ollama_model: str,
    base_url: str,
    timeout: float = 180.0,
) -> Dict:
    """Same classifier as the production code, but against ``base_url``."""
    model_name = ollama_model.replace("ollama/", "", 1)
    if all(isinstance(v, tuple) and len(v) >= 4 for v in mapping_dict.values()):
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
        "- MULTIPLE: predefined answer options, respondent can select more than one "
        "(e.g. 'select all that apply')\n"
        "- NUMERIC: no predefined options, expects a number (age, income, count, distance, etc.)\n"
        "- TEXT: no predefined options, expects free text (name, description, etc.)\n\n"
        "Respond with ONLY a valid JSON object where keys are variable names and values are the type.\n"
        'Example: {"AGE": "NUMERIC", "GENDER": "SINGLE", "HOBBIES": "MULTIPLE"}\n\n'
        "Variables:\n" + "\n\n".join(lines) + "\n\nJSON:"
    )

    url = base_url.rstrip("/") + "/api/generate"
    try:
        resp = requests.post(
            url,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json().get("response", "").strip()
        print(f"[LLM dtype @ {base_url}] Response snippet: {content[:200]}...")

        match = re.search(r"(\{.*\})", content, re.DOTALL)
        if not match:
            print("No JSON found in LLM dtype response. Using static fallback.")
            return _fallback_dtype(mapping_dict)

        json_str = re.sub(r",\s*}", "}", match.group(1))
        json_str = re.sub(r",\s*]", "]", json_str)
        dtype_map = json.loads(json_str)
        print(f"[LLM dtype] Classified {len(dtype_map)} variables.")
    except Exception as exc:  # noqa: BLE001
        print(f"LLM dtype inference at {base_url} failed: {exc}. Using static fallback.")
        return _fallback_dtype(mapping_dict)

    for header in mapping_dict:
        desc, question, answer_set = mapping_dict[header][:3]
        dtype = dtype_map.get(header, "TEXT")
        mapping_dict[header] = (desc, question, answer_set, dtype)
    return mapping_dict
