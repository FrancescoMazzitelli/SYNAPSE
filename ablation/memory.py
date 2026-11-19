"""Memory conditions for the ablation study.

The production pipeline (:class:`gp_survey_engine.AgenticSurveyPipeline`)
hard-codes a sliding window of the last 5 answers. Instead of forking it, this
module subclasses it and only overrides how the answer history is turned into
the prompt's memory block, so all the remaining behaviour (validation, retries,
response packaging) stays identical to the paper pipeline.

Supported conditions:

* ``none``   -- no previous answers are shown to the agent
* ``window`` -- the last ``window`` answers (``window=5`` reproduces the paper)
* ``full``   -- the whole answer history of the agent
"""

from __future__ import annotations

from functools import partial
import re
from typing import Dict, List, Optional, Tuple

from gp_survey_engine import AgenticSurveyPipeline

MEMORY_HEADER = "### RECENT ANSWERS (memory for consistency):"
VALID_MODES = ("none", "window", "full")


def parse_memory_condition(condition) -> Dict:
    """Normalise a spec entry into ``{name, mode, window}``.

    Accepts either a dict (``{"name": "window5", "mode": "window", "window": 5}``)
    or shorthands such as ``"none"``, ``"full"``, ``"window5"``/``"window:5"``.
    """
    if isinstance(condition, str):
        raw = condition.strip().lower()
        if raw in ("none", "no_memory", "no-memory", "absent"):
            return {"name": "none", "mode": "none", "window": 0}
        if raw in ("full", "complete", "all"):
            return {"name": "full", "mode": "full", "window": None}
        if raw.startswith("window"):
            tail = raw[len("window"):].lstrip(": _-")
            window = int(tail) if tail else 5
            return {"name": cond_name("window", window), "mode": "window", "window": window}
        raise ValueError(f"Unknown memory condition: {condition!r}")

    if isinstance(condition, dict):
        name = str(condition.get("name") or condition.get("mode") or "").strip()
        mode = str(condition.get("mode") or name).strip().lower()
        if ":" in mode:  # e.g. "window:5"
            mode, window_s = mode.split(":", 1)
            mode = mode.strip().lower()
            window = int(window_s) if window_s.strip() else 5
        else:
            window = condition.get("window", 5)
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown memory mode: {mode!r}")
        if mode == "window":
            window = int(window) if window is not None else 5
            if window < 0:
                raise ValueError(f"Memory window must be >= 0, got {window}")
        else:
            window = 0 if mode == "none" else None
        return {"name": name or cond_name(mode, window), "mode": mode, "window": window}

    raise TypeError(f"Memory condition must be str or dict, got {type(condition).__name__}")


def cond_name(mode: str, window: Optional[int]) -> str:
    if mode == "window":
        return f"window{window}"
    return mode


def condition_tag(condition: Dict, temperature: float) -> str:
    """Directory-safe tag for one memory x temperature cell."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(condition["name"])).strip("-._") or "mem"
    return f"mem-{name}_temp-{temperature:g}"


class MemoryAwareSurveyPipeline(AgenticSurveyPipeline):
    """Production pipeline with a configurable memory policy."""

    def __init__(self, *args, memory_mode: str = "window", memory_window: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        mode = (memory_mode or "window").strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown memory mode: {mode!r}")
        if mode == "window":
            if memory_window is None:
                memory_window = 5
            if int(memory_window) < 0:
                raise ValueError(f"Memory window must be >= 0, got {memory_window}")
            memory_window = int(memory_window)
        else:
            memory_window = 0 if mode == "none" else None
        self.memory_mode = mode
        self.memory_window = memory_window

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    @property
    def memory_enabled(self) -> bool:
        if self.memory_mode == "none":
            return False
        if self.memory_mode == "window" and not self.memory_window:
            return False
        return True

    def _generate_memory_context(self, history: List[Tuple[str, str]]) -> str:
        if not self.memory_enabled or not history:
            return ""

        recent = history if self.memory_mode == "full" else history[-self.memory_window:]

        context_block = "\n\n" + MEMORY_HEADER + "\n"
        for q, a in recent:
            context_block += f"Question: {q}\nAnswer: {a}\n"
        return context_block

    def _create_agent_prompt(self, *args, **kwargs) -> str:
        prompt = super()._create_agent_prompt(*args, **kwargs)
        if not self.memory_enabled:
            # The base prompt always renders the memory section ("None"); drop it
            # entirely for the memory-absent condition.
            prompt = strip_memory_block(prompt)
        return prompt


def strip_memory_block(prompt: str) -> str:
    """Remove the ``### RECENT ANSWERS`` section from a rendered prompt."""
    lines = prompt.split("\n")
    out: List[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == MEMORY_HEADER:
            i += 2  # header + the "None" content line
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def pipeline_factory(condition: Dict):
    """Build the callable accepted by ``ablation.pipeline.run_gp_survey(...)``."""
    return partial(
        MemoryAwareSurveyPipeline,
        memory_mode=condition["mode"],
        memory_window=condition.get("window"),
    )
