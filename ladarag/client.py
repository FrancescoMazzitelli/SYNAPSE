import json
import time
import threading
from typing import Optional

from .config import LADARAGConfig
from .db.storage import ServiceStorage
from .db.registry import ServiceRegistry
from .db.gateway import ServiceGateway
from .core.controlService import Controller
from .core.designerService import Designer


# Minimal OD classification prompt: asks the LLM whether a question is
# origin-destination related. Fast single-call, no plan generation.
OD_CLASSIFICATION_PROMPT = """You are a binary classifier for origin-destination travel queries.
Given a question, agent persona, and question description, determine if this question
is about origin-destination (OD) travel — i.e. the respondent's travel patterns
between locations, trip origins and destinations, route choices, or mode of transport
for specific trips.

Examples of OD questions:
- "How do you typically commute to work?"
- "What is your primary mode of transportation for daily travel?"
- "Which of the following best describes your usual route from home to work?"
- "How long does it take you to travel from home to your main activity location?"
- "Do you pass through any of these areas on your way to work?"

Examples of NON-OD questions:
- "What is your age?"
- "What is your household income?"
- "What is your level of education?"
- "Do you own a car?"
- "How many people live in your household?"
- "What is your occupation?"

Respond with ONLY a single word: YES or NO.

Question: {question_text}
Question description: {question_desc}
Agent persona: {agent_bio}"""


# Full OD query prompt with service catalog context
OD_QUERY_PROMPT = """You are an origin-destination travel survey respondent.
You have access to a multi-modal trip planning service catalog.

Persona:
{agent_bio}

Question: {question_text}
Question description: {question_desc}

Available trip planning services:
{catalog_context}

Based on the services available and your persona, briefly describe ONE specific,
realistic origin-destination trip scenario that matches both the question and your persona.

Output ONLY a JSON object with these fields:
  "trip_purpose": "string — why this trip is made (e.g. work commute, school run, grocery shopping)",
  "origin":       "string — approximate origin area or neighborhood",
  "destination":  "string — approximate destination area or neighborhood",
  "mode":         "string — primary mode of transport that would be chosen",
  "route_notes":  "string — any relevant routing considerations or constraints"
"""


class LADARAG:
    def __init__(self, config: LADARAGConfig):
        self._config = config
        self._lock = threading.Lock()
        self._controller: Optional[Controller] = None
        self._gateway: Optional[ServiceGateway] = None
        self._initialized = False

        if config.enabled:
            self._initialize()

    def _initialize(self):
        if self._initialized:
            return

        ollama_url = self._config.ollama_url
        model = self._config.model
        timeout = self._config.timeout

        storage = ServiceStorage()
        registry = ServiceRegistry()

        if self._config.services:
            storage.load_from_config(self._config.services)
            registry.load_from_config(self._config.services)

        gateway = ServiceGateway(
            storage=storage,
            registry=registry,
            ollama_url=ollama_url,
            model=model,
        )

        designer = Designer(fallback_model=model)

        controller = Controller(
            ollama_url=ollama_url,
            model=model,
            timeout=timeout,
            designer=designer,
        )

        self._gateway = gateway
        self._controller = controller
        self._storage = storage
        self._registry = registry
        self._initialized = True

    def is_enabled(self) -> bool:
        return self._config.enabled and self._initialized

    def classify_question(self, question_text: str,
                          question_desc: str,
                          agent_bio: str) -> bool:
        if not self.is_enabled():
            return False
        prompt = OD_CLASSIFICATION_PROMPT.format(
            question_text=question_text,
            question_desc=question_desc,
            agent_bio=agent_bio,
        )
        try:
            import requests
            resp = requests.post(
                f"{self._config.ollama_url}/api/chat",
                json={
                    "model": self._config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0.0, "num_predict": 10},
                    "stream": False,
                },
                timeout=15,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip().upper()
            return "YES" in content
        except Exception as e:
            print(f"[LADARAG] classify_question error: {e}")
            return False

    def query(self, question_text: str,
              question_desc: str,
              agent_bio: str) -> dict:
        if not self.is_enabled():
            return {}

        gateway = self._gateway
        services = gateway.search(question_text, top_k=5)

        catalog_lines = ["Service catalog:"]
        for s in services:
            sid = s.get("_id", "?")
            desc = s.get("description", "")
            caps = s.get("capabilities", {})
            caps_str = "; ".join(
                f"{op}: {desc}" for op, desc in list(caps.items())[:3]
            )
            catalog_lines.append(f"  - {sid}: {desc} | {caps_str}")

        catalog_context = "\n".join(catalog_lines)

        prompt = OD_QUERY_PROMPT.format(
            agent_bio=agent_bio,
            question_text=question_text,
            question_desc=question_desc,
            catalog_context=catalog_context,
        )

        try:
            import requests
            resp = requests.post(
                f"{self._config.ollama_url}/api/chat",
                json={
                    "model": self._config.model,
                    "format": {
                        "type": "object",
                        "properties": {
                            "trip_purpose": {"type": "string"},
                            "origin":       {"type": "string"},
                            "destination":  {"type": "string"},
                            "mode":         {"type": "string"},
                            "route_notes":  {"type": "string"},
                        },
                        "required": ["trip_purpose", "origin", "destination",
                                     "mode", "route_notes"],
                    },
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0.3, "num_predict": 500},
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
            return json.loads(content)
        except Exception as e:
            print(f"[LADARAG] query error: {e}")
            return {}

    def format_od_context(self, result: dict) -> str:
        if not result:
            return ""
        parts = [
            "### Origin-Destination Context",
            f"Trip purpose: {result.get('trip_purpose', 'N/A')}",
            f"Origin: {result.get('origin', 'N/A')}",
            f"Destination: {result.get('destination', 'N/A')}",
            f"Mode: {result.get('mode', 'N/A')}",
            f"Route notes: {result.get('route_notes', 'N/A')}",
        ]
        return "\n".join(parts)
