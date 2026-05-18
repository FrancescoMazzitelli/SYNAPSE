from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LADARAGConfig:
    enabled: bool = False
    ollama_url: str = "http://localhost:11434"
    model: str = "phi4-reasoning:14b"
    timeout: int = 120
    services: dict = field(default_factory=dict)

    def __init__(self, config_dict: Optional[dict] = None):
        d = config_dict or {}
        self.enabled = bool(d.get("enabled", False))
        self.ollama_url = str(d.get("ollama_url", "http://localhost:11434"))
        self.model = str(d.get("model", "phi4-reasoning:14b"))
        self.timeout = int(d.get("timeout", 120))
        self.services = dict(d.get("services", {}))
