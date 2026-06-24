from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValhallaConfig:
    enabled: bool = False
    tile_dir: str = "/tmp/valhalla_tiles"
    costing: str = "auto"

    def __init__(self, config_dict: Optional[dict] = None):
        d = config_dict or {}
        self.enabled = bool(d.get("enabled", False))
        self.tile_dir = str(d.get("tile_dir", "/tmp/valhalla_tiles"))
        self.costing = str(d.get("costing", "auto"))


@dataclass
class MapContextConfig:
    enabled: bool = False
    default_radius_m: float = 1500.0
    default_knn: int = 20
    cache_dir: Optional[str] = None

    def __init__(self, config_dict: Optional[dict] = None):
        d = config_dict or {}
        self.enabled = bool(d.get("enabled", False))
        self.default_radius_m = float(d.get("default_radius_m", 1500.0))
        self.default_knn = int(d.get("default_knn", 20))
        self.cache_dir = d.get("cache_dir")


@dataclass
class LADARAGConfig:
    enabled: bool = False
    ollama_url: str = "http://localhost:11434"
    model: str = "phi4-reasoning:14b"
    timeout: int = 120
    services: dict = field(default_factory=dict)
    valhalla: ValhallaConfig = field(default_factory=ValhallaConfig)
    map_context: MapContextConfig = field(default_factory=MapContextConfig)

    def __init__(self, config_dict: Optional[dict] = None):
        d = config_dict or {}
        self.enabled = bool(d.get("enabled", False))
        self.ollama_url = str(d.get("ollama_url", "http://localhost:11434"))
        self.model = str(d.get("model", "phi4-reasoning:14b"))
        self.timeout = int(d.get("timeout", 120))
        self.services = dict(d.get("services", {}))
        self.valhalla = ValhallaConfig(d.get("valhalla", {}))
        self.map_context = MapContextConfig(d.get("map_context", {}))
