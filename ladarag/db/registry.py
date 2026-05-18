import threading
from typing import Optional


class ServiceRegistry:
    def __init__(self):
        self._services: dict[str, dict] = {}
        self._lock = threading.Lock()

    def register(self, service_id: str, metadata: Optional[dict] = None):
        with self._lock:
            info = {"id": service_id, "service": service_id}
            if metadata:
                info["meta"] = {"service_doc_id": metadata.get("catalog_id", service_id)}
            self._services[service_id] = info

    def deregister(self, service_id: str):
        with self._lock:
            self._services.pop(service_id, None)

    def is_registered(self, service_id: str) -> bool:
        with self._lock:
            return service_id in self._services

    def list_services(self) -> list[dict]:
        with self._lock:
            return list(self._services.values())

    def get_registry_ids(self) -> set[str]:
        with self._lock:
            return set(self._services.keys())

    def load_from_config(self, services_config: dict):
        for sid in services_config:
            self.register(sid, {"catalog_id": sid})
