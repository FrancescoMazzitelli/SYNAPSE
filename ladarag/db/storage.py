from __future__ import annotations

import sqlite3
import json
import threading
from typing import Optional


class ServiceStorage:
    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS services (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                base_url TEXT NOT NULL DEFAULT '',
                capabilities TEXT NOT NULL DEFAULT '{}',
                endpoints TEXT NOT NULL DEFAULT '{}',
                parameters TEXT NOT NULL DEFAULT '{}',
                response_schemas TEXT NOT NULL DEFAULT '{}',
                request_schemas TEXT NOT NULL DEFAULT '{}',
                generated_description TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def _row_to_doc(self, row: tuple) -> dict:
        return {
            "_id": row[0],
            "name": row[1],
            "description": row[2],
            "base_url": row[3],
            "capabilities": json.loads(row[4]),
            "endpoints": json.loads(row[5]),
            "parameters": json.loads(row[6]),
            "response_schemas": json.loads(row[7]),
            "request_schemas": json.loads(row[8]),
            "generated_description": row[9],
        }

    def get(self, service_id: str) -> Optional[dict]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM services WHERE id = ?", (service_id,))
            row = cur.fetchone()
            return self._row_to_doc(row) if row else None

    def list_all(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM services")
            rows = cur.fetchall()
            return [self._row_to_doc(r) for r in rows]

    def upsert(self, doc: dict):
        service_id = doc.get("_id")
        if not service_id:
            raise ValueError("Document must have '_id' field")
        keys = ("name", "description", "base_url", "capabilities", "endpoints",
                "parameters", "response_schemas", "request_schemas", "generated_description")
        name = doc.get("name", "")
        description = doc.get("description", "")
        base_url = doc.get("base_url", "")
        capabilities = json.dumps(doc.get("capabilities", {}))
        endpoints = json.dumps(doc.get("endpoints", {}))
        parameters = json.dumps(doc.get("parameters", {}))
        response_schemas = json.dumps(doc.get("response_schemas", {}))
        request_schemas = json.dumps(doc.get("request_schemas", {}))
        generated_description = doc.get("generated_description", "")
        with self._lock:
            self._conn.execute("""
                INSERT INTO services (id, name, description, base_url, capabilities,
                    endpoints, parameters, response_schemas, request_schemas,
                    generated_description, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    base_url = excluded.base_url,
                    capabilities = excluded.capabilities,
                    endpoints = excluded.endpoints,
                    parameters = excluded.parameters,
                    response_schemas = excluded.response_schemas,
                    request_schemas = excluded.request_schemas,
                    generated_description = excluded.generated_description,
                    updated_at = CURRENT_TIMESTAMP
            """, (service_id, name, description, base_url, capabilities,
                  endpoints, parameters, response_schemas, request_schemas,
                  generated_description))
            self._conn.commit()

    def delete(self, service_id: str):
        with self._lock:
            self._conn.execute("DELETE FROM services WHERE id = ?", (service_id,))
            self._conn.commit()

    def update_schemas(self, service_id: str, schemas: dict):
        sets = []
        params = []
        for key in ("response_schemas", "request_schemas", "parameters", "generated_description"):
            if key in schemas:
                sets.append(f"{key} = ?")
                params.append(json.dumps(schemas[key]) if isinstance(schemas[key], dict) else schemas[key])
        if not sets:
            return
        params.append(service_id)
        with self._lock:
            self._conn.execute(f"UPDATE services SET {', '.join(sets)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?", params)
            self._conn.commit()

    def load_from_config(self, services_config: dict):
        for sid, svc_cfg in services_config.items():
            doc = {
                "_id": sid,
                "name": svc_cfg.get("name", sid),
                "description": svc_cfg.get("description", ""),
                "base_url": svc_cfg.get("url", ""),
                "capabilities": {},
                "endpoints": {},
                "parameters": {},
                "response_schemas": {},
                "request_schemas": {},
                "generated_description": svc_cfg.get("generated_description", ""),
            }
            for ep_key, ep_cfg in svc_cfg.get("endpoints", {}).items():
                method = "GET"
                path = ep_key
                if " " in ep_key:
                    parts = ep_key.split(" ", 1)
                    method = parts[0]
                    path = parts[1]
                op_key = f"{method} {ep_cfg.get('path', path)}"
                doc["capabilities"][op_key] = ep_cfg.get("capability", "")
                doc["endpoints"][op_key] = f"{doc['base_url'].rstrip('/')}{ep_cfg.get('path', path)}"
                doc["parameters"][op_key] = ep_cfg.get("parameters", "")
                doc["response_schemas"][op_key] = ep_cfg.get("response_schema", "")
                doc["request_schemas"][op_key] = ep_cfg.get("request_schema", "")
            self.upsert(doc)
