from __future__ import annotations

import json
import re
import time
import uuid
import requests
import numpy as np
from typing import Optional
from .storage import ServiceStorage
from .registry import ServiceRegistry


DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition assistant for a service discovery system.
Given a user query, decompose it into atomic information needs.
Each sub-query targets ONE type of information that can be satisfied by a single type of API/service.

RULES:
- Output 1 to 4 sub-queries.
- If the query asks for ONE type of information, output exactly one sub-query.
- Each sub-query is a short noun phrase (2-6 words), describing the resource/data type.
- Strip references to specific entities, locations, filters, conditions — keep only the resource type.
- Sub-queries must be independent (no references between them).
- Use the same language as the input query.
- Do NOT invent information not implied by the query.

EXAMPLES:
Input: "list all temperature sensors"
Output: {"sub_queries": ["temperature sensors"]}

Input: "find car parks near Arco di Traiano with charging stations nearby"
Output: {"sub_queries": ["parking spots", "tourist attractions", "charging stations"]}

Input: "show me air quality in zones with heavy traffic"
Output: {"sub_queries": ["air quality measurements", "traffic data by zone"]}

Input: "create a new user account"
Output: {"sub_queries": ["user account management"]}
"""

DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        }
    },
    "required": ["sub_queries"],
}


class ServiceGateway:
    def __init__(self, storage: ServiceStorage = None, registry: ServiceRegistry = None,
                 ollama_url: str = "http://localhost:11434",
                 model: str = "phi4-reasoning:14b"):
        self.storage = storage
        self.registry = registry
        self.ollama_url = ollama_url
        self.model = model
        self._embedding_dim = 1024
        self._index = None
        self._index_payloads: list[dict] = []
        self._index_built = False

    def set_storage(self, storage: ServiceStorage):
        self.storage = storage
        self._index_built = False

    def set_registry(self, registry: ServiceRegistry):
        self.registry = registry

    def _ollama_request(self, payload: dict, timeout: int = 30) -> dict:
        resp = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _get_embedding(self, text: str, prefix: str = "") -> list[float]:
        full_text = f"{prefix}{text}" if prefix else text
        resp = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.model, "prompt": full_text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(text, prefix="query: ")

    def embed_passage(self, text: str) -> list[float]:
        return self._get_embedding(text, prefix="passage: ")

    def _count_tokens(self, text: str) -> int:
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model, "prompt": text[:8000]},
                timeout=10,
            )
            resp.raise_for_status()
            return len(text) // 4
        except Exception:
            return len(text) // 4

    def _select_stage1_text(self, doc: dict) -> tuple[str, str]:
        description = (doc.get("description") or "").strip()
        generated = (doc.get("generated_description") or "").strip()
        if description and generated:
            return f"{description} {generated}", "description+generated"
        if len(description) >= 100:
            return description, "description"
        if generated:
            return generated, "generated_description"
        if description:
            return description, "description"
        return "", "none"

    def _build_enriched_text(self, http_op: str, capabilities: dict,
                              parameters: dict, response_schemas: dict,
                              request_schemas: dict) -> str:
        parts = [capabilities.get(http_op, "")]
        param_str = parameters.get(http_op, "") or ""
        if param_str:
            parts.append(f"| parameters: {param_str}")
        resp_str = response_schemas.get(http_op, "") or ""
        if resp_str:
            fields = re.findall(r'(\w+):', resp_str)
            if fields:
                parts.append(f"| response fields: {', '.join(fields)}")
        req_str = request_schemas.get(http_op, "") or ""
        if req_str and not http_op.startswith("GET"):
            fields = re.findall(r'(\w+):', req_str)
            if fields:
                parts.append(f"| request fields: {', '.join(fields)}")
        return " ".join(parts)

    def decompose_query(self, query_text: str) -> tuple[list, dict]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": query_text},
            ],
            "format": DECOMPOSITION_SCHEMA,
            "options": {"temperature": 0.0, "num_predict": 200},
            "stream": False,
        }
        t0 = time.perf_counter()
        try:
            result = self._ollama_request(payload, timeout=15)
            content = result["message"]["content"].strip()
            parsed = json.loads(content)
            sub_queries = parsed.get("sub_queries", [])
            seen, clean = set(), []
            for sq in sub_queries:
                sq = (sq or "").strip()
                if not sq or sq.lower() in seen:
                    continue
                seen.add(sq.lower())
                clean.append(sq)
            clean = clean[:4]
            meta = {"source": "llm" if clean else "fallback",
                    "latency_ms": int((time.perf_counter() - t0) * 1000)}
            return clean or [query_text], meta
        except Exception as e:
            return [query_text], {"source": "fallback",
                                  "latency_ms": int((time.perf_counter() - t0) * 1000),
                                  "reason": type(e).__name__}

    def _build_faiss_index(self):
        if self._index_built:
            return
        import faiss
        docs = self.storage.list_all()
        self._index_payloads = []
        vectors = []
        for doc in docs:
            text, source = self._select_stage1_text(doc)
            if not text:
                continue
            emb = self.embed_passage(text)
            if emb:
                vectors.append(np.array(emb, dtype=np.float32))
                self._index_payloads.append({"mongo_id": doc["_id"]})
        if vectors:
            dim = len(vectors[0])
            self._embedding_dim = dim
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(np.array(vectors))
        else:
            dim = self._embedding_dim
            self._index = faiss.IndexFlatIP(dim)
        self._index_built = True

    def _reindex(self):
        self._index_built = False
        self._build_faiss_index()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.storage:
            return []
        self._build_faiss_index()
        all_services = self.storage.list_all()
        if len(all_services) <= top_k:
            return all_services
        sub_queries, decomp_meta = self.decompose_query(query)
        stage1_scores: dict[str, float] = {}
        for sq in sub_queries:
            sq_emb = self.embed_query(sq)
            sq_vec = np.array([sq_emb], dtype=np.float32)
            if self._index.ntotal == 0:
                continue
            distances, indices = self._index.search(sq_vec, min(top_k, self._index.ntotal))
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self._index_payloads):
                    sid = self._index_payloads[idx]["mongo_id"]
                    score = float(dist)
                    if sid not in stage1_scores or score > stage1_scores[sid]:
                        stage1_scores[sid] = score
        if not stage1_scores:
            return all_services[:top_k]
        if self.registry:
            reg_ids = self.registry.get_registry_ids()
            stage1_scores = {k: v for k, v in stage1_scores.items() if k in reg_ids}
        services_data = {}
        for doc_id in stage1_scores:
            doc = self.storage.get(doc_id)
            if not doc:
                continue
            capabilities = doc.get("capabilities", {}) or {}
            ops = [op for op in capabilities.keys()
                   if op != "POST /register" and not op.endswith("/health")]
            if not ops:
                continue
            services_data[doc_id] = {
                "s1_score": stage1_scores[doc_id],
                **doc,
                "ops": ops,
            }
        if not services_data:
            return all_services[:top_k]
        all_pairs = []
        pair_map = []
        for doc_id, sdata in services_data.items():
            for op in sdata["ops"]:
                enriched = self._build_enriched_text(
                    op, sdata["capabilities"], sdata["parameters"],
                    sdata["response_schemas"], sdata["request_schemas"],
                )
                all_pairs.append((query, enriched))
                pair_map.append((doc_id, op))
        scores_by_service: dict = {doc_id: {} for doc_id in services_data}
        llm_scores = self._llm_rerank(all_pairs) if len(all_pairs) <= 20 else None
        if llm_scores:
            for (doc_id, op), score in zip(pair_map, llm_scores):
                scores_by_service[doc_id][op] = score
        TOP_ENDPOINTS = 4
        K_RRF = 60
        merged = {}
        for doc_id, sdata in services_data.items():
            ops = sdata["ops"]
            scored_ops = sorted(
                [(op, scores_by_service[doc_id].get(op, 0.0)) for op in ops],
                key=lambda x: x[1], reverse=True
            )
            relevant_ops = scored_ops[:TOP_ENDPOINTS]
            best_ep_score = max(scores_by_service[doc_id].values()) if scores_by_service[doc_id] else 0.0
            merged[doc_id] = {
                "_id": doc_id,
                "name": sdata.get("name"),
                "description": sdata.get("description"),
                "capabilities": {op: sdata["capabilities"][op] for op, _ in relevant_ops},
                "endpoints": {op: sdata["endpoints"].get(op) for op, _ in relevant_ops},
                "response_schemas": {op: sdata["response_schemas"].get(op) for op, _ in relevant_ops},
                "request_schemas": {op: sdata["request_schemas"].get(op) for op, _ in relevant_ops},
                "parameters": {op: sdata["parameters"].get(op) for op, _ in relevant_ops},
                "_stage1_score": sdata["s1_score"],
                "_best_ep_score": best_ep_score,
            }
        if not merged:
            return all_services[:top_k]
        services_list = list(merged.values())
        ranked_by_s1 = sorted(services_list, key=lambda x: x["_stage1_score"], reverse=True)
        ranked_by_ep = sorted(services_list, key=lambda x: x["_best_ep_score"], reverse=True)
        s1_rank = {s["_id"]: i for i, s in enumerate(ranked_by_s1, start=1)}
        ep_rank = {s["_id"]: i for i, s in enumerate(ranked_by_ep, start=1)}
        for s in services_list:
            s["_rrf_score"] = (1.0 / (K_RRF + s1_rank[s["_id"]])
                               + 1.0 / (K_RRF + ep_rank[s["_id"]]))
        ordered = sorted(services_list, key=lambda x: x["_rrf_score"], reverse=True)
        for s in ordered:
            s.pop("_stage1_score", None)
            s.pop("_best_ep_score", None)
            s.pop("_rrf_score", None)
        max_tokens = 5000
        current_tokens = 0
        top_results = []
        DICT_FIELDS = ("capabilities", "endpoints", "response_schemas",
                       "request_schemas", "parameters")

        def trim(svc, keep_ops):
            trimmed = {}
            for k, v in svc.items():
                if k in DICT_FIELDS and isinstance(v, dict):
                    trimmed[k] = {op: v[op] for op in keep_ops if op in v}
                else:
                    trimmed[k] = v
            return trimmed

        for s in ordered:
            ops = list(s.get("capabilities", {}).keys())
            inserted = False
            for n in range(len(ops), 0, -1):
                candidate = trim(s, ops[:n])
                tok = self._count_tokens(json.dumps(candidate))
                if current_tokens + tok <= max_tokens:
                    top_results.append(candidate)
                    current_tokens += tok
                    inserted = True
                    break
        return top_results

    def _llm_rerank(self, pairs: list[tuple]) -> Optional[list[float]]:
        try:
            scores = []
            for query, text in pairs:
                prompt = f"""On a scale of 0.0 to 1.0, how relevant is this service capability to answering this query? Respond with ONLY a number.

Query: {query}

Service capability: {text}

Relevance score:"""
                resp = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "options": {"temperature": 0.0, "num_ctx": 2048, "num_predict": 10},
                        "stream": False,
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                content = resp.json()["message"]["content"].strip()
                match = re.search(r'(\d+\.?\d*)', content)
                score = float(match.group(1)) if match else 0.5
                scores.append(min(max(score, 0.0), 1.0))
            return scores
        except Exception:
            return None
