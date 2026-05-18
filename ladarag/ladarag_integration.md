# LADARAG Integration Plan: In-Process Backend Swaps

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Jupyter Docker Container                        │
│                                                                    │
│  ┌─────────────────────┐  ┌─────────────────────────────────────┐ │
│  │   Survey Pipeline    │  │  LADARAG                             │ │
│  │   (gp_survey_engine) │  │                                       │ │
│  │                      │  │  ┌───────────────────────────────┐  │ │
│  │   For each agent,    │  │  │  core/                         │  │ │
│  │   for each question: │  │  │  ┌───────────────────────────┐│  │ │
│  │                      │  │  │  │ controlService.py         ││  │ │
│  │   1. Is it OD?       │──┼──┼─►│  - Controller (planner)   ││  │ │
│  │      (classifier)    │  │  │  │  - PlanValidator          ││  │ │
│  │   2. If yes, call    │  │  │  │  - JMESPath resolver      ││  │ │
│  │      client.query()  │  │  │  │  - Schema enforcement     ││  │ │
│  │   3. Inject OD data  │  │  │  └───────────────────────────┘│  │ │
│  │      into prompt     │  │  │  ┌───────────────────────────┐│  │ │
│  │   4. Agent answers   │  │  │  │ designerService.py        ││  │ │
│  │                      │  │  │  │  - Empty plan triage      ││  │ │
│  └─────────────────────┘  │  │  │  - API contract design     ││  │ │
│                            │  │  └───────────────────────────┘│  │ │
│  ┌─────────────────────┐  │  ├── db/                          │  │ │
│  │   Ollama             │  │  │  ┌───────────────────────────┐│  │ │
│  │   (phi4-reasoning)   │◄─┼──┼──│ gateway.py                ││  │ │
│  │                      │  │  │  │  - Query decomposition    ││  │ │
│  │   Embeddings via     │◄─┼──┼──│  - Stage 1 (FAISS +       ││  │ │
│  │   /api/embeddings    │  │  │  │      Ollama embeddings)   ││  │ │
│  └─────────────────────┘  │  │  │  - RRF ranking             ││  │ │
│                            │  │  │  - Token budget trimming   ││  │ │
│  ┌─────────────────────┐  │  │  │  - Endpoint selection      ││  │ │
│   Valhalla Routing      │  │  │  └───────────────────────────┘│  │ │
│   ├── port 8002         │◄─┼──┼──┐                            │  │
│   │   (standard)        │  │  │  │  registry.py               │  │ │
│   ├── port 8003         │◄─┼──┼──┤  (in-process service       │  │ │
│   │   (roadblocks)      │  │  │  │   registry, replaces       │  │ │
│   └─────────────────────┘  │  │  │   Consul)                  │  │ │
│                            │  │  ├───────────────────────────┤  │ │
│                            │  │  │  storage.py                │  │ │
│                            │  │  │  (SQLite service catalog,   │  │ │
│                            │  │  │   replaces MongoDB)         │  │ │
│                            │  │  └───────────────────────────┘  │ │
│                            │  ├── client.py                     │ │
│                            │  │  (called by survey pipeline)     │ │
│                            │  ├── config.py                     │ │
│                            │  │  (LADARAG configuration)        │ │
│                            │  └── README.md                     │ │
│                            └─────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## File Structure

```
synth-survey-gen/
├── ladarag/                          ← LADARAG package
│   ├── __init__.py                   ← Package exports
│   ├── README.md                     ← This document
│   ├── config.py                     ← LADARAG configuration
│   ├── client.py                     ← Survey pipeline client API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── controlService.py         ← From LADARAG-GP (adapted)
│   │   └── designerService.py        ← From LADARAG-GP (as-is)
│   └── db/
│       ├── __init__.py
│       ├── gateway.py                ← From LADARAG-GP (adapted)
│       ├── registry.py               ← New: in-process service registry
│       └── storage.py                ← New: SQLite service catalog
├── gp_survey_engine.py               ← Modified: calls LADARAG client
├── main.py                           ← Modified: initializes LADARAG
└── configs/Generic/config.json       ← Modified: LADARAG config section
```

## Component Details

### 1. `core/controlService.py` — LADARAG's Orchestrator (adapted)

**Origin:** `LADARAG-GP/control-unit/service/controlService.py`
**Lines:** ~1665

**Kept intact:**
- `PlanValidator` class (validates execution plans)
- `Controller.query_ollama()` (Ollama API call with guided decoding)
- `Controller._build_system_prompt()` (planning system prompt with examples A-F)
- `Controller._build_user_prompt()` (formats discovered services + query)
- `Controller.decompose_task()` (orchestrates plan generation)
- `Controller._empty_plan_detected()` (detects empty plans)
- `Controller.extract_agents()` (JSON parser with fallbacks)
- `Controller._resolve_expression()` (JMESPath resolution)
- `Controller.resolve_placeholders()` (placeholder substitution in URLs/input)
- `Controller._build_input_format_schema()` (request schema aggregation)
- `Controller._validate_plan()` (post-parse hallucination checking)

**Changes:**
- Remove `from service.discoveryService import Discovery`
- Remove `from service.designerService import Designer` (import locally)
- Remove `self.backend_mode = "MOCK"` (always REAL mode)
- Remove HC-12 section from system prompt (Microcks-specific)
- Add in-process `control()` method that accepts services + query directly
- Replace Flask request handling with a method that accepts parameters

### 2. `core/designerService.py` — Fallback Triage (as-is)

**Origin:** `LADARAG-GP/control-unit/service/designerService.py`
**Lines:** ~275

**No changes needed.** Uses only Ollama for LLM calls. Handles:
- Classification of empty plans (OUT_OF_DOMAIN, AMBIGUOUS, INVALID, UNKNOWN)
- API contract design for missing services (when OUT_OF_DOMAIN)

### 3. `db/gateway.py` — Retrieval Pipeline (adapted)

**Origin:** `LADARAG-GP/db-gateway/db-gateway.py`
**Lines:** ~888

**Kept intact:**
- Query decomposition with LLM (`decompose_query()`)
- Two-stage search logic (Stage 0 → Stage 1 → Stage 2)
- RRF ranking algorithm (`1/(K+rank)` with K=60)
- Token budget trimming with graceful endpoint scaling
- `select_stage1_text()` for service description selection
- `_build_enriched_text()` for CrossEncoder input construction
- `_trim_service()` for endpoint pruning
- `embed_query()` / `embed_passage()` interface (backed by Ollama now)
- `count_tokens()` using tokenizer

**Replaced:**
- `MongoClient` + pymongo → `storage.py` (SQLite)
- `QdrantClient` → FAISS in-memory index (via `faiss-cpu`)
- `SentenceTransformer` + `ONNXCrossEncoder` → Ollama `/api/embeddings`

#### MongoDB → SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS services (
    id TEXT PRIMARY KEY,                          -- e.g. "valhalla-routing"
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    base_url TEXT NOT NULL DEFAULT '',
    capabilities TEXT NOT NULL DEFAULT '{}',       -- JSON: {"GET /route": "..."}
    endpoints TEXT NOT NULL DEFAULT '{}',          -- JSON: {"GET /route": "http://..."}
    parameters TEXT NOT NULL DEFAULT '{}',         -- JSON: {"GET /route": "locations(str*), ..."}
    response_schemas TEXT NOT NULL DEFAULT '{}',   -- JSON: {"GET /route": "{distance:float, ...}"}
    request_schemas TEXT NOT NULL DEFAULT '{}',    -- JSON: {"GET /route": "{...}"}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_services_name ON services(name);
```

#### Qdrant → FAISS Mapping

| Qdrant | FAISS |
|--------|-------|
| `QdrantClient(host, port)` | `faiss.IndexFlatIP(dim)` |
| `create_collection(name, VectorParams)` | `index = faiss.IndexFlatIP(1024)` |
| `upsert(points=[PointStruct(id, vector, payload)])` | `index.add(vectors)` + `payload_list.append(payload)` |
| `query_points(collection, query, limit)` | `index.search(query_vector, k=limit)` |
| Search by cosine distance | Inner product on normalized vectors (equivalent) |

The FAISS index is rebuilt from SQLite on startup. Since there are only 2-3 services (and ~2-8 endpoints), the index is trivially small.

#### Embedding Model Swap

| Original | Replacement |
|----------|-------------|
| `SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')` → CPU, 600M params | `POST /api/embeddings` with existing Ollama model (phi4-reasoning) |
| `embed_query(text)` → `"query: " + text` → Qwen3 asymmetric prefix | `embed_query(text)` → Ollama embedding of `"query: " + text` |
| `embed_passage(text)` → `"passage: " + text` → Qwen3 asymmetric prefix | `embed_passage(text)` → Ollama embedding of `"passage: " + text` |
| `ONNXCrossEncoder('BAAI/bge-reranker-base')` → ONNX, ~1GB RAM | **Removed** (use LLM-based endpoint selection for small catalogs) |

Stage 2 reranking (the CrossEncoder) handles endpoint-level scoring. With 2-3 services and 2-4 endpoints each, the number of pairs is small enough that we can skip dedicated reranking and let the planner's LLM handle selection directly.

### 4. `db/registry.py` — In-Process Service Registry (new)

Replaces Consul's `GET /v1/agent/services`. Tracks which services are "registered and healthy."

```python
class ServiceRegistry:
    def __init__(self):
        self._services: dict[str, ServiceInfo] = {}

    def register(self, service_id, ...): ...
    def deregister(self, service_id): ...
    def list_services(self) -> list[dict]: ...
    def is_registered(self, service_id) -> bool: ...
```

### 5. `db/storage.py` — SQLite Service Catalog (new)

Replaces MongoDB CRUD. Same operations: `get`, `list`, `upsert`, `delete`, `update_schemas`.

```python
class ServiceStorage:
    def __init__(self, db_path: str = ":memory:"):
        self._init_db()

    def get(self, service_id) -> dict | None: ...
    def list_all(self) -> list[dict]: ...
    def upsert(self, service_doc: dict): ...
    def delete(self, service_id): ...
    def update_schemas(self, service_id, schemas: dict): ...
```

### 6. `client.py` — Survey Pipeline Client

```python
from ladarag.config import LADARAGConfig
from ladarag.core.controlService import Controller
from ladarag.db.gateway import ServiceGateway
from ladarag.db.registry import ServiceRegistry
from ladarag.db.storage import ServiceStorage
from ladarag.classifier import ODClassifier  # from previous implementation

class LADARAG:
    def __init__(self, config: LADARAGConfig):
        self.storage = ServiceStorage()
        self.registry = ServiceRegistry()
        self.gateway = ServiceGateway(storage, registry, ollama_url, model)
        self.controller = Controller(ollama_url, model)
        self.classifier = ODClassifier(ollama_url, model)

    def query(self, question: str, agent_bio: str = "") -> dict | None:
        # 1. Discover relevant services
        discovered = self.gateway.search(question)

        # 2. Filter by registry
        registered = [s for s in discovered if self.registry.is_registered(s["_id"])]

        # 3. Build plan (uses controlService.py's Controller)
        plan = self.controller.control(
            query=question,
            discovered_services=registered,
            agent_context=agent_bio,
            input_files=None,
        )

        # 4. Execute plan
        results = self.controller._execute_plan(plan)

        return {"plan": plan, "results": results}
```

### 7. Dependencies

**New (add to requirements.txt):**
```
faiss-cpu>=1.9.0      # Vector similarity search (replaces Qdrant)
```

**Existing (already in requirements.txt):**
```
ollama>=0.4.7         # LLM + embeddings
duckdb>=1.0.0         # SQL execution (for SQL task type)
jmespath>=1.1.0       # JMESPath expression resolution
requests              # HTTP client
pandas                # Data manipulation
numpy                 # Numerical ops
```

**Removed (no longer needed):**
- `ctransformers` (was used for socio-demographic detection with local GGUF)
- `sentence-transformers` (replaced by Ollama embeddings)
- `optimum[onnxruntime]` (no more ONNX CrossEncoder)

## Configuration

In `configs/Generic/config.json`:

```json
{
    "ladarag": {
        "enabled": false,
        "ollama_url": "http://localhost:11434",
        "model": "phi4-reasoning:14b",
        "timeout": 120,
        "storage": {
            "path": ":memory:"
        },
        "services": {
            "valhalla-routing": {
                "name": "Valhalla Routing Service",
                "description": "REST API for origin-destination routing using the Valhalla routing engine. Provides turn-by-turn directions, distance, duration, and geometry for routes between locations. Supports multiple costing models (auto, bicycle, pedestrian, transit) and can avoid features like tolls, ferries, and highways.",
                "url": "http://localhost:8002",
                "endpoints": {
                    "GET /route": {
                        "path": "/route",
                        "capability": "Calculate the optimal route between an origin and destination. Returns trip distance, duration, legs, and geometry.",
                        "parameters": "locations(string*), costing(string), directions_options(string), avoid_locations(string), date_time(string)",
                        "response_schema": "{trip:obj, legs:arr, distance:float, duration:float, geometry:str, locations:arr}"
                    }
                }
            },
            "valhalla-roadblocks": {
                "name": "Valhalla Roadblock-Aware Routing",
                "description": "REST API for origin-destination routing that accounts for roadblocks, road closures, and construction zones. Returns routes that avoid blocked roads and provides alternative paths.",
                "url": "http://localhost:8003",
                "endpoints": {
                    "GET /route": {
                        "path": "/route",
                        "capability": "Calculate a route between origin and destination that avoids known roadblocks and closures. Returns trip distance, duration, legs, geometry, and the number of roadblocks avoided.",
                        "parameters": "locations(string*), costing(string), exclude_polygons(string), directions_options(string)",
                        "response_schema": "{trip:obj, legs:arr, distance:float, duration:float, geometry:str, roadblocks_avoided:int}"
                    }
                }
            }
        }
    }
}
```

## Integration Flow

### Survey Pipeline (gp_survey_engine.py)

```
AgenticSurveyPipeline.run()
  │
  ├── for each agent:
  │     ├── for each question:
  │     │     ├── [LADARAG] classify_question(text, desc, bio)
  │     │     │               └── LLM: "Is this OD-trip related?"
  │     │     ├── if YES:
  │     │     │     ├── [LADARAG] query(text, bio)
  │     │     │     │               ├── gateway.search(query)
  │     │     │     │               │     ├── decompose_query(query)
  │     │     │     │               │     ├── FAISS: Stage 1 (descriptions)
  │     │     │     │               │     ├── RRF ranking
  │     │     │     │               │     ├── Token budget trim
  │     │     │     │               │     └── return top services
  │     │     │     │               ├── registry.filter(services)
  │     │     │     │               ├── controller.plan(query, services, bio)
  │     │     │     │               │     └── Self-consistent prompt + guided decoding
  │     │     │     │               ├── controller.execute(plan)
  │     │     │     │               │     ├── HTTP calls to Valhalla
  │     │     │     │               │     ├── JMESPath chaining
  │     │     │     │               │     ├── DuckDB SQL tasks
  │     │     │     │               │     └── return results
  │     │     │     │               └── return OD data
  │     │     │     └── [LADARAG] format_od_context(result)
  │     │     │                   └── "### RELEVANT TRIP DATA: ..."
  │     │     ├── agent.llm_response(prompt + od_context)
  │     │     └── store response
  │     └── save AgenticResponsePackage
  └── return results
```

### Controller.control() In-Process Flow

```
control(query, discovered_services, agent_context)
  │
  ├── decompose_task(query, services, agent_context)
  │     ├── _build_system_prompt(services)   ← same as LADARAG
  │     ├── _build_user_prompt(query, services) ← same as LADARAG
  │     ├── _build_input_format_schema(services) ← same as LADARAG
  │     ├── query_ollama(system_prompt, user_prompt, schema) ← same as LADARAG
  │     │     └── POST /api/chat with format=JSON schema
  │     ├── extract_agents(raw_response)    ← same as LADARAG
  │     │     └── JSON parser with </think> fallback
  │     ├── _validate_plan(plan, services)  ← same as LADARAG
  │     │     └── schema validation + param validation
  │     └── return plan
  │
  ├── [if empty plan] designerService.analyze(query, ...) ← same as LADARAG
  │     └── triage category + optional API contract
  │
  ├── execute_plan(plan)
  │     ├── for each task:
  │     │     ├── resolve placeholders ({{task.field}}) ← same as LADARAG
  │     │     │     └── JMESPath expressions
  │     │     ├── if SQL: duckdb.execute() ← same as LADARAG
  │     │     └── if HTTP: requests.get/post/put/delete ← same as LADARAG
  │     └── return results
  │
  └── return {plan, results}
```

## Implementation Order

1. **`ladarag/db/storage.py`** — SQLite service catalog (no dependencies, pure Python)
2. **`ladarag/db/registry.py`** — In-process service registry (no dependencies)
3. **`ladarag/db/gateway.py`** — Adapted retrieval (FAISS + Ollama, keep original logic)
4. **`ladarag/core/controlService.py`** — Adapted Controller (remove Consul, add in-process control())
5. **`ladarag/core/designerService.py`** — As-is copy
6. **`ladarag/config.py`** — Configuration wrapper
7. **`ladarag/client.py`** — Survey pipeline client API
8. **`ladarag/__init__.py`** — Package exports
9. **Modify `config.json`** — Add LADARAG section
10. **Modify `gp_survey_engine.py`** — Call LADARAG client
11. **Modify `main.py`** — Initialize LADARAG
12. **Update `requirements.txt`** — Add faiss-cpu
13. **`ladarag/README.md`** — Comprehensive documentation
