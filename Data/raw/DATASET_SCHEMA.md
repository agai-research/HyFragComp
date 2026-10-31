# HyFragComp B-READY Business Entry Dataset — Description & Schema

## 1. Purpose

This dataset is a reproducible experimental package built around the **World Bank B-READY 2024 Business Entry** domain. It is structured for research on service/workflow composition, fragment reuse, hypergraph retrieval, GraphRAG, LLM-based composition, GNN-based methods, and related approaches.

The package separates:

1. **Source-oriented data** — B-READY Business Entry observations and taxonomy.
2. **Derived data** — services, resources, requirements, constraints, fragments, and hypergraph relations.
3. **Synthetic experimental context** — controlled contextual information used to construct experiments.
4. **Evaluation instances** — requests, relevance labels, composition records, change scenarios, and scale configurations.

> **Provenance:** B-READY does not directly contain HyFragComp fragments or hypergraphs. These structures are derived experimental representations constructed from the B-READY domain.

---

## 2. Dataset scope

| Entity | Count |
|---|---:|
| Economies | 50 |
| Services | 18 |
| Resources | 9 |
| Requirements | 7 |
| Constraints | 5 |
| Fragments | 16 |
| Hyperedges | 17 |
| Controlled requests | 250 |
| Repository scales | 10, 20, 30, 40, 50 economies |

The dataset currently focuses on **Business Entry** and is designed so that additional B-READY topics can be integrated without changing the core schema.

---

## 3. Provenance model

```text
B-READY 2024 Business Entry
        |
        v
Source-oriented data
        |
        v
Domain transformation
        |
        +--> Economies
        +--> Services
        +--> Resources
        +--> Requirements
        +--> Constraints
        |
        v
Derived workflow representation
        |
        +--> Fragments
        +--> Hypergraph
        +--> Composition records
        |
        v
Experimental instances
        |
        +--> Requests
        +--> Retrieval relevance
        +--> Change scenarios
        +--> Scale-specific instances
```

This separation allows another approach to reuse the domain and evaluation data without adopting the HyFragComp implementation.

---

## 4. Directory structure

```text
HyFragComp_BREADY_BE_Dataset/
├── README.md
├── raw/
│   ├── business_entry_pillar_scores_2024.csv
│   ├── business_entry_scores_long.csv
│   ├── business_entry_taxonomy.json
│   └── economies.json
├── derived/
│   ├── service_catalog.json
│   ├── fragment_catalog.json
│   └── hypergraph.json
├── synthetic/
│   └── economy_service_context.json
├── instances/
│   ├── requests.json
│   ├── experiment_instances.json
│   └── evaluation_protocol.json
├── final_json/
│   ├── dataset.json
│   ├── economies.json
│   ├── services.json
│   ├── resources.json
│   ├── requirements.json
│   ├── constraints.json
│   ├── fragments.json
│   ├── hypergraph.json
│   ├── requests.json
│   ├── composition_records.json
│   ├── change_scenarios.json
│   ├── retrieval_relevance.json
│   └── service_context.json
├── final_csv/
│   ├── economies.csv
│   ├── services.csv
│   ├── resources.csv
│   ├── requirements.csv
│   ├── constraints.csv
│   ├── fragments.csv
│   ├── hyperedges.csv
│   ├── requests.csv
│   ├── composition_records.csv
│   ├── change_scenarios.csv
│   ├── retrieval_relevance.csv
│   ├── service_context.csv
│   ├── scale_10_economies.csv
│   ├── scale_10_requests.csv
│   ├── scale_20_economies.csv
│   ├── scale_20_requests.csv
│   ├── scale_30_economies.csv
│   ├── scale_30_requests.csv
│   ├── scale_40_economies.csv
│   ├── scale_40_requests.csv
│   ├── scale_50_economies.csv
│   └── scale_50_requests.csv
├── experiment_sets/
│   ├── scale_10.json
│   ├── scale_20.json
│   ├── scale_30.json
│   ├── scale_40.json
│   └── scale_50.json
└── docs/
    ├── validation_report.json
    ├── source_manifest.json
    └── checksums.json
```

---

# 5. Canonical JSON schema

The canonical reusable representation is under `final_json/`.

## 5.1 Economies

**File:** `final_json/economies.json`

Represents the geographical/business context.

```json
{
  "economy_id": "...",
  "name": "...",
  "code": "...",
  "region": "...",
  "year": 2024
}
```

## 5.2 Services

**File:** `final_json/services.json`

Represents service-level abstractions derived from the Business Entry domain.

```json
{
  "service_id": "...",
  "name": "...",
  "description": "...",
  "domain": "business_entry",
  "economy_scope": ["..."],
  "resource_ids": ["..."],
  "requirement_ids": ["..."],
  "constraint_ids": ["..."]
}
```

## 5.3 Resources

**File:** `final_json/resources.json`

```json
{
  "resource_id": "...",
  "name": "...",
  "type": "...",
  "description": "...",
  "service_ids": ["..."]
}
```

## 5.4 Requirements

**File:** `final_json/requirements.json`

```json
{
  "requirement_id": "...",
  "name": "...",
  "description": "...",
  "service_ids": ["..."]
}
```

## 5.5 Constraints

**File:** `final_json/constraints.json`

```json
{
  "constraint_id": "...",
  "name": "...",
  "description": "...",
  "scope": "...",
  "service_ids": ["..."]
}
```

---

# 6. Fragment schema

**File:** `final_json/fragments.json`

Fragments are **derived reusable workflow units**, not native B-READY entities.

```json
{
  "fragment_id": "...",
  "name": "...",
  "type": "atomic | composite",
  "description": "...",
  "service_ids": ["..."],
  "resource_ids": ["..."],
  "requirement_ids": ["..."],
  "constraint_ids": ["..."],
  "valid_from": "...",
  "valid_to": "...",
  "revision": 1,
  "scope": "...",
  "signature": "..."
}
```

---

# 7. Hypergraph schema

**File:** `final_json/hypergraph.json`

The hypergraph is a **derived representation** connecting domain entities and reusable fragments.

```json
{
  "nodes": [
    {
      "node_id": "...",
      "node_type": "service | resource | requirement | constraint | fragment"
    }
  ],
  "hyperedges": [
    {
      "hyperedge_id": "...",
      "type": "seq | and | xor | loop | der",
      "source": ["..."],
      "target": ["..."],
      "attributes": {}
    }
  ]
}
```

Supported relation types:

- `seq` — sequential relation
- `and` — conjunctive relation
- `xor` — alternative relation
- `loop` — iterative relation
- `der` — derivation relation

---

# 8. Request schema

**File:** `final_json/requests.json`

```json
{
  "request_id": "...",
  "economy_id": "...",
  "prompt": "...",
  "goals": ["..."],
  "requirements": ["..."],
  "constraints": ["..."],
  "expected_fragment_ids": ["..."]
}
```

The current package contains **250 controlled requests**.

---

# 9. Composition records

**File:** `final_json/composition_records.json`

```json
{
  "composition_id": "...",
  "request_id": "...",
  "fragment_ids": ["..."],
  "workflow": {},
  "valid": true,
  "sound": true,
  "reuse_ratio": 0.0,
  "delta_ratio": 0.0
}
```

---

# 10. Change scenarios

**File:** `final_json/change_scenarios.json`

```json
{
  "scenario_id": "...",
  "target_type": "service | resource | fragment",
  "target_id": "...",
  "change_type": "...",
  "description": "...",
  "affected_ids": ["..."]
}
```

These scenarios support invalidation, dependency propagation, substitution, repair, and delta recomposition.

---

# 11. Retrieval relevance

**File:** `final_json/retrieval_relevance.json`

```json
{
  "request_id": "...",
  "relevant_fragment_ids": ["..."],
  "relevance_grade": {}
}
```

Configured metrics include:

- Recall@K
- MRR
- nDCG@K

The relevance file is independent of the retrieval algorithm.

---

# 12. Service context

**File:** `final_json/service_context.json`

Contains contextual associations between economies and services and is useful for constructing controlled repository instances.

---

# 13. Scale-specific datasets

Five repository sizes are provided:

| Experiment set | Economies |
|---|---:|
| `scale_10` | 10 |
| `scale_20` | 20 |
| `scale_30` | 30 |
| `scale_40` | 40 |
| `scale_50` | 50 |

Each scale provides economy/request files and a JSON experiment definition.

---

# 14. CSV representation

`final_csv/` provides tabular equivalents:

| File | Role |
|---|---|
| `economies.csv` | Economy metadata |
| `services.csv` | Service catalog |
| `resources.csv` | Resource catalog |
| `requirements.csv` | Requirements |
| `constraints.csv` | Constraints |
| `fragments.csv` | Fragment catalog |
| `hyperedges.csv` | Hypergraph relations |
| `requests.csv` | Composition requests |
| `composition_records.csv` | Previous compositions |
| `change_scenarios.csv` | Change cases |
| `retrieval_relevance.csv` | Retrieval ground truth |
| `service_context.csv` | Economy/service context |

JSON preserves nested relations; CSV is convenient for pandas, statistical analysis, ML, and visualization.

---

# 15. Evaluation protocol

The current experimental configuration uses **five runs per condition**.

### Retrieval

- Recall@K
- MRR
- nDCG@K

### Composition quality

- coverage
- validity
- workflow soundness
- reuse ratio
- delta recomposition ratio

### Efficiency

- retrieval latency
- composition latency
- verification latency

The dataset contains inputs, instances, ground truth, and evaluation definitions—not fabricated performance results.

---

# 16. Reuse by other approaches

### Retrieval-only

```text
requests.json
fragments.json
hypergraph.json
retrieval_relevance.json
```

### Workflow composition

Additionally:

```text
services.json
resources.json
requirements.json
constraints.json
composition_records.json
```

### Change management

Additionally:

```text
change_scenarios.json
```

### Scalability

```text
experiment_sets/scale_10.json
experiment_sets/scale_20.json
experiment_sets/scale_30.json
experiment_sets/scale_40.json
experiment_sets/scale_50.json
```

---

# 17. Generic interface

```text
repository = load_dataset()

for request in requests:
    result = approach(
        request=request,
        repository=repository
    )

    metrics = evaluate(
        result=result,
        ground_truth=retrieval_relevance
    )
```

This allows replacement of HyFragComp with vector RAG, GraphRAG, GNN retrieval, hypergraph neural networks, classical IR, case-based reasoning, symbolic composition, LLM agents, or hybrid neuro-symbolic systems.

---

# 18. Reproducibility requirements

A study should report:

1. Dataset version.
2. Repository scale.
3. Request set.
4. Random seed(s).
5. Model/embedding configuration.
6. Retrieval configuration.
7. LLM configuration, if applicable.
8. Number of runs.
9. Evaluation metrics.
10. Any transformation applied to the canonical dataset.

The canonical files should remain unchanged; new representations should be stored as derived data.

---

# 19. Validation and integrity

The `docs/` directory contains:

- `validation_report.json`
- `source_manifest.json`
- `checksums.json`

These support structural validation, provenance tracking, and integrity checking.

---

# 20. Extension to other B-READY topics

The current release focuses on **Business Entry**. The schema can be extended to:

```text
Business Entry
Business Location
Utility Services
Labor
Financial Services
International Trade
Taxation
Dispute Resolution
Market Competition
Business Insolvency
```

A multi-topic extension should preserve the common entity vocabulary and add a topic attribute.

---

# 21. Recommended provenance statement

> We use the World Bank B-READY 2024 Business Entry domain as the source domain and construct a service-oriented experimental representation comprising services, resources, requirements, constraints, reusable fragments, and hypergraph relations. These latter structures are derived for experimental purposes and are not native B-READY entities.

---

# 22. Versioning

Recommended format:

```text
HyFragComp_BREADY_BE_vMAJOR.MINOR
```

Increment the version when source data, derived mappings, fragments, hypergraph relations, request sets, or evaluation ground truth change.

---

# 23. Quick-start loading

### JSON

```python
import json
from pathlib import Path

ROOT = Path("HyFragComp_BREADY_BE_Dataset/final_json")

with open(ROOT / "services.json", encoding="utf-8") as f:
    services = json.load(f)

with open(ROOT / "fragments.json", encoding="utf-8") as f:
    fragments = json.load(f)

with open(ROOT / "hypergraph.json", encoding="utf-8") as f:
    hypergraph = json.load(f)

with open(ROOT / "requests.json", encoding="utf-8") as f:
    requests = json.load(f)
```

### CSV

```python
import pandas as pd
from pathlib import Path

ROOT = Path("HyFragComp_BREADY_BE_Dataset/final_csv")

services = pd.read_csv(ROOT / "services.csv")
fragments = pd.read_csv(ROOT / "fragments.csv")
hyperedges = pd.read_csv(ROOT / "hyperedges.csv")
requests = pd.read_csv(ROOT / "requests.csv")
```

---

# 24. Design principle

> **Keep source observations, derived workflow structures, and experimental instances separate.**

This makes the dataset reusable by fundamentally different approaches while preserving provenance and experimental comparability.

---

# 25. Important limitation

This package is an **experimental dataset built around the B-READY Business Entry domain**, not a verbatim redistribution of the complete World Bank B-READY raw-data package.

Services, resources, requirements, constraints, fragments, hyperedges, controlled requests, composition records, and change scenarios are derived or experimental representations. For strict source-level reproducibility, the exact B-READY source files, version, processing scripts, and transformation rules should be archived alongside this dataset.
