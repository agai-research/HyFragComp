# HyFragComp

HyFragComp is a reuse layer for LLM-composed service workflows over a hypergraph of process fragments, built for smart governance (public-service procedures such as residence permits, company registration or licences). This repository accompanies the paper *Evolving Reuse of LLM-Composed Service Workflows for Smart Governance* (IEEE Transactions on Services Computing).

A request is abstracted into a template and decomposed into typed sub-goals. Stored composition records are aligned with these sub-goals (hybrid textual and AllSet-based structural matching) and re-validated against the current repository and context (checks V1-V5, with variant switching). Only the difference is recomposed, by an anchored beam search over the hypergraph (GraphRAG). Recurring regions are promoted to composite fragments, and repository or regulatory changes invalidate only the records and composites that depend on them.

## Environment

- Python 3.12 on CPU, no GPU. Install the pinned versions with `pip install -r requirements.txt`, or use the `Dockerfile`. Details in `environment.md`.
- Everything runs on NumPy, SciPy and scikit-learn: the autograd engine, the AllSet encoder, the HNSW index and the sentence encoder are implemented in this repository.
- Results are bit-identical across runs only with `PYTHONHASHSEED=0`. The entry points (`HyFragComp.py`, `main-exp.py`, `Exp/calibrate.py`, `reproduce.sh`) set it themselves.

## Language-model backends

The eight LLM roles (`c_cur`, `c_int`, `c_pln`, `c_ret`, `c_sel`, `c_com`, `c_ver`, `c_exp`) share one interface. `llm.backend` in `configs/default.yaml` selects how they are answered:

| Backend | Who answers the roles | LLM latency and tokens | Use |
|---|---|---|---|
| `rule` (default) | deterministic rule-and-embedding engine with the same typed outputs | computed from token counts with `llm.latency` | all experiments |
| `openai` | hosted model `llm.api.model` (GPT-5.6 Luna by default) | measured, logged to `results/logs/llm_calls.jsonl` | deployment, calibration |
| `shadow` | rule engine; every call is also sent to the hosted model | both logged to `results/logs/shadow.jsonl` | calibration |

The hosted backends read the API key from `OPENAI_API_KEY` and cache responses in `results/llm_cache`, so a rerun returns the recorded answers and latencies.

In every backend, the number of LLM calls is the number of role invocations decided by the pipeline, counted per role. The deterministic stages (retrieval, alignment, re-validation, verification) are always measured on the machine (`compute_ms`). With the `rule` backend, reported latencies are labelled *analytic*. `Exp/calibrate.py` compares the backends and fits `llm.latency` on measured calls (see `REPRODUCE.md`).

## Run

```bash
# Pipeline
python HyFragComp.py --selftest                     # calls each of the eight roles once
python HyFragComp.py --prompts --out out/prompts    # serves Prompts/prompts.json; writes JSON, Markdown and BPMN-style figures
python HyFragComp.py --text "..." --jur MAR --renew --priority   # one request (--method ours|whole|nore|scratch|cbr|cafc)

# Checks
python Test/first_test.py                           # scenario: rejections, variant switch, change events
python -m Test.smoke                                # every method on a few requests
python -m unittest Test.test_units Test.test_nn Test.test_cycles Test.test_llm_backend
python Test/static_check.py                         # static scan

# Data
python hypergraph.py --rebuild                      # rebuilds Data/main from the B-READY seed (byte-identical)

# Experiments
bash reproduce.sh smoke                             # tests and every series with small workloads (about 5 min)
bash reproduce.sh full                              # protocol of the paper, 5 seeds (several hours)
python main-exp.py --exp e1 e5 e6 e7 x1 x2 x3 --profile full --jobs 2
python main-exp.py --exp e1 --only figures          # re-draw from stored results

# Calibration against the hosted model (needs OPENAI_API_KEY)
python Exp/calibrate.py run && python Exp/calibrate.py analyze
```

Series: `e1` (Fig. 4, re-validation under staleness), `e5` (Fig. 5, repository growth), `e7` (Fig. 6, scalability), `e6` (Fig. 7, latency by stage), and `x1`-`x3` (matching quality, memory quality, privacy and explainability). `REPRODUCE.md` gives the protocol, run time and output files of each series. `DATA.md` describes the datasets and the workload generators.

## Configuration

Every parameter is in `configs/default.yaml`:

- matching: `theta_m`, `rho_min`, `K_a`, `lambdas`;
- memory: `k`, `k_u`, `R_max`, `W_s`, `tau_e`, `u_eager_pct`, sketch size and cost units;
- recomposition: `h`, `B_w`, `eps_max`, `K`, `theta_sim`, `eta`;
- encoder: `alpha`, `L`, dimensions and training;
- index: `M`, `ef_construction`, `ef_search`;
- planner thresholds;
- LLM backend, latency model and API;
- privacy: `pad_latency`, `pad_ms`, `pad_sleep`, `hide_scope_rejections`;
- baseline settings, stream generators, and the sweep values of each series under `experiments.smoke` and `experiments.full`.

The experiments load, for each method, the locked tuned overrides `configs/tuned_shared.yaml` and `configs/tuned_<method>.yaml` (hashes in `configs/tuned.lock.json`); `main-exp.py --no-tuned` uses the defaults only. `HyFragComp.py --config <file> ...` applies override files placed in `configs/`.

## Reproducibility

- Seeds: models and data `0`; experiment runs `0-4`; tuning `100-101` (disjoint from evaluation).
- Caches: trained encoder checkpoints in `results/ckpt`, embeddings and indexes in `results/cache`.
- Outputs of `main-exp.py`:
  - `results/raw`: per-request records, one file per series, method and seed; runs are resumable;
  - `results/agg`: mean, std, 95% CI and n, plus paired comparisons with bootstrap intervals and Cohen's d;
  - `results/tables` and `results/figs`: tables and figures.
- LLM call logs of the hosted backends are in `results/logs`.

## Repository layout

| Path | Content |
|---|---|
| `hfc/` | the approach: hypergraph, encoder, matching, re-validation, delta recomposition, memory, invalidation, LLM roles and backends |
| `Baselines/` | CBR-KG-Comp and CD-CompFrag |
| `Exp/` | experiment protocols, aggregation, figures, calibration |
| `Data/` | B-READY seed package, generated repository (`Data/main`), workload generators |
| `configs/` | default parameters and tuned overrides |
| `Prompts/`, `Test/`, `out/` | demonstration prompts, tests, sample outputs |
