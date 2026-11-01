# HyFragComp

HyFragComp is a reuse layer for LLM-composed service workflows over a **fragment hypergraph**, built for smart-governance procedures (residence permits, company registration, licences). Instead of recomposing a workflow for every request, it keeps fragments and past compositions in the hypergraph and reuses whatever can still be justified.

## Approach

For an incoming request:

1. **Decompose** the request into typed sub-goals.
2. **Align** stored composition records to the sub-goals.
3. **Re-validate** the aligned records against the current repository and request context (checks V1–V5; a *variant switch* replaces a fragment by a compatible sibling variant when a check fails).
4. **Recompose only the difference** with an anchored beam search over the hypergraph (graph-grounded retrieval), starting from the still-valid part of the record.
5. **Promote** frequently reused regions to composite fragments.
6. On repository change, **invalidate only the records that depend on the changed fragments**.

Eight LLM roles drive the pipeline through one interface: `c_cur` (curation), `c_int` (interpretation), `c_pln` (planning), `c_ret` (retrieval), `c_sel` (selection), `c_com` (composition), `c_ver` (verification), `c_exp` (explanation).

## Implementation

- Python 3.10+, CPU only. Dependencies: NumPy, SciPy, scikit-learn (`pip install -r requirements.txt`). Platform notes in `environment.md`.
- To stay light and offline, the autograd engine, the AllSet hypergraph encoder, the HNSW index and the sentence encoder are implemented in this repository.
- The LLM roles run by default on a local rule-and-embedding engine (`llm.backend: rule`). LLM latency and tokens are then priced analytically (`llm.latency`); measures that depend on them are labelled *analytic* in the results.
- The hypergraph in `Data/main` is seeded from World Bank B-READY procedure data.

```
HyFragComp.py   entry point (self-test, batch prompts, single request)
hypergraph.py   build / rebuild Data/main
main-exp.py     experiment driver (e1–e11)
Exp/tune.py     hyper-parameter tuning
configs/        default.yaml, tuned_*.yaml
Prompts/        prompts.json
Test/           scenario test, unit tests, static check
results/        ckpt, cache, raw, agg, figs, tables, logs
```

## Run

Run from the repository root.

```bash
python HyFragComp.py --selftest                     # calls the eight roles once
python HyFragComp.py --prompts --out out/prompts    # reads Prompts/prompts.json; writes JSON, Markdown, BPMN-style figures
python HyFragComp.py --text "..." --jur MAR --renew --priority   # single request
python Test/first_test.py                           # scenario: reuse, rejections, variant switch, change events
python -m unittest Test.test_units Test.test_nn     # unit tests
python Test/static_check.py                         # static scan
python hypergraph.py --rebuild                      # rebuild Data/main from the B-READY seed
```

## Experiments

```bash
python Exp/tune.py --trials 24                      # tuning on the validation split → configs/tuned_*.yaml
python main-exp.py --exp e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 --runs 5 --profile full
python main-exp.py --exp e1 --profile smoke         # reduced sweep, about one minute
python main-exp.py --exp e1 --only figures          # redraw figures and tables from stored results
```

`smoke` and `full` sweep values are set per experiment under `experiments.smoke` / `experiments.full`.

## Configuration

All parameters are in `configs/default.yaml`; overrides are passed with `--config` or placed in `configs/` as `tuned_*.yaml`.

| Group | Keys |
|---|---|
| matching | `theta_m`, `rho_min`, `K_a`, `lambdas` |
| memory | `k`, `k_u`, `R_max`, `W_s`, `tau_e`, `u_eager_pct`, cost units |
| recomposition | `h`, `B_w`, `eps_max`, `K`, `theta_sim`, `eta` |
| encoder | `alpha`, `L`, dimensions, training |
| index | `M`, `ef_construction`, `ef_search` |
| other | planner thresholds, `llm` (backend, latency model), privacy, baselines, stream generators, experiment sweeps |

## Reference

H. Mezni, Z. Brahmi, F. Ghedass, A. Albahli, H. Elmannai, S. Alalmaei, and A. Alsehaim, *Evolving Reuse of LLM-Composed Service Workflows for Smart Governance*, IEEE Transactions on Services Computing (under review).
