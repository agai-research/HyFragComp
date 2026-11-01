# HyFragComp

HyFragComp is a reuse layer for LLM-composed workflows over a fragment hypergraph, built for smart governance (public-service procedures such as residence permits, company registration or licences). A request is decomposed into typed sub-goals; stored composition records are aligned to the sub-goals, re-validated against the current repository and context (V1-V5, variant switch), and only the difference is recomposed by an anchored beam search over the hypergraph (GraphRAG). Frequent regions are promoted to composite fragments, and repository changes invalidate only the records that depend on them.

![approach](docs/approach.png)

## Contents

| Path | Role |
|---|---|
| `HyFragComp.py` | run the prototype on a request (`--text`, `--prompts`, `--selftest`) |
| `hypergraph.py` | build the fragment hypergraph from the dataset and print its statistics |
| `main-exp.py` | build instances and run experiment series e1-e11 |
| `hfc/` | prototype package: roles (`llm.py`), encoders (`emb.py`, `enc.py`, `allset.py`), HNSW (`index.py`), checks (`chk.py`), alignment and re-validation (`reuse.py`), delta recomposition (`delta.py`), memory and promotion (`mem.py`, `promo.py`), invalidation (`inval.py`), explanation (`explain.py`), drawing (`viz.py`) |
| `Baselines/` | `cbr` (Lu et al.) and `cafc` (Bucchiarone et al.); `scratch`, `whole`, `nore` are in `hfc/methods.py` and `hfc/system.py` |
| `Data/` | dataset build scripts, `main/` (50 economies, 3,250 procedures, 29,300 fragments, 52,400 hyperedges), `instances/` per experiment |
| `Exp/` | `e1.py` ... `e11.py`, `oracle.py`, `tune.py`, `plots.py`, `stats.py`, `core.py` |
| `Test/` | `first_test.py`, `test_units.py`, `test_nn.py`, `static_check.py` |
| `configs/` | `default.yaml` (all parameters), tuned override files after `Exp/tune.py` |
| `Prompts/prompts.json` | test requests (scenario, first application, unseen phrasing, terse, trap sentence) |
| `results/` | `raw/`, `agg/`, `figs/`, `tables/`, `logs/`, `tuning/` |
| `docs/` | `Implementation.tex`, `Results.tex`, `Dataset.tex/.docx`, `implementation report.docx`, figures |
| `notebooks/` | Colab notebooks; `python notebooks/run_nb.py notebooks/00_quickstart.ipynb` runs one without Jupyter |

## Environment

Python 3.10+ on CPU; `pip install -r requirements.txt`. Details in `environment.md`. Everything runs on NumPy, SciPy and scikit-learn: the autograd engine, AllSet encoder, HNSW index and the sentence encoder are implemented in this repository. The eight LLM roles (`c_cur`, `c_int`, `c_pln`, `c_ret`, `c_sel`, `c_com`, `c_ver`, `c_exp`) run on a local rule-and-embedding engine (`llm.backend: rule`). LLM latency and tokens are priced analytically (`llm.latency`); measures that depend on them are labelled *analytic* in the results.

## Run

```bash
python HyFragComp.py --selftest                    # calls the eight roles
python HyFragComp.py --prompts --out out/prompts   # serves Prompts/prompts.json, writes JSON, Markdown, BPMN-style figures
python HyFragComp.py --text "..." --jur MAR --renew --priority
python Test/first_test.py                          # scenario, rejections, variant switch, change events
python -m unittest Test.test_units Test.test_nn    # unit tests
python Test/static_check.py                        # static scan
python hypergraph.py --rebuild                     # rebuild Data/main from the B-READY seed
python Exp/tune.py --trials 24                     # tuning on the validation split, writes configs/tuned_*.yaml
python main-exp.py --exp e1 e2 e3 e4 e5 e6 e7 e8 e9 e10 e11 --runs 5 --profile full
python main-exp.py --exp e1 --profile smoke        # one-minute check
python main-exp.py --exp e1 --only figures         # re-draw from stored results
```

## Configuration

Every parameter is in `configs/default.yaml`: matching (`theta_m`, `rho_min`, `K_a`, `lambdas`), memory (`k`, `k_u`, `R_max`, `W_s`, `tau_e`, `u_eager_pct`, cost units), recomposition (`h`, `B_w`, `eps_max`, `K`, `theta_sim`, `eta`), encoder (`alpha`, `L`, dimensions, training), index (`M`, `ef_construction`, `ef_search`), planner thresholds, LLM latency model, privacy, baseline settings, stream generators and the sweep values of each experiment under `experiments.smoke` / `experiments.full`. Override files are passed with `--config` or placed in `configs/` (`tuned_*.yaml`).

## Reproducibility

Seeds: models and data `0`; experiment runs `0-4`; tuning `100-101`. Trained encoder checkpoints and indexes are cached in `results/ckpt` and `results/cache`. Raw per-request records are in `results/raw`, aggregates (mean, std, 95% CI, n) in `results/agg`, figures in `results/figs`, tables in `results/tables`, JSON-lines logs in `results/logs`.
