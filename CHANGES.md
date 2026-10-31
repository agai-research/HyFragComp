# Changes in the reproducibility revision

## Added
- `Exp/` experiment package (previously missing although `main-exp.py` imported it): protocols of the series in
  the paper (`e1` Fig. 4, `e5` Fig. 5, `e7` Fig. 6, `e6` Fig. 7) and of the additional experiments (`x1`-`x3`),
  resumable runs, aggregation with paired bootstrap intervals and Cohen's d, tables and figures.
- `llm.backend: openai` and `shadow` (`hfc/llm_api.py`, `hfc/llm.py`): every role can be answered by the hosted
  model (GPT-5.6 Luna by default) with measured latency and tokens, logged per call and cached for reruns.
- `Exp/calibrate.py`: replays a subset of the workloads with the rule, shadow and hosted backends, reports
  per-role and per-request agreement, and fits the latency model (`configs/latency_fitted.yaml`).
- `REPRODUCE.md` (command, protocol, run time and outputs of every figure and table), `DATA.md` (the three
  datasets and the workload generators), `reproduce.sh`, `Dockerfile`, `.gitignore`.
- Tests: `Test/test_cycles.py`, `Test/test_llm_backend.py` (offline, mock transport).

## Fixed
- `hfc/chk.py`: `ancestors` and `offsets` recursed forever on cyclic connections (RecursionError in long streams
  and during composite repair); they are now iterative, and a cycle is reported as a defect.
- Run-to-run nondeterminism: results depended on Python's per-process hash randomisation; the entry points now
  fix `PYTHONHASHSEED=0`.
- `hypergraph.py --rebuild` failed because the seed package was not extracted; `Data/build_base.py` now extracts
  `Data/raw/bready_dataset_v2.zip` and reproduces `Data/main` byte for byte.
- `privacy.pad_latency` was read but not applied; memory lookups are now padded to `privacy.pad_ms`
  (`pad_sleep: true` waits for real in deployment).

## Changed
- `requirements.txt` pins the tested versions.
- `configs/default.yaml`: `experiments` lists only the series reported in the paper; new `llm.api` section and
  `privacy.pad_sleep`, `privacy.hide_scope_rejections` (omit scope rejections from explanations; default off).
- README and `environment.md`: backends, cost measurement, datasets, commands; removed the references to the
  unpublished `Exp/tune.py` and `docs/` (the tuned overrides are the locked files of `configs/`).

With `llm.backend: rule` and `PYTHONHASHSEED=0`, the modified code returns the same workflows, call counts,
latencies and explanations as the original code (checked on 450 requests of three methods).
