# Environment

- Python 3.12.3 (tested), CPU only, no GPU needed. `Dockerfile` provides the same environment.
- Packages: exact versions in `requirements.txt` (NumPy, SciPy, scikit-learn, PyYAML, matplotlib, networkx, pandas, python-docx).
- `PYTHONHASHSEED=0`: Python randomises the iteration order of string sets per process, which changes tie-breaking; the entry points (`main-exp.py`, `HyFragComp.py`, `Exp/calibrate.py`, `reproduce.sh`) fix it.
- Optional: Graphviz `dot` (BPMN-style drawings are also written as `.dot` files).
- Language-model backend: `llm.backend` in `configs/default.yaml`: `rule` (default; the latency of a call is computed with `llm.latency`), `openai` (hosted model, `OPENAI_API_KEY`, measured latency) or `shadow` (calibration). See `REPRODUCE.md`.
- Seeds: `seed: 0` for models and data; the five experiment runs use seeds 0-4; tuning uses seeds 100-101.
- Memory: 4 GB is enough for the largest repository (20,000 fragments).
- Expected run times on two CPU cores: encoder training 2-3 min (cached in `results/ckpt`), `Test/first_test.py` 10 s, one smoke series 1 min, the full protocol several hours.
