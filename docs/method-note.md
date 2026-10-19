# Methodological note — Reproducible ML experiments with MLflow + Hydra

*Read this before starting. Only basic Python is assumed; no ML
background is required. Estimated reading time: 20 minutes.*

## 1. Why experiments become irreproducible

Typical pre-reproducibility workflow:

```python
model = RandomForestClassifier(n_estimators=300)   # was it 300? 100?
model.fit(X_train, y_train)                        # which split? which seed?
```

It's common for parameters to not be reported or tracker, afterwards its problematic to get the same result elsewhere.

- **parameters live in code** — changed by editing, not recorded;
- **no tracking** — runs are invisible;
- **config scattered** — half in the script, half in your head.

A contract helps: *every run is fully described by (a) a config
file and (b) an automatic record in a tracking store.* That is what you
will build.

## 2. Hydra in 15 minutes

Hydra turns configuration into composable files + command-line overrides.

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(cfg.seed)          # access values with attribute syntax
    print(cfg.model.n_estimators)

if __name__ == "__main__":
    main()
```

- `configs/config.yaml` is the **primary config**; its `defaults:` list
  composes group configs (`model: rf` → loads `configs/model/rf.yaml`).
- CLI overrides, no code edits:
  ```bash
  python train.py                       # use config values
  python train.py model=lr              # switch group file
  python train.py model.n_estimators=300
  python train.py seed=7
  ```
- Access nested values as `cfg.model.n_estimators`.
- A `???` in a config means *mandatory* — Hydra refuses to run until it
  is set. That is what the `???` in the skeleton configs are for.

## 3. MLflow in 15 minutes

MLflow Tracking is a log of runs. Minimal pattern:

```python
import mlflow

mlflow.set_experiment("reproducible-ml")
with mlflow.start_run():
    # Flatten the config so every knob becomes a separate param —
    # see §5 for why you should NOT pass the nested dict directly.
    mlflow.log_params({"seed": 42, "model.type": "rf",
                       "model.n_estimators": 100})
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    # The starter pickles model.pkl — log it as a generic artifact.
    # Do NOT use mlflow.sklearn.log_model (see §5 for why).
    mlflow.log_artifact("model.pkl")
```

- **params** — *what* you ran (every config value, flattened);
- **metrics** — *how well* it went;
- **artifacts** — *what it produced* (the model file).

View the log: `mlflow ui` → http://localhost:5000 — table of runs,
click a run to see params/metrics/artifacts, select several runs and
compare. Logging *all* config values as separate flattened params is
the most important habit; a run you can't identify is a run you
can't trust.

## 4. The 4 grading checks

| # | Check | How to self-verify |
|---|-------|--------------------|
| 1 | Run identifiability | `mlflow ui` → open any run → confirm params, metrics, and the `model.pkl` artifact are all there |
| 2 | Reproducibility | `python train.py` twice → two runs, same accuracy and f1 |
| 3 | CLI-only change | `python train.py model.n_estimators=300` → new run whose params show `n_estimators: 300` and whose metrics differ from baseline |
| 4 | Comparison | In the UI: select all 4 runs → compare → the table shows configs and metrics side by side |

## 5. Pitfalls

- **Flatten the config before logging.** If you call
  `mlflow.log_params(OmegaConf.to_container(cfg))`, MLflow receives a
  nested dict and string-casts each top-level value. Your `model` param
  becomes the string `"{'type': 'rf', 'n_estimators': 100, ...}"` — a
  single unreadable param instead of separate `model.type`,
  `model.n_estimators`, `model.max_depth`. Flat first:

  ```python
  def _flatten(obj, prefix, out):
      if isinstance(obj, dict):
          for k, v in obj.items():
              _flatten(v, f"{prefix}.{k}" if prefix else k, out)
      else:
          out[prefix] = obj

  flat = {}
  _flatten(OmegaConf.to_container(cfg, resolve=True), "", flat)
  mlflow.log_params(flat)
  ```

  The most important is that a run you can't identify/check because the config is not readable 
  is a run you can't trust.
- **Determinism**: the pipeline must be deterministic for check 2.
  `random_state` comes from config; never call anything random without it.
- **LogisticRegression** needs `max_iter=5000` (default 100 diverges with
  warnings on this dataset — warnings are not failures, but noise).
- **Hydra output dirs**: Hydra may write run outputs under `outputs/`.
  That is fine; do not log relative paths as params — log values.
- **Windows + `mlflow ui`**: if the UI won't start, use the provided
  `docker-compose.yml` and set `MLFLOW_TRACKING_URI=http://localhost:5000`.
- **Don't modify the training logic.** Packaging only. What is evaluated is the packing. If your metrics
  don't match §6 the tests will fail.
- **MLflow ≥ 3.0 and sklearn models**: `mlflow.sklearn.log_model` now
  requires explicit `trusted_types` for tree-based models (security
  feature). For this seminar, use `mlflow.log_artifact("model.pkl")`
  instead — the model is already pickled by the starter, and logging it
  as a generic artifact is equivalent for run identifiability.

## 6. Suggested experiment set + reference values

Computed with the fixed pipeline (seed 42, test_size 0.2, stratified
split, scikit-learn 1.5.2). **Your values must match these** — matching
numbers are the proof that the pipeline logic is intact. E1–E3 must
match exactly; E4's `f1` has a small tolerance because the random
forest's exact tree structure at `seed=7` is sensitive to floating-point
ordering in some environments.

| Run | Command | accuracy | f1 |
|-----|---------|----------|-----|
| E1 | `python train.py` | 0.9561 | 0.9655 |
| E2 | `python train.py model=lr` | 0.9649 | 0.9726 |
| E3 | `python train.py model.n_estimators=300` | 0.9474 | 0.9583 |
| E4 | `python train.py seed=7` | 0.9561 | 0.9645 |

(E1 run twice must give identical numbers — that is check 2.)

**Why the E4 `f1` tolerance:** the random forest at `seed=7` produces
a tree structure where one test sample lands on a tie between the
malignant and benign classes. Different platforms / BLAS backends can
break the tie differently, producing `f1` values in the range
0.9645–0.9650. The grading rubric (`docs/GRADING.md`) allows ±0.005
for E4's `f1` only. E1–E3 are platform-independent and must match
exactly.

## 7. Bonus part: data versioning with DVC (+1–2h, optional)

```bash
pip install dvc
dvc init
dvc add configs/                # version the configs alongside git
dvc remote add -d local /path/to/store
dvc push
```

Concept: large/binary data lives outside git; git stores small pointer
files; `dvc push/pull` sync the data. Show in your video that a fresh
clone + `dvc pull` restores the exact data snapshot of a run.
