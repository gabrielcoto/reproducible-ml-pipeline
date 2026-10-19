# Assignment — level-by-level contract

Work the levels in order. A level is done when its acceptance criterion
holds. Hours total ≤ 12 astronomical hours including the video.

There are 3 mandatory levels (basic / advanced / expert) plus an
optional bonus. Expert is required for
maximum score.

## Level 1 — Basic: starter + Hydra configuration (~3.5 h)

Set up the environment, verify the starter works, then wire Hydra so
every behavior-defining constant moves out of `train.py` and into
YAML configs.

### Setup

- Fork, clone, create branch `solution/repro-ml`.
- Create a venv: `python -m venv .venv && source .venv/bin/activate`.
- Install deps: `pip install -r requirements.lock`.
- Verify the starter runs: `python train.py` → prints
  `model=rf accuracy=0.9561 f1=0.9655`.

### Hydra wiring

1. Fill `configs/model/rf.yaml` (see the constants at the top of
   `train.py`) and create/fill `configs/model/lr.yaml`.
2. Add the `@hydra.main` decorator to `train.py`; read values via
   `cfg.seed`, `cfg.test_size`, `cfg.model.*`. Delete the module-level
   constants.
3. CLI overrides must work:
   - `python train.py model=lr` → trains logistic regression;
   - `python train.py model.n_estimators=300` → trains RF with 300 trees;
   - `python train.py seed=7` → trains with seed 7.
4. Metrics printed for the three commands must equal the reference table
   (method-note.md §6, rows E2, E3, E4).

### Acceptance

All 8 tests in `tests/test_basic.py` pass:

```bash
python -m pytest -m basic -v
```

The starter-runs-and-prints-baseline test must keep passing after your
Hydra work — if it fails, you broke the training logic (which must stay
identical; packaging only).

## Level 2 — Advanced: MLflow tracking (~2.5 h)

Log every run:

- `mlflow.set_experiment(<from config>)`;
- inside `mlflow.start_run()`: log ALL config values as params
  (flatten the OmegaConf container first — see method-note.md §5
  for the flatten helper; calling `mlflow.log_params` directly on the
  nested dict would stringify each top-level value into a single
  unreadable `model` param);
- log `accuracy` and `f1` as metrics;
- log the trained model via `mlflow.log_artifact("model.pkl")`.
  The starter already pickles `model.pkl` — we log it as a generic
  artifact. Do NOT use `mlflow.sklearn.log_model`: on MLflow ≥ 3.0
  that call raises a `trusted-types` error for tree-based models,
  which is out of scope for this seminar (see method-note.md §5);
- print the run id so runs are easy to find.

### Acceptance

The test in `tests/test_advanced.py` passes:

```bash
python -m pytest -m advanced -v
```

`mlflow ui` shows, for a run: full param list (flattened — including
`seed`, `model.type`, `model.n_estimators` / `model.C` as separate
keys, not a single stringified `model` blob), both metrics, and the
model artifact.

## Level 3 — Expert: 4 experiments + comparison (~1.5 h)

Run E1–E4 (see README "Execution order" and method-note.md §6). In
the UI, select all four runs and open the comparison view.

### Acceptance

All 6 tests in `tests/test_expert.py` pass:

```bash
python -m pytest -m expert -v
```

Reference metrics matched (E1–E3 exact, E4 ±0.005 tolerance for `f1`).
Reproducibility verified (same config → identical metrics). CLI
override verified (changes a logged param AND a metric). Comparison
view shows 4 runs with their differing params and metrics side by side.

## Bonus* — DVC: data/config versioning (+1.5 h, optional)

`dvc init`; `dvc add configs/`; commit the resulting pointer files; set
up a local remote and `dvc push`. Demonstrate in the video that a fresh
clone + `dvc pull` restores the exact config snapshot of run E3.

**Acceptance:** pointer files committed; fresh-clone + `dvc pull`
restores configs.

## Final — Report, video, submission (~1.5 h)

Fill `REPORT.md`. Record the video demonstrating the 4 checks from
`docs/GRADING.md` (≤ 4 min). Open the Pull Request per
`docs/SUBMISSION.md`.

**Acceptance:** PR opened with code, REPORT.md, and a working video
link.

---

## Summary

| Level | Tests | Hours (cumulative) |
|---|---|---|
| 1 — Basic | 8 (`tests/test_basic.py`) | ~3.5 h |
| 2 — Advanced | 1 (`tests/test_advanced.py`) | ~6 h |
| 3 — Expert | 6 (`tests/test_expert.py`) | ~7.5 h |
| + Bonus (DVC) | — (video-verified) | ~9 h |
| + Report + video + PR | — | ~10.5 h |

Total: ~10.5 h with bonus (cap 12 h). Mandatory part with submission:
~8 h.

Run all tests at once with:

```bash
python -m pytest tests/ -v
```

15 tests total.
