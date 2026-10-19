# Grading

There are 3 levels (basic / advanced / expert) plus an optional bonus.
Each level is verified by automated tests; the video demonstrates.

## Levels and weights

| Level | Tests pass | Video shows | Hours (est.) | Grade weight |
|---|---|---|---|---|
| 1 — Basic | `pytest -m basic -v` (8 tests) | Starter runs + Hydra config + CLI overrides | ~3.5 h | 30 % |
| 2 — Advanced | `pytest -m advanced -v` (1 test) | MLflow logs params (flattened) + metrics + artifact | ~2.5 h | 30 % |
| 3 — Expert | `pytest -m expert -v` (6 tests) | 4 experiments + reproducibility + CLI override + comparison | ~1.5 h | 40 % |
| + Bonus (DVC) | — (video-verified) | DVC config versioning demonstrated | +1.5 h | +10 % |

1–3 levels gets full marks (100 %). DVC is on top.

## The 4 verification checks (demonstrated in the video)

The control video (≤ 4 minutes) must show all four checks. Each is
pass/fail; all four must pass.

| # | Check | What must be visible in the video |
|---|-------|-----------------------------------|
| 1 | Run identifiability | In the MLflow UI: open any run — its full param list (flattened), both metrics, and the model artifact are shown |
| 2 | Reproducibility | `python train.py` executed twice — two runs with identical accuracy and f1 (reference: 0.9561 / 0.9655) |
| 3 | CLI-only change | `python train.py model.n_estimators=300` — a new run appears whose params show `n_estimators: 300` and whose metrics differ from baseline (0.9474 / 0.9583) |
| 4 | Comparison | The MLflow comparison view/table of all four runs E1–E4 |

Self-verify checks in the recording.

## Why reference values matter

Checks 2 and 3 rely on the reference numbers (method-note.md §6). If your numbers differ
significantly (more than ±0.005), the training logic was changed — the
task is packaging only, requires matching numbers for correctness.

**Tolerance:** E1–E3 must match exactly. E4 (`seed=7`) has a ±0.005
tolerance for `f1` because `RandomForestClassifier` can produce
slightly different trees across scikit-learn versions even with the
same seed. This is documented in `method-note.md` §6.

## Bonus (DVC)

DVC versioning demonstrated in the video (fresh clone + `dvc pull`
restores the config snapshot). Counts on top of advanced level, not a substitute.

## Report

`REPORT.md` must contain the experiment table (E1–E4 with commands and
metrics), one paragraph of observations (what changed and why it
mattered), and a list of problems you hit and how you solved them.
