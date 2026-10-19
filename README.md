# Reproducible ML experiments with MLflow and Hydra

A practical exercise on experiment reproducibility: configuration-driven
training, run tracking, and verification that results can be reproduced.
You receive a working training script and turn it into a fully configured,
fully tracked experiment pipeline.

**Estimated time:** 10–12 hours including the bonus stage and submission.
The mandatory is ~8 hours. This estimate is for someone familiar
with Python, the command line, and YAML basics.

**Repository:** https://github.com/gabrielcoto/reproducible-ml-pipeline/
**Submission:** Pull Request with code, a completed REPORT.md, and a link
to a video of up to 4 minutes.

## Context and Result

A saved model file without its history is very difficult to reproduce: Which code produced
it? Which data? Which parameters? Experiments need to be compared, audited, 
and re-run later. For this to work and be able to check upon it we need it to be described by
 (a) a config file and (b) an automatic record in a tracking store.

**After this task, you will be able to:**

- launch any training run from a config file, with no code edits;
- override any parameter from the command line (`python train.py model=lr`);
- open any past run and see its full config, metrics, and model artifact;
- re-run the same config and get *identical* metrics;
- compare several experiments side by side in the MLflow UI.

**Expected end result:** a configured experiment environment where
`python train.py <overrides>` runs the training end-to-end, logs
everything to MLflow, and 4 documented experiments can be compared in the
UI — plus a short video demonstrating the 4 verification checks.

## Project composition

| File | Purpose |
|------|---------|
| `train.py` | Working starter script (sklearn, fixed dataset). No config system, no tracking — your job. |
| `configs/config.yaml` | Primary config skeleton with gaps (TODO) |
| `configs/model/rf.yaml`, `configs/model/lr.yaml` | Model parameter configs with gaps (TODO) |
| `method-note.md` | Theory: Hydra, MLflow, pitfalls, reference values. **Read first.** |
| `pytest.ini` | Test marker definitions (basic / advanced / expert) |
| `tests/test_basic.py` | Level 1 — starter runs + Hydra config filled + decorator + CLI overrides |
| `tests/test_advanced.py` | Level 2 — MLflow logs params / metrics / artifact |
| `tests/test_expert.py` | Level 3 — 4 experiments + reproducibility + reference values |
| `docs/ASSIGNMENT.md` | Level-by-level contract with acceptance criteria |
| `docs/GRADING.md` | The 4 grading checks and bonus, in detail |
| `docs/SUBMISSION.md` | Where and how to submit (fork, branch, MR, video) |
| `.github/pull_request_template.md` | Auto-fills the PR description when you open it |
| `.github/workflows/tests.yml` | CI recipe — runs the test suite automatically on every PR and push to `solution/*` |
| `REPORT.md` | Report template — fill it in |
| `requirements.txt` | Unpinned deps (lighter install) |
| `requirements.lock` | Pinned deps (full reproducibility — recommended) |
| `docker-compose.yml` | Optional fallback MLflow server for Windows |

## Execution order

| Level | What to do | Time reference |
|-------|-----------|----------------|
| 1 — Basic | Set up environment, verify starter runs, then wrap `train.py` in Hydra: complete the configs, add `@hydra.main`, CLI overrides work | 3.5 h |
| 2 — Advanced | Add MLflow tracking: params (flattened), metrics, model artifact per run | 2.5 h |
| 3 — Expert | Run the 4 experiments, verify against reference values, build the comparison in the UI | 1.5 h |
| + Bonus | Version configs/data with DVC | +1.5 h |
| Final | Fill REPORT.md, record the video (4 checks), open a Pull Request | 1.5 h |

Total: ~10.5 h with bonus (cap 12 h); mandatory part with submission ~8 h.
Precise acceptance criteria per level: `docs/ASSIGNMENT.md`.

## Getting your copy

1. On GitHub, click **Fork** on the seminar repository (keep the name
   unchanged). Your fork lives at `github.com/<your-username>/<repo>`.
2. Clone your fork and create the working branch:
3. All work happens on branch `solution/repro-ml`. `origin` is your fork;
   `upstream` is the original repo.

## Necessary knowledge and environment

You need: Python 3.10+ basics (functions, dicts), the command line,
YAML syntax (5-minute primer is in the note §2.1), Git (fork/branch/commit).
**No ML background is assumed** — the model and dataset are fixed.

Recommended environment: Linux, macOS, or Windows with WSL2 or Git Bash.
Everything installs via pip (see Quick Launch); Docker is needed only for
the optional MLflow fallback. The first run needs access to PyPI.

## Quick Launch

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.lock                       # pinned versions (recommended)
python -m pytest -m basic -v                          # 1 passes (starter), 7 fail (need your Hydra work)
python train.py
# expected starter output:
# model=rf accuracy=0.9561 f1=0.9655
```

Open the tracking UI (after Level 2):

```bash
mlflow ui    # → http://localhost:5000
```

## Checks

Automated tests are organised by level — 3 levels, 15 tests total:

```bash
python -m pytest -m basic -v       # Level 1 — starter runs + Hydra config + decorator + CLI overrides
python -m pytest -m advanced -v    # Level 2 — MLflow logs params/metrics/artifact
python -m pytest -m expert -v      # Level 3 — 4 experiments + reproducibility + reference values
python -m pytest -v                # All levels at once (15 tests)
```

On the untouched starter, exactly one test passes (the starter-runs
sanity check inside `test_basic.py`). All other tests fail until you
complete the corresponding level — they're your automated feedback
loop before recording the video.

**CI runs automatically on every PR** After pushing, open your Merge
Request — you'll see a green ✅ or red ❌ badge near the top showing
whether the test suite passes. If red, click through to see exactly
which tests failed and why. No need to run pytest locally if CI is
working.

- Reference values for the 4 experiments: `method-note.md` §6. E1–E3
  must match exactly; E4 has a ±0.005 tolerance (sklearn version
  sensitivity — see `method-note.md` §6 for details).
- The 4 grading checks are demonstrated in your video, not by auto-tests:
  `docs/GRADING.md`. The auto-tests verify the same properties at the
  source level, so passing `pytest -m expert` is strong evidence your
  video will pass too.

## Settings

| Variable (config key) | Default | Meaning |
|-----------------------|---------|---------|
| `seed` | 42 | Random seed for the data split and RF |
| `test_size` | 0.2 | Held-out fraction of the dataset |
| `experiment_name` | reproducible-ml | MLflow experiment name |
| `model.type` | rf | `rf` or `lr` |
| `model.n_estimators` | 100 | Random forest size (rf only) |
| `model.max_depth` | null | RF depth limit (rf only) |
| `model.C` | 1.0 | Logistic regression regularization (lr only) |

## Typical problems

- `mlflow ui` won't start or shows nothing on Windows → use the fallback:
  `docker compose up -d mlflow`, then
  `export MLFLOW_TRACKING_URI=http://localhost:5000` (PowerShell:
  `$env:MLFLOW_TRACKING_URI="http://localhost:5000"`) and rerun `train.py`.
- Port 5000 is busy → `mlflow ui --port 5001`.
- Metrics don't match the reference table → you changed the training
  logic. Packaging only. Restore and redo.
- Hydra complains about `???` → that is the point: fill the config first.
- "Command not found" after venv activation → reactivate; check `which python`.

## What to hand in

Code: your implementation + completed `REPORT.md` on branch
`solution/repro-ml` of your fork.
Video: up to 4 minutes, hosted anywhere accessible by link. 
Do not commit the video file to Git.
Open a Pull Request from your fork's `solution/repro-ml` into the
original repo's `main`. The PR link **is** your submission.
Full mechanics: `docs/SUBMISSION.md`. Problems with the assignment itself
(not your solution): open an Issue in the original repo.

## Materials

- Hydra tutorial: https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/
- MLflow tracking guide: https://mlflow.org/docs/latest/tracking.html
- OmegaConf (config syntax): https://omegaconf.readthedocs.io/
- Dataset (fixed, ships with sklearn): https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
