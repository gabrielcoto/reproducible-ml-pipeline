"""Level 3 (expert) tests — 4 experiments + reproducibility + comparison.

Verifies the 4 grading checks at the source level:
  - Check 1 (run identifiability): already covered in test_advanced.
  - Check 2 (reproducibility): E1 run twice → identical metrics.
  - Check 3 (CLI-only override): `model.n_estimators=300` changes metrics.
  - Check 4 (4 runs + reference values): all 4 experiments produce
        reference metrics (within tolerance).

Run:  pytest -m expert -v
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_train(args: list[str], cwd: Path, env: dict) -> tuple[float, float]:
    """Helper: run `python train.py <args>` and parse the accuracy + f1."""
    proc = subprocess.run(
        [sys.executable, "train.py", *args],
        capture_output=True, text=True, timeout=300, cwd=str(cwd), env=env,
    )
    assert proc.returncode == 0, f"train.py failed:\n{proc.stderr}"
    m = re.search(r"accuracy=(0\.\d+) f1=(0\.\d+)", proc.stdout)
    assert m, f"Could not parse accuracy/f1 from output: {proc.stdout!r}"
    return float(m.group(1)), float(m.group(2))


# ── Check 2: reproducibility ───────────────────────────────────────

@pytest.mark.expert
def test_e1_reproducible_across_reruns(tmp_mlflow_db):
    """E1 run twice → identical accuracy and f1 (within 1e-9)."""
    import os
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_mlflow_db}"}
    acc1, f1_1 = _run_train([], REPO_ROOT, env)
    acc2, f1_2 = _run_train([], REPO_ROOT, env)
    assert abs(acc1 - acc2) < 1e-9, (
        f"accuracy differs across reruns: {acc1} vs {acc2}"
    )
    assert abs(f1_1 - f1_2) < 1e-9, (
        f"f1 differs across reruns: {f1_1} vs {f1_2}"
    )


# ── Check 3: CLI override changes the metric ──────────────────────

@pytest.mark.expert
def test_cli_n_estimators_changes_metric(tmp_mlflow_db):
    """`python train.py model.n_estimators=300` must produce different
    metrics than the baseline (E3 vs E1)."""
    import os
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_mlflow_db}"}
    acc1, f1_1 = _run_train([], REPO_ROOT, env)
    acc3, f1_3 = _run_train(["model.n_estimators=300"], REPO_ROOT, env)
    # Either accuracy or f1 must differ.
    moved = abs(acc1 - acc3) > 1e-9 or abs(f1_1 - f1_3) > 1e-9
    assert moved, (
        "n_estimators override had no effect on metrics — the CLI override "
        "path is broken. Check that build_model() reads cfg.model.n_estimators."
    )


# ── Check 4: reference values match (with E4 tolerance) ────────────

# Reference values from method-note.md §6 / SOLUTION/REFERENCE_VALUES.md.
# Calibrated on scikit-learn 1.5.2.
# E4's f1 has a ±0.005 tolerance (see method-note.md §6 for why).
REFERENCE = {
    "E1": (0.9561, 0.9655),  # python train.py
    "E2": (0.9649, 0.9726),  # python train.py model=lr
    "E3": (0.9474, 0.9583),  # python train.py model.n_estimators=300
    "E4": (0.9561, 0.9645),  # python train.py seed=7  (±0.005 tolerance for f1)
}


@pytest.mark.expert
def test_e1_matches_reference(tmp_mlflow_db):
    import os
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_mlflow_db}"}
    acc, f1 = _run_train([], REPO_ROOT, env)
    ref_acc, ref_f1 = REFERENCE["E1"]
    assert abs(acc - ref_acc) < 1e-4, f"E1 accuracy {acc} != {ref_acc}"
    assert abs(f1 - ref_f1) < 1e-4, f"E1 f1 {f1} != {ref_f1}"


@pytest.mark.expert
def test_e2_matches_reference(tmp_mlflow_db):
    import os
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_mlflow_db}"}
    acc, f1 = _run_train(["model=lr"], REPO_ROOT, env)
    ref_acc, ref_f1 = REFERENCE["E2"]
    assert abs(acc - ref_acc) < 1e-4, f"E2 accuracy {acc} != {ref_acc}"
    assert abs(f1 - ref_f1) < 1e-4, f"E2 f1 {f1} != {ref_f1}"


@pytest.mark.expert
def test_e3_matches_reference(tmp_mlflow_db):
    import os
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_mlflow_db}"}
    acc, f1 = _run_train(["model.n_estimators=300"], REPO_ROOT, env)
    ref_acc, ref_f1 = REFERENCE["E3"]
    assert abs(acc - ref_acc) < 1e-4, f"E3 accuracy {acc} != {ref_acc}"
    assert abs(f1 - ref_f1) < 1e-4, f"E3 f1 {f1} != {ref_f1}"


@pytest.mark.expert
def test_e4_matches_reference_with_tolerance(tmp_mlflow_db):
    """E4 (seed=7) has a ±0.005 tolerance for f1 due to sklearn version
    sensitivity — see method-note.md §6."""
    import os
    env = {**os.environ, "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_mlflow_db}"}
    acc, f1 = _run_train(["seed=7"], REPO_ROOT, env)
    ref_acc, ref_f1 = REFERENCE["E4"]
    assert abs(acc - ref_acc) < 1e-4, f"E4 accuracy {acc} != {ref_acc}"
    # ±0.005 tolerance for f1.
    assert abs(f1 - ref_f1) <= 0.005, (
        f"E4 f1 {f1} differs from reference {ref_f1} by more than ±0.005. "
        "This is usually a scikit-learn version difference — see method-note.md §6."
    )
