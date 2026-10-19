"""Level 1 (basic) tests — starter runs + Hydra configuration.

This level verifies that:
  - The untouched starter runs and prints the baseline metrics
    (must keep passing after your Hydra work — same training logic).
  - `configs/model/rf.yaml` is filled in (no `???`).
  - `configs/model/lr.yaml` is filled in (no `???`).
  - `train.py` is decorated with `@hydra.main`.
  - CLI override `model=lr` produces a different model type in the output.
  - CLI override `model.n_estimators=300` changes the printed n_estimators
    (via `--cfg job`).

Once all 8 tests in this file pass, you're ready for Level 2 (advanced).

Run:  pytest -m basic -v
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.basic


# ── Sanity: the starter runs and prints the baseline ───────────────
# This must pass on the untouched starter AND keep passing after your
# Hydra/MLflow work (same training logic, same numbers). If it fails
# after your changes — restore and redo.

def test_starter_runs_and_prints_baseline():
    """`python train.py` runs and prints the baseline metrics.
    """
    proc = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"train.py failed:\n{proc.stderr}"
    m = re.search(r"model=(\w+) accuracy=(0\.\d+) f1=(0\.\d+)", proc.stdout)
    assert m, f"unexpected output: {proc.stdout!r}"
    assert m.group(1) == "rf"
    assert float(m.group(2)) == 0.9561
    assert float(m.group(3)) == 0.9655


# ── Config files are filled in (no `???` placeholders) ─────────────

def test_rf_config_is_filled():
    """`configs/model/rf.yaml` must declare type, n_estimators, max_depth."""
    p = REPO_ROOT / "configs" / "model" / "rf.yaml"
    assert p.exists(), f"Missing {p.relative_to(REPO_ROOT)}"
    content = p.read_text(encoding="utf-8")
    assert "???" not in content, (
        "configs/model/rf.yaml still has `???` placeholders — fill them in."
    )
    for key in ("type", "n_estimators", "max_depth"):
        assert re.search(rf"^{key}\s*:", content, re.MULTILINE), (
            f"configs/model/rf.yaml is missing the `{key}` field."
        )


def test_lr_config_is_filled():
    """`configs/model/lr.yaml` must declare type and C."""
    p = REPO_ROOT / "configs" / "model" / "lr.yaml"
    assert p.exists(), f"Missing {p.relative_to(REPO_ROOT)}"
    content = p.read_text(encoding="utf-8")
    assert "???" not in content, (
        "configs/model/lr.yaml still has `???` placeholders — fill them in."
    )
    for key in ("type", "C"):
        assert re.search(rf"^{key}\s*:", content, re.MULTILINE), (
            f"configs/model/lr.yaml is missing the `{key}` field."
        )


# ── train.py is decorated with @hydra.main ─────────────────────────

def test_train_uses_hydra_decorator():
    """`train.py` must have `@hydra.main(...)` decorator."""
    src = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    assert "hydra.main" in src, (
        "train.py is not decorated with @hydra.main. See method-note.md §2."
    )
    assert re.search(r'config_name\s*=\s*["\']config["\']', src), (
        "Hydra decorator must reference config_name='config'."
    )


def test_no_module_level_constants_in_train():
    """The TODO(1) constants (MODEL_TYPE, N_ESTIMATORS, etc.) must be
    removed from train.py — they should live in configs/ now."""
    src = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    for name in ("MODEL_TYPE", "N_ESTIMATORS", "MAX_DEPTH", "C =", "SEED =",
                 "TEST_SIZE =", "EXPERIMENT_NAME ="):
        bad = re.search(rf"^{re.escape(name.rstrip())}\s*=", src, re.MULTILINE)
        assert not bad, (
            f"train.py still has a module-level constant `{name}` — "
            f"it should live in configs/. See TODO(1) in the starter."
        )


# ── CLI overrides work end-to-end ──────────────────────────────────

def test_cli_model_lr_produces_lr_run():
    """`python train.py model=lr` must train LR (not RF)."""
    proc = subprocess.run(
        [sys.executable, "train.py", "model=lr"],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"train.py failed:\n{proc.stderr}"
    m = re.search(r"model=(\w+)", proc.stdout)
    assert m, f"unexpected output: {proc.stdout!r}"
    assert m.group(1) == "lr", (
        f"Expected model=lr, got model={m.group(1)}. "
        "Check that configs/model/lr.yaml has type: lr and that build_model() "
        "dispatches on cfg.model.type."
    )


def test_cli_n_estimators_override_resolves():
    """`python train.py model.n_estimators=300 --cfg job` must show
    n_estimators: 300 in the resolved config."""
    proc = subprocess.run(
        [sys.executable, "train.py", "model.n_estimators=300", "--cfg", "job"],
        capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"--cfg job failed:\n{proc.stderr}"
    assert re.search(r"n_estimators:\s*300", proc.stdout), (
        "Hydra override `model.n_estimators=300` did not resolve. "
        "Check the dotted path matches your YAML structure."
    )


def test_cli_seed_override_resolves():
    """`python train.py seed=7 --cfg job` must show seed: 7."""
    proc = subprocess.run(
        [sys.executable, "train.py", "seed=7", "--cfg", "job"],
        capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, f"--cfg job failed:\n{proc.stderr}"
    assert re.search(r"^seed:\s*7", proc.stdout, re.MULTILINE), (
        "Hydra override `seed=7` did not resolve."
    )
