"""Shared pytest configuration: markers + temp MLflow fixture.

Markers
-------
- @basic    : level 1 — starter runs + Hydra config filled + decorator + CLI overrides
- @advanced : level 2 — MLflow logs params/metrics/artifact per run
- @expert   : level 3 — 4 experiments + reproducibility + comparison

Run a single level with:  pytest -m basic -v
Run all levels at once:    pytest -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# ── Path setup ─────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Fixture: temp MLflow tracking DB ────────────────────────────────
@pytest.fixture
def tmp_mlflow_db(tmp_path, monkeypatch):
    """Point MLflow at a temp sqlite DB so tests don't pollute the repo.

    Returns the path to the temp DB.
    """
    db = tmp_path / "mlflow.db"
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{db}")
    # Also delete any existing experiment with our default name, so
    # tests start clean even if the repo's mlflow.db leaked in.
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        for exp in client.search_experiments():
            if exp.name == "reproducible-ml":
                client.delete_experiment(exp.experiment_id)
    except Exception:
        pass
    return db

