"""Level 2 (advanced) tests — MLflow tracking.

Verifies that:
  - A single `python train.py` creates an MLflow run.
  - The run logs params (at least seed, model.type, model.n_estimators or
    model.C).
  - The run logs metrics (accuracy, f1).
  - The run logs at least one artifact (model.pkl).

Run:  pytest -m advanced -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.advanced
def test_train_creates_mlflow_run_with_params_metrics_artifact(tmp_mlflow_db, monkeypatch):
    """A single `python train.py` must produce an MLflow run with:
    - params populated (≥3, including seed, model.type, model.n_estimators or model.C)
    - metrics populated (accuracy, f1)
    - at least one artifact (model.pkl)
    """
    # Run train.py from the repo root, pointing at our temp MLflow DB.
    env = {
        **dict(__import__("os").environ),
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_mlflow_db}",
    }
    proc = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=True, text=True, timeout=300, cwd=str(REPO_ROOT),
        env=env,
    )
    assert proc.returncode == 0, f"train.py failed:\n{proc.stderr}"
    # Print the last line so we can see the run id.
    assert "run_id=" in proc.stdout, (
        "train.py did not print a run_id. Make sure mlflow.start_run() is "
        "active and you print mlflow.active_run().info.run_id."
    )

    # Pull the run from MLflow and verify it's identifiable.
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(f"sqlite:///{tmp_mlflow_db}")
    client = MlflowClient()
    exp = client.get_experiment_by_name("reproducible-ml")
    assert exp is not None, (
        "Experiment 'reproducible-ml' was not created. Did you call "
        "mlflow.set_experiment(cfg.experiment_name)?"
    )

    runs = client.search_runs([exp.experiment_id], max_results=10)
    assert len(runs) >= 1, "No runs were logged to MLflow."
    run = runs[0]

    # (a) params populated — at least seed, model.type, and one model-specific knob.
    assert len(run.data.params) >= 3, (
        f"Expected ≥3 params, got {len(run.data.params)}. "
        "Did you log ALL config values via mlflow.log_params(...)?"
    )
    assert "seed" in run.data.params, "Missing `seed` param."
    assert "model.type" in run.data.params, (
        "Missing `model.type` param. Check that you logged the full flattened config."
    )

    # (b) metrics populated.
    assert "accuracy" in run.data.metrics, "Missing `accuracy` metric."
    assert "f1" in run.data.metrics, "Missing `f1` metric."

    # (c) at least one artifact.
    arts = client.list_artifacts(run.info.run_id)
    assert len(arts) >= 1, (
        "No artifacts were logged. Did you call mlflow.log_artifact('model.pkl')?"
    )
    art_names = {a.path for a in arts}
    assert "model.pkl" in art_names, (
        f"Expected `model.pkl` artifact, got {art_names}."
    )
