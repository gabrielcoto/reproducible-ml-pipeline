"""Starter training script — WORKS AS-IS, requires a config system and tracking.

Your job (see method-note.md):
  TODO(1)  make every constant below come from configs/ (Hydra), not from this file
  TODO(2)  wire seed and test_size from config into load_data()
  TODO(3)  model selection + hyperparameters from configs/model/*.yaml
  TODO(4)  add the @hydra.main decorator and config_path/config_name
  TODO(5)  add MLflow tracking: params, metrics, model artifact — per run
The training LOGIC must stay identical.
"""
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# --- TODO(1): these constants move to configs/ -------------------------------
MODEL_TYPE = "rf"        # "rf" | "lr"
N_ESTIMATORS = 100       # RF only
MAX_DEPTH = None         # RF only
C = 1.0                  # LR only
SEED = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "reproducible-ml"
# -----------------------------------------------------------------------------


def load_data(seed, test_size):
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_model():
    if MODEL_TYPE == "rf":
        return RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=SEED)
    if MODEL_TYPE == "lr":
        return LogisticRegression(C=C, max_iter=5000)
    raise ValueError(f"unknown model: {MODEL_TYPE}")


def main():
    X_train, X_test, y_train, y_test = load_data(SEED, TEST_SIZE)   # TODO(2)
    model = build_model()                                           # TODO(3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"model={MODEL_TYPE} accuracy={acc:.4f} f1={f1:.4f}")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    # TODO(5): log params / metrics / artifact to MLflow here


if __name__ == "__main__":
    main()
