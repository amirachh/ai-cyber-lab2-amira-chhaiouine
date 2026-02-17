"""Model evaluation for phishing detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src import data
from src.utils import save_json, save_plot


RESULTS_DIR = Path("results")
MODEL_PATH = RESULTS_DIR / "model.joblib"
METRICS_PATH = RESULTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"


def _unpack_preprocessing_output(output: Any) -> Tuple[Any, Any, Any, Any]:
    """Accept tuple/list or dict output from preprocessing()."""
    if isinstance(output, dict):
        required_keys = ("X_train", "X_test", "y_train", "y_test")
        missing = [key for key in required_keys if key not in output]
        if missing:
            raise KeyError(f"preprocessing() output dict is missing keys: {missing}")
        return output["X_train"], output["X_test"], output["y_train"], output["y_test"]

    if isinstance(output, (tuple, list)) and len(output) == 4:
        return output[0], output[1], output[2], output[3]

    raise TypeError(
        "preprocessing() must return either a dict with keys "
        "X_train/X_test/y_train/y_test or a 4-item tuple/list."
    )


def _resolve_pos_label(y_true: Iterable[Any]) -> Any:
    """Choose a positive label for binary metrics when labels are not {0, 1}."""
    unique_labels = sorted(set(y_true))
    if len(unique_labels) != 2:
        raise ValueError(f"Expected binary labels, got: {unique_labels}")

    if set(unique_labels) == {0, 1}:
        return 1

    return unique_labels[-1]


def evaluate() -> Dict[str, float]:
    """Load model, evaluate on test data, and save metrics/artifacts."""
    # Ensure results directory exists.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    preprocessing_fn = getattr(data, "preprocessing", data.prepare_splits)
    X_train, X_test, y_train, y_test = _unpack_preprocessing_output(preprocessing_fn())

    y_pred = model.predict(X_test)
    pos_label = _resolve_pos_label(y_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary", pos_label=pos_label),
        "recall": recall_score(y_test, y_pred, average="binary", pos_label=pos_label),
        "f1_score": f1_score(y_test, y_pred, average="binary", pos_label=pos_label),
    }

    save_json(metrics, METRICS_PATH)

    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    save_plot(plt, CONFUSION_MATRIX_PATH)
    plt.close()

    print(
        "Evaluation complete. Saved metrics to "
        f"{METRICS_PATH} and confusion matrix to {CONFUSION_MATRIX_PATH}."
    )

    return metrics


if __name__ == "__main__":
    evaluate()
