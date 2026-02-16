"""Model training script for phishing detection."""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from src.data import preprocessing
from src.utils import ensure_directory


RESULTS_DIR = Path("results")
MODEL_PATH = RESULTS_DIR / "model.joblib"


def _extract_training_split(preprocessed_data):
    """Extract the training split from common preprocessing return formats."""
    if isinstance(preprocessed_data, dict):
        return preprocessed_data["X_train"], preprocessed_data["y_train"]

    if isinstance(preprocessed_data, (tuple, list)):
        if len(preprocessed_data) >= 2:
            return preprocessed_data[0], preprocessed_data[1]
        raise ValueError("preprocessing() must return at least X_train and y_train.")

    raise TypeError(
        "Unsupported output from preprocessing(). Expected dict, tuple, or list."
    )


def train_model():
    """Train and persist a Logistic Regression model."""
    preprocessed_data = preprocessing()
    X_train, y_train = _extract_training_split(preprocessed_data)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    ensure_directory(str(RESULTS_DIR))
    joblib.dump(model, MODEL_PATH)

    print(f"Training complete. Model saved to {MODEL_PATH}.")


if __name__ == "__main__":
    train_model()
