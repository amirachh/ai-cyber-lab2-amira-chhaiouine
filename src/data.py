"""Data loading and preprocessing utilities for phishing detection."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_CANDIDATES = (
    "label",
    "labels",
    "class",
    "target",
    "result",
    "is_phishing",
    "phishing",
    "y",
)


def load_dataset(dataset_path: str = "data/processed/dataset.csv") -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing rows and remove non-numeric columns automatically."""
    cleaned = df.dropna().copy()
    return cleaned.select_dtypes(include=["number", "bool"]).copy()


def identify_label_column(df: pd.DataFrame) -> str:
    """Identify a label column name from common phishing dataset conventions."""
    lower_to_original = {column.lower(): column for column in df.columns}

    for candidate in LABEL_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    for column in df.columns:
        lowered = column.lower()
        if "label" in lowered or "class" in lowered or "result" in lowered or "target" in lowered:
            return column

    raise ValueError(
        "Unable to identify label column. Expected a column like: "
        "label, class, result, target, phishing, or similar."
    )


def to_binary_label(series: pd.Series) -> pd.Series:
    """Convert label values to binary 0/1 representation."""
    values = series.dropna().unique()
    if len(values) != 2:
        raise ValueError(f"Label must be binary. Found {len(values)} unique values: {values}")

    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        sorted_values = sorted(values)
        mapping = {sorted_values[0]: 0, sorted_values[1]: 1}
        return series.map(mapping).astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    canonical_map = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "benign": 0,
        "legitimate": 0,
        "safe": 0,
        "phishing": 1,
        "malicious": 1,
        "fraud": 1,
    }

    if set(normalized.unique()).issubset(canonical_map):
        return normalized.map(canonical_map).astype(int)

    unique_values = sorted(normalized.unique())
    mapping = {unique_values[0]: 0, unique_values[1]: 1}
    return normalized.map(mapping).astype(int)


def prepare_splits(
    dataset_path: str = "data/processed/dataset.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, clean, label-normalize, and split the phishing dataset."""
    df = load_dataset(dataset_path)

    label_column = identify_label_column(df)
    y_raw = df[label_column]
    X_raw = df.drop(columns=[label_column])

    combined = pd.concat([X_raw, y_raw.rename("label")], axis=1).dropna()
    y = to_binary_label(combined["label"])

    X = clean_features(combined.drop(columns=["label"]))
    aligned = X.join(y.rename("label"), how="inner")

    X_final = aligned.drop(columns=["label"])
    y_final = aligned["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y_final,
        test_size=0.2,
        random_state=42,
    )

    return X_train, X_test, y_train, y_test
