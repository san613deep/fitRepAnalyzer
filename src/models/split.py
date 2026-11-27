"""Leakage-aware dataset splitting for participant-based sensor classification."""
from __future__ import annotations

from sklearn.model_selection import GroupShuffleSplit


NON_FEATURE_COLUMNS = {"label", "participant", "category", "set"}


def split_by_participant(data, test_size: float = 0.25, random_state: int = 42):
    """Split rows while ensuring no participant appears in both partitions.

    The participant identifier is used only as a grouping variable and is never
    returned as a model feature. The function returns X_train, X_test, y_train,
    and y_test, preserving the original row indices.
    """
    required = {"label", "participant"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")

    feature_columns = [
        column for column in data.columns if column not in NON_FEATURE_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No model feature columns were found")

    X = data[feature_columns]
    y = data["label"]
    groups = data["participant"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_indices, test_indices = next(splitter.split(X, y, groups=groups))

    return (
        X.iloc[train_indices].copy(),
        X.iloc[test_indices].copy(),
        y.iloc[train_indices].copy(),
        y.iloc[test_indices].copy(),
    )
