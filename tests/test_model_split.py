"""Tests for leakage-aware participant-level train/test splitting."""
import pandas as pd
import pytest

from src.models.split import split_by_participant


def _sample_data():
    return pd.DataFrame(
        {
            "participant": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "category": ["strength"] * 8,
            "set": [1, 2] * 4,
            "label": ["squat", "bench"] * 4,
            "acc_x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "gyr_y": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
        }
    )


def test_train_and_test_participants_are_disjoint():
    data = _sample_data()
    X_train, X_test, y_train, y_test = split_by_participant(data)

    train_participants = set(data.loc[X_train.index, "participant"])
    test_participants = set(data.loc[X_test.index, "participant"])

    assert train_participants
    assert test_participants
    assert train_participants.isdisjoint(test_participants)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_participant_metadata_is_not_a_model_feature():
    X_train, X_test, _, _ = split_by_participant(_sample_data())

    for frame in (X_train, X_test):
        assert "participant" not in frame.columns
        assert "category" not in frame.columns
        assert "set" not in frame.columns
        assert "label" not in frame.columns


def test_missing_group_column_raises_clear_error():
    with pytest.raises(ValueError, match="participant"):
        split_by_participant(_sample_data().drop(columns=["participant"]))


def test_invalid_test_size_raises_clear_error():
    with pytest.raises(ValueError, match="test_size"):
        split_by_participant(_sample_data(), test_size=1.0)
