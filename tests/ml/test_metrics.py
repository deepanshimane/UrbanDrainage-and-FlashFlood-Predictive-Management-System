"""Tests for B2 - Score card metrics.

Run with: .venv\\Scripts\\pytest tests/ml/test_metrics.py -v
"""

import numpy as np
import pytest

from src.ml.metrics import (
    mae_cm,
    rmse_cm,
    binary_metrics,
    evaluate_lead_times,
    evaluate_flooded_only,
    FLOOD_THRESH,
)


def test_perfect_prediction():
    """When prediction perfectly matches ground truth, F1=1.0, IoU=1.0, MAE=0.0."""
    y_true = np.array([0.0, 0.05, 0.15, 0.50], dtype=np.float32)
    y_pred = np.array([0.0, 0.05, 0.15, 0.50], dtype=np.float32)

    res = binary_metrics(y_pred, y_true, threshold=FLOOD_THRESH)
    assert res["f1"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["iou"] == 1.0
    assert mae_cm(y_pred, y_true) == 0.0
    assert rmse_cm(y_pred, y_true) == 0.0


def test_all_zero_prediction():
    """When ground truth has floods but model predicts all zeros, recall=0.0, F1=0.0."""
    y_true = np.array([0.0, 0.20, 0.40], dtype=np.float32)
    y_pred = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    res = binary_metrics(y_pred, y_true, threshold=FLOOD_THRESH)
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0
    assert res["iou"] == 0.0


def test_all_dry_perfect():
    """When neither ground truth nor prediction has any floods."""
    y_true = np.array([0.0, 0.02, 0.05], dtype=np.float32)
    y_pred = np.array([0.0, 0.01, 0.04], dtype=np.float32)

    res = binary_metrics(y_pred, y_true, threshold=FLOOD_THRESH)
    assert res["f1"] == 1.0 or res["f1"] == 0.0  # clean edge case handled
    assert res["fp"] == 0
    assert res["fn"] == 0


def test_units_conversion_cm():
    """0.10 metres error must equal 10.0 cm."""
    y_true = np.array([0.0], dtype=np.float32)
    y_pred = np.array([0.10], dtype=np.float32)

    assert abs(mae_cm(y_pred, y_true) - 10.0) < 1e-4
    assert abs(rmse_cm(y_pred, y_true) - 10.0) < 1e-4


def test_evaluate_lead_times_structure():
    """evaluate_lead_times must return keys '1h', '2h', '3h' with all metrics."""
    # shape [B=2, T=12, N=5]
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0.0, 0.3, size=(2, 12, 5)).astype(np.float32)
    y_pred = y_true + rng.normal(0, 0.02, size=(2, 12, 5)).astype(np.float32)
    y_pred = np.clip(y_pred, 0, None)

    res = evaluate_lead_times(y_pred, y_true)
    for lead in ["1h", "2h", "3h"]:
        assert lead in res
        for m in ["mae_cm", "rmse_cm", "precision", "recall", "f1", "iou"]:
            assert m in res[lead]
            assert isinstance(res[lead][m], float)


def test_evaluate_flooded_only():
    """evaluate_flooded_only isolates samples that contain true flooding."""
    # 2 samples: sample 0 has no flood, sample 1 has a flood at 3h
    y_true = np.zeros((2, 12, 3), dtype=np.float32)
    y_true[1, 11, 0] = 0.50  # 3h flooded
    y_pred = np.zeros((2, 12, 3), dtype=np.float32)
    y_pred[1, 11, 0] = 0.50

    res = evaluate_flooded_only(y_pred, y_true)
    assert res["3h"]["samples_with_flood"] == 1
    assert res["3h"]["f1"] == 1.0
