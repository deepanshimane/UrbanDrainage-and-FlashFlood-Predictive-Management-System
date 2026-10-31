"""B2 - Score card (how wrong are our flood predictions?).

All flood depths are in metres in the dataset.
We convert errors to CENTIMETRES (cm) for easy human reading.
A node is considered FLOODED if water depth > 0.10 m (10 cm).

Lead times:
  +1h -> step 4 (index 3)
  +2h -> step 8 (index 7)
  +3h -> step 12 (index 11)
"""

import numpy as np

FLOOD_THRESH = 0.10  # metres (10 cm)
LEAD_STEPS = {
    "1h": 3,   # 4th step (0-indexed: 3)
    "2h": 7,   # 8th step (0-indexed: 7)
    "3h": 11,  # 12th step (0-indexed: 11)
}


def mae_cm(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Absolute Error in centimetres (cm)."""
    return float(np.mean(np.abs(y_pred - y_true)) * 100.0)


def rmse_cm(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root Mean Squared Error in centimetres (cm)."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)) * 100.0)


def binary_metrics(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = FLOOD_THRESH):
    """Compute Precision, Recall, F1, and IoU for flood / no-flood classification.

    Parameters
    ----------
    y_pred, y_true : np.ndarray
        Depths in metres. Can be any shape, will be flattened.
    threshold : float
        Threshold above which a node is counted as flooded (default 0.10m).

    Returns
    -------
    dict with precision, recall, f1, iou, tp, fp, fn, tn
    """
    pred_flood = (y_pred > threshold).astype(bool).ravel()
    true_flood = (y_true > threshold).astype(bool).ravel()

    tp = int(np.sum(pred_flood & true_flood))
    fp = int(np.sum(pred_flood & ~true_flood))
    fn = int(np.sum(~pred_flood & true_flood))
    tn = int(np.sum(~pred_flood & ~true_flood))

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if (tp == 0 and fn == 0) else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if (tp == 0 and fp == 0) else 0.0)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if fn == 0 and fp == 0 else 0.0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate_lead_times(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = FLOOD_THRESH):
    """Compute score card across all lead times (+1h, +2h, +3h).

    Parameters
    ----------
    y_pred, y_true : np.ndarray
        Shape [..., 12, N] or [B, 12, N] where axis -2 is the 12 forecast horizon steps.

    Returns
    -------
    dict of metrics keyed by '1h', '2h', '3h'.
    """
    results = {}
    for label, step_idx in LEAD_STEPS.items():
        # slice at step_idx
        p_step = y_pred[..., step_idx, :]
        t_step = y_true[..., step_idx, :]

        clf = binary_metrics(p_step, t_step, threshold=threshold)
        results[label] = {
            "mae_cm": round(mae_cm(p_step, t_step), 2),
            "rmse_cm": round(rmse_cm(p_step, t_step), 2),
            "precision": round(clf["precision"], 4),
            "recall": round(clf["recall"], 4),
            "f1": round(clf["f1"], 4),
            "iou": round(clf["iou"], 4),
        }

    return results


def evaluate_flooded_only(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = FLOOD_THRESH):
    """Compute F1 score restricted only to windows/samples where at least one node flooded.

    Because floods are rare events, this prevents 'sunny day' samples from
    skewing the score.
    """
    # shape [B, 12, N]
    results = {}
    for label, step_idx in LEAD_STEPS.items():
        p_step = y_pred[..., step_idx, :]  # [B, N]
        t_step = y_true[..., step_idx, :]  # [B, N]

        # find samples where at least one node in ground truth exceeds threshold
        has_flood = (t_step > threshold).any(axis=-1)
        if not np.any(has_flood):
            results[label] = {"f1": 0.0, "samples_with_flood": 0}
            continue

        clf = binary_metrics(p_step[has_flood], t_step[has_flood], threshold=threshold)
        results[label] = {
            "f1": round(clf["f1"], 4),
            "precision": round(clf["precision"], 4),
            "recall": round(clf["recall"], 4),
            "samples_with_flood": int(np.sum(has_flood)),
        }

    return results
