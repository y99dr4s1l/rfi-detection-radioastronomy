import numpy as np
import time


def compute_metrics(mask_true: np.ndarray, mask_pred: np.ndarray) -> dict:
    """
    Computes binary classification metrics for RFI detection.

    Args:
        mask_true: Boolean ground truth array of any shape.
        mask_pred: Boolean predicted mask array, same shape as mask_true.

    Returns:
        dict with keys: precision, recall, f1, TP, FP, FN, TN.
    """
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    TP = np.sum(mask_pred & mask_true)
    FP = np.sum(mask_pred & ~mask_true)
    FN = np.sum(~mask_pred & mask_true)
    TN = np.sum(~mask_pred & ~mask_true)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN)
    }