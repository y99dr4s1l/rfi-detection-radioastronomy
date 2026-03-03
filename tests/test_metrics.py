import numpy as np
import pytest
from evaluation.metrics import compute_metrics


def test_compute_metrics_perfect_prediction():
    mask_true = np.array([True, True, False, False])
    mask_pred = np.array([True, True, False, False])
    result = compute_metrics(mask_true, mask_pred)
    assert result['precision'] == pytest.approx(1.0)
    assert result['recall'] == pytest.approx(1.0)
    assert result['f1'] == pytest.approx(1.0)

def test_compute_metrics_all_wrong():
    mask_true = np.array([True, True, False, False])
    mask_pred = np.array([False, False, True, True])
    result = compute_metrics(mask_true, mask_pred)
    assert result['precision'] == pytest.approx(0.0)
    assert result['recall'] == pytest.approx(0.0)
    assert result['f1'] == pytest.approx(0.0)

def test_compute_metrics_no_positive_pred():
    """Nessuna predizione positiva — precision indefinita, recall zero."""
    mask_true = np.array([True, True, False, False])
    mask_pred = np.array([False, False, False, False])
    result = compute_metrics(mask_true, mask_pred)
    assert result['precision'] == pytest.approx(0.0)
    assert result['recall'] == pytest.approx(0.0)
    assert result['f1'] == pytest.approx(0.0)

def test_compute_metrics_counts():
    mask_true = np.array([True, True, False, False])
    mask_pred = np.array([True, False, True, False])
    result = compute_metrics(mask_true, mask_pred)
    assert result['TP'] == 1
    assert result['FP'] == 1
    assert result['FN'] == 1
    assert result['TN'] == 1

def test_compute_metrics_2d_arrays():
    mask_true = np.zeros((10, 10), dtype=bool)
    mask_pred = np.zeros((10, 10), dtype=bool)
    mask_true[3:6, 3:6] = True
    mask_pred[3:6, 3:6] = True
    result = compute_metrics(mask_true, mask_pred)
    assert result['f1'] == pytest.approx(1.0)