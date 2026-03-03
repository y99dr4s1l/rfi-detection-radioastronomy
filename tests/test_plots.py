import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend non interattivo per i test
import pytest
from visualization.plots import plot_detection_result


def test_plot_detection_result_runs():
    spec = np.random.randn(64, 64).astype(np.float32)
    gt = np.zeros((64, 64), dtype=bool)
    pred = np.zeros((64, 64), dtype=bool)
    plot_detection_result(spec, gt, pred)

def test_plot_detection_result_3d_input():
    """Deve gestire input con dimensione canale (h, w, 1)."""
    spec = np.random.randn(64, 64, 1).astype(np.float32)
    gt = np.zeros((64, 64, 1), dtype=bool)
    pred = np.zeros((64, 64, 1), dtype=bool)
    plot_detection_result(spec, gt, pred)

def test_plot_detection_result_save(tmp_path):
    spec = np.random.randn(64, 64).astype(np.float32)
    gt = np.zeros((64, 64), dtype=bool)
    pred = np.zeros((64, 64), dtype=bool)
    save_path = str(tmp_path / 'test_plot.png')
    plot_detection_result(spec, gt, pred, save_path=save_path)
    import os
    assert os.path.exists(save_path)