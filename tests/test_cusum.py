import numpy as np
from methods.statistical.cusum import CUSUM

def test_cusum_output_shape():
    arr = np.random.randn(10, 100)
    mask = CUSUM(arr, k=0.3, h=0.5, output=False)
    assert mask.shape == arr.shape

def test_cusum_binary_mask():
    arr = np.random.randn(10, 100)
    mask = CUSUM(arr, k=0.3, h=0.5, output=False)
    assert set(np.unique(mask)).issubset({0, 1})

def test_cusum_output_true():
    arr = np.random.randn(10, 100)
    C = CUSUM(arr, k=0.3, h=0.5, output=True)
    assert C.shape == arr.shape