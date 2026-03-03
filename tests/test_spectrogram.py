import numpy as np
import pytest
from preprocessing.spectrogram import (
    polynomial_detrend,
    extract_and_split_patches,
    balance_dataset
)
@pytest.fixture
def synthetic_spectrogram():
    # usiamo patch piccole nei test per evitare memory issues
    data = np.random.randn(20, 20).astype(np.float32)
    masks = np.zeros((20, 20), dtype=bool)
    masks[5:8, 5:8] = True
    return data, masks

def test_extract_and_split_patches_shapes(synthetic_spectrogram):
    data, masks = synthetic_spectrogram
    train_d, train_m, test_d, test_m = extract_and_split_patches(
        data, masks, patch_size=(4, 4), train_size=3
    )
    assert train_d.shape[1:] == (4, 4, 1)
    assert train_m.shape[1:] == (4, 4, 1)
    assert train_d.shape[0] == 3

def test_extract_and_split_patches_no_overlap(synthetic_spectrogram):
    data, masks = synthetic_spectrogram
    train_d, _, test_d, _ = extract_and_split_patches(
        data, masks, patch_size=(4, 4), train_size=3
    )
    assert train_d.shape[0] == 3
    assert test_d.shape[0] > 0

def test_extract_and_split_patches_reproducible(synthetic_spectrogram):
    data, masks = synthetic_spectrogram
    result1 = extract_and_split_patches(data, masks, patch_size=(4, 4), random_seed=42)
    result2 = extract_and_split_patches(data, masks, patch_size=(4, 4), random_seed=42)
    np.testing.assert_array_equal(result1[0], result2[0])