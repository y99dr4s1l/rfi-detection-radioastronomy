import numpy as np
import pytest
from methods.statistical.sum_threshold import chi_threshold5, sumthreshold


# --- chi_threshold5 ---

def test_chi_threshold5_keys():
    """I keys devono essere potenze di 2 da 2^1 a 2^i."""
    chi = chi_threshold5(i=4, chi1=2.4, target_value=0)
    assert set(chi.keys()) == {2, 4, 8, 16}

def test_chi_threshold5_decay():
    """I valori devono decrescere all'aumentare della finestra."""
    chi = chi_threshold5(i=4, chi1=2.4, target_value=0)
    values = [chi[k] for k in sorted(chi.keys())]
    assert all(values[i] > values[i+1] for i in range(len(values)-1))

def test_chi_threshold5_target_value_zero():
    """Con target_value=chi1 tutti i valori devono essere zero."""
    chi = chi_threshold5(i=3, chi1=2.4, target_value=2.4)
    for v in chi.values():
        assert v == pytest.approx(0.0)


# --- sumthreshold ---

def test_sumthreshold_output_shape():
    arr = np.random.randn(10, 100).astype(np.float32)
    chi = chi_threshold5(i=3, chi1=2.4, target_value=0)
    mask = sumthreshold(arr, chi, output=False)
    assert mask.shape == arr.shape

def test_sumthreshold_output_bool():
    arr = np.random.randn(10, 100).astype(np.float32)
    chi = chi_threshold5(i=3, chi1=2.4, target_value=0)
    mask = sumthreshold(arr, chi, output=False)
    assert mask.dtype == bool

def test_sumthreshold_output_true_returns_tuple():
    arr = np.random.randn(10, 100).astype(np.float32)
    chi = chi_threshold5(i=3, chi1=2.4, target_value=0)
    result = sumthreshold(arr, chi, output=True)
    assert isinstance(result, tuple)
    assert len(result) == 3

def test_sumthreshold_high_signal_flagged():
    """Un segnale con spike evidenti deve produrre almeno un pixel flaggato."""
    arr = np.zeros((10, 100), dtype=np.float32)
    arr[:, 50] = 100.0
    chi = chi_threshold5(i=3, chi1=2.4, target_value=0)
    mask = sumthreshold(arr, chi, output=False)
    assert mask.sum() > 0

def test_sumthreshold_clean_signal_not_flagged():
    """Un segnale piatto e basso non deve produrre flag."""
    arr = np.zeros((10, 100), dtype=np.float32)
    chi = chi_threshold5(i=3, chi1=2.4, target_value=0)
    mask = sumthreshold(arr, chi, output=False)
    assert mask.sum() == 0