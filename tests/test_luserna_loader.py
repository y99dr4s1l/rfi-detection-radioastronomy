import os
import numpy as np
import pandas as pd
import pytest
import tempfile
from data.luserna_loader import load_luserna, load_luserna_truth


@pytest.fixture
def synthetic_h5(tmp_path):
    """Crea file h5 sintetici per il test."""
    data = pd.DataFrame(
        np.random.randn(100, 50),
        index=np.arange(100, dtype=np.int64) * 1_000_000
    )
    spec_path = tmp_path / 'PG01.h5'
    data.to_hdf(spec_path, key='data')

    truth = pd.DataFrame(
        np.zeros((100, 50), dtype=bool),
        index=data.index
    )
    truth_path = tmp_path / 'truth_flag.h5'
    truth.to_hdf(truth_path, key='bool_data')

    return tmp_path


def test_load_luserna_shape(synthetic_h5):
    data = load_luserna('PG01', path=str(synthetic_h5))
    assert data.shape == (100, 50)


def test_load_luserna_index_normalized(synthetic_h5):
    data = load_luserna('PG01', path=str(synthetic_h5))
    assert data.index[0] == pytest.approx(0.0)


def test_load_luserna_powers(synthetic_h5):
    data_db = load_luserna('PG01', path=str(synthetic_h5), powers=False)
    data_pw = load_luserna('PG01', path=str(synthetic_h5), powers=True)
    assert not np.allclose(data_db.values, data_pw.values)


def test_load_luserna_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_luserna('NONEXISTENT', path=str(tmp_path))


def test_load_luserna_truth_shape(synthetic_h5):
    truth = load_luserna_truth(path=str(synthetic_h5))
    assert truth.shape == (100, 50)


def test_load_luserna_truth_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_luserna_truth(path=str(tmp_path))