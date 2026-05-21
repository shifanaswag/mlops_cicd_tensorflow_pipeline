# tests/test_data.py

import numpy as np
from sklearn.datasets import load_iris

def test_iris_data_shape():
    data = load_iris()

    X = data.data
    y = data.target

    # Check feature shape
    assert X.shape == (150, 4)

    # Check labels
    assert len(np.unique(y)) == 3


def test_no_missing_values():
    data = load_iris()
    X = data.data

    assert not np.isnan(X).any()