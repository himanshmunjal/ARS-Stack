"""
tests/test_serving.py

Unit tests for src/serving/app.py's request vectorization.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.serving.app import _vectorize


def test_vectorize_scales_only_the_columns_the_scaler_was_fit_on():
    """Pre-fix bug: creditcard's scaler is fit on Time/Amount only, but was
    applied to all 30 features, which raised at request time."""
    train = pd.DataFrame({"Time": [0.0, 10.0, 20.0], "Amount": [1.0, 2.0, 3.0]})
    scaler = StandardScaler().fit(train)
    bundle = {"feature_columns": ["Time", "V1", "Amount"], "scaler": scaler}

    row = _vectorize(bundle, {"Time": 10.0, "V1": -1.5, "Amount": 3.0})

    expected_amount = scaler.transform(pd.DataFrame({"Time": [10.0], "Amount": [3.0]}))[0, 1]
    assert row.shape == (1, 3)
    assert row[0, 0] == 0.0          # Time scaled (it's the training mean)
    assert row[0, 1] == -1.5         # V1 untouched
    assert np.isclose(row[0, 2], expected_amount)


def test_vectorize_without_scaler_keeps_column_order():
    bundle = {"feature_columns": ["b", "a"], "scaler": None}
    assert _vectorize(bundle, {"a": 1.0, "b": 2.0}).tolist() == [[2.0, 1.0]]
