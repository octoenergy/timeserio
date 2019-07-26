import pytest

import pandas as pd

from timeserio.pipeline.pipeline import _y_as_string


@pytest.fixture
def df():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [1, 4, 9],
    })
    return df


def test_fit_decorator(df):
    """Test we can pass the target column by name."""
    @_y_as_string
    def fit(df, y):
        return df["x"] + y

    fit1 = fit(df, df["y"])
    fit2 = fit(df, "y")

    pd.testing.assert_series_equal(fit1, fit2)
