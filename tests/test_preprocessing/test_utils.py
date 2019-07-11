import pytest

from timeserio.preprocessing.utils import (
    _as_list_of_str,
    IdentityRegressor
)


@pytest.mark.parametrize(
    'columns_in, columns_out', [
        ([], []),
        (None, []),
        ('col', ['col']),
        (['col'], ['col']),
        (['col1', 'col2'], ['col1', 'col2']),
    ]
)
def test_as_list_of_str(columns_in, columns_out):
    assert _as_list_of_str(columns_in) == columns_out


identity_regressor = IdentityRegressor()

X = [[0], [1], [None]]


def test_identity_regressor_fit():
    assert identity_regressor.fit(X) is identity_regressor


def test_identity_regressor_predict():
    assert identity_regressor.predict(X) == X
