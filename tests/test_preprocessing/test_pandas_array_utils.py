import pytest

import numpy as np
import numpy.testing as npt

import timeserio.ini as ini
from timeserio.preprocessing.pandas import array_to_dataframe
from timeserio.data.mock import mock_fit_data


N_TIMES = 3
N_IDS = 2
N_ROWS = N_IDS * N_TIMES
N_DIMS = 2


@pytest.fixture
def df():
    return mock_fit_data(periods=N_TIMES, ids=np.arange(N_IDS))


def arr1d():
    return np.random.rand(N_ROWS)


def arr2d():
    return np.random.rand(N_ROWS, N_DIMS)


@pytest.mark.parametrize('arr, n_dims', [
    (arr1d(), 1),
    (arr2d(), 2),
])
def test_create_df_from_array(arr, n_dims):
    df = array_to_dataframe(arr, 'col')
    npt.assert_array_equal(df.values, arr.reshape(-1, n_dims))
    assert df.columns.levels[0] == ['col']
    assert all(df.columns.levels[1] == list(range(n_dims)))


@pytest.mark.parametrize('arr, n_dims', [
    (arr1d(), 1),
    (arr2d(), 2),
])
def test_insert_into_flat_idx_df(df, arr, n_dims):
    df = df[[ini.Columns.target]]
    df = array_to_dataframe(arr, 'col', df=df)
    df = df[[ini.Columns.target, 'col']]
    assert df.values.shape == (N_ROWS, n_dims + 1)


@pytest.mark.parametrize('arr, n_dims', [
    (arr1d(), 1),
    (arr2d(), 2),
])
def test_insert_into_multi_idx_df(df, arr, n_dims):
    df = df[['embedding']]
    df = array_to_dataframe(arr, 'col', df=df)
    df = df[['embedding', 'col']]
    assert df.values.shape == (N_ROWS, n_dims + 2)
