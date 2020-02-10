import datetime as dt

import numpy as np
import pandas as pd
import dask.dataframe as dd

from .. import ini

DEF_N = 48
DEF_EMB_DIM = 2
DEF_SEQ_LENGTH = 3
DEF_FREQ = '0.5H'


def mock_datetime_range(periods=DEF_N, start=None):
    """Sample datetime range with a half-hour period."""
    if not start:
        start = dt.datetime.now()
    return pd.date_range(start=start, freq=DEF_FREQ, periods=periods)


def _single_user_fit_df(
    periods=DEF_N,
    start_date=None,
    id=0,
    embedding_dim=DEF_EMB_DIM,
    seq_length=DEF_SEQ_LENGTH
):
    embeddings = {
        i: np.random.rand(periods)
        for i in range(embedding_dim)
    }
    dt_range = mock_datetime_range(periods, start=0)
    seq_dt = {
        i: dt_range + pd.Timedelta(i / 2, unit='h')
        for i in range(seq_length)
    }
    seq_usage = {
        i: np.random.rand(periods)
        for i in range(seq_length)
    }
    df = pd.concat({
        'embedding': pd.DataFrame(embeddings),
        f'seq_{ini.Columns.datetime}': pd.DataFrame(seq_dt),
        f'seq_{ini.Columns.target}': pd.DataFrame(seq_usage),
    }, axis=1)
    df[ini.Columns.datetime] = mock_datetime_range(periods, start=start_date)
    df['weather_temperature'] = 30 * np.random.rand(periods)
    df[ini.Columns.target] = np.random.rand(periods)
    df[ini.Columns.id] = id
    df['cluster'] = 13
    return df


def mock_fit_data(periods=DEF_N,
                  start_date=None,
                  ids=[0],
                  embedding_dim=DEF_EMB_DIM,
                  seq_length=DEF_SEQ_LENGTH):
    """Create example fit data in the tall DataFrame format."""
    user_dfs = [
        _single_user_fit_df(
            periods=periods,
            start_date=start_date,
            id=id,
            embedding_dim=embedding_dim,
            seq_length=seq_length
        )
        for id in ids
    ]
    df = pd.concat(user_dfs, axis=0)
    df.reset_index(inplace=True, drop=True)
    return df


def mock_dask_fit_data(
    periods=DEF_N,
    start_date=None,
    ids=[0],
    embedding_dim=DEF_EMB_DIM,
    seq_length=DEF_SEQ_LENGTH
):
    """Create example fit data as a dask DataFrame.

    DataFrame is partitioned by ID.
    """
    df = mock_fit_data(
        periods=periods,
        start_date=start_date,
        ids=ids,
        embedding_dim=embedding_dim,
        seq_length=seq_length
    )
    ddf = dd.from_pandas(df, chunksize=periods)
    return ddf


def _single_user_raw_df(
    periods=DEF_N,
    start_date=None,
    id=0,
):
    df = pd.DataFrame({
        ini.Columns.id: id,
        ini.Columns.datetime: mock_datetime_range(periods, start=start_date),
        ini.Columns.target: np.random.rand(periods)
    })
    return df


def mock_raw_data(periods=DEF_N,
                  start_date=None,
                  ids=[0]):
    """Create example raw data in the tall DataFrame format."""
    user_dfs = [
        _single_user_raw_df(
            periods=periods,
            start_date=start_date,
            id=id,
        )
        for id in ids
    ]
    df = pd.concat(user_dfs, axis=0)
    df.reset_index(inplace=True, drop=True)
    return df


def mock_dask_raw_data(
    periods=DEF_N,
    start_date=None,
    ids=[0]
):
    """Create example fit data as a dask DataFrame.

    DataFrame is partitioned by ID.
    """
    df = mock_raw_data(
        periods=periods,
        start_date=start_date,
        ids=ids,
    )
    ddf = dd.from_pandas(df, chunksize=periods)
    return ddf


def mock_predict_data(periods=DEF_N, start_date=None):
    """Create example predict data in the tall DataFrame format."""
    df = mock_fit_data(periods=periods, start_date=start_date)
    return df.drop('usage', axis=1)
