import pytest
import datetime

import numpy as np
import numpy.testing as npt
import pandas as pd

import timeserio.ini as ini
from timeserio.data.mock import mock_raw_data, DEF_FREQ
from timeserio.preprocessing import LagFeaturizer


@pytest.fixture
def df():
    return mock_raw_data(start_date=datetime.datetime(2017, 1, 1, 1, 0))


def test_fit_single_lag(df):
    feat = LagFeaturizer(
        datetime_column=ini.Columns.datetime,
        columns=ini.Columns.target,
        lags=DEF_FREQ
    )
    df2 = feat.fit_transform(df)

    values = df[ini.Columns.target].values
    values_transformed = df2[ini.Columns.target].values
    values_lagged = df2[f"{ini.Columns.target}_{DEF_FREQ}"].values

    npt.assert_equal(values, values_transformed)
    npt.assert_equal(values[:-1], values_lagged[1:])
    npt.assert_equal(values_lagged[:1], np.nan)


def test_fit_multiple_lags(df):
    feat = LagFeaturizer(
        datetime_column=ini.Columns.datetime,
        columns=ini.Columns.target,
        lags=['0.5H', '1H']
    )
    df2 = feat.fit_transform(df)

    assert len(df2) == len(df)


def test_transform_from_datetime(df):
    feat = LagFeaturizer(
        datetime_column=ini.Columns.datetime,
        columns=ini.Columns.target,
        lags=DEF_FREQ
    )
    feat.fit(df)
    df2 = df[[ini.Columns.datetime]]
    result = feat.transform(df2)

    assert f"{ini.Columns.target}_{DEF_FREQ}" in result


def test_columns(df):
    feat = LagFeaturizer(
        datetime_column=ini.Columns.datetime,
        columns=ini.Columns.target,
        lags=DEF_FREQ
    )
    assert feat.required_columns == set([
        ini.Columns.datetime, ini.Columns.target
    ])
    assert feat.transformed_columns(feat.required_columns) == set([
        ini.Columns.datetime, ini.Columns.target,
        f"{ini.Columns.target}_{DEF_FREQ}"
    ])


def test_transform_with_duplicates(df):
    """Transform a df with duplicate datetimes."""
    df_test = pd.concat([df, df], ignore_index=True)
    feat = LagFeaturizer(
        datetime_column=ini.Columns.datetime,
        columns=ini.Columns.target,
        lags=DEF_FREQ
    )

    feat.fit(df)
    df_test_transformed = feat.transform(df_test)

    assert len(df_test_transformed) == len(df_test)


def test_fit_with_duplicates_raises(df):
    """Transform a df with duplicate datetimes."""
    df_test = pd.concat([df, df], ignore_index=True)
    feat = LagFeaturizer(
        datetime_column=ini.Columns.datetime,
        columns=ini.Columns.target,
        lags=DEF_FREQ
    )

    with pytest.raises(ValueError):
        _ = feat.fit(df_test)


def test_fit_with_duplicates_with_agg(df):
    """Transform a df with duplicate datetimes."""
    df_test = pd.concat([df, df], ignore_index=True)
    feat = LagFeaturizer(
        datetime_column=ini.Columns.datetime,
        columns=ini.Columns.target,
        lags=DEF_FREQ,
        duplicate_agg='mean'
    )

    df_test_transformed = feat.fit_transform(df_test)

    assert len(feat.df_) == len(df)
    assert len(df_test_transformed) == len(df_test)
