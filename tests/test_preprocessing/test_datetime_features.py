import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
import datetime
from pandas.api.types import is_numeric_dtype

import timeserio.ini as ini
from timeserio.data.mock import mock_fit_data
from timeserio.preprocessing import PandasDateTimeFeaturizer
from timeserio.preprocessing.datetime import (
    get_fractional_day_from_series, get_fractional_hour_from_series,
    get_fractional_year_from_series, truncate_series,
    get_zero_indexed_month_from_series, get_time_is_in_interval_from_series,
    get_is_holiday_from_series
)


datetime_column = ini.Columns.datetime
seq_column = f'seq_{ini.Columns.datetime}'
usage_column = ini.Columns.target


@pytest.fixture
def df():
    return mock_fit_data(start_date=datetime.datetime(2017, 1, 1, 1, 0))


@pytest.fixture
def featurizer():
    return PandasDateTimeFeaturizer()


def test_get_fractional_hour_from_series():
    series = pd.Series(
        pd.date_range(start='2000-01-01', freq='0.5H', periods=48)
    )
    fractionalhour = get_fractional_hour_from_series(series)
    expected = pd.Series(np.linspace(0, 23.5, 48))
    pdt.assert_series_equal(fractionalhour, expected)


def test_get_fractional_day_from_series():
    series = pd.Series(pd.date_range(start='2000-01-01', freq='6H', periods=5))
    fractional_day = get_fractional_day_from_series(series)
    expected = pd.Series([0, 0.25, 0.5, 0.75, 0])
    pdt.assert_series_equal(fractional_day, expected)


def test_get_fractional_year_from_series():
    series = pd.Series(
        pd.date_range(start='2000-01-01', freq='31D', periods=5)
    )
    fractional_year = get_fractional_year_from_series(series)
    expected = pd.Series([0, 1, 2, 3, 4]) * 31 / 365.
    pdt.assert_series_equal(fractional_year, expected)


def test_get_is_holiday_from_series():
    series = pd.Series(pd.date_range(start='2000-01-01', freq='D', periods=5))
    is_holiday = get_is_holiday_from_series(series)
    expected = pd.Series([1, 1, 1, 1, 0])
    pdt.assert_series_equal(is_holiday, expected)


def test_get_zero_indexed_month_from_series():
    series = pd.Series(
        pd.date_range(start='2000-01-01', freq='1M', periods=12)
    )
    month0 = get_zero_indexed_month_from_series(series)
    expected = pd.Series(range(12))
    pdt.assert_series_equal(month0, expected)


@pytest.mark.parametrize(
    'series_data, truncation_period, expected_data',
    [
        ([pd.Timestamp(2019, 1, 1, 1, 9)], 'H', [pd.Timestamp(2019, 1, 1, 1)]),
        ([pd.Timestamp(2019, 1, 2, 1)], 'd', [pd.Timestamp(2019, 1, 2)]),
        ([pd.Timestamp(2019, 1, 1)], 'W', [pd.Timestamp(2018, 12, 31)]),
        ([pd.Timestamp(2019, 1, 1)], 'W-FRI', [pd.Timestamp(2018, 12, 29)]),
        ([pd.Timestamp(2019, 1, 1)], 'W-TUE', [pd.Timestamp(2018, 12, 26)]),
        ([pd.Timestamp(2019, 2, 8)], 'm', [pd.Timestamp(2019, 2, 1)]),
        ([pd.Timestamp(2019, 3, 4)], 'Y', [pd.Timestamp(2019, 1, 1)]),
        (
            [pd.Timestamp(2019, 1, 1, 1, 30), pd.Timestamp(2019, 1, 1, 2, 30)],
            'H',
            [pd.Timestamp(2019, 1, 1, 1), pd.Timestamp(2019, 1, 1, 2)],
        ),
    ]
)
def test_truncate_series(series_data, truncation_period, expected_data):
    out = truncate_series(pd.Series(series_data), truncation_period)
    expected = pd.Series(expected_data)

    pdt.assert_series_equal(out, expected)


def test_set_get_params(featurizer):
    # FixMe: move to generic test_transformer or sth (IG)
    featurizer.set_params(column='wrong_column')
    params = featurizer.get_params()
    assert 'attributes' in params
    assert 'column' in params
    assert params['column'] == 'wrong_column'
    assert params['column'] == featurizer.column
    assert params['attributes'] == featurizer.attributes


def test_with_unknown_attribute(df, featurizer):
    featurizer.set_params(attributes='unknown_attribute')
    with pytest.raises(KeyError):
        featurizer.transform(df)


def test_with_unknown_column(df, featurizer):
    featurizer.set_params(column='unknown_column')
    with pytest.raises(KeyError):
        featurizer.transform(df)


def test_with_non_dt_column(df, featurizer):
    featurizer.set_params(column=ini.Columns.target)
    with pytest.raises(AttributeError):
        featurizer.transform(df)


def test_featurizer(df, featurizer):
    df = featurizer.transform(df)
    assert len(featurizer.attributes)
    for attr in featurizer.attributes:
        assert attr in df
        assert is_numeric_dtype(df[attr])


def test_featurizer_callable(df, featurizer):
    df = featurizer(df)
    assert len(featurizer.attributes)
    for attr in featurizer.attributes:
        assert attr in df
        assert is_numeric_dtype(df[attr])


def test_seq_datetime(df, featurizer):
    featurizer.set_params(
        column=seq_column,
        attributes=['dayofweek', 'fractionalhour']
    )
    df = featurizer.transform(df)
    assert 'dayofweek' in df
    assert 'fractionalhour' in df
    assert df['dayofweek'].shape == df[seq_column].shape
    assert df['fractionalhour'].shape == df[seq_column].shape
    npt.assert_array_equal(
        df['fractionalhour', 0][1:], df['fractionalhour', 1][:-1]
    )


@pytest.mark.parametrize('periods', [48, 96])
def test_get_time_is_in_interval_from_series(periods):
    df = mock_fit_data(
        start_date=datetime.datetime(2017, 1, 1, 0, 0), periods=periods
    )
    is_peak = get_time_is_in_interval_from_series(
        start_time='17:00', end_time='19:00', series=df[datetime_column]
    )
    not_peak = get_time_is_in_interval_from_series(
        start_time='19:00', end_time='17:00', series=df[datetime_column]
    )
    assert len(is_peak) == periods
    assert np.sum(not_peak) == 44 * len(df) / 48


@pytest.mark.parametrize('periods', [48, 96])
def test_featurize_is_in_interval(featurizer, periods):
    df = mock_fit_data(
        start_date=datetime.datetime(2017, 1, 1, 0, 0), periods=periods
    )
    featurizer.set_params(
        column=datetime_column,
        attributes='is_in_interval',
        kwargs={
            'start_time': '17:00',
            'end_time': '19:00',
        }
    )
    df = featurizer.transform(df)
    assert np.sum(df['is_in_interval']) == 4 * len(df) / 48


@pytest.mark.parametrize('periods', [48, 96])
def test_get_is_morning_peak_from_series(featurizer, periods):
    df = mock_fit_data(
        start_date=datetime.datetime(2017, 1, 1, 0, 0), periods=periods
    )
    featurizer.set_params(
        column=datetime_column,
        attributes=['is_peak', 'is_daytime', 'is_morningpeak']
    )
    df = featurizer.transform(df)
    assert 'is_peak' in df
    assert 'is_daytime' in df
    assert 'is_morningpeak' in df
    assert np.sum(df['is_peak']) == 6 * len(df) / 48
    assert np.sum(df['is_daytime']) == 34 * len(df) / 48
    assert np.sum(df['is_morningpeak']) == 10 * len(df) / 48


@pytest.mark.parametrize(
    'transformer, required_columns',
    [
        (
            PandasDateTimeFeaturizer(column=datetime_column),
            {datetime_column}
        )
    ]
)
def test_required_columns(transformer, required_columns):
    assert hasattr(transformer, 'required_columns')
    assert transformer.required_columns == required_columns


@pytest.mark.parametrize(
    'transformer, transformed_columns', [
        (
            PandasDateTimeFeaturizer(
                column=datetime_column,
                attributes=['hour']
            ),
            {datetime_column, 'hour'}
        )
    ]
)
def test_transformed_columns_from_required(transformer, transformed_columns):
    output_columns = transformer.transformed_columns(
        list(transformer.required_columns)
    )
    assert output_columns == transformed_columns


@pytest.mark.parametrize(
    'transformer, transformed_columns, input_columns', [
        (
            PandasDateTimeFeaturizer(
                column=datetime_column,
                attributes=['hour']
            ),
            {datetime_column, 'hour'}, {datetime_column}
        ),
        (
            PandasDateTimeFeaturizer(
                column=datetime_column,
                attributes=['hour']
            ),
            {usage_column, datetime_column, 'hour'},
            {usage_column, datetime_column}
        )
    ]
)
def test_transformed_columns(transformer, transformed_columns, input_columns):
    output_columns = transformer.transformed_columns(input_columns)
    assert output_columns == transformed_columns


def test_invalid_columns():
    transformer = PandasDateTimeFeaturizer(column=datetime_column)
    with pytest.raises(ValueError):
        transformer.transformed_columns(usage_column)
