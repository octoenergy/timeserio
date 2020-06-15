import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import List, Union

from .. import ini
from .utils import _as_list_of_str, CallableMixin

import holidays

DAYS_IN_YEAR = 365
HOURS_IN_DAY = 24
MINUTES_IN_HOUR = 60
SECONDS_IN_MINUTE = 60
MINUTES_IN_DAY = MINUTES_IN_HOUR * HOURS_IN_DAY

PEAK_INTERVAL = ('16:00', '19:00')
DAY_INTERVAL = ('06:00', '23:00')
MORNING_INTERVAL = ('5:00', '10:00')


def get_fractional_hour_from_series(series: pd.Series) -> pd.Series:
    """
    Return fractional hour in range 0-24, e.g. 12h30m --> 12.5.

    Accurate to 1 minute.
    """
    hour = series.dt.hour
    minute = series.dt.minute
    return hour + minute / MINUTES_IN_HOUR


def get_fractional_day_from_series(series: pd.Series) -> pd.Series:
    """
    Return fractional day in range 0-1, e.g. 12h30m --> 0.521.

    Accurate to 1 minute
    """
    fractional_hours = get_fractional_hour_from_series(series)
    return fractional_hours / HOURS_IN_DAY


def get_fractional_year_from_series(series: pd.Series) -> pd.Series:
    """
    Return fractional year in range 0-1.

    Accurate to 1 day
    """
    return (series.dt.dayofyear - 1) / 365


def get_is_holiday_from_series(
    series: pd.Series, country: str = "UnitedKingdom"
) -> pd.Series:
    """Return 1 if day is a public holiday.

    By default, uses UK holidays, but can specify a country by string name in
    `country` arg. See `holidays.list_supported_countries()` for list of
    supported countries.
    """
    years = series.dt.year.unique()
    holiday_dates = holidays.CountryHoliday(country, years=years)
    return series.dt.date.isin(holiday_dates).astype(int)


def get_zero_indexed_month_from_series(series: pd.Series) -> pd.Series:
    """Return months, in the range 0-11."""
    return series.dt.month - 1


def get_is_weekday_from_series(series: pd.Series) -> pd.Series:
    """Return 1 if day is weekday."""
    dow = series.dt.dayofweek
    return dow.isin([0, 1, 2, 3, 4])


def get_time_is_in_interval_from_series(
        series: pd.Series,
        *,
        start_time,
        end_time,
) -> pd.Series:
    """Return if time of day is in a given time interval."""
    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time).time()
    if isinstance(end_time, str):
        end_time = pd.to_datetime(end_time).time()

    times = series.dt.time

    wrap = False
    if start_time > end_time:
        wrap = True
        start_time, end_time = end_time, start_time

    start_condition = times >= start_time
    end_condition = times < end_time
    interval_condition = start_condition & end_condition
    if wrap:
        interval_condition = ~interval_condition

    return interval_condition


def get_is_peak_hour_from_series(series: pd.Series) -> pd.Series:
    """Return if time in peak hour interval."""
    interval = PEAK_INTERVAL
    return get_time_is_in_interval_from_series(series,
                                               start_time=interval[0],
                                               end_time=interval[1])


def get_is_daytime_from_series(series: pd.Series) -> pd.Series:
    """Return if time in daytime interval."""
    interval = DAY_INTERVAL
    return get_time_is_in_interval_from_series(series,
                                               start_time=interval[0],
                                               end_time=interval[1])


def get_is_morning_peak_from_series(series: pd.Series) -> pd.Series:
    """Return if time in morning peak interval."""
    interval = MORNING_INTERVAL
    return get_time_is_in_interval_from_series(series,
                                               start_time=interval[0],
                                               end_time=interval[1])


def truncate_series(series: pd.Series, truncation_period: str) -> pd.Series:
    return series.dt.to_period(truncation_period).dt.to_timestamp()


SUPPORTED_DATETIME_ATTRS = [
    'time',
    'hour',
    'month',
    'day',
    'dayofweek',
    'dayofyear',
    'weekday_name',
]

# Custom datetime featurizers such as daylight etc.
CUSTOM_ATTRIBUTES = {
    'fractionalday': get_fractional_day_from_series,
    'fractionalhour': get_fractional_hour_from_series,
    'fractionalyear': get_fractional_year_from_series,
    'month0': get_zero_indexed_month_from_series,
    'is_holiday': get_is_holiday_from_series,
    'is_weekday': get_is_weekday_from_series,
    'is_in_interval': get_time_is_in_interval_from_series,
    'is_peak': get_is_peak_hour_from_series,
    'is_daytime': get_is_daytime_from_series,
    'is_morningpeak': get_is_morning_peak_from_series,
    'dt_truncated': truncate_series
}


class PandasDateTimeFeaturizer(BaseEstimator, TransformerMixin, CallableMixin):
    """Featurize datetime column by adding specified attributes."""

    valid_attributes = (
        SUPPORTED_DATETIME_ATTRS + list(CUSTOM_ATTRIBUTES.keys())
    )

    def __init__(
        self,
        column=ini.Columns.datetime,
        attributes=['month0', 'dayofweek', 'fractionalday'],
        kwargs=None
    ):
        self.column = column
        self.attributes = attributes
        self.kwargs = kwargs

    @property
    def attributes_(self):
        """Check attributes to be featurized."""
        return _as_list_of_str(self.attributes)

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Input must be a DataFrame.')
        df = df.copy()
        column = df[self.column]
        if isinstance(column, pd.DataFrame):  # hierarchical index
            column_df = df[[self.column]].stack()
            new_features_df = self.transform(column_df).unstack()
            new_features_df = new_features_df[self.attributes_]
            df = pd.concat([df, new_features_df], axis=1)
            return df
        for attr in self.attributes_:
            if attr in SUPPORTED_DATETIME_ATTRS:
                df[attr] = column.dt.__getattribute__(attr)
            elif attr in CUSTOM_ATTRIBUTES:
                df[attr] = CUSTOM_ATTRIBUTES[attr](
                    column, **self.kwargs if self.kwargs else {}
                )
            else:
                raise KeyError(
                    f'Unknown attribute "{attr}"; '
                    'see `PandasDateTimeFeaturizer.valid_attributes`'
                )
        return df

    @property
    def required_columns(self):
        return set([self.column])

    def transformed_columns(self, input_columns):
        input_columns = set(_as_list_of_str(input_columns))
        if not self.required_columns <= input_columns:
            raise ValueError(f'Required columns are {self.required_columns}')
        return input_columns | set(self.attributes_)


class _BaseLagFeaturizer(BaseEstimator, TransformerMixin, CallableMixin):
    datetime_column: str
    columns: Union[str, List[str]]
    lags: List
    refit: bool
    duplicate_agg: str

    def __init__(
        self,
        datetime_column,
        columns,
        lags,
        refit = True,
        duplicate_agg = "raise"
    ):
        self.datetime_column = datetime_column
        self.columns = columns
        self.lags = lags
        self.refit = refit
        self.duplicate_agg = duplicate_agg

    def fit(self, df, y=None, **fit_params):
        if hasattr(self, 'df_') and not self.refit:
            return self
        columns = _as_list_of_str(self.columns)
        self.df_ = df.set_index(self.datetime_column)[columns]
        if self.duplicate_agg == 'raise':
            if any(self.df_.index.duplicated()):
                raise ValueError(
                    "Input dataframe contains duplicate entries "
                    "with the same %s", self.datetime_column
                )
        else:
            self.df_ = self.df_.groupby(level=0).agg(self.duplicate_agg)
        return self

    @property
    def required_columns(self):
        columns = [self.datetime_column] + _as_list_of_str(self.columns)
        return set(columns)

    def transformed_columns(self, input_columns):
        input_columns = set(_as_list_of_str(input_columns))
        if not self.required_columns <= input_columns:
            raise ValueError(f'Required columns are {self.required_columns}')
        lags = _as_list_of_str(self.lags)
        columns = _as_list_of_str(self.columns)
        new_columns = [f"{col}_{lag}" for lag in lags for col in columns]
        return input_columns | set(new_columns)


class LagFeaturizer(_BaseLagFeaturizer):
    """Add lag features to a DataFrame.

    Note that None will be inserted where no historic data is available.

    Parameters:
        datetime_column: str
        columns: str or List[str]
            Feature column to lag according to datetime_column.
            See pandas.DataFrame.groupby
        lags: List[DateOffset, tseries.offsets, timedelta, or str]
            List of lags to apply.
            See pandas.DataFrame.shift
        refit: bool, default True
            If set to False, a fitted LagFeaturizer will not be refitted
            on subsequent calls to `fit()`
        duplicate_agg: str, default 'raise'
            Aggregation functions to apply to values for duplicate datetimes.
            By default, an error is raised if duplicates
            See pandas.DataFrame.groupby().aggregate()

    This Transformer is stateful.

    """

    def transform(self, df):
        check_is_fitted(self, 'df_')
        lags = _as_list_of_str(self.lags)
        df = df.copy().set_index(self.datetime_column)
        for lag in lags:
            lag_df = self.df_.shift(freq=lag).add_suffix(f"_{lag}")
            df = pd.merge(
                df, lag_df,
                how="left", left_index=True, right_index=True,
                suffixes=("", "")
            )
        return df.reset_index()


class RollingMeanFeaturizer(_BaseLagFeaturizer):
    """Add rolling mean features to a DataFrame.

        Note that None will be inserted where no historic data is available.

    Parameters:
        datetime_column: str
        columns: str or List[str]
            Feature column to lag according to datetime_column.
            See pandas.DataFrame.groupby
        windows: List[int or offset]
            List of moving window sizes.
            See pandas.DataFrame.rolling
        min_periods: int, default ``None``
            Minimum number of observations in window required to have a value
        center: bool, default False
            Set the labels at the center of the window
        win_type: str, default ``None``
            Provide a window type. If ``None``, all points are
            evenly weighted
        closed: str, default ``None``
            Make the interval closed on the 'right', 'left', 'both'
            or 'neither' endpoints.
        refit: bool, default True
            If set to False, a fitted LagFeaturizer will not be refitted
            on subsequent calls to `fit()`
        duplicate_agg: str, default 'raise'
            Aggregation functions to apply to values for duplicate datetimes.
            By default, an error is raised if duplicates
            See pandas.DataFrame.groupby().aggregate()

    This Transformer is stateful.

    """

    def __init__(
        self,
        datetime_column: str,
        columns: Union[str, List[str]],
        windows: List,
        min_periods: int = None,
        center: bool = False,
        win_type: str = None,
        closed: str = None,
        refit: bool = True,
        duplicate_agg: str = 'raise'
    ):
        super().__init__(
            datetime_column,
            columns,
            windows,
            refit,
            duplicate_agg
        )
        self.min_periods = min_periods
        self.center = center
        self.win_type = win_type
        self.closed = closed

    def transform(self, df):
        check_is_fitted(self, 'df_')
        windows = _as_list_of_str(self.lags)
        df = df.copy().set_index(self.datetime_column)
        for window in windows:
            window_df = self.df_.rolling(
                window=window,
                min_periods=self.min_periods,
                center=self.center,
                win_type=self.win_type,
                closed=self.closed
            ).mean().add_suffix(f"_{window}")
            df = pd.merge(
                df, window_df,
                how="left", left_index=True, right_index=True,
                suffixes=("", "")
            )
        return df.reset_index()
