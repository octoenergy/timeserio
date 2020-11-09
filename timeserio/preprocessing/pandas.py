import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .utils import _as_list_of_str


def array_to_dataframe(
    array: np.ndarray, column: str, df=None
) -> pd.DataFrame:
    """Add 1D or 2D array as column in df.

    If no df provided, return a new one.
    """
    if len(array.shape) < 0 or len(array.shape) > 2:
        raise ValueError('Expecting 1D or 2D array')
    if len(array.shape) == 1:
        array = array.reshape(-1, 1)
    n_columns = array.shape[1]
    cidx = pd.MultiIndex.from_arrays(
        [
            [column] * n_columns,
            range(n_columns),
        ]
    )
    df_arr = pd.DataFrame(array, columns=cidx)
    if df is None:
        df = df_arr
    else:
        df = _join_multilevel_dataframes([df, df_arr])
    return df


def _join_multilevel_dataframes(df_list):
    """Concat multiple dataframes.

    Support a combination of 1- and 2-deep indices.
    """
    minx_df = []
    for df in df_list:
        if isinstance(df.columns, pd.MultiIndex):
            minx_df.append(df)
        else:
            df.columns = pd.MultiIndex.from_product([df.columns, ['']])
            minx_df.append(df)
    # Join all dataframes together
    multi_concat = pd.concat(minx_df, axis=1)
    return multi_concat


class PandasColumnSelector(BaseEstimator, TransformerMixin):
    """Select a sub-set of columns from a pandas DataFrame."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df):
        columns = _as_list_of_str(self.columns)
        subframe = df[columns]
        return subframe.copy()

    @property
    def required_columns(self):
        return set(_as_list_of_str(self.columns))

    def transformed_columns(self, input_columns):
        input_columns = set(_as_list_of_str(input_columns))
        if not self.required_columns <= input_columns:
            raise ValueError(f'Required columns are {self.required_columns}')
        return self.required_columns


def _get_column_as_tensor(s: pd.Series):
    """Get every normal or TensorArray column as a 2D array."""
    try:
        return s.tensor.values
    except AttributeError:  # normal column
        return s.values.reshape(-1, 1)


class PandasValueSelector(BaseEstimator, TransformerMixin):
    """Select scalar - or vector-valued feature cols, and return np.array.

    Optionally, cast the resulting arry to dtype.
    """

    def __init__(self, columns=None, dtype=None):
        self.columns = columns
        self.dtype = dtype

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df):
        columns = _as_list_of_str(self.columns)
        any_tensors = any(hasattr(df[col], "tensor") for col in columns)
        if not any_tensors:
            subarray = df[columns].values
        else:  # support a mix of compatible tensors and regular columns
            blocks = [_get_column_as_tensor(df[col]) for col in columns]
            subarray = np.hstack(blocks)
        if self.dtype:
            subarray = subarray.astype(self.dtype)
        return subarray

    @property
    def required_columns(self):
        return set(_as_list_of_str(self.columns))

    def transformed_columns(self, input_columns):
        input_columns = set(_as_list_of_str(input_columns))
        if not self.required_columns <= input_columns:
            raise ValueError(f'Required columns are {self.required_columns}')
        return {None}


class PandasIndexValueSelector(BaseEstimator, TransformerMixin):
    """Select index levels as feature cols, and return np.array.

    Optionally, cast the resulting arry to dtype.
    """

    def __init__(self, levels=None, dtype=None):
        self.levels = levels
        self.dtype = dtype

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df):
        levels = self.levels or []
        if isinstance(levels, str):
            levels = [levels]
        try:
            iter(levels)
        except TypeError:
            levels = [levels]
        blocks = [
            df.index.get_level_values(level).values.reshape(-1, 1)
            for level in levels
        ]
        subarray = np.hstack(blocks) if blocks else np.empty((len(df), 0))
        if self.dtype:
            subarray = subarray.astype(self.dtype)
        return subarray


class PandasSequenceSplitter(BaseEstimator, TransformerMixin):
    """Split sequence columns in two."""

    def __init__(self, columns=None, index=0):
        self.columns = columns
        self.index = index

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df):
        columns = _as_list_of_str(self.columns)
        index = self.index
        for col in columns:
            values = df[col].values
            df = array_to_dataframe(values[:, :index], f'{col}_pre', df=df)
            df = array_to_dataframe(values[:, index:], f'{col}_post', df=df)
        return df

    @property
    def required_columns(self):
        columns = set(_as_list_of_str(self.columns))
        return columns

    def transformed_columns(self, input_columns):
        columns = _as_list_of_str(input_columns)
        input_columns = set(columns)
        if not self.required_columns <= input_columns:
            raise ValueError(f'Required columns are {self.required_columns}')
        to_change = _as_list_of_str(self.columns)
        columns = [f'{col}_post' for col in to_change]
        columns2 = [f'{col}_pre' for col in to_change]
        return set(np.concatenate([columns, columns2]))
