from typing import List, Union

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold


class PandasTimeSeriesSplit(_BaseKFold):
    """Apply a sklearn TimeSeriesSplit to multiple timeseries in a single DF.

    The dataframe should be ordered by date ascending for each time series, and
    the index should be unique.

    Parameters:
        groupby : string or array of strings
            The column name(s) to group the input dataframe by - each group
            should hold a monotonically increasing time series.

        datetime_col : string
            The column name of the datetime column - used to validate that the
            dataframe is groups of time series.

        n_splits : int, default = 3
            Number of splits. Must be at least 2.

        max_train_size : int, optional
            Maximum size for a single training set.

    """

    def __init__(
        self,
        groupby: Union[str, List[str]],
        datetime_col: str,
        n_splits: int = 3,
        max_train_size: int = None,
    ):
        self.groupby = groupby
        self.datetime_col = datetime_col
        self.n_splits = n_splits
        self.max_train_size = max_train_size

    def split(self, df, y=None, groups=None):
        self._validate_df(df)
        groups = df.groupby(self.groupby).indices
        splits = {}
        while True:
            X_idxs, y_idxs = [], []
            for key, sub_idx in groups.items():
                sub_df = df.iloc[sub_idx]
                sub_y = y[sub_idx] if y is not None else None

                if key not in splits:
                    splitter = TimeSeriesSplit(
                        self.n_splits, self.max_train_size
                    )
                    splits[key] = splitter.split(sub_df, sub_y)

                try:
                    X_idx, y_idx = next(splits[key])
                    X_idx = np.array(
                        [df.index.get_loc(i) for i in sub_df.iloc[X_idx].index]
                    )
                    y_idx = np.array(
                        [df.index.get_loc(i) for i in sub_df.iloc[y_idx].index]
                    )
                    X_idxs.append(X_idx)
                    y_idxs.append(y_idx)
                except StopIteration:
                    pass

            if len(X_idxs) == 0:
                break

            yield np.concatenate(X_idxs), np.concatenate(y_idxs)

    def _validate_df(self, df):
        if df.index.duplicated().any():
            raise ValueError("Dataframe has non-unique index.")
        shift_date = df.groupby(self.groupby)[self.datetime_col].shift()
        if (shift_date >= df[self.datetime_col]).any():
            raise ValueError(
                "Dataframe not in ascending order for each time series."
            )
