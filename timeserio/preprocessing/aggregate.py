from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .utils import CallableMixin, _as_list_of_str


class AggregateFeaturizer(BaseEstimator, TransformerMixin, CallableMixin):
    """Create features as aggregates of all points sharing `groupby` keys.

    Parameters:
        groupby: str or List[str]
            Specify keys/column names to group by.
            See pandas.DataFrame.groupby

        aggregate: dict
            Aggregation functions to apply to each column after grouping.
            See pandas.DataFrame.groupby().aggregate()

    This Transformer is statefull, and the number of output columns depends
    on the input.

    """

    def __init__(self,
                 *,
                 aggregate: Dict[str, Union[str, List[str]]],
                 groupby: Optional[Union[str, List[str]]] = None):
        self.groupby = groupby
        self.aggregate = aggregate

    def fit(self, df, y=None):
        groupby = _as_list_of_str(self.groupby)
        to_aggregate = df.groupby(groupby) if groupby else df
        aggregated = self._agg(to_aggregate)
        aggregated.columns = _flatten_columns(aggregated.columns)
        self.df_ = aggregated
        return self

    def _agg(self, df):
        groupby = _as_list_of_str(self.groupby)
        aggregated = {}
        for column, methods in self.aggregate.items():
            methods = [methods] if type(methods) is str else methods
            for method in methods:
                single_agg = df[column].aggregate(method)
                if type(single_agg) is not pd.Series:
                    single_agg = [single_agg]
                aggregated[f'{column}_{method}'] = single_agg
        aggregated = pd.DataFrame(aggregated)
        if groupby:
            # Flatten the DF into a single row
            aggregated = pd.DataFrame(aggregated.unstack(groupby)).T
        return aggregated

    def transform(self, df):
        check_is_fitted(self, 'df_')
        return _cartesian_product(df, self.df_)

    @property
    def required_columns(self):
        required_columns = set(self.aggregate.keys())
        groupby = _as_list_of_str(self.groupby)
        required_columns |= set(groupby)
        return required_columns


def _flatten_columns(columns: pd.Index) -> Union[pd.Index, List[str]]:
    if type(columns) is not pd.MultiIndex:
        return columns
    height, width = np.array(columns.codes).shape
    new_cols = []
    for j in range(width):
        col = ''
        for i in range(height):
            name = columns.names[i]
            level = str(columns.levels[i][columns.codes[i][j]])
            col += f'{name}_{level}' if name else level
            col += '_'
        new_cols.append(col[:-1])
    return new_cols


def _cartesian_product(df_1: pd.DataFrame, df_2: pd.DataFrame) -> pd.DataFrame:
    df_1, df_2 = df_1.copy(), df_2.copy()
    df_1['dummy_key'] = 1
    df_2['dummy_key'] = 1
    return df_1.merge(df_2, on='dummy_key',
                      suffixes=('', '_agg')).drop('dummy_key', axis=1)
