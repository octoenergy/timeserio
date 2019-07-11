import numpy as np
import pandas as pd
import pytest

from timeserio.preprocessing import aggregate

test_df = pd.DataFrame(dict(
    col1=[1, 2, 3, 4],
    col2=[2, 4, 6, 8],
    team=[1, 1, 2, 2],
    group=[1, 2, 1, 2],
))

a0 = {'col1': ['sum', 'mean']}
a1 = {'col1': 'sum'}
a2 = {'col1': 'sum', 'col2': 'sum'}
a3 = {'col1': ['sum', 'mean'], 'col2': 'mean'}

g0 = ['team']
g1 = ['team', 'group']


@pytest.mark.parametrize(
    "column_tuples, names, expected", [
        ([('col1', 'mean'), ('col1', 'sum'), ('col2', 'mean')], [None, None],
         ['col1_mean', 'col1_sum', 'col2_mean']),
        ([('col1', 'mean', 1), ('col1', 'sum', 2), ('col2', 'mean', 1)],
         [None, None, 'team'],
         ['col1_mean_team_1', 'col1_sum_team_2', 'col2_mean_team_1']),
        ([('col1_mean', 1), ('col1_sum', 2), ('col2_mean', 1)],
         [None, 'team'],
         ['col1_mean_team_1', 'col1_sum_team_2', 'col2_mean_team_1']),
        ([('col1', 'mean', 1, 1), ('col1', 'sum', 2, 2),
          ('col2', 'sum', 1, 1)],
         [None, None, 'team', 'group'],
         ['col1_mean_team_1_group_1', 'col1_sum_team_2_group_2',
          'col2_sum_team_1_group_1']),
        ([('col1_mean', 1, 1), ('col1_sum', 2, 2), ('col2_sum', 1, 1)],
         [None, 'team', 'group'],
         ['col1_mean_team_1_group_1', 'col1_sum_team_2_group_2',
          'col2_sum_team_1_group_1']),
    ]
)
def test_flatten_columns(column_tuples, names, expected):
    columns = pd.MultiIndex.from_tuples(column_tuples, names=names)
    out = aggregate._flatten_columns(columns)
    assert out == expected


def test_flatten_column_ignores_flat():
    flat = pd.Index([1, 2, 3, 4])
    assert (flat == aggregate._flatten_columns(flat)).all()


@pytest.mark.parametrize(
    "df_1_dict, df_2_dict, expected_dict", [
        ({'a': [1, 2]}, {'b': [3, 4]}, {'a': [1, 1, 2, 2], 'b': [3, 4, 3, 4]}),
        ({'a': [1, 2]}, {'a': [3, 4]}, {'a': [1, 1, 2, 2],
                                        'a_agg': [3, 4, 3, 4]}),
        ({'a': [1, 2]}, {'b': [5]}, {'a': [1, 2], 'b': [5, 5]}),
    ]
)
def test_cartesian_product(df_1_dict, df_2_dict, expected_dict):
    df_1 = pd.DataFrame(df_1_dict)
    df_2 = pd.DataFrame(df_2_dict)
    expected = pd.DataFrame(expected_dict)

    out = aggregate._cartesian_product(df_1, df_2)
    out = out.sort_values(list(expected.columns))

    pd.testing.assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    "df, agg_dict, groupby, expected", [
        (test_df, a0, None, [10, 2.5]),
        (test_df, a1, None, [10]),
        (test_df, a2, None, [10, 20]),
        (test_df, a3, None, [10, 2.5, 5]),
        (test_df, a0, g0, [3, 7, 1.5, 3.5]),
        (test_df, a1, g0, [3, 7]),
        (test_df, a2, g0, [3, 7, 6, 14]),
        (test_df, a3, g0, [3, 7, 1.5, 3.5, 3, 7]),
        (test_df, a0, g1, [1, 2, 3, 4, 1, 2, 3, 4]),
        (test_df, a1, g1, [1, 2, 3, 4]),
        (test_df, a2, g1, [1, 2, 3, 4, 2, 4, 6, 8]),
        (test_df, a3, g1, [1, 2, 3, 4, 1, 2, 3, 4, 2, 4, 6, 8]),
    ]
)
def test_aggregate_values(df, agg_dict, groupby, expected):
    f = aggregate.AggregateFeaturizer(groupby=groupby, aggregate=agg_dict)
    aggregated = f.fit(df).df_
    expected = np.array(expected)[np.newaxis, :]
    np.testing.assert_array_equal(aggregated.values, expected)


c1s = 'col1_sum'
c1m = 'col1_mean'
c2s = 'col2_sum'
c2m = 'col2_mean'


@pytest.mark.parametrize(
    "df, agg_dict, groupby, expected", [
        (test_df, a0, None, [c1s, c1m]),
        (test_df, a1, None, [c1s]),
        (test_df, a2, None, [c1s, c2s]),
        (test_df, a3, None, [c1s, c1m, c2m]),
        (test_df, a0, g0, [c1s, c1s, c1m, c1m]),
        (test_df, a1, g0, [c1s, c1s]),
        (test_df, a2, g0, [c1s, c1s, c2s, c2s]),
        (test_df, a3, g0, [c1s, c1s, c1m, c1m, c2m, c2m]),
        (test_df, a0, g1, [c1s, c1s, c1s, c1s, c1m, c1m, c1m, c1m]),
        (test_df, a1, g1, [c1s, c1s, c1s, c1s]),
        (test_df, a2, g1, [c1s, c1s, c1s, c1s, c2s, c2s, c2s, c2s]),
        (test_df, a3, g1, [c1s, c1s, c1s, c1s, c1m, c1m, c1m, c1m, c2m, c2m,
                           c2m, c2m]),
    ]
)
def test_aggregate_col_names(df, agg_dict, groupby, expected):
    f = aggregate.AggregateFeaturizer(groupby=groupby, aggregate=agg_dict)
    to_aggregate = df.groupby(groupby) if groupby else df
    aggregated_cols = f._agg(to_aggregate).columns.get_level_values(0).values
    np.testing.assert_array_equal(aggregated_cols, expected)


@pytest.mark.parametrize(
    'agg_dict, groupby, required_columns', [
        (a0, None, {'col1'}),
        (a2, None, {'col1', 'col2'}),
        (a0, g0, {'col1', 'team'}),
        (a2, g0, {'col1', 'col2', 'team'}),
        (a0, g1, {'col1', 'team', 'group'}),
        (a2, g1, {'col1', 'col2', 'team', 'group'}),
    ]
)
def test_required_columns(agg_dict, groupby, required_columns):
    f = aggregate.AggregateFeaturizer(groupby=groupby, aggregate=agg_dict)
    assert hasattr(f, 'required_columns')
    assert f.required_columns == required_columns
