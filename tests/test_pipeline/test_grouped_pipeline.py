import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline as Pipeline_sk
from sklearn.preprocessing import FunctionTransformer

from timeserio import ini
from timeserio.data import mock
from timeserio.pipeline import GroupedPipeline, Pipeline
from timeserio.preprocessing import (PandasColumnSelector,
                                     PandasDateTimeFeaturizer,
                                     PandasValueSelector, utils)


@pytest.fixture
def input_df():
    df = mock.mock_raw_data(ids=[0, 1])
    df['group'] = np.random.randint(2, size=len(df))
    return df


col_selector = ('select', PandasColumnSelector([ini.Columns.target]))
val_selector = ('select', PandasValueSelector([ini.Columns.target]))
identity = ('identity', utils.IdentityRegressor())
lagger = ('lag', FunctionTransformer(lambda x: x.shift(1), validate=False))


@pytest.mark.parametrize(
    'pipeline, groupby, is_estimator',
    [
        (Pipeline([val_selector]), 'id', False),
        (Pipeline([val_selector]), ['id'], False),
        (Pipeline([val_selector]), ['id', 'group'], False),
        (Pipeline_sk([val_selector]), ['id'], False),
        (Pipeline_sk([val_selector]), ['id', 'group'], False),
        (Pipeline([val_selector, identity]), ['id'], True),
        (Pipeline([val_selector, identity]), ['id', 'group'], True),
        (Pipeline_sk([val_selector, identity]), ['id'], True),
        (Pipeline_sk([val_selector, identity]), ['id', 'group'], True),
    ]
)
def test_grouped_returns_numpy(pipeline, groupby, is_estimator, input_df):
    gp = GroupedPipeline(groupby=groupby, pipeline=pipeline)
    if is_estimator:
        out = gp.fit_predict(input_df)
    else:
        out = gp.fit_transform(input_df)
    assert type(out) is np.ndarray


@pytest.mark.parametrize(
    'pipeline, groupby',
    [
        (Pipeline([col_selector]), ['id']),
        (Pipeline([col_selector]), ['id', 'group']),
        (Pipeline_sk([col_selector]), ['id']),
        (Pipeline_sk([col_selector]), ['id', 'group']),
    ]
)
def test_grouped_returns_pandas(pipeline, groupby, input_df):
    gp = GroupedPipeline(groupby=groupby, pipeline=pipeline)
    out = gp.fit_transform(input_df)
    assert type(out) is pd.DataFrame


@pytest.mark.parametrize(
    'pipeline, groupby, y, y_mode',
    [
        (Pipeline([val_selector, identity]), ['id'], 'id', 'string'),
        (Pipeline([val_selector, identity]), ['id'], 'id', 'series'),
        (Pipeline([val_selector, identity]), ['id'], 'id', 'array'),
        (Pipeline_sk([val_selector, identity]), ['id'], 'id', 'string'),
        (Pipeline_sk([val_selector, identity]), ['id'], 'id', 'series'),
        (Pipeline_sk([val_selector, identity]), ['id'], 'id', 'array'),
    ]
)
def test_grouped_with_y(pipeline, groupby, y, input_df, y_mode):
    if y_mode == 'series':
        y = input_df[y]
    elif y_mode == 'array':
        y = input_df[y].values
    gp = GroupedPipeline(groupby=groupby, pipeline=pipeline)

    out = gp.fit_predict(input_df, y)

    assert type(out) is np.ndarray
    assert len(out) == len(input_df)


@pytest.mark.parametrize(
    'pipeline, groupby, is_estimator',
    [
        (Pipeline([col_selector]), ['id'], False),
        (Pipeline([col_selector]), ['id', 'group'], False),
        (Pipeline_sk([col_selector]), ['id'], False),
        (Pipeline_sk([col_selector]), ['id', 'group'], False),
        (Pipeline([col_selector, identity]), ['id'], True),
        (Pipeline([col_selector, identity]), ['id', 'group'], True),
        (Pipeline_sk([col_selector, identity]), ['id'], True),
        (Pipeline_sk([col_selector, identity]), ['id', 'group'], True),
    ]
)
def test_grouped_order(pipeline, groupby, is_estimator, input_df):
    gp = GroupedPipeline(groupby=groupby, pipeline=pipeline)
    if is_estimator:
        out = gp.fit_predict(input_df).values
        expected = input_df[ini.Columns.target].values
    else:
        out = gp.fit_transform(input_df).values
        expected = input_df[[ini.Columns.target]].values
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    'pipeline, groupby, is_estimator',
    [
        (Pipeline([col_selector, lagger]), ['id'], False),
        (Pipeline([col_selector, lagger]), ['id', 'group'], False),
        (Pipeline_sk([col_selector, lagger]), ['id'], False),
        (Pipeline_sk([col_selector, lagger]), ['id', 'group'], False),
        (Pipeline([col_selector, lagger, identity]), ['id'], True),
        (Pipeline([col_selector, lagger, identity]), ['id', 'group'], True),
        (Pipeline_sk([col_selector, lagger, identity]), ['id'], True),
        (Pipeline_sk([col_selector, lagger, identity]), ['id', 'group'], True),
    ]
)
def test_grouped_values(mocker, pipeline, groupby, is_estimator, input_df):
    gp = GroupedPipeline(groupby=groupby, pipeline=pipeline)
    if is_estimator:
        out = gp.fit_predict(input_df)
    else:
        out = gp.fit_transform(input_df)

    input_df['out'] = out
    for _, df in input_df.groupby(groupby):
        expected = df[ini.Columns.target].shift(1)
        np.testing.assert_array_equal(expected.values, df['out'].values)


def test_raises_when_missing_key(input_df):
    transform_df = mock.mock_raw_data(ids=[0, 1, 2])
    gp = GroupedPipeline(groupby=['id'], pipeline=Pipeline([col_selector]))
    gp.fit(input_df)
    with pytest.raises(KeyError, message="Missing key 2 in fitted pipelines"):
        gp.transform(transform_df)


def test_one_group_missing_return_none(input_df):
    transform_df = mock.mock_raw_data(ids=[0, 1, 2])
    gp = GroupedPipeline(groupby=['id'], pipeline=Pipeline([val_selector]),
                         errors='return_empty')
    gp.fit(input_df)
    out = gp.transform(transform_df)
    assert out.shape[1] == 1

    transformed_part = transform_df[ini.Columns.target].values[:96]
    np.testing.assert_array_equal(out[:96, 0], transformed_part)
    assert np.isnan(out[96:]).all()


def test_one_groups_missing_return_df(input_df):
    transform_df = mock.mock_raw_data(ids=[0, 1, 2])

    dt_feat = PandasDateTimeFeaturizer(attributes='month')
    gp = GroupedPipeline(groupby=['id'], pipeline=dt_feat, errors='return_df')
    gp.fit(input_df)
    out = gp.transform(transform_df)
    set(out.columns) == {
        'id', ini.Columns.datetime, ini.Columns.target, 'month'
    }
    assert (~out[out.id == 0].month.isnull()).all()
    assert (~out[out.id == 1].month.isnull()).all()
    assert (out[out.id == 2].month.isnull()).all()

    orig_cols = ['id', ini.Columns.datetime, ini.Columns.target]
    pd.testing.assert_frame_equal(out[orig_cols], transform_df)


@pytest.mark.parametrize(
    'errors',
    ('return_empty')
)
def test_all_groups_missing_raises(input_df, errors):
    transform_df = mock.mock_raw_data(ids=[2, 3])
    gp = GroupedPipeline(groupby=['id'], pipeline=Pipeline([col_selector]),
                         errors=errors)
    gp.fit(input_df)
    with pytest.raises(KeyError,
                       message='All keys missing in fitted pipelines'):
        gp.transform(transform_df)
