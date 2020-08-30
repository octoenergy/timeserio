import numpy as np
import pandas as pd
import pytest

import timeserio.ini as ini
from timeserio.data.mock import mock_fit_data
from timeserio.preprocessing import (
    PandasColumnSelector, PandasValueSelector, PandasSequenceSplitter
)


datetime_column = ini.Columns.datetime
usage_column = ini.Columns.target


@pytest.fixture
def df(use_tensor_extension):
    return mock_fit_data(use_tensor_extension=use_tensor_extension)


@pytest.mark.parametrize(
    'columns, num_columns', [
        (None, 0),
        ([], 0),
        ('cluster', 1),
        ([datetime_column], 1),
        (['weather_temperature', usage_column], 2),
        ([datetime_column, usage_column], 2),
        ('embedding', 1),
        (['embedding', usage_column], 2),
    ]
)
def test_column_selector(df, columns, num_columns):
    subframe = PandasColumnSelector(columns=columns).transform(df)
    assert isinstance(subframe, pd.DataFrame)
    assert len(subframe) == len(df)
    try:
        new_columns = subframe.columns.get_level_values(0).unique()
    except AttributeError:  # not a MultiIindex
        new_columns = subframe.columns
    assert len(new_columns) == num_columns


@pytest.mark.parametrize(
    'columns, shape1', [
        (None, 0),
        ([], 0),
        ('cluster', 1),
        ([datetime_column], 1),
        (['weather_temperature', usage_column], 2),
        ([datetime_column, usage_column], 2),
        ('embedding', 2),
        (['embedding', usage_column], 3),
    ]
)
def test_value_selector(df, columns, shape1):
    expected_shape = (len(df), shape1)
    subarray = PandasValueSelector(columns=columns).transform(df)
    assert isinstance(subarray, np.ndarray)
    assert subarray.shape == expected_shape


@pytest.mark.parametrize(
    'transformer, required_columns', [
        (PandasColumnSelector('col1'), {'col1'}),
        (PandasColumnSelector(['col1', 'col2']), {'col1', 'col2'}),
        (PandasValueSelector('col1'), {'col1'}),
        (PandasSequenceSplitter(['col1']), {'col1'}),
        (PandasSequenceSplitter(['col1', 'col2']), {'col1', 'col2'})
    ]
)
def test_single_transformer_required_columns(transformer, required_columns):
    assert hasattr(transformer, 'required_columns')
    assert transformer.required_columns == required_columns


@pytest.mark.parametrize(
    'transformer, transformed_columns', [
        (PandasColumnSelector('col1'), {'col1'}),
        (PandasColumnSelector(['col1', 'col2']), {'col1', 'col2'}),
        (PandasValueSelector('col1'), {None}),
        (PandasSequenceSplitter(['col1']), {'col1_post', 'col1_pre'}),
        (
            PandasSequenceSplitter(['col1', 'col2']),
            {'col1_post', 'col2_post', 'col1_pre', 'col2_pre'}
        )
    ]
)
def test_single_transformer_transformed_from_required_only(
    transformer, transformed_columns
):
    output_columns = transformer.transformed_columns(
        list(transformer.required_columns)
    )
    assert output_columns == transformed_columns


@pytest.mark.parametrize(
    'transformer, input_columns, transformed_columns', [
        (PandasColumnSelector('col1'), {'col1', 'col2'}, {'col1'}),
    ]
)
def test_single_transformer_transformed_columns(
    transformer, input_columns, transformed_columns
):
    output_columns = transformer.transformed_columns(input_columns)
    assert output_columns == transformed_columns


@pytest.mark.parametrize(
    'transformer',
    [(PandasColumnSelector('column')), (PandasSequenceSplitter(['col1']))]
)
def test_missing_required_columns(transformer):
    with pytest.raises(ValueError):
        transformer.transformed_columns(['random'])
