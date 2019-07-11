import pytest

from sklearn.preprocessing import MinMaxScaler

from timeserio.preprocessing import (
    PandasValueSelector, PandasColumnSelector, PandasDateTimeFeaturizer,
    StatelessOneHotEncoder
)
from timeserio.pipeline import Pipeline, FeatureUnion, GroupedPipeline

test_pipeline1 = Pipeline(
    [
        (
            'select',
            PandasDateTimeFeaturizer(column='datetime', attributes='hour')
        ), ('select2', PandasValueSelector(['hour', 'id']))
    ]
)

test_pipeline2 = Pipeline(
    [
        (
            'select',
            PandasDateTimeFeaturizer(column='datetime', attributes='hour')
        ), ('select2', PandasValueSelector(['id']))
    ]
)

test_pipeline3 = Pipeline(
    [
        (
            'select',
            PandasDateTimeFeaturizer(column='datetime', attributes='hour')
        ), ('select2', PandasColumnSelector(['hour', 'id']))
    ]
)

test_grouped_pipeline1 = GroupedPipeline(groupby='team',
                                         pipeline=test_pipeline1)
test_grouped_pipeline1_multiple = GroupedPipeline(groupby=['week', 'team'],
                                                  pipeline=test_pipeline1)
test_grouped_pipeline2 = GroupedPipeline(groupby='team',
                                         pipeline=test_pipeline2)
test_grouped_pipeline3 = GroupedPipeline(groupby='team',
                                         pipeline=test_pipeline3)

# feature union
pipeline1 = Pipeline(
    [
        (
            'featurize',
            PandasDateTimeFeaturizer(column='datetime', attributes='hour')
        ), ('select', PandasColumnSelector(columns=['hour']))
    ]
)
pipeline2 = Pipeline([('select2', PandasColumnSelector('id'))])
pipeline3 = Pipeline([('select3', PandasValueSelector('id'))])
pipeline4 = Pipeline(
    [
        (
            'featurize',
            PandasDateTimeFeaturizer(column='datetime', attributes='hour')
        ), ('select', PandasValueSelector(columns=['hour']))
    ]
)
test_feature_union = FeatureUnion([('datetime', pipeline1), ('id', pipeline2)])
test_feature_union2 = FeatureUnion(
    [('datetime', pipeline4), ('select', pipeline3)]
)

pipeline_m2 = Pipeline([('id', PandasValueSelector(columns='id'))])

pipeline_mixed = Pipeline(
    [
        (
            'featurize',
            PandasDateTimeFeaturizer(column='datetime', attributes='hour')
        ), ('select', PandasValueSelector(columns=['hour']))
    ]
)

pipeline_m1 = Pipeline(
    [
        (
            'datetime',
            PandasDateTimeFeaturizer(column='datetime', attributes='month0')
        ), ('select', PandasValueSelector(columns='month0')),
        (
            'onehot',
            StatelessOneHotEncoder(
                n_features=1,
                n_values=12,
                sparse=False
            )
        )
    ]
)
test_feature_union_mixed = FeatureUnion(
    [('id', pipeline_m2), ('month', pipeline_m1)]
)


@pytest.mark.parametrize(
    'transformer, required_columns', [
        (test_pipeline2, {'datetime', 'id'}),
        (test_pipeline1, {'datetime', 'id'}),
    ]
)
def test_pipeline_required(transformer, required_columns):
    assert hasattr(transformer, 'required_columns')
    assert transformer.required_columns == required_columns


@pytest.mark.parametrize(
    'transformer, input_columns, transformed_columns', [
        (test_pipeline2, ['datetime', 'id'], {None}),
        (test_pipeline1, {'datetime', 'id'}, {None}),
        (test_pipeline3, {'datetime', 'id'}, {'hour', 'id'})
    ]
)
def test_pipeline_transformed(transformer, input_columns, transformed_columns):
    assert transformer.transformed_columns(input_columns) == \
        transformed_columns


@pytest.mark.parametrize(
    'transformer, required_columns', [
        (test_feature_union, {'datetime', 'id'}),
        (test_feature_union2, {'datetime', 'id'})
    ]
)
def test_feature_union_required_columns(transformer, required_columns):
    assert hasattr(transformer, 'required_columns')
    assert transformer.required_columns == required_columns


@pytest.mark.parametrize(
    'transformer, transformed_columns, input_columns', [
        (test_feature_union, {'hour', 'id'}, {'datetime', 'id'}),
        (test_feature_union2, {None}, {'datetime', 'id'})
    ]
)
def test_feature_union_transformed_columns(
    transformer, transformed_columns, input_columns
):
    assert transformer.transformed_columns(input_columns) == \
        transformed_columns


# numpy pipeline and feature union
@pytest.mark.parametrize(
    'pipeline, input_columns, expected', [
        (pipeline_mixed, ['datetime', 'usage'], {None}),
        (pipeline_m1, ['datetime', 'usage'], {None}),
        (test_feature_union_mixed, ['id', 'datetime'], {None})
    ]
)
def test_pandas_numpy_mixed_pipeline(pipeline, input_columns, expected):
    assert pipeline.transformed_columns(input_columns) == expected


pipeline_scaler = Pipeline([('scale', MinMaxScaler)])
pipeline_scaler2 = Pipeline(
    [
        (
            'featurize',
            PandasDateTimeFeaturizer('datetime', attributes='month0')
        ), ('select', PandasValueSelector(columns='month0')),
        ('scale', MinMaxScaler())
    ]
)


# test unknown transformer
@pytest.mark.parametrize(
    'pipeline, required_columns, transformed_columns', [
        (pipeline_scaler, {None}, {None}),
        (pipeline_scaler2, {'datetime'}, {None})
    ]
)
def test_pipeline_no_attributes_required_or_transformed(
    pipeline, required_columns, transformed_columns
):
    assert pipeline.required_columns == required_columns
    assert pipeline.transformed_columns(pipeline.required_columns) == \
        transformed_columns


def test_invalid_pipeline():
    invalid_pipeline = Pipeline(
        [
            (
                'featurize',
                PandasDateTimeFeaturizer('datetime', attributes='month0')
            ), ('select', PandasValueSelector(columns='dayofweek')),
            ('scale', MinMaxScaler())
        ]
    )
    with pytest.raises(ValueError):
        invalid_pipeline.transformed_columns(['usage', 'datetime'])


@pytest.mark.parametrize(
    'transformer, required_columns', [
        (test_grouped_pipeline2, {'team', 'datetime', 'id'}),
        (test_grouped_pipeline1, {'team', 'datetime', 'id'}),
        (test_grouped_pipeline1_multiple, {'team', 'week', 'datetime', 'id'}),
    ]
)
def test_grouped_pipeline_required(transformer, required_columns):
    assert hasattr(transformer, 'required_columns')
    assert transformer.required_columns == required_columns


@pytest.mark.parametrize(
    'transformer, input_columns, transformed_columns', [
        (test_grouped_pipeline2, ['team', 'datetime', 'id'], {None}),
        (test_grouped_pipeline1, {'team', 'datetime', 'id'}, {None}),
        (test_grouped_pipeline1_multiple, {'team', 'week', 'datetime', 'id'},
         {None}),
        (test_grouped_pipeline3, {'team', 'datetime', 'id'}, {'hour', 'id'})
    ]
)
def test_grouped_pipeline_transformed(transformer, input_columns,
                                      transformed_columns):
    assert transformer.transformed_columns(input_columns) == \
        transformed_columns
