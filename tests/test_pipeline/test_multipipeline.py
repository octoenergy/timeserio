import pytest
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from timeserio.pipeline import Pipeline, MultiPipeline
from timeserio.preprocessing import (
    PandasValueSelector, PandasDateTimeFeaturizer
)


@pytest.fixture
def multipipeline():
    multipipeline = MultiPipeline(
        {
            'scaler': StandardScaler(),
            'poly': PolynomialFeatures(degree=2)
        }
    )
    return multipipeline


@pytest.fixture()
def multipipeline_validation():
    pipeline_id = Pipeline([('id', PandasValueSelector(columns='id'))])
    pipeline_hour = Pipeline(
        [
            (
                'featurize',
                PandasDateTimeFeaturizer(column='datetime', attributes='hour')
            ), ('select', PandasValueSelector(columns=['hour']))
        ]
    )
    multipipeline = MultiPipeline(
        {
            'id_pipeline': pipeline_id,
            'hour_pipeline': pipeline_hour
        }
    )
    return multipipeline


def test_get_param_names(multipipeline):
    assert multipipeline._get_param_names() == ['poly', 'scaler']


def test_get_params(multipipeline):
    params = multipipeline.get_params()
    assert 'poly' in params
    assert 'poly__degree' in params
    assert 'scaler' in params
    assert 'scaler__with_mean' in params


def test_set_params(multipipeline):
    params = multipipeline.get_params()
    assert params['poly__degree'] == 2
    multipipeline.set_params(poly__degree=10)
    params = multipipeline.get_params()
    assert params['poly__degree'] == 10


def test_get_attr(multipipeline):
    assert 'degree' in multipipeline.poly.get_params()


def test_get_item(multipipeline):
    assert multipipeline['poly'] is multipipeline.poly


def test_pipeline_validation(multipipeline_validation):
    required_columns = {'id', 'datetime'}
    assert multipipeline_validation.transformed_columns(required_columns) == \
        {None}
    assert multipipeline_validation.required_columns == required_columns
