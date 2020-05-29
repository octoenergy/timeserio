import pytest

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

from timeserio.preprocessing import (PandasValueSelector, PandasColumnSelector)
from timeserio.pipeline import Pipeline, MultiPipeline
from timeserio.validation import is_valid_multimodel
from timeserio.multimodel import MultiModel
from timeserio.keras.multinetwork import MultiNetworkBase


class SimpleMultiNetwork(MultiNetworkBase):
    """A simple feature-based forecaster"""

    def _model(
        self,
        forecaster_features=1,
        forecaster_hidden_units=(8, 8),
    ):
        # Forecaster
        forecaster_input = Input(
            (forecaster_features, ), name='forecaster_input_features'
        )
        forecaster_output = forecaster_input
        forecaster_dense_units = list(forecaster_hidden_units) + [
            1
        ]  # append final output
        for idx, units in enumerate(forecaster_dense_units):
            forecaster_output = Dense(
                units=units,
                activation='relu',
                name='forecaster_dense_{}'.format(idx)
            )(forecaster_output)
        forecaster_model = Model(
            forecaster_input, forecaster_output, name='Forecaster'
        )
        optimizer = SGD(lr=0.001)
        forecaster_model.compile(loss='mse', optimizer=optimizer)
        return {
            'forecaster': forecaster_model,
        }

    def _callbacks(self):
        return {'forecaster': []}


@pytest.fixture
def multinetwork():
    return SimpleMultiNetwork(forecaster_features=1, verbose=0)


@pytest.fixture
def multimodel(multinetwork):
    id_pipeline = Pipeline([('select', PandasValueSelector(columns='id'))])
    usage_pipeline = Pipeline([('select', PandasValueSelector('usage'))])
    multipipeline = MultiPipeline(
        {
            'id_pipeline': id_pipeline,
            'usage_pipeline': usage_pipeline
        }
    )
    multimodel = MultiModel(
        multinetwork=multinetwork,
        multipipeline=multipipeline,
        manifold={'combined': (['id', 'usage'])}
    )
    return multimodel


@pytest.fixture
def invalid_multimodel(multinetwork):
    id_pipeline = Pipeline(
        [
            ('s1', PandasColumnSelector('datetime')),
            ('select', PandasValueSelector('hour'))
        ]
    )
    usage_pipeline = Pipeline([('select', PandasValueSelector('usage'))])
    multipipeline = MultiPipeline(
        {
            'id_pipeline': id_pipeline,
            'usage_pipeline': usage_pipeline
        }
    )
    multimodel = MultiModel(
        multinetwork=multinetwork,
        multipipeline=multipipeline,
        manifold={'combined': (['id', 'usage'])}
    )
    return multimodel


def test_valid_multimodel(multimodel):
    assert is_valid_multimodel(multimodel)


def test_invalid_multimodel(invalid_multimodel):
    assert not is_valid_multimodel(invalid_multimodel)
