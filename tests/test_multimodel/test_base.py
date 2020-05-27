import pytest
import tempfile

import numpy.testing as npt
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD

import timeserio.ini as ini
from timeserio.data.mock import mock_fit_data
from timeserio.keras.multinetwork import MultiNetworkBase
from timeserio.multimodel import MultiModel
from timeserio.preprocessing import PandasValueSelector
from timeserio.batches.single.row import RowBatchGenerator
from timeserio.pipeline import Pipeline, MultiPipeline

from timeserio.utils.pickle import dumpf, loadf
from timeserio.validation.validation import is_valid_multimodel

EPOCHS = 1


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
def multinetwork(random):
    return SimpleMultiNetwork(forecaster_features=1, verbose=0)


@pytest.fixture
def multipipeline():
    multipipeline = MultiPipeline(
        {
            'features': PandasValueSelector('weather_temperature'),
            'usage': PandasValueSelector(ini.Columns.target)
        }
    )
    return multipipeline


@pytest.fixture
def manifold():
    manifold = {'forecaster': ('usage', 'usage')}
    return manifold


@pytest.fixture
def multimodel(multinetwork, multipipeline, manifold):
    multimodel = MultiModel(
        multinetwork=multinetwork,
        multipipeline=multipipeline,
        manifold=manifold
    )
    return multimodel


@pytest.fixture
def multimodel_validation(multinetwork):

    multipipeline = MultiPipeline(
        {
            'features_pipeline':
            Pipeline(
                [('weather', PandasValueSelector('weather_temperature'))]
            ),
            'usage_pipeline':
            Pipeline([('usage', PandasValueSelector('usage'))])
        }
    )

    multimodel = MultiModel(
        multinetwork=multinetwork,
        multipipeline=multipipeline,
        manifold={'forecaster': ('usage', 'usage')}
    )
    return multimodel


@pytest.fixture
def df():
    return mock_fit_data(periods=13)


@pytest.fixture
def validation_df():
    return mock_fit_data(periods=13)


def test_fit(multimodel, df):
    error0 = multimodel.evaluate(df=df, model='forecaster')
    multimodel.fit(df=df, model='forecaster', batch_size=100, epochs=EPOCHS)
    error = multimodel.evaluate(df=df, model='forecaster')
    assert error < error0


@pytest.mark.parametrize('batch_size', [1, 2, 2 ** 10])
def test_fit_generator(multimodel, df, batch_size):
    generator = RowBatchGenerator(df=df, batch_size=batch_size)
    error0 = multimodel.evaluate(df=df, model='forecaster')
    multimodel.fit_generator(
        df_generator=generator, model='forecaster', epochs=3
    )
    error = multimodel.evaluate(df=df, model='forecaster')
    assert error < error0


def test_fit_with_validation_data(multimodel, df, validation_df):
    multimodel.fit(
        df=df,
        model='forecaster',
        batch_size=100,
        epochs=EPOCHS,
        validation_data=validation_df
    )
    history = multimodel.multinetwork.history[0]['history']
    assert history['val_loss'][0] > 0
    assert len(history['val_loss']) == EPOCHS


def test_fit_with_validation_data_none(multimodel, df):
    error0 = multimodel.evaluate(df=df, model='forecaster')
    multimodel.fit(
        df=df,
        model='forecaster',
        batch_size=100,
        epochs=EPOCHS,
        validation_data=None
    )
    error = multimodel.evaluate(df=df, model='forecaster')
    assert error < error0


@pytest.mark.parametrize('num_epochs', [3])
@pytest.mark.parametrize('batch_size', [1, 2, 2 ** 10])
def test_fit_generator_with_validation_data(
    multimodel, df, batch_size, num_epochs
):
    fit_generator = RowBatchGenerator(df=df, batch_size=batch_size)
    val_data = df
    multimodel.fit_generator(
        df_generator=fit_generator,
        model='forecaster',
        epochs=num_epochs,
        validation_data=val_data
    )
    history = multimodel.multinetwork.history[0]['history']
    assert len(history['val_loss']) == num_epochs
    assert history['val_loss'][0] > history['loss'][-1] > 0


@pytest.mark.parametrize('num_epochs', [3])
@pytest.mark.parametrize('batch_size', [1, 2, 2 ** 10])
def test_fit_generator_with_validation_gen(
    multimodel, df, batch_size, num_epochs
):
    fit_generator = RowBatchGenerator(df=df, batch_size=batch_size)
    val_generator = RowBatchGenerator(df=df, batch_size=batch_size)
    multimodel.fit_generator(
        df_generator=fit_generator,
        model='forecaster',
        epochs=num_epochs,
        validation_data=val_generator
    )
    history = multimodel.multinetwork.history[0]['history']
    assert len(history['val_loss']) == num_epochs
    assert history['val_loss'][0] > history['loss'][-1] > 0


def test_evaluate(multimodel, df):
    multimodel.fit(df=df, model='forecaster', batch_size=100, epochs=EPOCHS)
    loss = multimodel.evaluate(df, model='forecaster')
    history = multimodel.multinetwork.history[0]['history']
    assert 0 < loss < history['loss'][0]


@pytest.mark.parametrize('batch_size', [1, 2, 1024])
def test_evaluate_generator(multimodel, df, batch_size):
    df_generator = RowBatchGenerator(df=df, batch_size=batch_size)
    loss0 = multimodel.evaluate(df, model='forecaster')
    loss = multimodel.evaluate_generator(df_generator, model='forecaster')
    npt.assert_allclose(loss0, loss, rtol=1e-6)


def test_fit_frozen(multimodel, df):
    multimodel.trainable_models = None
    error0 = multimodel.evaluate(df=df, model='forecaster')
    multimodel.fit(df=df, model='forecaster', batch_size=100, epochs=EPOCHS)
    error = multimodel.evaluate(df=df, model='forecaster')
    assert error == error0


def test_fit_un_frozen(multimodel, df):
    multimodel.trainable_models = 'forecaster'
    error0 = multimodel.evaluate(df=df, model='forecaster')
    multimodel.fit(df=df, model='forecaster', batch_size=100, epochs=EPOCHS)
    error = multimodel.evaluate(df=df, model='forecaster')
    assert error < error0


def test_multimodel_pipeline_validation(multimodel_validation):
    for pipeline in multimodel_validation.multipipeline.pipelines:
        assert hasattr(
            multimodel_validation.multipipeline[pipeline], 'required_columns'
        )
        assert hasattr(
            multimodel_validation.multipipeline[pipeline],
            'transformed_columns'
        )
    assert multimodel_validation.multipipeline.required_columns == \
        {'usage', 'weather_temperature'}
    assert multimodel_validation.multipipeline.transformed_columns(
        ['usage', 'weather_temperature']
    ) == {None}


def test_multimodel_pickle_pipeline(multimodel_validation):
    with tempfile.NamedTemporaryFile(suffix='.pickle', delete=True) as f:
        dumpf(multimodel_validation, f.name)
        obj2 = loadf(f.name)
        is_valid_multimodel(obj2)
