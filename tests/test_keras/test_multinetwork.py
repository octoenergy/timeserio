import itertools

import numpy as np
import numpy.testing as npt
import pytest
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import (
    Input,
    Dense,
    Concatenate,
    Embedding,
    Flatten,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K  # noqa

from timeserio.keras.batches import ArrayBatchGenerator
from timeserio.keras.multinetwork import MultiNetworkBase
from timeserio.keras.utils import iterlayers, seed_random

from timeserio.utils.pickle import dumps, loads


class MinimalSubClass(MultiNetworkBase):
    def _model(self):
        return super()._model()


class SimpleMultiNetwork(MultiNetworkBase):
    """A simple feature-based forecaster."""

    def _model(
        self,
        forecaster_features=1,
        forecaster_hidden_units=(8, ),
        lr=0.1,
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
                activation=None,
                name='forecaster_dense_{}'.format(idx)
            )(forecaster_output)
        forecaster_model = Model(
            forecaster_input, forecaster_output, name='Forecaster'
        )
        optimizer = SGD(lr=lr)
        forecaster_model.compile(
            optimizer=optimizer, loss='mse', metrics=['mae']
        )
        return {
            'forecaster': forecaster_model,
        }

    def _callbacks(
        self, *, lr_params=dict(monitor='loss', patience=1, factor=0.01)
    ):
        learning_rate_reduction = ReduceLROnPlateau(**lr_params)
        return {
            'forecaster': [learning_rate_reduction],
        }


class BatchNormNetwork(MultiNetworkBase):
    """This network changes get_weights/set_weights order when frozen."""

    def _model(self):
        # Encoder
        encoder_input = Input((1, ))
        encoder_output = encoder_input
        encoder_output = BatchNormalization()(encoder_output)
        encoder_output = Dense(1)(encoder_output)
        encoder_model = Model(
            encoder_input,
            encoder_output,
        )
        # Combined model
        combined_output = encoder_model(encoder_input)
        combined_model = Model(
            encoder_input,
            combined_output,
        )

        return {'combined': combined_model}


class EmbedderForecasterNetwork(MultiNetworkBase):
    """A simple feature-based forecaster."""

    def _model(
            self,
            *,
            embedding_dim=2,
            embedding_max_size=10000,
            forecaster_features=1,
            forecaster_hidden_units=(8, 8),
            lr=0.1
    ):
        # Embedder
        embedding_initialize = RandomUniform(minval=-1, maxval=1)
        customer_in = Input((1,))
        embedding_out = Embedding(input_dim=embedding_max_size,
                                  output_dim=embedding_dim, input_length=1,
                                  embeddings_initializer=embedding_initialize
                                  )(customer_in)
        embedding_out = Flatten()(embedding_out)
        embedding_model = Model(
            customer_in,
            embedding_out,
            name='Embedder'
        )

        # Forecaster
        features_in = Input((forecaster_features,))
        embedding_in = Input((embedding_dim,))
        forecaster_output = Concatenate()([features_in, embedding_in])
        # append final output
        forecaster_dense_units = list(forecaster_hidden_units) + [1]
        for idx, units in enumerate(forecaster_dense_units):
            forecaster_output = Dense(
                units=units,
                activation='relu',
                name='forecaster_dense_{}'.format(idx)
            )(forecaster_output)
        forecaster_model = Model(
            [features_in, embedding_in],
            forecaster_output,
            name='Forecaster'
        )

        # Combined model
        combined_output = forecaster_model(
            [features_in, embedding_model(customer_in)]
        )
        combined_model = Model(
            [features_in, customer_in],
            combined_output,
            name='Combined'
        )
        optimizer = Adam(lr=lr)
        combined_model.compile(optimizer=optimizer, loss='mse')
        return {
            'forecaster': forecaster_model,
            'embedder': embedding_model,
            'combined': combined_model
        }

    def _callbacks(
            self,
            *,
            es_params={
                'patience': 20,
                'monitor': 'val_loss'
            },
            lr_params={
                'monitor': 'val_loss',
                'patience': 4,
                'factor': 0.2
            }
    ):
        early_stopping = EarlyStopping(**es_params)
        learning_rate_reduction = ReduceLROnPlateau(**lr_params)
        return {
            'forecaster': [],
            'embedder': [],
            'combined': [
                early_stopping, learning_rate_reduction
            ]
        }


class TestBaseClass:
    """Test asbtract base class."""

    def test_base_class_abstract(self):
        with pytest.raises(TypeError):
            MultiNetworkBase(param='value')

    def test_sub_class_illegal_param(self):
        with pytest.raises(ValueError):
            MinimalSubClass(param='value')

    def test_sub_class_not_implemented_model(self):
        with pytest.raises(NotImplementedError):
            base = MinimalSubClass()
            base.model = base._model()

    def test_sub_class_get_set_params(self):
        params = dict(epochs=66, batch_size=1024)
        base = MinimalSubClass(**params)
        assert base.get_params().items() >= params.items()
        params2 = dict(epochs=42, batch_size=128)
        base.set_params(**params2)
        assert base.get_params().items() >= params2.items()


def test_reproducibility():
    results = []
    for _ in range(5):
        with seed_random():
            multinetwork = SimpleMultiNetwork()
            x = np.random.rand(13, 1)
            y = np.random.rand(13, 1)
            multinetwork.fit(x, y, model='forecaster', batch_size=1, epochs=1)
            loss = multinetwork.history[-1]['history']['loss'][-1]
        results.append(loss)
    assert min(results) == max(results)


class TestSubClass:
    """Test subclass."""

    @pytest.fixture
    def testclass(self):
        return SimpleMultiNetwork

    @pytest.fixture
    def multinetwork(self, random):
        return SimpleMultiNetwork()

    def test_default_params(self, multinetwork):
        params = multinetwork.get_params()
        assert 'forecaster_features' in params

    def test_history_is_appended(self, multinetwork):
        """Multinetwork history gets overwritten at the end of training."""
        x = np.random.rand(13, 1)
        y = np.random.rand(13, 1)
        multinetwork.fit(x, y, model='forecaster', batch_size=100, epochs=1)
        assert len(multinetwork.history[-1]['history']['loss']) == 1
        multinetwork.fit(x, y, model='forecaster', batch_size=100, epochs=2)
        assert len(multinetwork.history[-1]['history']['loss']) == 2
        assert len(multinetwork.history) == 2

    def test_history_is_reset(self, multinetwork):
        """Multinetwork history gets overwritten at the end of training."""
        x = np.random.rand(13, 1)
        y = np.random.rand(13, 1)
        multinetwork.fit(x, y, model='forecaster', batch_size=100, epochs=1)
        multinetwork.fit(
            x,
            y,
            model='forecaster',
            reset_history=True,
            batch_size=100,
            epochs=1
        )
        assert len(multinetwork.history) == 1

    def test_model_names(self, multinetwork):
        assert multinetwork.model_names == ['forecaster']

    def test_model_subnets(self, multinetwork):
        for name in multinetwork.model_names:
            assert name in multinetwork.model
            assert isinstance(multinetwork.model[name], Model)

    def test_predict_defaut(self, multinetwork):
        x = np.random.rand(13, 1)
        y = multinetwork.predict(x, model='forecaster')
        assert y.shape == (13, 1)

    def test_predict_set_params(self, multinetwork):
        multinetwork.set_params(forecaster_features=2)
        x = np.random.rand(13, 2)
        y = multinetwork.predict(x, model='forecaster')
        assert y.shape == (13, 1)

    def test_fit_default(self, multinetwork):
        x = np.random.rand(13, 1)
        y = x * 2
        metrics0 = multinetwork.evaluate(x, y, model='forecaster')
        multinetwork.fit(x, y, model='forecaster', batch_size=100, epochs=1)
        metrics = multinetwork.evaluate(x, y, model='forecaster')
        for metric0, metric in zip(metrics0, metrics):
            assert metric < metric0

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_fit_generator(self, multinetwork, batch_size):
        x = np.random.rand(13, 1)
        y = np.random.rand(13, 1)
        metrics0 = multinetwork.evaluate(x, y, model='forecaster')
        generator = ArrayBatchGenerator(x, y, batch_size=batch_size)
        multinetwork.fit_generator(generator, model='forecaster', epochs=1)
        metrics = multinetwork.evaluate(x, y, model='forecaster')
        for metric0, metric in zip(metrics0, metrics):
            assert metric < metric0

    def test_validation_data(self, multinetwork):
        x = np.random.rand(13, 1)
        y = np.random.rand(13, 1)
        x_val = np.random.rand(13, 1)
        y_val = np.random.rand(13, 1)
        num_epochs = 1
        multinetwork.fit(
            x,
            y,
            model='forecaster',
            batch_size=100,
            epochs=num_epochs,
            validation_data=(x_val, y_val)
        )
        history = multinetwork.history[-1]['history']
        assert len(history['val_loss']) == (num_epochs)
        assert history['val_loss'][0] > 0
        assert history['val_loss'][0] >= history['val_loss'][-1]

    def test_score(self, multinetwork):
        x = np.random.rand(13, 1)
        y = np.random.rand(13, 1)
        num_epochs = 1
        model = 'forecaster'
        multinetwork.fit(x, y, model=model, batch_size=100, epochs=num_epochs)
        score_value = multinetwork.score(x, y, model=model)
        history = multinetwork.history[-1]['history']
        assert score_value >= 0
        assert history['loss'][0] >= (score_value)

    def test_evaluate(self, multinetwork):
        x = np.random.rand(13, 1)
        y = np.random.rand(13, 1)
        num_epochs = 2
        model = 'forecaster'
        multinetwork.fit(x, y, model=model, batch_size=100, epochs=num_epochs)
        mse, mae = multinetwork.evaluate(x, y, model=model)
        history = multinetwork.history[-1]['history']
        assert mse >= 0
        assert mae >= 0
        assert history['loss'][0] >= mse
        if "mean_absolute_error" in history:
            assert history['mean_absolute_error'][0] >= mae
        else:
            assert history['mae'][0] >= mae

    @pytest.mark.parametrize('batch_size', [1, 2, 2 ** 10])
    def test_evaluate_generator(self, multinetwork, batch_size):
        x = np.random.rand(12, 1)
        y = np.random.rand(12, 1)
        # evaluate_generator appears to average batch losses while
        # ignoring the number of samples per batch when different
        # This is apparent when the last batch is smaller.
        generator = ArrayBatchGenerator(x, y, batch_size=batch_size)
        error0 = multinetwork.evaluate(x, y, model='forecaster')
        error = multinetwork.evaluate_generator(generator, model='forecaster')
        npt.assert_allclose(error, error0, rtol=1e-6)

    @pytest.mark.parametrize('models', [None, 'forecaster', ['forecaster']])
    def test_trainable_models_sets_internal_state(self, multinetwork, models):
        multinetwork.trainable_models = models
        assert multinetwork._trainable_models == models

    @pytest.mark.parametrize('model', [None, '', []])
    def test__freeze_sets_trainable_none(self, multinetwork, model):
        multinetwork._freeze_models_except(model)
        all_models = multinetwork.model_names
        for m in all_models:
            for layer in iterlayers(multinetwork.model[m]):
                assert layer.trainable is False

    def test__freeze_sets_trainable_all(self, multinetwork):
        all_models = multinetwork.model_names
        multinetwork._freeze_models_except(all_models)
        for m in all_models:
            for layer in iterlayers(multinetwork.model[m]):
                assert layer.trainable is True

    def test__freeze_sets_trainable_except(self, multinetwork):
        model = 'forecaster'
        all_models = multinetwork.model_names
        multinetwork._freeze_models_except(model)
        for m in all_models:
            if m == model:
                for layer in iterlayers(multinetwork.model[m]):
                    assert layer.trainable is True
            else:
                for layer in iterlayers(multinetwork.model[m]):
                    assert layer.trainable is False

    def test_freeze(self, multinetwork, mocker):
        multinetwork._freeze_models_except = mocker.MagicMock()
        multinetwork._freeze()
        assert multinetwork._freeze_models_except.called_once_with(
            multinetwork.trainable_models
        )

    def test_freeze_with_arg(self, multinetwork, mocker):
        multinetwork._freeze_models_except = mocker.MagicMock()
        trainable_models = ['forecaster', 'classifier']
        multinetwork._freeze(trainable_models)
        assert multinetwork._freeze_models_except.called_once_with(
            trainable_models
        )

    def test_unfreeze(self, multinetwork, mocker):
        multinetwork._freeze_models_except = mocker.MagicMock()
        multinetwork._unfreeze()
        assert multinetwork._freeze_models_except.called_once_with(
            multinetwork.model_names
        )

    def test_training_context(self, multinetwork, mocker):
        multinetwork._freeze = mocker.MagicMock()
        multinetwork._unfreeze = mocker.MagicMock()

        with multinetwork._training_context():
            assert multinetwork._freeze.called_once_with()
            assert not multinetwork._unfreeze.called

        assert multinetwork._freeze.called_once_with()
        assert multinetwork._unfreeze.called_once_with()

    def test_training_context_with_trainable_models(
        self, multinetwork, mocker
    ):
        multinetwork._freeze = mocker.MagicMock()
        multinetwork._unfreeze = mocker.MagicMock()
        trainable_models = ['forecaster', 'classifier']
        with multinetwork._training_context(trainable_models=trainable_models):
            assert multinetwork._freeze.called_once_with(
                trainable_models=trainable_models
            )
            assert not multinetwork._unfreeze.called

        assert multinetwork._freeze.called_once_with()
        assert multinetwork._unfreeze.called_once_with()

    def test_training_context_preserves_loss_and_metrics(self, multinetwork):
        loss = multinetwork.model['forecaster'].loss
        metrics = multinetwork.model['forecaster'].metrics

        with multinetwork._training_context():
            context_loss = multinetwork.model['forecaster'].loss
            context_metrics = multinetwork.model['forecaster'].metrics
            assert context_loss == loss
            for metric, context_metric in itertools.zip_longest(
                metrics, context_metrics
            ):
                assert metric.get_config() == context_metric.get_config()

    def test_fit_frozen(self, multinetwork):
        """Assert fitting a frozen model does nothing.

        - all weights remain same.
        - error remains same.
        """
        x = np.random.rand(13, 1)
        y = np.random.rand(13, 1)
        error0 = multinetwork.evaluate(x, y, model='forecaster')
        weights0 = multinetwork.model['forecaster'].get_weights()
        multinetwork.trainable_models = None
        multinetwork.fit(x, y, model='forecaster', batch_size=100, epochs=3)
        error = multinetwork.evaluate(x, y, model='forecaster')
        weights = multinetwork.model['forecaster'].get_weights()
        for w0, w in zip(weights0, weights):
            npt.assert_equal(w0, w)
        assert error == error0

    def test_fit_frozen_via_kwarg(self, multinetwork):
        """Assert fitting a frozen model does nothing.

        Pass trainable_models as kwarg to fit.
        - all weights remain same.
        - error remains same.
        """
        x = np.random.rand(13, 1)
        y = np.random.rand(13, 1)
        error0 = multinetwork.evaluate(x, y, model='forecaster')
        weights0 = multinetwork.model['forecaster'].get_weights()

        assert multinetwork.trainable_models == ['forecaster']

        multinetwork.fit(
            x, y, model='forecaster',
            trainable_models=[],
            batch_size=100, epochs=3
        )
        error = multinetwork.evaluate(x, y, model='forecaster')
        weights = multinetwork.model['forecaster'].get_weights()
        for w0, w in zip(weights0, weights):
            npt.assert_equal(w0, w)
        assert error == error0


class TestMultiNetworkSerialization:
    @pytest.fixture
    def multinetwork(self, random):
        return SimpleMultiNetwork()

    @pytest.fixture
    def bn_multinetwork(self, random):
        return BatchNormNetwork()

    @pytest.fixture
    def ef_multinetwork(self, random):
        return EmbedderForecasterNetwork()

    def assert_models_same(self, model, model2):
        """Assert two keras models same."""
        if model.optimizer:
            assert (
                model.optimizer.get_config() == model2.optimizer.get_config()
            )
        assert len(model.layers) == len(model2.layers)  # shallow comparison
        layers = list(iterlayers(model))
        layers2 = list(iterlayers(model2))
        assert len(layers) == len(layers2)  # deep comparison

    def assert_model_dicts_same(self, model_dict, model_dict_2):
        assert model_dict.keys() == model_dict_2.keys()

        for model_name in model_dict.keys():
            model = model_dict[model_name]
            model2 = model_dict_2[model_name]
            self.assert_models_same(model, model2)

    def test_deserialized_params(self, ef_multinetwork):
        params = ef_multinetwork.get_params()
        s = dumps(ef_multinetwork)
        new_multinetwork = loads(s)
        new_params = new_multinetwork.get_params()
        assert new_params == params

    def test_deserialized_gradients(self, ef_multinetwork):
        ef_multinetwork._init_model()
        s = dumps(ef_multinetwork)
        ef_multinetwork = loads(s)
        model = ef_multinetwork.model['combined']
        grads = K.gradients(model.total_loss, model.trainable_weights)
        assert all([g is not None for g in grads])

    def test_deserialized_batch_norm(self, bn_multinetwork):
        multinetwork = bn_multinetwork
        multinetwork.trainable_models = None
        s = dumps(multinetwork)
        new_multinetwork = loads(s)
        self.assert_model_dicts_same(
            multinetwork.model, new_multinetwork.model
        )

    def test_layer_sharing(self, ef_multinetwork):
        multinetwork = ef_multinetwork
        s = dumps(multinetwork)
        new_multinetwork = loads(s)
        self.assert_model_dicts_same(
            multinetwork.model, new_multinetwork.model
        )

    def test_history_preserved(self, multinetwork):
        record = {'model': None}
        multinetwork.history = [record]
        s = dumps(multinetwork)
        new_multinetwork = loads(s)
        assert new_multinetwork.history == [record]

    def test_optimizer_state(self, multinetwork):
        with multinetwork._training_context():
            lr_init = K.get_value(
                multinetwork.model['forecaster'].optimizer.lr
            )
            lr_changed = lr_init + 1.
            K.set_value(
                multinetwork.model['forecaster'].optimizer.lr, lr_changed
            )
        s = dumps(multinetwork)
        new_multinetwork = loads(s)
        lr_new = K.get_value(new_multinetwork.model['forecaster'].optimizer.lr)
        assert np.allclose(lr_new, lr_changed), 'Optimizer loaded incorrectly'

    def test_serialize_no_keras(self, multinetwork, mocker):
        import timeserio.keras.multinetwork

        # replace `keras` with a broken non-module
        mocker.patch.object(
            timeserio.keras.multinetwork, "keras", None
        )

        params = multinetwork.get_params()
        s = dumps(multinetwork)

        # loading fails because HABEMUS_KERAS is still True
        with pytest.raises(AttributeError):
            new_multinetwork = loads(s)

        # if both are mocked correctly, "unsafe" unpickling works again
        mocker.patch.object(
            timeserio.keras.multinetwork, "HABEMUS_KERAS", False
        )
        new_multinetwork = loads(s)
        new_params = new_multinetwork.get_params()

        assert new_params == params
