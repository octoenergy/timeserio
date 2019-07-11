import pytest
import keras.backend as K  # noqa
from keras.layers import Concatenate, Dense, Input
from keras.models import Model, Sequential

from timeserio.keras.utils import iterlayers


def model_single_layer():
    model = Dense(10)
    return model


def model_sequential():
    model = Sequential()
    model.add(Dense(10))
    model.add(Dense(20))
    return model


def model_compiled():
    model = Sequential()
    model.add(Dense(10))
    model.add(Dense(20))
    model.compile(optimizer='sgd', loss='mse')
    return model


def model_with_shared_layer():
    x1 = Input((10, ))
    x2 = Input((10, ))
    dense = Dense(1)
    concat = Concatenate()([dense(x1), dense(x2)])
    model = Model([x1, x2], concat)
    return model


def model_with_submodel():
    submodel = Sequential()
    submodel.add(Dense(10))
    submodel.add(Dense(20))
    model = Sequential()
    model.add(submodel)
    model.add(Dense(30))
    return model


def model_with_shared_in_submodel():
    x = Input((10, ))
    dense = Dense(10)
    submodel = Model(x, dense(x))
    model = Model(x, dense(submodel(x)))
    return model


class TestIterLayers:
    """Test iteration over complex model layers."""

    @pytest.mark.parametrize(
        'model, n_layers',
        [
            (None, 1),  # Let's see if that's what we want
            (model_single_layer(), 1),
            (model_sequential(), 2),
            (model_compiled(), 2),
            (model_with_submodel(), 3)
        ]
    )
    def test_model(self, model, n_layers):
        layers = list(iterlayers(model))
        assert len(layers) == n_layers

    def test_multimodel(self, multimodel, multimodel_num_layers):
        for name, model in multimodel.items():
            layers = list(iterlayers(model))
            expected_num_layers = multimodel_num_layers[name]
            assert len(layers) == expected_num_layers
