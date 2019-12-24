from typing import Dict

import pytest
from keras.layers import LSTM, Concatenate, Dense, Input
from keras.models import Model


@pytest.fixture
def multimodel() -> Dict[str, Model]:
    """Toy example of inter-connected `keras` models."""
    encoder_features = 1
    encoder_lstm_units = (8, )
    forecaster_hidden_units = (8, 8)
    forecaster_features = 5

    encoder_input = Input(
        (None, encoder_features), name='encoder_input_sequence'
    )
    encoder_output = encoder_input
    for idx, units in enumerate(encoder_lstm_units):
        encoder_output = LSTM(
            units=units,
            return_sequences=True
            if idx < len(encoder_lstm_units) - 1 else False,
            name='encoder_lstm_{}'.format(idx)
        )(encoder_output)
    encoder_model = Model(encoder_input, encoder_output, name='Encoder')

    # Forecaster
    forecaster_input_encoding = Input(
        (encoder_model.output_shape[-1], ), name='forecaster_input_encoding'
    )
    forecaster_input_features = Input(
        (forecaster_features, ), name='forecaster_input_features'
    )
    forecaster_output = Concatenate(name='forecaster_concatenate_inputs')(
        [forecaster_input_encoding, forecaster_input_features]
    )
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
        [forecaster_input_encoding, forecaster_input_features],
        forecaster_output,
        name='Forecaster'
    )

    # Combined model
    combined_output = forecaster_model(
        [encoder_model(encoder_input), forecaster_input_features]
    )
    combined_model = Model(
        [encoder_input, forecaster_input_features], combined_output
    )
    combined_model.compile(optimizer='sgd', loss='mse')

    return {
        'encoder': encoder_model,
        'forecaster': forecaster_model,
        'combined': combined_model,
    }


@pytest.fixture
def multimodel_num_layers() -> Dict[str, int]:
    """Number of layers in each of sub-models in `multimodel` fixture."""
    return {
        'encoder': 2,
        'forecaster': 6,
        'combined': 10,
    }
