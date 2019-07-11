"""
A set of tools for determining output latency (delay and stride).

For temporal NN models.
"""
import keras.layers as kl
import numpy as np


def _n(num_or_tuple):
    """Get value from number or one-elt tuple/list."""
    try:
        num = num_or_tuple[0]
    except TypeError:
        num = num_or_tuple
    return num


def total_latency(relative_latencies):
    """
    Compute absolute delay + stride for last layer in a stack.

    Arguments:
    ----------
    relative_latencies: list of tuples
        each tuple is (rel_delay, rel_stride)

    Returns:
    --------
    delay: float
    stride: int

    """
    return absolute_latencies(relative_latencies)[-1]


def absolute_latencies(relative_latencies):
    """
    Compute absolute delay + stride for a each in a stack of layers.

    Arguments:
    ----------
    relative_latencies: list of tuples
        each tuple is (rel_delay, rel_stride)

    Returns:
    --------
    absolute_latencies: list of tuples
        each tuple is (abs_stride, abs_stride)

    """
    abs_delay, abs_stride = 0, 1
    absolute_latencies = []
    for idx, (rel_delay, rel_stride) in enumerate(relative_latencies):
        abs_delay += abs_stride * rel_delay
        abs_stride *= rel_stride
        absolute_latencies.append((abs_delay, abs_stride))
    return absolute_latencies


def layer_latency(layer):
    """
    Get latency for a given keras layer.

    Arguments:
    ----------
    layer: Layer
        a Keras layer

    Returns:
    --------
    delay: float
    stride: int

    """
    if isinstance(layer, kl.Conv1D):
        # https://keras.io/layers/convolutional/#conv1d
        stride = _n(layer.strides)
        if layer.padding == 'valid':
            delay = (_n(layer.kernel_size) - 1) * _n(layer.dilation_rate) / 2
        elif layer.padding == 'same':
            delay = 0
        else:
            raise ValueError('Padding {} not supported'.format(layer.padding))
    elif isinstance(layer, (kl.MaxPool1D, kl.AvgPool1D)):
        # https://keras.io/layers/pooling/#maxpooling1d
        stride = _n(layer.strides) if layer.strides else layer.padding
        if layer.padding == 'valid':
            delay = (_n(layer.pool_size) - 1) / 2
        elif layer.padding == 'same':
            delay = 0
    else:
        delay, stride = 0, 1

    return delay, stride


def model_latency(model):
    """
    Get latency for a given keras temporal model.

    Arguments:
    ----------
    model: keras.Model
        a Keras multi-layer temporal model

    Returns:
    --------
    delay: float
    stride: int

    """
    relative_latencies = [layer_latency(layer) for layer in model.layers]
    delay, stride = total_latency(relative_latencies)
    return delay, stride


def model_output_length(model, input_shape) -> int:
    """
    Get number of output timesteps for keras temporal model.

    Arguments:
    ----------
    model: keras.Model
        a Keras multi-layer temporal model

    input_shape: tuple
        shape of a single input

    """
    try:
        len(input_shape)
    except TypeError:
        input_shape = (input_shape, )

    if len(input_shape) == 1:
        batch_shape = (1, input_shape[0], 1)
    elif len(input_shape) == 2:
        batch_shape = (1, input_shape[0], input_shape[1])
    elif len(input_shape) == 3:
        batch_shape = input_shape
    else:
        raise ValueError(
            'input_shape must specify shape of single sample or batch, '
            'but found too many dimensions'
        )

    input_ = np.zeros(batch_shape)
    output = model.predict(input_)

    output_length = output.shape[1]
    return output_length
