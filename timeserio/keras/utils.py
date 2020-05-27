from contextlib import contextmanager
import os
import random as py_random
from typing import Iterable

import numpy as np

from timeserio.externals import keras
from timeserio.externals import tensorflow as tf


def iterlayers(model: "keras.engine.Layer") -> Iterable["keras.engine.Layer"]:
    """
    Return iterable over all layers (and sub-layers) of a model.

    This works because a keras Model is a sub-class of Layer.
    Can be used for freezing/un-freezing layers, etc.
    """
    if hasattr(model, 'layers'):
        for layer in model.layers:
            yield from iterlayers(layer)
    else:
        yield model


@contextmanager
def seed_random(seed=42):
    """Seed all random number generators to ensure repeatable tests.

    Sets python, `numpy`, and `tensorflow` random seeds
    to a repeatable states. This is useful in tests, but should not be
    used in production.

    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """
    os.environ['PYTHONHASHSEED'] = f'{seed}'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    py_random.seed(seed)
    np.random.seed(seed)
    tf.reset_default_graph()

    graph = tf.Graph()
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    session = tf.Session(graph=graph, config=config)
    keras.backend.set_session(session)
    with tf.device("/cpu:0"), graph.as_default(), session.as_default():
        tf.set_random_seed(seed)
        graph.seed = seed
        yield
