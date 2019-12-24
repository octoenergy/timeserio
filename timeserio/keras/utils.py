import os
import random
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


def seed_random():
    """Seed all random number generators to ensure repeatable tests.

    Sets python, `numpy`, and `tensorflow` random seeds
    to a repeatable states. This is useful in tests, but should not be
    used in production.

    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    random.seed(12345)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    tf.set_random_seed(1234)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    keras.backend.set_session(sess)
