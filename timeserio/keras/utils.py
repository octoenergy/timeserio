from contextlib import contextmanager
import inspect
import os
import random as py_random
from typing import Iterable

import numpy as np

from timeserio.externals import tensorflow as tf, keras


def iterlayers(model: keras.layers.Layer) -> Iterable[keras.layers.Layer]:
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


def has_arg(fn, name, accept_all=False):
    """Check if a callable accepts a given keyword argument.

    See https://github.com/tensorflow/tensorflow/pull/37004

    Arguments:
      fn: Callable to inspect.
      name: Check if `fn` can be called with `name` as a keyword argument.
      accept_all: What to return if there is no parameter called `name` but the
        function accepts a `**kwargs` argument.
    Returns:
      bool, whether `fn` accepts a `name` keyword argument.
    """
    arg_spec = inspect.getfullargspec(fn)
    if accept_all and arg_spec.varkw is not None:
        return True
    return name in arg_spec.args or name in arg_spec.kwonlyargs


@contextmanager
def seed_random(seed=42):
    """Seed all random number generators to ensure repeatable tests.

    Sets python, `numpy`, and `tensorflow` random seeds
    to a repeatable states. This is useful in tests, but should not be
    used in production.

    https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    """
    py_random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    yield
