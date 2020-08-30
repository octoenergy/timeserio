import pytest

from timeserio.keras.utils import seed_random
from timeserio.externals import HABEMUS_TENSOR_EXT


@pytest.fixture
def random():
    """Seed all random number generators.

    Ensures repeatable tests.
    """
    with seed_random():
        yield


@pytest.fixture(params=[False, True] if HABEMUS_TENSOR_EXT else [])
def use_tensor_extension(request):
    return request.param
