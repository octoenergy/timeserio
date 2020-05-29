import pytest

from timeserio.keras.utils import seed_random


@pytest.fixture
def random():
    """Seed all random number generators.

    Ensures repeatable tests.
    """
    with seed_random():
        yield
