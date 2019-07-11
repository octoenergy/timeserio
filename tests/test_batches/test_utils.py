import pytest

from timeserio.batches.utils import ceiling_division


@pytest.mark.parametrize(
    'dividend, divisor, result', [
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (1, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (2, 1, 2),
    ]
)
def test_ceiling_division(dividend, divisor, result):
    assert ceiling_division(dividend, divisor) == result
