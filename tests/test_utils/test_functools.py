import pytest

from timeserio.utils.functools import get_default_args


def func1(a, b):
    return


def func2(a, *, b):
    return


def func3(a, b=3):
    return


def func4(a='a', b=3):
    return


@pytest.mark.parametrize(
    'func, args', [
        (func1, {}),
        (func2, {}),
        (func3, {
            'b': 3
        }),
        (func4, {
            'a': 'a',
            'b': 3
        }),
    ]
)
def test_get_default_args(func, args):
    assert get_default_args(func) == args
