"""See
https://github.com/cloudpipe/cloudpickle/blob/master/tests/cloudpickle_test.py
Our aim is not to test cloudpickle but to make sure we use an implementation
that can pickle lambdas etc.
"""
import tempfile

import pytest

from timeserio.utils.pickle import dumpf, loadf, dumps, loads


def assert_is(a, b):
    assert a is b


def assert_eq(a, b):
    assert a == b


def assert_lambda_eq(a, b):
    for x in range(10):
        assert a(x) == b(x)


@pytest.mark.parametrize(
    'obj, assert_object_same', [
        (None, assert_is),
        ([1, 2, 3], assert_eq),
        (lambda x: x * x, assert_lambda_eq)
    ]
)
class TestPickling:
    def test_pickle_to_string(self, obj, assert_object_same):
        s = dumps(obj)
        obj2 = loads(s)
        assert_object_same(obj, obj2)

    def test_pickle_to_file(self, obj, assert_object_same):
        with tempfile.NamedTemporaryFile(suffix='.pickle', delete=True) as f:
            dumpf(obj, f.name)
            obj2 = loadf(f.name)
        assert_object_same(obj, obj2)

    def test_pickle_to_s3(
        self,
        s3, test_bucket_name,
        obj, assert_object_same
    ):
        filename = f's3://{test_bucket_name}/path/file.pickle'
        dumpf(obj, filename)
        obj2 = loadf(filename)
        assert_object_same(obj, obj2)


if __name__ == '__main__':
    pytest.main()
