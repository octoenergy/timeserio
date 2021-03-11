import cloudpickle as cp
import fsspec

__all__ = ['dumpf', 'loadf']


def dumpf(obj, filename):
    """Serialize object to file of given name."""
    with fsspec.open(filename, 'wb') as file:
        cp.dump(obj, file)


def loadf(filename):
    """Load serialized object from file of given name."""
    with fsspec.open(filename, 'rb') as file:
        obj = cp.load(file)
    return obj


def dumps(obj):
    """Serialize object to string."""
    return cp.dumps(obj)


def loads(s):
    """Load serialized object from string."""
    return cp.loads(s)
