import s3fs
import joblib.externals.cloudpickle as cp

__all__ = ['dumpf', 'loadf']


def open_url(filename, mode):
    """Open file from local drive or s3 bucket.

    S3 filename must start with `s3://`.
    """
    if filename.startswith('s3://'):
        s3 = s3fs.S3FileSystem()
        file = s3.open(filename, mode)
    else:
        file = open(filename, mode)
    return file


def dumpf(obj, filename):
    """Serialize object to file of given name."""
    with open_url(filename, 'wb') as file:
        cp.dump(obj, file)


def loadf(filename):
    """Load serialized object from file of given name."""
    with open_url(filename, 'rb') as file:
        obj = cp.load(file)
    return obj


def dumps(obj):
    """Serialize object to string."""
    return cp.dumps(obj)


def loads(s):
    """Load serialized object from string."""
    return cp.loads(s)
