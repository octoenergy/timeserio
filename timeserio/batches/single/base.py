import abc

from timeserio.externals import keras


class BatchGenerator(keras.utils.Sequence, abc.ABC):
    pass
