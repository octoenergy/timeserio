import abc

from keras.utils import Sequence


class BatchGenerator(Sequence, abc.ABC):
    pass
