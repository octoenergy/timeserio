from keras.utils import Sequence


class ArrayBatchGenerator(Sequence):
    """Generate batches from X, y arrays.

    Mainly used for testing `fit_generator` and related functions
    """

    def __init__(self, x, y, batch_size=4):
        self.x = x
        self.y = y
        if len(self.y.shape) == 1:
            self.y = self.y.reshape((-1, 1))
        assert self.x.shape[0] == self.y.shape[0]
        self.batch_size = batch_size
        self.n_rows = self.x.shape[0]

    def __len__(self):
        return self.n_rows // self.batch_size + (
            1 if self.n_rows % self.batch_size else 0
        )

    def __getitem__(self, item):
        item = item % len(self)
        start_row = item * self.batch_size
        end_row = min((item + 1) * self.batch_size, self.n_rows)
        return self.x[start_row:end_row, :], self.y[start_row:end_row, :]
