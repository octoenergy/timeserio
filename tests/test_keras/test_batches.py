import pytest

import numpy as np

from timeserio.keras.batches import ArrayBatchGenerator


def x_y_data(n_points):
    x = np.random.rand(n_points, 2)
    y = np.random.rand(n_points, 1)
    return x, y


class TestArrayBatchGenerator:
    @pytest.mark.parametrize(
        'n_points, batch_size, expected_nb_batches', [
            (0, 2, 0),
            (1, 2, 1),
            (2, 2, 1),
            (4, 2, 2),
            (5, 2, 3),
        ]
    )
    def test_nb_batches(self, n_points, batch_size, expected_nb_batches):
        x, y = x_y_data(n_points)
        generator = ArrayBatchGenerator(
            x=x, y=y, batch_size=batch_size
        )
        assert len(generator) == expected_nb_batches

    @pytest.mark.parametrize(
        'n_points, batch_size, expected_last_batch_size',
        [(4, 2, 2), (5, 2, 1), (7, 3, 1)]
    )
    def test_batch_size(self, n_points, batch_size, expected_last_batch_size):
        x, y = x_y_data(n_points)
        generator = ArrayBatchGenerator(
            x=x, y=y, batch_size=batch_size
        )
        for batch_id in range(len(generator) - 1):
            x_batch, y_batch = generator[batch_id]
            assert len(x_batch) == batch_size
            assert len(y_batch) == batch_size
            assert x_batch.shape[1] == x.shape[1]
            assert y_batch.shape[1] == y.shape[1]

        for batch_id in range(len(generator) - 1, len(generator)):
            x_batch, y_batch = generator[batch_id]
            assert len(x_batch) == expected_last_batch_size
            assert len(y_batch) == expected_last_batch_size
            assert x_batch.shape[1] == x.shape[1]
            assert y_batch.shape[1] == y.shape[1]
