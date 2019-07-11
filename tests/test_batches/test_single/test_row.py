import pytest

import timeserio.ini as ini
from timeserio.data.mock import mock_fit_data
from timeserio.batches.single.row import RowBatchGenerator


class TestRowBatchGenerator:
    @pytest.mark.parametrize(
        'n_points, batch_size, expected_nb_batches', [
            (0, 2, 0),
            (1, 2, 1),
            (2, 2, 1),
            (4, 2, 2),
            (5, 2, 3),
            (0, None, 0),
            (5, None, 1),
        ]
    )
    def test_nb_batches(self, n_points, batch_size, expected_nb_batches):
        df = mock_fit_data(periods=n_points, ids=[0])
        generator = RowBatchGenerator(
            df=df, batch_size=batch_size, columns=[ini.Columns.target]
        )
        assert len(generator) == expected_nb_batches

    @pytest.mark.parametrize(
        'n_points, batch_size, expected_last_batch_size',
        [(4, None, 4), (4, 2, 2), (5, 2, 1), (7, 3, 1)]
    )
    def test_batch_size(self, n_points, batch_size, expected_last_batch_size):
        df = mock_fit_data(periods=n_points, ids=[0])
        generator = RowBatchGenerator(
            df=df, batch_size=batch_size, columns=[ini.Columns.target]
        )
        for batch_id in range(len(generator) - 1):
            batch = generator[batch_id]
            assert len(batch) == batch_size

        assert len(generator[-1]) == expected_last_batch_size
