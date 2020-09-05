import numpy as np
import pytest

import timeserio.ini as ini
from timeserio.data.mock import mock_fit_data, mock_raw_data
from timeserio.batches.chunked.pandas import (
    SequenceForecastBatchGenerator,
    RowBatchGenerator,
)


class TestRowBatchGenerator:
    @pytest.mark.parametrize(
        'n_points, batch_size, batch_aggregator, expected_nb_batches', [
            (5, 2, 1, 6),
            (5, None, 1, 2),
            (5, None, 2, 1),
        ]
    )
    def test_nb_batches(
        self, n_points, batch_size, batch_aggregator, expected_nb_batches
    ):
        n_customers = 2
        ids = np.arange(n_customers)
        df = mock_raw_data(periods=n_points, ids=ids)
        generator = RowBatchGenerator(
            df=df,
            batch_size=batch_size,
            columns=[ini.Columns.target],
            batch_aggregator=batch_aggregator
        )
        assert len(generator) == expected_nb_batches


class TestSequenceForecastBatchGeneratorMultiID:
    @pytest.mark.parametrize('n_customers', [
        1,
        2,
        3,
        4,
        5,
    ])
    def test_n_subgens(self, n_customers, use_tensor_extension):
        ids = np.arange(n_customers)
        df = mock_fit_data(
            periods=4, ids=ids, use_tensor_extension=use_tensor_extension
        )
        generator = SequenceForecastBatchGenerator(
            df=df,
            sequence_length=2,
            forecast_steps_max=1,
            batch_size=2 ** 10,
        )
        assert len(generator.chunks) == n_customers
        assert len(generator.subgen_lengths) == n_customers
        assert len(generator.subgen_index_bounds) == n_customers + 1

    @pytest.mark.parametrize('n_customers', [
        1,
        2,
    ])
    @pytest.mark.parametrize(
        'batch_size, exp_sg_len', [
            (1, 2),
            (2 ** 10, 1),
            (None, 1),
        ]
    )
    def test_subgen_lengths(
        self, n_customers, batch_size, exp_sg_len, use_tensor_extension
    ):
        n_customers = 3
        ids = np.arange(n_customers)
        df = mock_fit_data(
            periods=3, ids=ids, use_tensor_extension=use_tensor_extension
        )
        generator = SequenceForecastBatchGenerator(
            df=df,
            sequence_length=1,
            forecast_steps_max=1,
            batch_size=batch_size,
        )
        assert all(sgl == exp_sg_len for sgl in generator.subgen_lengths)

    @pytest.mark.parametrize(
        'batch_size, batch_idx, exp_subgen_idx, exp_idx_in_subgen', [
            (1, 0, 0, 0),
            (1, 1, 0, 1),
            (1, 2, 1, 0),
            (1, 3, 1, 1),
            (2 ** 10, 0, 0, 0),
            (2 ** 10, 1, 1, 0),
            (2 ** 10, 2, 2, 0),
            (None, 2, 2, 0),
        ]
    )
    def test_find_batch_in_subgens(
        self, batch_size, batch_idx, exp_subgen_idx, exp_idx_in_subgen,
        use_tensor_extension
    ):
        n_customers = 3
        ids = np.arange(n_customers)
        df = mock_fit_data(
            periods=3, ids=ids, use_tensor_extension=use_tensor_extension
        )
        generator = SequenceForecastBatchGenerator(
            df=df,
            sequence_length=1,
            forecast_steps_max=1,
            batch_size=batch_size,
        )
        subgen_idx, idx_in_subgen = generator.find_subbatch_in_subgens(
            batch_idx
        )
        assert subgen_idx == exp_subgen_idx
        assert idx_in_subgen == exp_idx_in_subgen

    def test_find_batch_raises_outside_subgens(self, use_tensor_extension):
        n_customers = 3
        ids = np.arange(n_customers)
        df = mock_fit_data(
            periods=3, ids=ids, use_tensor_extension=use_tensor_extension
        )
        generator = SequenceForecastBatchGenerator(
            df=df,
            sequence_length=1,
            forecast_steps_max=1,
            batch_size=2 ** 10,
        )
        batch_idx = 2 ** 10
        with pytest.raises(IndexError):
            generator.find_subbatch_in_subgens(batch_idx)

    def test_aggregate_ids(self, use_tensor_extension):
        n_customers = 2
        ids = np.arange(n_customers)
        df = mock_fit_data(
            periods=3, ids=ids, use_tensor_extension=use_tensor_extension
        )
        generator = SequenceForecastBatchGenerator(
            df=df,
            sequence_length=2,
            forecast_steps_max=1,
            batch_size=2,
            batch_aggregator=2
        )
        assert len(generator) == 1
        batch = generator[0]
        assert len(batch) == 2
