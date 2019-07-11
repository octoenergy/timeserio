import pytest

import numpy as np

from timeserio import ini
from timeserio.data.mock import mock_dask_raw_data
from timeserio.batches.chunked.parquet import (
    RowBatchGenerator, SequenceForecastBatchGenerator
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
        self, n_points, batch_size, batch_aggregator, expected_nb_batches,
        tmpdir
    ):
        n_customers = 2
        ids = np.arange(n_customers)
        ddf = mock_dask_raw_data(periods=n_points, ids=ids)
        path = str(tmpdir.mkdir('data'))
        ddf.to_parquet(path)
        generator = RowBatchGenerator(
            path=path,
            batch_size=batch_size,
            columns=[ini.Columns.target],
            batch_aggregator=batch_aggregator
        )
        assert len(generator) == expected_nb_batches


class TestSequenceForecastBatchGeneratorFromParquet:
    @pytest.mark.parametrize('n_customers', [
        1,
        2,
        3,
        4,
        5,
    ])
    def test_n_subgens(self, n_customers, tmpdir):
        ids = np.arange(n_customers)
        periods = 4
        ddf = mock_dask_raw_data(periods=periods, ids=ids)
        path = str(tmpdir.mkdir('data'))
        ddf.to_parquet(path)
        generator = SequenceForecastBatchGenerator(
            path=path,
            sequence_length=2,
            forecast_steps_max=1,
        )
        assert len(generator.files) == n_customers
        assert len(generator.subgen_lengths) == n_customers
        assert len(generator.subgen_index_bounds) == n_customers + 1

    def test_subgen_lengths(self, tmpdir):
        exp_sg_len = 1
        n_customers = 3
        ids = np.arange(n_customers)
        ddf = mock_dask_raw_data(periods=3, ids=ids)
        path = str(tmpdir.mkdir('data'))
        ddf.to_parquet(path)
        generator = SequenceForecastBatchGenerator(
            path=path,
            sequence_length=1,
            forecast_steps_max=1,
        )
        assert all(sgl == exp_sg_len for sgl in generator.subgen_lengths)

    @pytest.mark.parametrize(
        'batch_size, batch_idx, exp_subgen_idx, exp_idx_in_subgen', [
            (None, 0, 0, 0),
            (None, 1, 1, 0),
            (None, 2, 2, 0),
        ]
    )
    def test_find_batch_in_subgens(
        self, batch_size, batch_idx, exp_subgen_idx, exp_idx_in_subgen, tmpdir
    ):
        n_customers = 3
        ids = np.arange(n_customers)
        ddf = mock_dask_raw_data(periods=3, ids=ids)
        path = str(tmpdir.mkdir('data'))
        ddf.to_parquet(path)
        generator = SequenceForecastBatchGenerator(
            path=path,
            sequence_length=1,
            forecast_steps_max=1,
            batch_size=batch_size,
        )
        subgen_idx, idx_in_subgen = generator.find_subbatch_in_subgens(
            batch_idx
        )
        assert subgen_idx == exp_subgen_idx
        assert idx_in_subgen == exp_idx_in_subgen

    def test_find_batch_raises_outside_subgens(self, tmpdir):
        n_customers = 3
        ids = np.arange(n_customers)
        ddf = mock_dask_raw_data(periods=3, ids=ids)
        path = str(tmpdir.mkdir('data'))
        ddf.to_parquet(path)
        generator = SequenceForecastBatchGenerator(
            path=path,
            sequence_length=1,
            forecast_steps_max=1,
        )
        batch_idx = 2 ** 10
        with pytest.raises(IndexError):
            generator.find_subbatch_in_subgens(batch_idx)

    @pytest.mark.parametrize('batch_aggregator, exp_gen_len', [(1, 2), (2, 1)])
    def test_aggregate_ids(self, batch_aggregator, exp_gen_len, tmpdir):
        n_customers = 2
        ids = np.arange(n_customers)
        ddf = mock_dask_raw_data(periods=3, ids=ids)
        path = str(tmpdir.mkdir('data'))
        ddf.to_parquet(path)
        generator = SequenceForecastBatchGenerator(
            path=path,
            sequence_length=2,
            forecast_steps_max=1,
            batch_aggregator=batch_aggregator
        )
        assert len(generator) == exp_gen_len
        batch = generator[0]
        assert len(batch) == batch_aggregator
