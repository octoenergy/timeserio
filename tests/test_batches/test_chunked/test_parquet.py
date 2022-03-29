import pytest

import boto3
import moto
import numpy as np
from pytest_lazyfixture import lazy_fixture
import tentaclio as tio

from timeserio import ini
from timeserio.data.mock import mock_dask_raw_data
from timeserio.batches.chunked.parquet import (
    RowBatchGenerator, SequenceForecastBatchGenerator
)


@pytest.fixture
def local_writable_url(tmpdir):
    yield f"file://{tmpdir.mkdir('data')}"


@pytest.fixture
def s3_writable_url():
    test_bucket_name = "test_bucket"
    with moto.mock_s3():
        client = boto3.client("s3", "us-east-1")
        client.create_bucket(Bucket=test_bucket_name)
        url = f"s3://{test_bucket_name}"
        yield url


def _ddf_to_parquet(ddf, path):
    for i, df in enumerate(ddf.partitions):
        with tio.open(f"{path}/chunk_{i}.parquet", "wb") as fh:
            df.compute().to_parquet(fh)


class TestRowBatchGenerator:
    @pytest.mark.parametrize(
        'n_points, batch_size, batch_aggregator, expected_nb_batches', [
            (5, 2, 1, 6),
            (5, None, 1, 2),
            (5, None, 2, 1),
        ]
    )
    @pytest.mark.parametrize(
        'writable_url',
        [lazy_fixture("local_writable_url"), lazy_fixture("s3_writable_url")],
    )
    def test_nb_batches(
        self, n_points, batch_size, batch_aggregator, expected_nb_batches,
        writable_url
    ):
        n_customers = 2
        ids = np.arange(n_customers)
        ddf = mock_dask_raw_data(periods=n_points, ids=ids)
        _ddf_to_parquet(ddf, writable_url)
        generator = RowBatchGenerator(
            path=writable_url,
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
    @pytest.mark.parametrize(
        'writable_url',
        [lazy_fixture("local_writable_url"), lazy_fixture("s3_writable_url")],
    )
    def test_n_subgens(self, n_customers, writable_url):
        ids = np.arange(n_customers)
        periods = 4
        ddf = mock_dask_raw_data(periods=periods, ids=ids)
        _ddf_to_parquet(ddf, writable_url)
        generator = SequenceForecastBatchGenerator(
            path=writable_url,
            sequence_length=2,
            forecast_steps_max=1,
        )
        assert len(generator.files) == n_customers
        assert len(generator.subgen_lengths) == n_customers
        assert len(generator.subgen_index_bounds) == n_customers + 1

    @pytest.mark.parametrize(
        'writable_url',
        [lazy_fixture("local_writable_url"), lazy_fixture("s3_writable_url")],
    )
    def test_subgen_lengths(self, writable_url):
        exp_sg_len = 1
        n_customers = 3
        ids = np.arange(n_customers)
        ddf = mock_dask_raw_data(periods=3, ids=ids)
        _ddf_to_parquet(ddf, writable_url)
        generator = SequenceForecastBatchGenerator(
            path=writable_url,
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
    @pytest.mark.parametrize(
        'writable_url',
        [lazy_fixture("local_writable_url"), lazy_fixture("s3_writable_url")],
    )
    def test_find_batch_in_subgens(
        self,
        batch_size,
        batch_idx,
        exp_subgen_idx,
        exp_idx_in_subgen,
        writable_url
    ):
        n_customers = 3
        ids = np.arange(n_customers)
        ddf = mock_dask_raw_data(periods=3, ids=ids)
        _ddf_to_parquet(ddf, writable_url)
        generator = SequenceForecastBatchGenerator(
            path=writable_url,
            sequence_length=1,
            forecast_steps_max=1,
            batch_size=batch_size,
        )
        subgen_idx, idx_in_subgen = generator.find_subbatch_in_subgens(
            batch_idx
        )
        assert subgen_idx == exp_subgen_idx
        assert idx_in_subgen == exp_idx_in_subgen

    @pytest.mark.parametrize(
        'writable_url',
        [lazy_fixture("local_writable_url"), lazy_fixture("s3_writable_url")],
    )
    def test_find_batch_raises_outside_subgens(self, writable_url):
        n_customers = 3
        ids = np.arange(n_customers)
        ddf = mock_dask_raw_data(periods=3, ids=ids)
        _ddf_to_parquet(ddf, writable_url)
        generator = SequenceForecastBatchGenerator(
            path=writable_url,
            sequence_length=1,
            forecast_steps_max=1,
        )
        batch_idx = 2 ** 10
        with pytest.raises(IndexError):
            generator.find_subbatch_in_subgens(batch_idx)

    @pytest.mark.parametrize('batch_aggregator, exp_gen_len', [(1, 2), (2, 1)])
    @pytest.mark.parametrize(
        'writable_url',
        [lazy_fixture("local_writable_url"), lazy_fixture("s3_writable_url")],
    )
    def test_aggregate_ids(self, batch_aggregator, exp_gen_len, writable_url):
        n_customers = 2
        ids = np.arange(n_customers)
        ddf = mock_dask_raw_data(periods=3, ids=ids)
        _ddf_to_parquet(ddf, writable_url)
        generator = SequenceForecastBatchGenerator(
            path=writable_url,
            sequence_length=2,
            forecast_steps_max=1,
            batch_aggregator=batch_aggregator
        )
        assert len(generator) == exp_gen_len
        batch = generator[0]
        assert len(batch) == batch_aggregator
