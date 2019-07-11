import numpy as np
import pandas as pd
import pytest

import timeserio.ini as ini
from timeserio.data.mock import mock_fit_data, mock_raw_data
from timeserio.batches.single.sequence import (
    SamplingForecastBatchGenerator, SequenceForecastBatchGenerator
)
from numpy.testing import assert_array_equal


class TestForecastBatchGeneratorBase:
    pass


class TestSamplingForecastBatchGenerator:
    def test_get_sequence_values(self):
        n_points, sequence_length = 10, 2
        df = mock_fit_data(periods=n_points, ids=[0])
        gen = SamplingForecastBatchGenerator(
            df=df,
            sequence_length=sequence_length,
        )
        start_indices = np.array([0, 3, 1])
        num_indices = len(start_indices)
        seq_values = gen._get_sequence_values(ini.Columns.id, start_indices)
        assert seq_values.shape == (num_indices, sequence_length)


class TestSequenceForecastBatchGenerator:
    @pytest.mark.parametrize(
        'n_points, seq_length, fc_max, n_sequences_expected', [
            (0, 2, 1, 0), (1, 2, 1, 0), (2, 2, 1, 0), (3, 2, 1, 1),
            (4, 2, 1, 1), (5, 2, 1, 2), (5, 2, 2, 1), (5, 2, 3, 1),
            (5, 2, 4, 0)
        ]
    )
    def test_num_examples(
        self, n_points, seq_length, fc_max, n_sequences_expected
    ):
        df = mock_fit_data(periods=n_points, ids=[0])
        generator = SequenceForecastBatchGenerator(
            df=df,
            sequence_length=seq_length,
            forecast_steps_min=1,
            forecast_steps_max=fc_max,
        )
        assert generator.num_examples == n_sequences_expected

        generator.batch_offset = True
        assert generator.num_examples == max(0, n_sequences_expected - 1)

    @pytest.mark.parametrize(
        'id_column, sequence_columns, last_step_columns, expected_columns', [
            (None, [], [], []),
            (
                None, [ini.Columns.target], [],
                [ini.Columns.target, 'seq_' + ini.Columns.target]
            ),
            (
                ini.Columns.id, [ini.Columns.target], [], [
                    ini.Columns.id, ini.Columns.target,
                    'seq_' + ini.Columns.target
                ]
            ),
        ]
    )
    def test_columns(
        self, id_column, sequence_columns, last_step_columns, expected_columns
    ):
        df = mock_raw_data(periods=10, ids=[0])
        generator = SequenceForecastBatchGenerator(
            df=df,
            batch_size=2,
            sequence_length=2,
            forecast_steps_min=1,
            forecast_steps_max=1,
            id_column=id_column,
            sequence_columns=sequence_columns,
            sequence_prefix='seq_',
            last_step_columns=last_step_columns,
            last_step_prefix='end_of_'
        )
        batch = generator[0]
        batch_columns = {
            col[0] if isinstance(col, tuple) else col
            for col in batch.columns
        }
        assert batch_columns == set(expected_columns)

    @pytest.mark.parametrize(
        'start_time_idx, expected_start_time_idx', [
            (None, 0),
            (0, 0),
            (4, 4),
        ]
    )
    def test_start_time(self, start_time_idx, expected_start_time_idx):
        df = mock_fit_data(periods=1344, ids=[0])
        df = df.sort_values(by=[ini.Columns.datetime])
        if start_time_idx is None:
            start_time = None
        else:
            start_time = df[ini.Columns.datetime][start_time_idx].time()
        expected_start_time = df[ini.Columns.datetime
                                 ][expected_start_time_idx].time()
        generator = SequenceForecastBatchGenerator(
            df=df,
            sequence_length=48,
            sequence_columns=[ini.Columns.datetime],
            batch_offset=False,
            start_time=start_time
        )
        batch = generator[0]
        actual_start_time = batch[f'seq_{ini.Columns.datetime}'][0][0].time()
        assert actual_start_time == expected_start_time

    def test_invalid_start_time(self):
        df = mock_fit_data(periods=1344, ids=[0])
        df = df.sort_values(by=[ini.Columns.datetime])
        start_time = (df[ini.Columns.datetime][0] +
                      pd.Timedelta(1, unit='m')).time()
        generator = SequenceForecastBatchGenerator(
            df=df,
            sequence_length=48,
            sequence_columns=[ini.Columns.datetime],
            start_time=start_time
        )
        with pytest.raises(ValueError):
            generator[0]

    def test_random_offset(self, random):
        df = mock_fit_data(periods=101, ids=[0])
        generator = SequenceForecastBatchGenerator(
            df=df,
            batch_offset=True,
            sequence_length=10,
        )
        with pytest.raises(AssertionError):
            assert_array_equal(generator[0], generator[0])

    @pytest.mark.parametrize(
        'seq_len, period, expected_max_offset', [
            (1, 1, 0),
            (2, 2, 0),
            (10, 1, 9),
            (10, 5, 5),
            (12, 4, 8),
        ]
    )
    def test_random_offset_value_with_period(
        self, random, seq_len, period, expected_max_offset
    ):
        df = mock_fit_data(periods=101, ids=[0])
        generator = SequenceForecastBatchGenerator(
            df=df,
            sequence_length=seq_len,
            batch_offset=True,
            batch_offset_period=period
        )
        offsets = [generator.random_offset_value for _ in range(100)]
        assert min(offsets) == 0
        assert max(offsets) == expected_max_offset
        assert all(offset % period == 0 for offset in offsets)

    @pytest.mark.parametrize('seq_len, period', [
        (10, 3),
        (10, 15),
    ])
    def test_incompatible_period(self, seq_len, period):
        with pytest.raises(ValueError):
            SequenceForecastBatchGenerator(
                df=None,
                sequence_length=seq_len,
                batch_offset=True,
                batch_offset_period=period
            )

    @pytest.mark.parametrize(
        'n_points, seq_length, fc_max, batch_size, n_batches_expected', [
            (2, 2, 1, 1, 0),
            (3, 2, 1, 1, 1),
            (3, 2, 1, 2, 1),
            (5, 2, 1, 1, 2),
            (5, 2, 1, 2, 1),
            (5, 2, 1, None, 1),
        ]
    )
    def test_n_batches(
        self, n_points, seq_length, fc_max, batch_size, n_batches_expected
    ):
        df = mock_fit_data(periods=n_points, ids=[0])
        generator = SequenceForecastBatchGenerator(
            df=df,
            batch_size=batch_size,
            sequence_length=seq_length,
            forecast_steps_min=1,
            forecast_steps_max=fc_max,
        )
        assert len(generator) == n_batches_expected

    @pytest.mark.parametrize(
        'n_points, seq_length, fc_max, batch_size, n_batches_expected', [
            (3, 2, 1, 1, 0),
            (5, 2, 1, None, 1),
        ]
    )
    def test_n_batches_with_offset(
        self, n_points, seq_length, fc_max, batch_size, n_batches_expected
    ):
        df = mock_fit_data(periods=n_points, ids=[0])
        generator = SequenceForecastBatchGenerator(
            df=df,
            batch_size=batch_size,
            sequence_length=seq_length,
            forecast_steps_min=1,
            forecast_steps_max=fc_max,
            batch_offset=True,
        )
        assert len(generator) == n_batches_expected

    @pytest.mark.parametrize(
        'n_points, seq_length, fc_max, batch_size, expected_last_batch_size', [
            (5, 2, 1, 2, 2),
            (7, 2, 1, 2, 1),
        ]
    )
    def test_batch_size(
        self, n_points, seq_length, fc_max, batch_size,
        expected_last_batch_size
    ):
        df = mock_fit_data(periods=n_points, ids=[0])
        generator = SequenceForecastBatchGenerator(
            df=df,
            batch_size=batch_size,
            sequence_columns=[ini.Columns.target],
            last_step_columns=[],
            sequence_length=seq_length,
            forecast_steps_min=1,
            forecast_steps_max=fc_max,
        )
        for batch_idx in range(len(generator) - 1):
            assert len(generator[batch_idx]) == batch_size
        assert len(generator[-1]) == expected_last_batch_size

    def test_single_batch(self):
        df = mock_fit_data(periods=9, ids=[0])
        seq_length = 2
        generator = SequenceForecastBatchGenerator(
            df=df,
            batch_size=4,
            sequence_length=seq_length,
            id_column='id',
            sequence_columns=[ini.Columns.datetime, ini.Columns.target],
            sequence_prefix='seq_',
            last_step_columns=[],
            forecast_steps_min=1,
            forecast_steps_max=1,
        )
        assert len(generator) == 1
        batch = generator[0]
        assert isinstance(batch, pd.DataFrame)
        expected_columns = [
            'id', ini.Columns.datetime, ini.Columns.target,
            f'seq_{ini.Columns.datetime}', f'seq_{ini.Columns.target}'
        ]
        for col in expected_columns:
            assert col in batch
        sequence_columns = [
            f'seq_{ini.Columns.datetime}', f'seq_{ini.Columns.target}'
        ]
        for sequence_column in sequence_columns:
            sequenced = batch[sequence_column]
            assert sequenced.values.shape[1] == seq_length

    def test_single_batch_with_last_step(self):
        df = mock_fit_data(periods=9, ids=[0])
        seq_length = 2
        generator = SequenceForecastBatchGenerator(
            df=df,
            batch_size=4,
            sequence_length=seq_length,
            id_column='id',
            sequence_columns=[ini.Columns.datetime, ini.Columns.target],
            sequence_prefix='seq_',
            last_step_columns=[ini.Columns.datetime],
            last_step_prefix='last_step_',
            forecast_steps_min=1,
            forecast_steps_max=1,
        )
        assert len(generator) == 1
        batch = generator[0]
        assert isinstance(batch, pd.DataFrame)
        expected_columns = [
            'id', ini.Columns.datetime, ini.Columns.target,
            f'seq_{ini.Columns.datetime}', f'seq_{ini.Columns.target}',
            f'last_step_{ini.Columns.datetime}'
        ]
        for col in expected_columns:
            assert col in batch
        sequence_columns = [
            f'seq_{ini.Columns.datetime}', f'seq_{ini.Columns.target}'
        ]
        for sequence_column in sequence_columns:
            values = batch[sequence_column].values
            assert values.shape[1] == seq_length
        last_step_columns = [f'last_step_{ini.Columns.datetime}']
        for column in last_step_columns:
            values = batch[column].values
            assert len(values.shape) == 1
