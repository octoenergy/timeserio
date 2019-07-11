import abc
import functools
from typing import Union

import numpy as np

from ... import ini
from ...preprocessing.pandas import array_to_dataframe
from ..utils import ceiling_division
from .base import BatchGenerator


class ForecastBatchGeneratorBase(BatchGenerator):
    """Generate batches of sequence forecast examples.

    Assume single continuous timeseries.
    """

    def __init__(
        self,
        *,
        df,
        batch_size: Union[None, int] = None,
        sequence_length=2,
        columns=None,
        sequence_columns=None,
        sequence_prefix='seq_',
        last_step_columns=None,
        last_step_prefix='end_of_',
        forecast_steps_min=1,
        forecast_steps_max=1,
    ):
        self.df = df
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.columns = columns
        self.sequence_columns = sequence_columns
        self.sequence_prefix = sequence_prefix
        self.last_step_columns = last_step_columns
        self.last_step_prefix = last_step_prefix

        if sequence_columns:
            if not sequence_prefix:
                raise ValueError('`sequence_prefix` must be non-empty')

        if last_step_columns:
            if not (set(last_step_columns) <= set(sequence_columns)):
                raise ValueError('`last_step_columns` must be a subset of '
                                 '`sequence_columns`')
            if not last_step_prefix:
                raise ValueError('`last_step_prefix` must be non-empty')
            if last_step_prefix == sequence_prefix:
                raise ValueError('`last_step_prefix` must be '
                                 'different from `sequence_prefix`')

        self.forecast_steps_min = forecast_steps_min
        self.forecast_steps_max = forecast_steps_max

    @property
    def num_points(self):
        """Return number of rows in original timeseries."""
        return len(self.df)

    @abc.abstractclassmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def batch_seq_start_indices(self, batch_idx):
        raise NotImplementedError

    def _get_sequence_values(self, column, start_indices):
        values = self.df[column].values
        cols = [
            values[start_indices + s] for s in range(self.sequence_length)
        ]
        seq_values = np.vstack(cols).T
        return seq_values

    def __getitem__(self, batch_idx):
        if not len(self):
            raise IndexError('Batch index out of range: Empty batch generator')
        batch_idx = batch_idx % len(self)
        start_indices = self.batch_seq_start_indices(batch_idx)
        batch_size = len(start_indices)
        end_indices = start_indices + self.sequence_length
        fc_indices = end_indices + np.random.randint(
            self.forecast_steps_min - 1, self.forecast_steps_max, batch_size
        )
        cols = self.columns if self.columns else []
        sequence_columns = self.sequence_columns or []
        last_step_columns = self.last_step_columns or []

        cols = cols + sequence_columns
        batch_df = self.df[cols].iloc[fc_indices].copy()
        batch_df.reset_index(drop=True, inplace=True)
        for column in sequence_columns:
            seq_values = self._get_sequence_values(
                column, start_indices
            )
            seq_col_name = self.sequence_prefix + column
            batch_df = array_to_dataframe(
                seq_values,
                column=seq_col_name,
                df=batch_df
            )
        for column in last_step_columns:
            seq_col_name = self.sequence_prefix + column
            last_step_col_name = self.last_step_prefix + column
            batch_df[last_step_col_name] = batch_df[seq_col_name].iloc[:, -1]

        return batch_df


class SamplingForecastBatchGenerator(ForecastBatchGeneratorBase):
    """Generate batches of sequence forecast examples.

    Assume single continuous timeseries.
    """

    def __init__(
        self,
        *,
        df,
        batch_size: Union[None, int] = None,
        sequence_length=2,
        columns=None,
        sequence_columns=None,
        sequence_prefix='seq_',
        last_step_columns=None,
        last_step_prefix='end_of_',
        forecast_steps_min=1,
        forecast_steps_max=1,
        oversampling=1,
    ):
        super().__init__(
            df=df,
            batch_size=batch_size,
            sequence_length=sequence_length,
            columns=columns,
            sequence_columns=sequence_columns,
            sequence_prefix=sequence_prefix,
            last_step_columns=last_step_columns,
            last_step_prefix=last_step_prefix,
            forecast_steps_min=forecast_steps_min,
            forecast_steps_max=forecast_steps_max,
        )
        self.oversampling = oversampling or 1

    @property
    def num_examples(self):
        """Return number of examples to yield in one epoch."""
        num_examples = (
            self.num_points -
            self.forecast_steps_max
        ) // self.sequence_length * self.oversampling
        return max(0, num_examples)

    @property
    def _eff_batch_size(self):
        return self.batch_size or self.num_examples

    def __len__(self):
        return ceiling_division(self.num_examples, self._eff_batch_size)

    def batch_seq_start_indices(self, batch_idx):
        start_indices = np.random.randint(
            low=0,
            high=(
                self.num_points + 1 -
                self.sequence_length -
                self.forecast_steps_max
            ),
            size=self._eff_batch_size
        )
        return start_indices


class SequenceForecastBatchGenerator(ForecastBatchGeneratorBase):
    """Generate batches of sequence forecast examples.

    Assume single continuous timeseries.
    """

    def __init__(
        self,
        *,
        df,
        batch_size: Union[None, int] = None,
        sequence_length=2,
        id_column=None,
        sequence_columns=[ini.Columns.datetime, ini.Columns.target],
        sequence_prefix='seq_',
        last_step_columns=[ini.Columns.datetime],
        last_step_prefix='end_of_',
        forecast_steps_min=1,
        forecast_steps_max=1,
        batch_offset=False,
        batch_offset_period=1,
        dt_column=ini.Columns.datetime,
        start_time=None,
    ):
        super().__init__(
            df=df,
            batch_size=batch_size,
            sequence_length=sequence_length,
            columns=[id_column] if id_column else None,
            sequence_columns=sequence_columns,
            sequence_prefix=sequence_prefix,
            last_step_columns=last_step_columns,
            last_step_prefix=last_step_prefix,
            forecast_steps_min=forecast_steps_min,
            forecast_steps_max=forecast_steps_max,
        )
        self.batch_offset = batch_offset
        self.batch_offset_period = batch_offset_period
        self.dt_column = dt_column
        self.start_time = start_time
        if self.sequence_length % self.batch_offset_period != 0:
            raise ValueError(f'sequence_length not divisible'
                             f' by batch_offset_period')

    @property  # type: ignore
    @functools.lru_cache(None)
    def first_index(self):
        if self.start_time is None:
            return 0
        times = self.df[self.dt_column].dt.time.values
        first_idx = np.argmax(times == self.start_time)
        if not first_idx and times[0] != self.start_time:
            raise ValueError(f'Start time {self.start_time} not found in df')
        return first_idx

    @property
    def num_examples(self):
        """Return number of examples to yield in one epoch."""
        num_examples = (
            self.num_points -
            self.forecast_steps_max -
            self.first_index
        ) // self.sequence_length
        if self.batch_offset:
            num_examples -= 1  # to deal with random offset
        return max(0, num_examples)

    @property
    def _eff_batch_size(self):
        return self.batch_size or self.num_examples

    def __len__(self):
        return ceiling_division(self.num_examples, self._eff_batch_size)

    def batch_seq_start_indices(self, batch_idx):
        start_indices = np.arange(
            self.sequence_length * self._eff_batch_size * batch_idx,
            self.sequence_length *
            min(self._eff_batch_size * (batch_idx + 1), self.num_examples),
            self.sequence_length
        )
        return self.first_index + start_indices + self.random_offset_value

    @property
    def random_offset_value(self):
        if not self.batch_offset:
            return 0
        offset = np.random.randint(
            low=0,
            high=ceiling_division(
                self.sequence_length,
                self.batch_offset_period
            )
        ) * self.batch_offset_period
        return offset

    def _get_sequence_values(self, column, start_indices):
        """Efficient implementation for consecutive sequences."""
        seq_values = self.df[column].values[
            start_indices[0]:start_indices[-1] + self.sequence_length
        ]
        seq_values = seq_values.reshape((-1, self.sequence_length))
        return seq_values
