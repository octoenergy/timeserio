"""Generate batches from pandas DataFrame.

Experimental.
"""
from .base import ChunkedBatchGenerator
from ... import ini
# from ..single import row as single_row
from ..single import sequence as single_sequence


class SequenceForecastBatchGenerator(ChunkedBatchGenerator):
    """Generate batches from individual partitions of a Dask DataFrame."""

    def __init__(
        self,
        *,
        ddf,
        batch_size=2,
        sequence_length=2,
        id_column=ini.Columns.id,
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
        batch_aggregator=1
    ):
        super().__init__()
        self.ddf = ddf
        self.subgens = []  # to enable setters
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.id_column = id_column
        self.sequence_columns = sequence_columns
        self.sequence_prefix = sequence_prefix
        self.last_step_columns = last_step_columns
        self.last_step_prefix = last_step_prefix
        self.forecast_steps_min = forecast_steps_min
        self.forecast_steps_max = forecast_steps_max
        self.batch_offset = batch_offset
        self.batch_offset_period = batch_offset_period
        self.dt_column = dt_column
        self.start_time = start_time
        self.batch_aggregator = batch_aggregator

    @property
    def chunks(self):
        return list(self.ddf.partitions)

    def make_subgen(self, chunk):
        partition = chunk
        subgen = single_sequence.SequenceForecastBatchGenerator(
            df=partition.compute(),
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            id_column=self.id_column,
            sequence_columns=self.sequence_columns,
            sequence_prefix=self.sequence_prefix,
            last_step_columns=self.last_step_columns,
            last_step_prefix=self.last_step_prefix,
            forecast_steps_min=self.forecast_steps_min,
            forecast_steps_max=self.forecast_steps_max,
            batch_offset=self.batch_offset,
            batch_offset_period=self.batch_offset_period,
            dt_column=self.dt_column,
            start_time=self.start_time,
        )
        return subgen
